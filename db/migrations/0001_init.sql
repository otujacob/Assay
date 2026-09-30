-- Assay 0001: ingestion + lineage core (PRD 25, FR-01..05, FR-32, FR-40).
-- Guarantees enforced in the database, not only in application code:
--   * tenant isolation: row-level security keyed on current_setting('app.tenant_id')
--   * append-only: no UPDATE/DELETE/TRUNCATE (trigger), and the app role has no such grants
--   * tamper evidence: per-tenant hash chain, row_hash = sha256(prev|id|tenant|payload_hash)
-- The application sets the tenant per transaction: SELECT set_config('app.tenant_id', $1, true);

-- Requires PostgreSQL 13+ (built-in gen_random_uuid(), sha256()).

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'assay_app') THEN
    CREATE ROLE assay_app NOLOGIN NOBYPASSRLS;
  END IF;
END $$;

CREATE TABLE ingestion_events (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  event_id           text NOT NULL,
  kind               text NOT NULL CHECK (kind IN ('transaction', 'outcome')),
  source_id          text NOT NULL,
  event_time         timestamptz,
  recorded_at        timestamptz NOT NULL,
  event_payload_hash text NOT NULL,
  validation_result  text NOT NULL CHECK (validation_result IN ('accepted', 'rejected', 'quarantined')),
  reasons            jsonb NOT NULL DEFAULT '[]',
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL
);
-- Idempotency: one accepted/quarantined record per event_id per tenant.
CREATE UNIQUE INDEX ingestion_events_idem
  ON ingestion_events (tenant_id, event_id)
  WHERE validation_result <> 'rejected' AND event_id <> '';

CREATE TABLE transactions (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text NOT NULL,
  txn_id             text NOT NULL,
  customer_pid       text NOT NULL,
  account_pid        text NOT NULL,
  beneficiary_pid    text,
  merchant_id        text,
  amount             numeric(18, 2) NOT NULL CHECK (amount > 0),
  currency           char(3) NOT NULL,
  channel            text NOT NULL,
  device_hash        text,
  ip_hash            text,
  country            char(2),
  event_time         timestamptz NOT NULL,
  recorded_at        timestamptz NOT NULL,
  ingestion_event_id uuid NOT NULL REFERENCES ingestion_events (id),
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL,
  UNIQUE (tenant_id, txn_id)
);

CREATE TABLE outcomes (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text NOT NULL,
  txn_id             text NOT NULL,
  outcome_type       text NOT NULL CHECK (outcome_type IN
                       ('confirmed_fraud', 'confirmed_legitimate', 'chargeback', 'dispute_closed')),
  source             text NOT NULL,
  event_time         timestamptz NOT NULL,
  maturity_state     text NOT NULL CHECK (maturity_state IN ('pending', 'matured')),
  matured_at         timestamptz,
  recorded_at        timestamptz NOT NULL,
  ingestion_event_id uuid NOT NULL REFERENCES ingestion_events (id),
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL,
  -- A maturity change is a new row, never an update: latest row per (tenant, txn) wins in views.
  FOREIGN KEY (tenant_id, txn_id) REFERENCES transactions (tenant_id, txn_id)
);

CREATE TABLE quarantine (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  txn_id             text NOT NULL,
  outcome_type       text NOT NULL,
  source             text NOT NULL,
  event_time         timestamptz NOT NULL,
  reason             text NOT NULL,
  recorded_at        timestamptz NOT NULL,
  ingestion_event_id uuid NOT NULL REFERENCES ingestion_events (id),
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL
);

CREATE TABLE audit_log (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq          bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id    text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  created_by   text NOT NULL,
  schema_version text,
  actor        text NOT NULL,
  action       text NOT NULL,
  object       text NOT NULL,
  result       text NOT NULL,
  time         timestamptz NOT NULL,
  payload_hash text NOT NULL,
  prev_hash    text NOT NULL,
  row_hash     text NOT NULL
);

-- Hash chain: serialised per (table, tenant) by an advisory lock held to end of transaction.
CREATE FUNCTION assay_chain_row() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
  prev text;
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended(TG_TABLE_NAME || ':' || NEW.tenant_id, 0));
  EXECUTE format('SELECT row_hash FROM %I.%I WHERE tenant_id = $1 ORDER BY seq DESC LIMIT 1',
                 TG_TABLE_SCHEMA, TG_TABLE_NAME)
    INTO prev USING NEW.tenant_id;
  NEW.prev_hash := coalesce(prev, repeat('0', 64));
  NEW.row_hash := encode(sha256(convert_to(
      NEW.prev_hash || '|' || NEW.id::text || '|' || NEW.tenant_id || '|' || NEW.payload_hash,
      'UTF8')), 'hex');
  RETURN NEW;
END $$;

CREATE FUNCTION assay_block_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME USING ERRCODE = 'insufficient_privilege';
END $$;

-- Returns the seq of the first row whose chain link or hash is wrong, or NULL if intact (FR-32).
CREATE FUNCTION assay_verify_chain(p_table text, p_tenant text) RETURNS bigint
LANGUAGE plpgsql STABLE AS $$
DECLARE
  bad bigint;
BEGIN
  IF p_table NOT IN ('ingestion_events', 'transactions', 'outcomes', 'quarantine', 'audit_log') THEN
    RAISE EXCEPTION 'unknown table %', p_table;
  END IF;
  EXECUTE format($q$
    SELECT seq FROM (
      SELECT seq, id, payload_hash, prev_hash, row_hash,
             lag(row_hash) OVER (ORDER BY seq) AS expected_prev
      FROM %I WHERE tenant_id = $1
    ) t
    WHERE prev_hash IS DISTINCT FROM coalesce(expected_prev, repeat('0', 64))
       OR row_hash <> encode(sha256(convert_to(
            prev_hash || '|' || id::text || '|' || $1 || '|' || payload_hash, 'UTF8')), 'hex')
    ORDER BY seq LIMIT 1
  $q$, p_table) INTO bad USING p_tenant;
  RETURN bad;
END $$;

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['ingestion_events', 'transactions', 'outcomes', 'quarantine', 'audit_log']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format($p$CREATE POLICY tenant_isolation ON %I
        USING (tenant_id = current_setting('app.tenant_id', true))
        WITH CHECK (tenant_id = current_setting('app.tenant_id', true))$p$, t);
    EXECUTE format('CREATE TRIGGER %I BEFORE INSERT ON %I FOR EACH ROW EXECUTE FUNCTION assay_chain_row()',
                   t || '_chain', t);
    EXECUTE format('CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION assay_block_mutation()',
                   t || '_no_mutate', t);
    EXECUTE format('CREATE TRIGGER %I BEFORE TRUNCATE ON %I FOR EACH STATEMENT EXECUTE FUNCTION assay_block_mutation()',
                   t || '_no_truncate', t);
    EXECUTE format('REVOKE ALL ON %I FROM assay_app', t);
    EXECUTE format('GRANT SELECT, INSERT ON %I TO assay_app', t);
  END LOOP;
END $$;
