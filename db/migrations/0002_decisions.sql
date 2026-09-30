-- Assay 0002: decision lineage objects (PRD 14.1, 25.2; FR-20, FR-31, FR-33).
-- Same guarantees as 0001 (tenant RLS, append-only, per-tenant hash chain). A correction or a
-- late-arriving component is a NEW row that references the original: trust_assessments carry a
-- version_no, and a later policy_decision references the newer assessment.

CREATE TABLE model_bundles (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,
  schema_version  text,
  bundle_id       text NOT NULL,
  model_version   text,
  dataset_id      text,
  code_commit     text,
  status          text NOT NULL,
  artifact_sha256 text NOT NULL,
  manifest        jsonb NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  UNIQUE (tenant_id, bundle_id)
);

CREATE TABLE feature_vectors (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                  bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id            text NOT NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  created_by           text NOT NULL,
  schema_version       text,
  txn_id               text NOT NULL,
  feature_set_version  text NOT NULL,
  definition_versions  jsonb NOT NULL,
  as_of_time           timestamptz NOT NULL,
  feature_names        jsonb NOT NULL,
  feature_values       jsonb NOT NULL,
  graph_snapshot_id    text,          -- empty in the MVP (no graph store)
  payload_hash         text NOT NULL,
  prev_hash            text NOT NULL,
  row_hash             text NOT NULL,
  UNIQUE (tenant_id, txn_id),  -- a transaction is scored once; re-scoring is a replay, not a new vector
  FOREIGN KEY (tenant_id, txn_id) REFERENCES transactions (tenant_id, txn_id)
);

CREATE TABLE predictions (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                   bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id             text NOT NULL,
  created_at            timestamptz NOT NULL DEFAULT now(),
  created_by            text NOT NULL,
  schema_version        text,
  txn_id                text NOT NULL,
  feature_vector_id     uuid NOT NULL REFERENCES feature_vectors (id),
  bundle_id             text NOT NULL,
  raw_score             double precision NOT NULL,
  calibrated_risk       double precision NOT NULL CHECK (calibrated_risk BETWEEN 0 AND 1),
  member_spread         double precision NOT NULL,
  distance_to_threshold double precision NOT NULL,
  payload_hash          text NOT NULL,
  prev_hash             text NOT NULL,
  row_hash              text NOT NULL,
  UNIQUE (tenant_id, txn_id),
  FOREIGN KEY (tenant_id, bundle_id) REFERENCES model_bundles (tenant_id, bundle_id)
);

CREATE TABLE explanations (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  prediction_id      uuid NOT NULL REFERENCES predictions (id),
  method             text NOT NULL,
  params             jsonb NOT NULL,
  background_version text NOT NULL,
  seed               integer NOT NULL,
  attributions       jsonb NOT NULL,
  stability          double precision,
  sensitivity        double precision,
  faithfulness       double precision,
  reproducible       boolean NOT NULL,
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL
);

CREATE TABLE trust_assessments (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,
  schema_version  text,
  prediction_id   uuid NOT NULL REFERENCES predictions (id),
  version_no      integer NOT NULL CHECK (version_no >= 1),
  mode            text NOT NULL,
  state           text NOT NULL CHECK (state IN ('high', 'moderate', 'low', 'insufficient_evidence')),
  ti              double precision,
  ti_low          double precision,
  ti_high         double precision,
  reason_codes    jsonb NOT NULL DEFAULT '[]',
  components      jsonb NOT NULL,
  weights_version text NOT NULL,
  weights_used    jsonb NOT NULL,
  evidence        jsonb NOT NULL,   -- inputs needed to replay: drift vector, dq inputs, explain flag
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  UNIQUE (tenant_id, prediction_id, version_no),
  -- "never manufacture a confident score": a state with no score has no number, and vice versa
  CHECK ((state = 'insufficient_evidence') = (ti IS NULL)),
  CHECK (state <> 'insufficient_evidence' OR jsonb_array_length(reason_codes) > 0)
);

CREATE TABLE policy_decisions (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                  bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id            text NOT NULL,
  created_at           timestamptz NOT NULL DEFAULT now(),
  created_by           text NOT NULL,
  schema_version       text,
  txn_id               text NOT NULL,
  trust_assessment_id  uuid NOT NULL REFERENCES trust_assessments (id),
  policy_version       text NOT NULL,
  risk_band            text NOT NULL,
  gate                 text NOT NULL,
  queue                text,
  recommended_action   text NOT NULL,
  automation_level     integer NOT NULL CHECK (automation_level = 0),  -- MVP: recommend only (FR-23)
  payload_hash         text NOT NULL,
  prev_hash            text NOT NULL,
  row_hash             text NOT NULL
);

CREATE INDEX policy_decisions_txn ON policy_decisions (tenant_id, txn_id, seq);
-- Point-in-time feature history: a customer's past transactions, and recent payments to a beneficiary.
CREATE INDEX transactions_customer ON transactions (tenant_id, customer_pid, event_time);
CREATE INDEX transactions_beneficiary ON transactions (tenant_id, beneficiary_pid, event_time)
  WHERE beneficiary_pid IS NOT NULL;

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['model_bundles', 'feature_vectors', 'predictions', 'explanations',
                           'trust_assessments', 'policy_decisions']
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

-- assay_verify_chain (0001) accepts only the 0001 tables; widen its whitelist.
CREATE OR REPLACE FUNCTION assay_verify_chain(p_table text, p_tenant text) RETURNS bigint
LANGUAGE plpgsql STABLE AS $$
DECLARE
  bad bigint;
BEGIN
  IF p_table NOT IN ('ingestion_events', 'transactions', 'outcomes', 'quarantine', 'audit_log',
                     'model_bundles', 'feature_vectors', 'predictions', 'explanations',
                     'trust_assessments', 'policy_decisions') THEN
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
