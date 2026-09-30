-- Assay 0004: human review and feedback capture (PRD 10.5, 11, 25.2; FR-24 to FR-30).
-- Case status is DERIVED from these append-only rows (open, escalated, decided, conflicted,
-- adjudicated); nothing is updated in place. Every analyst decision is kept (PRD 8.4).

CREATE TABLE review_cases (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  txn_id             text NOT NULL,
  first_decision_id  uuid NOT NULL REFERENCES policy_decisions (id),
  queue              text NOT NULL,
  blind              boolean NOT NULL,          -- decided once, at enqueue, and recorded (FR-28)
  sla_minutes        integer NOT NULL CHECK (sla_minutes > 0),
  enqueued_at        timestamptz NOT NULL,
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL,
  UNIQUE (tenant_id, txn_id),
  FOREIGN KEY (tenant_id, txn_id) REFERENCES transactions (tenant_id, txn_id)
);

CREATE TABLE analyst_actions (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  txn_id             text NOT NULL,
  policy_decision_id uuid NOT NULL REFERENCES policy_decisions (id),
  analyst_pid        text NOT NULL,              -- pseudonymous
  role               text NOT NULL,
  action             text NOT NULL CHECK (action IN
                       ('approve', 'block', 'escalate', 'request_review', 'override', 'unsure', 'adjudicate')),
  final_decision     text CHECK (final_decision IN ('approve', 'block')),
  reason_code        text,
  confidence         double precision CHECK (confidence BETWEEN 0 AND 1),
  evidence_checklist jsonb NOT NULL DEFAULT '{}',
  notes              text,
  display_state      jsonb NOT NULL,             -- what the analyst was shown, recorded by the SERVER
  blind_flag         boolean NOT NULL,
  seconds_to_decision double precision,
  action_time        timestamptz NOT NULL,
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL,
  -- an override or a final decision that differs from the recommendation needs a reason (FR-26)
  CHECK (action <> 'override' OR reason_code IS NOT NULL),
  CHECK (action NOT IN ('override', 'adjudicate') OR final_decision IS NOT NULL)
);
CREATE INDEX analyst_actions_txn ON analyst_actions (tenant_id, txn_id, seq);

CREATE TABLE feedback_records (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,
  schema_version     text,
  analyst_action_id  uuid NOT NULL REFERENCES analyst_actions (id),
  aas                double precision,           -- null until the analyst has enough matured cases
  fcs                double precision NOT NULL,
  lvs                double precision NOT NULL,
  crs                double precision,
  fqs                double precision NOT NULL,
  formula_version    text NOT NULL,
  disposition        text NOT NULL CHECK (disposition IN ('accept', 'reject', 'defer')),
  disposition_reason text NOT NULL,
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL
);

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['review_cases', 'analyst_actions', 'feedback_records']
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
