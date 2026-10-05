-- Assay 0007: candidate models and their lifecycle (PRD 12): candidate -> validated -> shadow -> approved
-- -> canary -> champion, with rollback. Append-only like every decision table: a candidate is one row, and
-- every step after it is a LATER row in model_lifecycle_events, so nothing is updated and the order of
-- events is the history. The database itself refuses an approval by the person who created the candidate
-- (separation of duties, like policy approval) and refuses to record a model's promotion without an approval.

CREATE TABLE model_candidates (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,                 -- who created the candidate
  schema_version  text,
  candidate_id    text NOT NULL,                 -- the candidate's bundle id
  base_bundle_id  text NOT NULL,                 -- the champion it was compared with
  artefact_path   text,                          -- where the signed bundle lives
  pool_summary    jsonb NOT NULL,                -- what feedback was accepted, rejected, deferred
  training        jsonb NOT NULL,                -- dataset, labels used, windows
  gates           jsonb NOT NULL,                -- the validation gate report
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  UNIQUE (tenant_id, candidate_id),
  UNIQUE (tenant_id, candidate_id, created_by)
);

CREATE TABLE model_lifecycle_events (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,                 -- who did this step
  schema_version  text,
  candidate_id    text NOT NULL,
  created_by_candidate text NOT NULL,            -- the candidate's creator, copied so the database can compare
  kind            text NOT NULL,
  detail          jsonb NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  -- the creator recorded here must be the real creator: the key includes created_by on the candidate row
  FOREIGN KEY (tenant_id, candidate_id, created_by_candidate)
    REFERENCES model_candidates (tenant_id, candidate_id, created_by),
  CHECK (kind IN ('validated', 'validation_failed', 'shadow_started', 'approved', 'canary_started',
                  'promoted', 'rolled_back', 'rejected')),
  CHECK (kind <> 'approved' OR created_by <> created_by_candidate),     -- separation of duties
  CHECK (kind <> 'promoted' OR detail ? 'approval_event')               -- no promotion without an approval
);

-- What a candidate would have said on live traffic while it takes no action (shadow mode, PRD 12.2).
CREATE TABLE shadow_scores (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,
  schema_version  text,
  txn_id          text NOT NULL,
  candidate_id    text NOT NULL,
  champion_id     text NOT NULL,
  candidate_risk  double precision NOT NULL,
  champion_risk   double precision NOT NULL,
  candidate_call  boolean NOT NULL,              -- would it have been flagged
  champion_call   boolean NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  UNIQUE (tenant_id, txn_id, candidate_id)
);

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['model_candidates', 'model_lifecycle_events', 'shadow_scores']
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
