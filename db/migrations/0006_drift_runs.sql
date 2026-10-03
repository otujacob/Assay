-- Assay 0006: population drift runs (PRD 9.1, 9.3, 26.4; FR-17).
-- Each run compares the model's reference sample with the most recent scored feature vectors and
-- stores the per-feature drift. Scoring reads the latest run, so the signal survives a restart and
-- is shared by every API worker; an alarm is a row with alarm = true, never an update.

CREATE TABLE drift_runs (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,
  schema_version  text,
  bundle_id       text NOT NULL,
  n_reference     integer NOT NULL CHECK (n_reference > 0),
  n_window        integer NOT NULL CHECK (n_window > 0),
  window_start    timestamptz NOT NULL,
  window_end      timestamptz NOT NULL,
  feature_names   jsonb NOT NULL,
  drift           jsonb NOT NULL,                 -- scaled KS statistic per feature, each in [0, 1]
  max_drift       double precision NOT NULL CHECK (max_drift BETWEEN 0 AND 1),
  alarm_level     double precision NOT NULL,      -- the level in force for this run (a parameter)
  alarm           boolean NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  FOREIGN KEY (tenant_id, bundle_id) REFERENCES model_bundles (tenant_id, bundle_id),
  CHECK (alarm = (max_drift >= alarm_level))
);

DO $$
DECLARE
  t text := 'drift_runs';
BEGIN
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
END $$;
