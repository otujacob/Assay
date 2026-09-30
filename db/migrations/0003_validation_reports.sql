-- Assay 0003: validation reports stored with lineage (PRD 25.2, FR-39).
-- A report is linked to the model bundle it evaluated and the dataset it was evaluated on, so the
-- evidence for enabling Calibrated mode or setting thresholds is auditable.

CREATE TABLE validation_reports (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,
  schema_version  text,
  bundle_id       text NOT NULL,
  dataset_id      text NOT NULL,
  mode            text NOT NULL,
  n_cases         integer NOT NULL,
  measures        jsonb NOT NULL,
  baselines       jsonb NOT NULL,
  ablations       jsonb NOT NULL,
  stress_results  jsonb,
  pass_criteria   jsonb NOT NULL,
  limitations     jsonb NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  FOREIGN KEY (tenant_id, bundle_id) REFERENCES model_bundles (tenant_id, bundle_id)
);

DO $$
DECLARE
  t text := 'validation_reports';
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

-- Replace the chain verifier's hard-coded table list (0001, 0002) with a check that the named
-- relation is a table in the current schema with a row_hash column, so new chained tables need no
-- edit here. The name is still quoted with %I, and only real chained tables are accepted.
CREATE OR REPLACE FUNCTION assay_verify_chain(p_table text, p_tenant text) RETURNS bigint
LANGUAGE plpgsql STABLE AS $$
DECLARE
  bad bigint;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_attribute a
    JOIN pg_class c ON c.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relname = p_table AND n.nspname = current_schema() AND c.relkind = 'r'
      AND a.attname = 'row_hash' AND NOT a.attisdropped
  ) THEN
    RAISE EXCEPTION 'unknown chained table %', p_table;
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
