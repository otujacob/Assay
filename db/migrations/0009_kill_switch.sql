-- Assay 0009: the kill switch (PRD 12.4). While it is engaged for a tenant, every case goes to human review whatever the model
-- says. Append-only like everything else: engaging and releasing are rows, the latest row is the state, and the history of who
-- did what, when and why is never rewritten. Releasing needs a reason too.

CREATE TABLE kill_switch_events (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,                 -- a person, or system:monitor
  schema_version  text,
  kind            text NOT NULL CHECK (kind IN ('engaged', 'released')),
  reason          text NOT NULL CHECK (length(btrim(reason)) > 0),
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL
);

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['kill_switch_events']
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
