-- Assay 0008: analysts can flag an entity-graph relationship as wrong (PRD 13.7). A flag is one append-only row; it
-- down-weights that relationship from the time it was recorded and deletes nothing, so the graph as it was at any past
-- decision can still be reproduced. Only the flagger's identity (created_by) and the reason are kept; the graph itself
-- is rebuilt from stored transactions, so there is no second copy of relationship data to fall out of step.

CREATE TABLE graph_edge_flags (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,                 -- the analyst who flagged it
  schema_version  text,
  rel             text NOT NULL CHECK (rel IN ('USED_DEVICE', 'ACCESSED_FROM', 'PAID', 'TRANSACTED_AT')),
  src             text NOT NULL,                 -- the customer
  dst             text NOT NULL,                 -- the device, IP address, beneficiary or merchant
  reason          text NOT NULL CHECK (length(reason) > 0),
  flagged_at      timestamptz NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL
);

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['graph_edge_flags']
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
