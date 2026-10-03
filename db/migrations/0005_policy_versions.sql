-- Assay 0005: versioned, effective-dated decision policies that need a second person's approval
-- (PRD 10.4, 25.2; FR-21). Append-only like every decision table: a proposal is one row in
-- policy_versions and its approval is a LATER row in policy_approvals, so nothing is updated.
-- The database itself refuses an approval by the proposer and any automation above level 0 (FR-23).

CREATE TABLE policy_versions (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq             bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id       text NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  created_by      text NOT NULL,                 -- the proposer
  schema_version  text,
  version         text NOT NULL,                 -- policy-1, policy-2, ...
  payload         jsonb NOT NULL,
  effective_from  timestamptz NOT NULL,
  payload_hash    text NOT NULL,
  prev_hash       text NOT NULL,
  row_hash        text NOT NULL,
  UNIQUE (tenant_id, version),
  UNIQUE (tenant_id, id, created_by),
  CHECK ((payload ->> 'automation_level')::integer = 0)  -- MVP recommend-only (FR-23)
);

CREATE TABLE policy_approvals (
  id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  seq                bigint GENERATED ALWAYS AS IDENTITY,
  tenant_id          text NOT NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  created_by         text NOT NULL,              -- the approver
  schema_version     text,
  policy_version_id  uuid NOT NULL,
  proposed_by        text NOT NULL,
  approved_by        text NOT NULL,
  approved_at        timestamptz NOT NULL,
  payload_hash       text NOT NULL,
  prev_hash          text NOT NULL,
  row_hash           text NOT NULL,
  UNIQUE (tenant_id, policy_version_id),         -- approved once
  -- proposed_by must be the real proposer: the key includes created_by, so it cannot be spoofed
  FOREIGN KEY (tenant_id, policy_version_id, proposed_by)
    REFERENCES policy_versions (tenant_id, id, created_by),
  CHECK (approved_by <> proposed_by)             -- separation of duties (FR-21)
);

DO $$
DECLARE
  t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['policy_versions', 'policy_approvals']
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
