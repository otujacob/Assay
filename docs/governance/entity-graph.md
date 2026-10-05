# The entity graph (PRD 13, V1)

What it is, what it is for, what it is not for, and what is not done. The numbers named here are working defaults, not
validated values (PRD 13.3).

## What it is

A temporal graph of the pseudonymised identifiers already on every transaction: customer, account, device, IP address,
merchant and beneficiary. It is **rebuilt from the stored transactions, outcomes and analyst flags** (incrementally as they
arrive), so there is no second copy of relationship data that could disagree with the append-only tables.

- **Observed relationships:** customer USED_DEVICE device, ACCESSED_FROM IP address, PAID beneficiary, TRANSACTED_AT merchant,
  and customer OWNS account.
- **Derived links:** two customers who used the same device, IP address or beneficiary are linked (SHARES_DEVICE, SHARES_IP,
  SHARES_BENEFICIARY). A merchant is recorded but does not link customers: two people who bought from the same large merchant are
  barely connected.
- **Bitemporal (PRD 13.4):** each observation has the time it was true (the transaction's event time) and the time Assay learned of
  it. A query names both and sees only what satisfies both, so the view of a past decision is the graph as it was then, whatever
  arrived later.
- **Confidence (PRD 13.3):** `(1 - exp(-k n)) x exp(-lambda x age) x 1 / (1 + ln(1 + degree))`: evidence, recency, and specificity (a
  node used by many customers is a weak link). It says how sure the graph is that a relationship is real and meaningful. It is
  **not a fraud probability** and is never labelled as one.
- **Expiry:** an edge whose confidence falls below a floor is treated as expired at that time. Nothing is deleted; a later
  sighting revives it.
- **Hubs:** a node used by more than 60 customers does not link them to each other. A public Wi-Fi address would otherwise connect
  strangers.
- **Analyst flags (PRD 13.7):** an analyst can flag a relationship as wrong, with a reason. It is stored (`graph_edge_flags`,
  append-only, tenant-isolated) and down-weights that relationship (to 20%) from the time it was recorded. It does not change the
  view of any decision made before the flag.

## What it is for

**Investigator context.** On a case, "Show linked entities" lists the other customers connected to its device, IP address and
beneficiary, how sure the graph is of each link, whether a connected customer has a confirmed fraud (as known at the time), the size
of the connected group, and a warning when the view leans on weak links. `GET /v1/decisions/{id}/graph`. Same access rules as the
explanation: analysts (not on a blind case) and auditors; reading is audited. `POST /v1/graph/edges/flag` flags a link.

## What it is not for

- **Not a model input.** The pre-set test of whether graph features beat the same data used as flat features was **not supported as
  specified**: they help where rings exist, and cost a little on other fraud
  ([docs/validation/README.md](../validation/README.md)). The PRD's rule for that outcome is that the graph stays investigator
  context, and it does. Nothing in scoring reads it.
- **Not evidence of guilt.** Families share devices; strangers share networks. The view says so every time, and so does the API.
- **Not for customers.** The note on every response says it must not be shared with one.
- **Not the analyst graph.** The PRD's restricted analyst nodes (for feedback-integrity checks) are not built.

## Features, and the flat baseline they were compared with

`assay/graph/features.py`: confidence-weighted counts of other customers per device, IP and beneficiary; proximity to a customer with a
confirmed fraud known at the time (two hops, the second halved); the size, density and fraud-linked share of the customer's connected
group; a ring score (dense, fraud-linked, recently formed); a new-link flag (first use of a structure others already use); and the
weakest confidence behind the features. Every feature of a transaction is computed from the graph **before** that transaction, and
no outcome counts until it matured; tests check both. Transactions with the same timestamp do not see each other.

## Known limits

- **Communities are connected groups, not Louvain or Leiden.** Customers reachable within two hops through links above a confidence
  threshold. Simple, deterministic and dependency-free, and able to merge two rings joined by one strong link.
- **Scale is untested.** The investigator view rebuilds from stored transactions into an in-process cache. On the synthetic data the
  feature builder runs at about 1.6 ms per transaction on one core, and the cache grows with the number of transactions, so a
  multi-process deployment holds one graph per process. OPD-14 (graph technology) asks for a latency test at pilot volume; none has
  been run. A dedicated graph store is the alternative if this does not hold.
- **Identity attributes are not used** (phone, address). Synthetic-identity detection (PRD 13.6) needs them, and whether they may
  lawfully be used is a legal question (OPD-14).
- **Mule-account and ring detection as products** (V2) are not built; this is the store and the investigator view.
- **No cross-tenant edges**, by design (PRD 15.3).
