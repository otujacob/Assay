# How a model changes (PRD 11 and 12)

Assay never retrains or promotes a model by itself. This page says what the steps are, who takes them, what the
database enforces, and what is **not** done yet. All thresholds named here are working defaults, not validated
values (OPD-11, OPD-12).

## The path

1. **Feedback pool.** `python -m assay.learning pool --tenant T` (or the Model Updates panel) shows what the stored
   analyst decisions would contribute: accepted, rejected, deferred, and why. Each case is re-assessed from the raw
   actions every time, using each analyst's accuracy on matured outcomes as known now.
2. **Create a candidate.** `python -m assay.learning create ...` trains a candidate on verified outcomes plus the
   accepted pool, signs and stores it, scores it against the champion, and records the gate report. This is a job,
   not an API call, because it trains a model. The person who runs it is recorded as the creator and **cannot approve
   the candidate later**.
3. **Gates.** See the table below. A candidate with a failed gate stops there.
4. **Shadow.** An administrator or approver starts it. The candidate scores live traffic and takes no action; its
   output is stored and used by nothing else. One candidate at a time per tenant.
5. **Approval.** A different person (an approver) approves it, and only when the shadow evidence is sufficient (G7), they
   give a reason, they confirm they reviewed the results by segment (G6), and they waive, by name, every gate that
   could not be judged. The waiver is stored with the approval.
6. **Canary.** The approver sets a share of traffic (at most 50%) that the candidate decides. A transaction always
   takes the same path (a hash of its id), so a replay agrees.
7. **Promotion.** Needs a canary that decided enough cases. The previous champion stays deployable.
8. **Rollback.** Restores the previous champion at any point after the canary starts. A reason is required. A rolled-back
   or rejected candidate cannot be revived: create a new one.

Every step is an append-only row (`model_candidates`, `model_lifecycle_events`, `shadow_scores`). The history is the
order of the rows, and nothing is updated. Each worker reads the stored events at each scoring call, so all of them
agree on which model decides, whichever one handled the request.

## What the database refuses, in addition to the service

- an approval by the person who created the candidate (a check on the row, and a foreign key stops a caller lying
  about who the creator was);
- a promotion that does not cite an approval event;
- any update or delete of these rows, and tenant mixing (row-level security, as for every other table).

## Gates (`assay/learning/gates.py`)

| Gate | What is checked | Can it say "could not judge"? |
|---|---|---|
| G1 Data | single-tenant, manifest complete, leak check passed at training, matured labels only, class balance documented, analyst labels only from the accepted pool | no |
| G2 Performance | PR-AUC, precision and recall at each model's own operating point, overall and by channel, not worse than the champion beyond tolerance (2 points; 5 points of recall per channel). A channel with fewer than 15 verified frauds is shown but not judged | no |
| G3 Calibration | calibration error at most 0.05 and within 0.01 of the champion's | no |
| G4 Trust | high-trust error rate and error discrimination not degraded | **yes**, with fewer than 50 High-trust cases |
| G5 Explanation | median explanation reliability not degraded | **yes**, with fewer than 30 explained cases |
| G6 Segments | the table by channel, for a person to review. Assay makes no fairness claim | never passes by itself |
| G7 Shadow | at least 7 days and 200 cases, no alarm (flag rate within 0.05 and mean risk within 0.05 of the champion's) | pending until judged |
| G8 Integrity | artefact signed, lineage recorded | no |
| G9 Approval | a second person's recorded sign-off | pending until given |

Candidate and champion are scored on the **same holdout**: the candidate's test window, which only verified outcomes
reach. Analyst labels can enter the training window and nothing else, so a candidate is never judged on labels that
analysts influenced (PRD 11.4).

## What the feedback pool does (`assay/learning/pool.py`)

Label levels: 1 verified outcome, 2 adjudicated decision, 3 decision a second analyst agreed with, 4 single decision.
A verified outcome always wins over an analyst decision. An adjudicated decision is accepted once it is 7 days old. A
corroborated decision is accepted if its quality score is not below the reject threshold and it is at least 7 days
old. A single decision is accepted only if its quality score reaches the accept threshold, which in practice needs an analyst with a proven
record. Unsure and conflicted cases are deferred. A decision with no reason code where the analyst disagreed with
the model is rejected.

Integrity checks quarantine an analyst's uncorroborated labels when most of their decisions are implausibly fast,
when nearly all are the same call, or when their accuracy on matured cases is below 50% with enough evidence. A cap
stops one analyst supplying more than 35% of the accepted unverified labels. Not checked: concentration on particular
merchants or beneficiaries (needs the graph store) and sudden shifts in an analyst's accuracy.

## What was found, and what was not

On simulated analysts the pool removed about 97% of injected wrong labels, and the model trained on what it kept beat
the model trained on everything once noise was heavy. It did **not** meet the pre-set rule: with clean analysts it is
not shown to be as good as learning from everything, because it accepts only about 18% of cases
([docs/validation/README.md](../validation/README.md)). Real analysts have not been tested, and a pilot is where the
accept threshold, the corroboration requirement and the gate tolerances should be set.

## Not built

- Schedules or triggers that create candidates (OPD-13). A person runs `create`.
- Bounded automatic recalibration (V1 in the PRD), and any automatic promotion.
- A kill switch that falls back to human review for all cases when no safe bundle exists. Rollback restores the
  previous champion only.
- Rollback triggers evaluated automatically from matured outcomes (calibration breach, high-trust error rate,
  degraded performance). Rollback is a person's decision; the numbers that should prompt it are on the dashboard
  and in the validation reports.
- Approval authority is a role (`approver`). Whether that is the institution's model risk owner, Assay, or both is
  OPD-12.
