# The case summary (PRD 19, V2 "AI Fraud Copilot", first step)

A short summary of a case for an investigator, on the case screen ("Summarise this case"), `GET /v1/decisions/{id}/summary`.

## What it is, and what it is not

**It is written by fixed rules from the stored record. No language model is used and nothing leaves the system.** That is the PRD's default
answer to OPD-19 (no personal data to a third-party model), and it means nothing has to be decided before it can be used.

It restates what is already on the case screen, one sourced sentence at a time: the model's risk against its flagging threshold; the
Trust Index (or why there is none) and the reasons recorded; the weakest reliability components and any that are missing; the three
largest drivers with their direction and the explanation's stability; what the policy recommended and which step decided it (or that the
kill switch is engaged); and what the entity graph shows. Every sentence names its source (and the stored fields behind it, on hover).

It does not weigh the evidence, say why anything happened, guess at intent, advise, or answer questions. The wording is fixed so it can
never be more confident than its source: provisional trust is called a ranking, not a probability; a graph link is called "evidence of a
connection, not of wrongdoing". It leaves out what-ifs (slow to compute, so still on request), analyst notes and earlier decisions (they
could anchor the reader), and anything not in the record. It is not for a customer.

Access is the same as the explanation: analysts (not on a blind case, which gets a refusal) and auditors. Reading it is audited.

## How it is tested

Unit tests check each sentence against the record; that **every number in a summary is a number in its source**; that provisional
trust is never described as a probability; that no advice, cause or guess appears in the wording; and that the kill switch changes what
is said. Mutation checks confirmed that an invented number, an overclaim, "probability" wording and advice are each caught. API tests
cover roles, other tenants, unknown ids, blind review and the audit entry.

## What a language-model copilot would still need

Not built, and not started: free-form questions, drafting notes, or any model call. It needs the OPD-19 decision first (may personal data
reach a model, and which one, hosted where), and a pre-set evaluation of whether its statements stay grounded in the record, which cannot
be done without the real model. The structured, sourced facts this summary produces are what such a copilot would be limited to quoting.
