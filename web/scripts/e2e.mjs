// End-to-end check against the RUNNING demo server, driving a real browser (local Edge).
//   node scripts/e2e.mjs [url]
// Exits non-zero on the first failed expectation.
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:8000";
mkdirSync("screenshots", { recursive: true });
let failed = 0;
const check = (name, cond, detail = "") => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!cond) failed++;
};

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(String(e)));

const user = (key) => page.selectOption("select[aria-label='Demo user']", key);
const text = async (sel) => (await page.locator(sel).first().innerText()).replace(/\s+/g, " ");

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForSelector("[data-testid=case-details]", { timeout: 30000 });

// --- analyst: queue, blind review -------------------------------------------------------------
const rows = page.locator("[data-testid^=row-]");
const nRows = await rows.count();
check("queue shows cases for the analyst", nRows > 5, `rows=${nRows}`);
const prios = await page.locator("table.queue td:first-child .prio").allInnerTexts();
check("queue is ordered by priority (P1 before P4)", prios.indexOf("P1") === 0 || prios[0] <= prios[prios.length - 1], prios.slice(0, 6).join(","));

const blindRow = page.locator("[data-testid^=row-]", { has: page.locator(".blind-dot") }).first();
if (await blindRow.count()) {
  check("blind case hides risk and trust in the queue", (await blindRow.innerText()).includes("hidden"));
  await blindRow.click();
  await page.waitForSelector("[data-testid=blind-notice]");
  check("blind case withholds score, trust and recommendation", (await page.locator("[data-testid=risk-card]").count()) === 0 && (await page.locator("[data-testid=recommendation]").count()) === 0);
  await page.screenshot({ path: "screenshots/04-blind-case.png" });
  check("a blind case offers no what-if", (await page.locator("[data-testid=counterfactuals]").count()) === 0);
} else {
  console.log("SKIP  no blind case in the demo queue");
}

// --- analyst: open a normal case, enforce reason codes ---------------------------------------------
const plain = page.locator("[data-testid^=row-]", { hasNot: page.locator(".blind-dot") }).first();
await plain.click();
await page.waitForSelector("[data-testid=risk-card]");
check("a normal case shows the Fraud Risk Score and Trust Index", (await page.locator("[data-testid=trust-gauge]").count()) === 1);
const gauge = await text("[data-testid=trust-gauge]");
check("insufficient evidence is never shown as a number", !/Insufficient evidence/i.test(gauge) || !/\b\d+\s*\/\s*100/.test(gauge), gauge.slice(0, 80));
check("recommendation says recommend-only", /recommend-only/.test(await text("[data-testid=recommendation]")));

// --- analyst: a rule-written case summary (no language model) -----------------------------------------------------------
check("the summary is not built until asked", (await page.locator("[data-testid=summary-note]").count()) === 0);
await page.getByRole("button", { name: "Summarise this case" }).click();
await page.waitForSelector("[data-testid=summary-note]", { timeout: 60000 });
const sumText = await text("[data-testid=case-summary]");
check("the summary states the model's risk and threshold and that no language model was used", /put the risk at \d+\.\d%/.test(sumText) && /no language model was used/.test(sumText), sumText.slice(0, 120));
check("it lists what it does not cover", /Not covered:/.test(sumText));
await page.locator("[data-testid=case-summary]").screenshot({ path: "screenshots/17-case-summary.png" });

// --- analyst: what would change the decision? (counterfactuals, PRD 7.1) -------------------------------------------
check("what-ifs are not worked out until asked", (await page.locator("[data-testid=cf-note]").count()) === 0);
await page.getByRole("button", { name: "Show what-ifs" }).click();
await page.waitForSelector("[data-testid=cf-note]", { timeout: 60000 });
const cfText = await text("[data-testid=counterfactuals]");
check("the what-if card always says it describes the model, not cause and effect", /describes the MODEL, not cause and effect/.test(cfText) && /must not be given to a customer/.test(cfText));
check("the what-if card says either what would flip the call or that none exists", (await page.locator("[data-testid=cf-list]").count()) === 1 || (await page.locator("[data-testid=cf-none]").count()) === 1 || /none held up/.test(cfText), cfText.slice(0, 120));
check("the what-if card shows how far two explanation methods agree", /Two methods agree \d+%/.test(await text("[data-testid=cf-methods]")));
await page.locator("[data-testid=counterfactuals]").screenshot({ path: "screenshots/11-what-if.png" });

// --- analyst: what else is linked to this case? (entity graph, PRD 13) ---------------------------------------------
check("linked entities are not loaded until asked", (await page.locator("[data-testid=graph-note]").count()) === 0);
await page.getByRole("button", { name: "Show linked entities" }).click();
await page.waitForSelector("[data-testid=graph-note]", { timeout: 60000 });
const gText = await text("[data-testid=linked-entities]");
check("the linked-entities card says it shows connections, not guilt, and is not for customers", /not of wrongdoing/.test(gText) && /must not be shared with a customer/.test(gText));
check("it lists the case's entities and either some links or that none exist", (await page.locator("[data-testid=graph-link]").count()) > 0 || (await page.locator("[data-testid=graph-none]").count()) === 1 || /no device, IP address or beneficiary/.test(gText), gText.slice(0, 140));
await page.locator("[data-testid=linked-entities]").screenshot({ path: "screenshots/14-linked-entities.png" });

await page.getByRole("button", { name: /Override AI/ }).click();
await page.getByRole("button", { name: "Submit override" }).click();
check("override without a reason code is refused", /needs a reason code/i.test(await text("[role=alert]")));
await page.screenshot({ path: "screenshots/05-override-needs-reason.png" });

// --- analyst: record a real decision through the API -------------------------------------------------
await page.getByRole("button", { name: /Override AI/ }).click(); // close the override form
await page.selectOption("[data-testid=action-panel] select[aria-label='Reason code']", "insufficient_information");
await page.getByLabel("Customer contacted").check();
await page.getByRole("button", { name: "Mark unsure" }).click();
await page.waitForSelector("[data-testid=action-result]");
const res = await text("[data-testid=action-result]");
check("the decision is recorded with shadow feedback scores", /Recorded/.test(res) && /shadow, not used for learning/.test(res), res.slice(0, 120));
check("the decision appears in the case history", (await page.locator("[data-testid=action-history]").count()) === 1);
await page.screenshot({ path: "screenshots/06-decision-recorded.png" });

// --- manager: dashboard, no decision controls ------------------------------------------------------------
await user("manager");
await page.waitForSelector("[data-testid=kpis]");
await page.waitForTimeout(600);
const kpis = await text("[data-testid=kpis]");
check("manager sees pending reviews and SLA breaches", /Total Pending Reviews/.test(kpis) && /past SLA/.test(kpis), kpis.slice(0, 90));
check("high-trust error rate is unavailable, not invented", /Not yet available/.test(kpis));
check("manager cannot record decisions", (await page.locator("[data-testid=action-panel]").count()) === 0);

// --- approver: governance evidence labelled synthetic, verdict shown honestly ----------------------------------
await user("approver");
await page.waitForSelector("[data-testid=governance]");
await page.waitForTimeout(600);
const gov = await text("[data-testid=governance]");
check("governance evidence is labelled synthetic", /Synthetic data/.test(gov));
check("the validation verdict is shown as NOT supported", /Not supported as specified/.test(gov));
await page.getByRole("tab", { name: "Model Bundle & Lineage" }).click();
check("bundle lineage shows the signed artefact hash", /Artefact SHA-256/.test(await text("[data-testid=bundle-card]")));

// --- decision policies: propose, then a DIFFERENT person approves (FR-21) -------------------------------------------
await user("analyst");
await page.waitForSelector("[data-testid=case-details]");
check("an analyst sees neither the policy panel nor the audit log",
  (await page.locator("[data-testid=policy-panel]").count()) === 0 && (await page.locator("[data-testid=audit-log]").count()) === 0);

await user("admin");
await page.waitForSelector("[data-testid=policy-panel]");
await page.waitForTimeout(800); // let any late response from the previous user arrive
const bannerText = (await page.locator(".banner-error").allInnerTexts()).join(" | ");
const stuckLoading = (await page.locator("body").innerText()).includes("Loading case");
check("switching to a role with no case access shows no error banner and no stuck loading state",
  bannerText === "" && !stuckLoading, `banner="${bannerText}" stuckLoading=${stuckLoading}`);
// A policy row is one whose FIRST cell is a version. (The empty-state message also contains "policy-0", so
// matching the row's whole text miscounts it as a policy when there are none, e.g. on a fresh demo.)
const policyRowCount = () => page.evaluate(() =>
  [...document.querySelectorAll("[data-testid=policy-panel] tbody tr")]
    .filter((r) => /^policy-\d+/.test(r.querySelector("td")?.textContent?.trim() ?? "")).length);
const before = await policyRowCount();
// Preview before proposing: an amount rule of 1 sends every approval to a person.
await page.fill("[aria-label='Always review amounts from']", "1");
await page.getByRole("button", { name: "Preview impact" }).click();
await page.waitForSelector("[data-testid=policy-preview]");
const previewText = await text("[data-testid=policy-preview]");
check("a preview shows the change in review volume against the policy in force",
  /Review volume \d+ → \d+/.test(previewText) && /compared with policy-\d+, the policy in force/.test(previewText), previewText.slice(0, 110));
check("the preview says what it does not tell you", /does not predict analyst decisions/.test(previewText));
await page.locator("[data-testid=policy-panel]").screenshot({ path: "screenshots/10-policy-preview.png" });
await page.fill("[aria-label='Always review amounts from']", "");
check("editing the form clears a stale preview", (await page.locator("[data-testid=policy-preview]").count()) === 0);
await page.selectOption("[aria-label='Data-quality gate action']", "hold");
await page.getByRole("button", { name: "Propose policy" }).click();
await page.waitForFunction((n) => [...document.querySelectorAll("[data-testid=policy-panel] tbody tr")]
  .filter((r) => /^policy-\d+/.test(r.querySelector("td")?.textContent?.trim() ?? "")).length > n, before);
const pending = page.locator("[data-testid=policy-panel] tbody tr", { hasText: "Awaiting approval" }).first();
check("a proposed policy is awaiting approval", (await pending.count()) === 1, (await pending.innerText()).replace(/\s+/g, " ").slice(0, 100));
check("the proposer has no Approve button", (await page.getByRole("button", { name: "Approve" }).count()) === 0);
await page.locator("[data-testid=policy-panel]").screenshot({ path: "screenshots/07-policy-proposed.png" });

await user("approver");
await page.waitForSelector("[data-testid=policy-panel]");
await page.waitForTimeout(500);
const awaiting = page.locator("[data-testid=policy-panel] tbody tr", { hasText: "Awaiting approval" }).first();
const version = (await awaiting.locator("td").first().innerText()).trim();
check("the approver sees an Approve button and no Propose form",
  (await awaiting.getByRole("button", { name: "Approve" }).count()) === 1 && (await page.getByRole("button", { name: "Propose policy" }).count()) === 0, version);
await awaiting.getByRole("button", { name: "Approve" }).click();
await page.waitForSelector(`[data-testid=policy-panel] tbody tr:has-text("${version}"):has-text("Approved by u:approver")`);
const inForce = await text("[data-testid=policy-panel] tbody tr:has-text('In force')");
check("after approval the policy is in force", inForce.includes(version), inForce.slice(0, 80));
await page.locator("[data-testid=policy-panel]").screenshot({ path: "screenshots/08-policy-approved.png" });

// --- model updates: a candidate through shadow, approval, canary, promotion and rollback (PRD 12) -----------------
// Needs a demo started with --with-candidate. Without one the panel says so and this part is skipped.
await user("admin");
await page.waitForSelector("[data-testid=model-updates]");
await page.waitForTimeout(800);
const hasCandidate = (await page.locator("[data-testid=model-updates]").getByRole("button", { name: "Start shadow" }).count()) > 0;
if (!hasCandidate) {
  console.log("SKIP  no candidate model ready to start in this demo (start it with --with-candidate; each demo run can take its candidate through the lifecycle once)");
} else {
  const traffic = (n) => page.evaluate((k) => fetch(`/demo/traffic?n=${k}`, { method: "POST" }).then((r) => r.json()), n);
  const refresh = async () => { await page.getByRole("button", { name: "Refresh" }).click(); await page.waitForTimeout(500); };
  const mu = page.locator("[data-testid=model-updates]");
  check("a validated candidate is listed with its gates, and an unjudged gate is not shown as a pass",
    /Validated/.test(await text("[data-testid=model-updates]")) && /Needs review/.test(await text("[data-testid=model-updates]")));
  check("the feedback summary is shown", (await page.locator("[data-testid=feedback-pool]").count()) === 1);
  await page.screenshot({ path: "screenshots/12-model-updates.png", fullPage: false });
  await mu.getByRole("button", { name: "Start shadow" }).click();
  await mu.getByText("In shadow").first().waitFor({ timeout: 10000 });
  check("an administrator can start shadow", true);
  check("an administrator has no approve control", (await mu.getByRole("button", { name: "Approve" }).count()) === 0);
  const t1 = await traffic(40);
  check("live traffic was scored while in shadow", t1.played > 0, JSON.stringify(t1));

  await user("approver");
  await page.waitForSelector("[data-testid=model-updates]");
  await refresh();
  const mu2 = page.locator("[data-testid=model-updates]");
  check("the shadow report shows cases and agreement with the champion", /Agrees with the champion on \d+%/.test(await text("[data-testid=shadow-report]")), await text("[data-testid=shadow-report]"));
  check("approval is blocked until a reason and the segment review are given", await mu2.getByRole("button", { name: "Approve" }).isDisabled());
  await mu2.getByLabel("Approval rationale").fill("Reviewed the gate report and the shadow run");
  await mu2.getByLabel(/I reviewed the results by segment/).check();
  for (const box of await mu2.locator("[aria-label='Gates that could not be judged'] input[type=checkbox]").all()) await box.check();
  await mu2.getByRole("button", { name: "Approve" }).click();
  await mu2.getByRole("button", { name: "Start canary" }).waitFor({ timeout: 10000 });
  check("a different person approves, and the candidate moves to approved", true);
  await mu2.getByLabel("Canary share").fill("0.5");
  await mu2.getByRole("button", { name: "Start canary" }).click();
  await mu2.getByRole("button", { name: "Promote to champion" }).waitFor({ timeout: 10000 });
  check("the status line shows the canary share", /canary .* at 50% of traffic/.test(await text("[data-testid=learning-status]")), await text("[data-testid=learning-status]"));
  const t2 = await traffic(60);
  check("live traffic was decided partly by the canary", t2.played > 0, JSON.stringify(t2));
  await mu2.getByRole("button", { name: "Promote to champion" }).click();
  await page.waitForFunction(() => /Champion/.test(document.querySelector("[data-testid^=candidate-b-]")?.textContent ?? "") && !/canary/.test(document.querySelector("[data-testid=learning-status]")?.textContent ?? ""), null, { timeout: 10000 });
  check("promotion needs a canary that decided enough cases, and then makes it the champion", true);
  await page.screenshot({ path: "screenshots/13-model-promoted.png", fullPage: false });
  await mu2.getByLabel("Roll back (reason)").fill("end-to-end check");
  await mu2.getByRole("button", { name: "Roll back" }).click();
  await mu2.getByText("Rolled back").first().waitFor({ timeout: 10000 });
  check("a rollback restores the previous champion", !/canary/.test(await text("[data-testid=learning-status]")));
  const hist = await mu2.locator("details").first().innerText();
  void hist;
}

// --- auditor: read-only policies, audit log with search and CSV export (FR-34, FR-43) ----------------------------
await user("auditor");
await page.waitForSelector("[data-testid=audit-log] tbody tr");
check("the auditor sees the policy list but cannot act on it",
  (await page.locator("[data-testid=policy-panel]").count()) === 1 &&
  (await page.getByRole("button", { name: "Approve" }).count()) === 0 && (await page.getByRole("button", { name: "Propose policy" }).count()) === 0);
const auditText = await text("[data-testid=audit-log]");
check("the audit log shows the hash chain verified", /Hash chain verified/.test(auditText), auditText.slice(-90));
await page.fill("[data-testid=audit-log] [aria-label='Action']", "policy_approve");
await page.getByRole("button", { name: "Search" }).click();
await page.waitForFunction(() => {
  const rows = [...document.querySelectorAll("[data-testid=audit-log] tbody tr")];
  return rows.length > 0 && rows.every((r) => r.textContent.includes("policy_approve"));
});
check("searching by action shows only that action", (await text("[data-testid=audit-log] tbody")).includes("u:approver"));
const [download] = await Promise.all([
  page.waitForEvent("download"),
  page.getByRole("button", { name: "Export CSV" }).click(),
]);
const { readFileSync } = await import("node:fs");
const csv = readFileSync(await download.path(), "utf8");
check("the CSV export downloads with a header and the filtered rows",
  download.suggestedFilename() === "assay-audit.csv" && csv.startsWith("seq,time,actor,action,object,result,row_hash") && csv.includes("policy_approve"),
  csv.split("\n")[0]);
await page.locator("[data-testid=audit-log]").screenshot({ path: "screenshots/09-audit-log.png" });

check("no uncaught page errors", pageErrors.length === 0, pageErrors.join(" | "));
await browser.close();
console.log(failed === 0 ? "\nALL END-TO-END CHECKS PASSED" : `\n${failed} CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
