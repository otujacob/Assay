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

check("no uncaught page errors", pageErrors.length === 0, pageErrors.join(" | "));
await browser.close();
console.log(failed === 0 ? "\nALL END-TO-END CHECKS PASSED" : `\n${failed} CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
