// Takes screenshots of the running app using the locally installed Edge (no browser download).
//   node scripts/screenshot.mjs [url] [outDir]
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:8000";
const out = process.argv[3] ?? "screenshots";
mkdirSync(out, { recursive: true });

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto(url, { waitUntil: "networkidle" });
await page.waitForSelector("[data-testid=case-details]", { timeout: 20000 });
await page.screenshot({ path: `${out}/01-analyst-full.png`, fullPage: true });

// Manager view (dashboard KPIs, queue, governance)
await page.selectOption("select[aria-label='Demo user']", "manager");
await page.waitForSelector("[data-testid=kpis]", { timeout: 20000 });
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/02-manager-full.png`, fullPage: true });

// Governance tabs as the approver
await page.selectOption("select[aria-label='Demo user']", "approver");
await page.waitForSelector("[data-testid=governance]");
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/03-approver-governance.png`, fullPage: true });

console.log(JSON.stringify({ consoleErrors: errors }));
await browser.close();
