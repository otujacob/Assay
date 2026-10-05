// End-to-end check of the browser SIGN-IN against the SSO demo (backend/scripts/demo_sso.py), in a real browser (local Edge).
//   node scripts/e2e-sso.mjs [app-url]
// The provider is a STAND-IN: this shows our client and server agree, not that any real provider will work (docs/security/sso.md).
// Exits non-zero if any expectation fails.
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const url = process.argv[2] ?? "http://127.0.0.1:8001";
mkdirSync("screenshots", { recursive: true });
let failed = 0;
const check = (name, cond, detail = "") => {
  console.log(`${cond ? "PASS" : "FAIL"}  ${name}${detail ? "  " + detail : ""}`);
  if (!cond) failed++;
};

const browser = await chromium.launch({ channel: "msedge", headless: true });
const errors = [];

/** A fresh browser context per scenario, so each starts signed out with its own sessionStorage. */
async function scenario(fn) {
  const ctx = await browser.newContext({ viewport: { width: 1500, height: 1000 } });
  const page = await ctx.newPage();
  const apiCalls = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => { if (new URL(r.url()).pathname.startsWith("/api/")) apiCalls.push({ path: new URL(r.url()).pathname, headers: r.headers() }); });
  try { await fn(page, apiCalls); } finally { await ctx.close(); }
}

const signInAs = async (page, who) => {
  await page.goto(url);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForSelector(`a[data-person=${who}]`);
  await page.click(`a[data-person=${who}]`);
};
const text = async (page, sel) => (await page.locator(sel).first().innerText()).replace(/\s+/g, " ");

// --- not signed in: nothing of the app, and no API calls --------------------------------------------------------------
await scenario(async (page, calls) => {
  await page.goto(url);
  await page.waitForSelector("[data-testid=sign-in]");
  check("a person who has not signed in sees only the sign-in screen", (await page.locator("[data-testid=case-details]").count()) === 0 && (await page.locator(".demo-banner").count()) === 0);
  check("there is no demo-user switch in single sign-on mode", (await page.locator("select[aria-label='Demo user']").count()) === 0);
  check("no case or queue data was requested before sign-in", calls.every((c) => c.path === "/api/auth/config"), calls.map((c) => c.path).join(","));
  await page.screenshot({ path: "screenshots/15-sign-in.png" });
});

// --- an analyst signs in -------------------------------------------------------------------------------------------
await scenario(async (page, calls) => {
  await signInAs(page, "ada");
  await page.waitForSelector("[data-testid=signed-in-user]", { timeout: 60000 });
  await page.waitForSelector("[data-testid^=row-]", { timeout: 60000 });
  check("after the provider, the app shows who signed in and the review queue", /Ada/.test(await text(page, "[data-testid=signed-in-user]")) && (await page.locator("[data-testid^=row-]").count()) > 5);
  check("the sign-in code is gone from the address bar", !/code=|state=/.test(page.url()), page.url());
  check("there is no demo banner and no demo-user switch", (await page.locator(".demo-banner").count()) === 0 && (await page.locator("select[aria-label='Demo user']").count()) === 0);
  const authed = calls.filter((c) => c.path !== "/api/auth/config");
  check("every API request carries a bearer token and none carries a demo user", authed.length > 0 && authed.every((c) => /^Bearer ey/.test(c.headers["authorization"] ?? "") && !c.headers["x-demo-user"]), `${authed.length} requests`);
  check("an analyst sees neither the policy panel nor the audit log", (await page.locator("[data-testid=policy-panel]").count()) === 0 && (await page.locator("[data-testid=audit-log]").count()) === 0);
  await page.locator("[data-testid^=row-]").first().click();
  await page.waitForSelector("[data-testid=case-details]");
  check("the analyst can open a case", true);
  await page.screenshot({ path: "screenshots/16-signed-in.png" });

  // a reload keeps the session within the tab
  await page.reload();
  await page.waitForSelector("[data-testid=signed-in-user]", { timeout: 60000 });
  check("a reload does not sign the person out", /Ada/.test(await text(page, "[data-testid=signed-in-user]")));

  // the API refuses a token that was altered, whatever the browser holds
  const token = await page.evaluate(() => JSON.parse(sessionStorage.getItem("assay.oidc.session")).token);
  const status = await page.evaluate(async (t) => (await fetch("/api/me", { headers: { Authorization: `Bearer ${t.slice(0, -4)}AAAA` } })).status, token);
  check("the API refuses a token whose signature was altered", status === 401, `status ${status}`);
  const noHeader = await page.evaluate(async () => (await fetch("/api/me")).status);
  check("and refuses a request with no credentials", noHeader === 401, `status ${noHeader}`);

  // a corrupted session in storage is not trusted: the API says no, and the person is asked to sign in again
  await page.evaluate(() => {
    const s = JSON.parse(sessionStorage.getItem("assay.oidc.session"));
    s.token = s.token.slice(0, -4) + "AAAA";
    sessionStorage.setItem("assay.oidc.session", JSON.stringify(s));
  });
  await page.reload();
  await page.waitForSelector("[data-testid=sign-in]", { timeout: 30000 });
  check("a tampered session is refused and the person is asked to sign in again", /could not be confirmed/.test(await text(page, "[role=alert]")));
  check("and the bad token is not kept", (await page.evaluate(() => sessionStorage.getItem("assay.oidc.session"))) === null);
});

// --- signing out ends the session at both ends ----------------------------------------------------------------------
await scenario(async (page) => {
  await signInAs(page, "ada");
  await page.waitForSelector("[data-testid=signed-in-user]", { timeout: 60000 });
  await page.getByRole("button", { name: "Sign out" }).click();
  await page.waitForSelector("[data-testid=sign-in]", { timeout: 30000 });
  check("sign-out returns to the sign-in screen", true);
  check("and removes the session from the tab", (await page.evaluate(() => sessionStorage.getItem("assay.oidc.session"))) === null);
  await page.reload();
  await page.waitForSelector("[data-testid=sign-in]");
  check("a reload after sign-out stays signed out", (await page.locator("[data-testid=case-details]").count()) === 0);
});

// --- an expired session is not used ---------------------------------------------------------------------------------
await scenario(async (page) => {
  await signInAs(page, "ada");
  await page.waitForSelector("[data-testid=signed-in-user]", { timeout: 60000 });
  await page.evaluate(() => {
    const s = JSON.parse(sessionStorage.getItem("assay.oidc.session"));
    s.expiresAt = Date.now() - 1000;
    sessionStorage.setItem("assay.oidc.session", JSON.stringify(s));
  });
  await page.reload();
  await page.waitForSelector("[data-testid=sign-in]", { timeout: 30000 });
  check("a session past its expiry is dropped and the person signs in again", (await page.evaluate(() => sessionStorage.getItem("assay.oidc.session"))) === null);
});

// --- MFA, no role, cancelling --------------------------------------------------------------------------------------
await scenario(async (page) => {
  await signInAs(page, "nomfa");
  await page.waitForSelector("[data-testid=sign-in]", { timeout: 30000 });
  check("a sign-in with a password only is refused with the MFA message", /multi-factor/i.test(await text(page, "[role=alert]")), await text(page, "[role=alert]"));
  check("and nothing of the app is shown", (await page.locator("[data-testid=case-details]").count()) === 0);
});
await scenario(async (page) => {
  await signInAs(page, "ghost");
  await page.waitForSelector("[data-testid=no-access]", { timeout: 30000 });
  check("someone signed in with no mapped group is told they have no access", /none of your groups gives you a role/.test(await text(page, "[data-testid=no-access]")));
  check("and sees no queue", (await page.locator("[data-testid^=row-]").count()) === 0);
});
await scenario(async (page) => {
  await page.goto(url);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForSelector("a[data-person=deny]");
  await page.click("a[data-person=deny]");
  await page.waitForSelector("[data-testid=sign-in]", { timeout: 30000 });
  check("cancelling at the provider returns to sign-in with the reason", /User cancelled/.test(await text(page, "[role=alert]")));
  check("and the address bar is clean", !/error=|code=/.test(page.url()), page.url());
});

// --- roles come from the server: an approver and an auditor see their own screens ----------------------------------------
await scenario(async (page) => {
  await signInAs(page, "morgan");
  await page.waitForSelector("[data-testid=signed-in-user]", { timeout: 60000 });
  await page.waitForSelector("[data-testid=policy-panel]", { timeout: 30000 });
  check("a model-risk approver sees the policy panel and model updates", (await page.locator("[data-testid=model-updates]").count()) === 1);
  check("and not the review queue", /does not include the review queue/.test(await text(page, ".left-col")));
});
await scenario(async (page) => {
  await signInAs(page, "audrey");
  await page.waitForSelector("[data-testid=audit-log]", { timeout: 60000 });
  check("an auditor sees the audit log", (await page.locator("[data-testid=audit-log]").count()) === 1);
  check("and cannot propose or approve policies", (await page.getByRole("button", { name: "Propose policy" }).count()) === 0 && (await page.getByRole("button", { name: "Approve" }).count()) === 0);
});

check("no uncaught page errors", errors.length === 0, errors.join(" | "));
await browser.close();
console.log(failed === 0 ? "\nALL SIGN-IN END-TO-END CHECKS PASSED" : `\n${failed} CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
