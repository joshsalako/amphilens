const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const webRoot = path.join(__dirname, "..", "src", "amphilens", "web");
const statePath = path.join(webRoot, "navigation_state.js");
const appPath = path.join(webRoot, "app.js");
const indexPath = path.join(webRoot, "index.html");
const appSource = fs.readFileSync(appPath, "utf8");
const indexSource = fs.readFileSync(indexPath, "utf8");

function loadNavigationState(storage) {
  const window = { sessionStorage: storage };
  if (fs.existsSync(statePath)) {
    vm.runInNewContext(fs.readFileSync(statePath, "utf8"), { window });
  }
  return window.AmphiLensNavigationState;
}

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    get(key) { return values; },
  };
}

test("restores a saved workflow view and persists it across a page reload", () => {
  const storage = memoryStorage({ "amphilens.active-view": "predict" });
  const firstLoad = loadNavigationState(storage);

  assert.ok(firstLoad, "navigation state must be available before app startup");
  if (!firstLoad) return;
  assert.equal(typeof firstLoad.load, "function");
  assert.equal(firstLoad.load(), "predict");
  firstLoad.save("training");

  const refreshedPage = loadNavigationState(storage);
  assert.equal(refreshedPage.load(), "training");
});

test("uses Home when the saved view is missing or unsupported", () => {
  const emptyState = loadNavigationState(memoryStorage());
  const invalidState = loadNavigationState(memoryStorage({ "amphilens.active-view": "not-a-view" }));
  assert.ok(emptyState, "navigation state must be available before app startup");
  assert.ok(invalidState, "navigation state must be available before app startup");
  if (!emptyState || !invalidState) return;
  assert.equal(emptyState.load(), "home");
  assert.equal(invalidState.load(), "home");
});

test("the app initializes its current view from navigation state", () => {
  assert.ok(
    /view:\s*window\.AmphiLensNavigationState\?\.load\(\)\s*\|\|\s*"home"/.test(appSource),
    "app startup must restore a saved view, with Home as the fallback",
  );
});

test("navigating to a view saves it in tab-scoped navigation state", () => {
  assert.ok(
    /AmphiLensNavigationState\?\.save\(view\)/.test(appSource),
    "the selected view must be saved whenever navigation changes",
  );
});

test("the navigation state script loads before the app script", () => {
  const stateScript = indexSource.indexOf('src="/static/navigation_state.js"');
  const appScript = indexSource.indexOf('src="/static/app.js"');

  assert.notEqual(stateScript, -1);
  assert.ok(appScript > stateScript);
});
