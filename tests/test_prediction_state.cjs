const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const sourcePath = path.join(__dirname, "..", "src", "amphilens", "web", "prediction_state.js");

function createStorage() {
  const values = new Map();
  return {
    getItem(key) { return values.has(key) ? values.get(key) : null; },
    setItem(key, value) { values.set(key, String(value)); },
    removeItem(key) { values.delete(key); },
  };
}

function loadPreferences(storage) {
  const window = { localStorage: storage };
  vm.runInNewContext(fs.readFileSync(sourcePath, "utf8"), { window });
  return window.AmphiLensPredictionPreferences;
}

function control(name, value, type = "text", options = []) {
  return { name, value, type, options, tagName: type === "select-one" ? "SELECT" : "INPUT" };
}

function form(...elements) {
  return { elements };
}

test("prediction preferences persist within a project, but not across projects or consent fields", () => {
  const storage = createStorage();
  const preferences = loadPreferences(storage);
  const firstForm = form(
    control("image_root", "/data/images"),
    control("model_choice", "hosted:amphilens-yolo26-m", "select-one", [
      { value: "preset:yolo26" },
      { value: "hosted:amphilens-yolo26-m" },
    ]),
    control("confidence", "0.65", "range"),
    control("uploads_dataset", "on", "checkbox"),
    control("token_secret", "do-not-save", "password"),
  );

  preferences.save("/projects/grass", firstForm);

  const secondForm = form(
    control("image_root", "/other/images"),
    control("model_choice", "preset:yolo26", "select-one", [
      { value: "preset:yolo26" },
      { value: "hosted:amphilens-yolo26-m" },
    ]),
    control("confidence", "0.25", "range"),
    control("uploads_dataset", "", "checkbox"),
    control("token_secret", "", "password"),
  );
  preferences.restore("/projects/grass", secondForm);

  assert.equal(secondForm.elements[0].value, "/data/images");
  assert.equal(secondForm.elements[1].value, "hosted:amphilens-yolo26-m");
  assert.equal(secondForm.elements[2].value, "0.65");
  assert.equal(secondForm.elements[3].value, "");
  assert.equal(secondForm.elements[4].value, "");

  const otherProjectForm = form(control("confidence", "0.25", "range"));
  preferences.restore("/projects/field-study", otherProjectForm);
  assert.equal(otherProjectForm.elements[0].value, "0.25");
  assert.doesNotMatch(
    storage.getItem("amphilens:prediction-preferences:v1:/projects/grass"),
    /do-not-save/,
  );
});

test("stale select values do not replace current model options", () => {
  const storage = createStorage();
  const preferences = loadPreferences(storage);
  storage.setItem(
    "amphilens:prediction-preferences:v1:/projects/grass",
    JSON.stringify({ model_choice: "hosted:removed-model" }),
  );
  const currentForm = form(
    control("model_choice", "preset:yolo26", "select-one", [{ value: "preset:yolo26" }]),
  );

  preferences.restore("/projects/grass", currentForm);

  assert.equal(currentForm.elements[0].value, "preset:yolo26");
});
