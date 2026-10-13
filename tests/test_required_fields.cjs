const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const sourcePath = path.join(__dirname, "..", "src", "amphilens", "web", "required_fields.js");

function loadRequiredFields() {
  const window = {};
  vm.runInNewContext(fs.readFileSync(sourcePath, "utf8"), { window });
  return window.AmphiLensRequiredFields;
}

function label(control, { requiredWhenCloud = false, execution = "local" } = {}) {
  const attributes = new Set(requiredWhenCloud ? ["data-required-when-cloud"] : []);
  const controlAttributes = new Set();
  return {
    control: {
      required: control.required,
      setAttribute(name) { controlAttributes.add(name); },
      removeAttribute(name) { controlAttributes.delete(name); },
      hasAttribute(name) { return controlAttributes.has(name); },
    },
    closest(selector) {
      assert.equal(selector, "form");
      return { elements: { execution: { value: execution } } };
    },
    hasAttribute(name) { return attributes.has(name); },
    toggleAttribute(name, enabled) {
      if (enabled) attributes.add(name);
      else attributes.delete(name);
    },
    getAttribute(name) { return attributes.has(name) ? "" : null; },
  };
}

test("required markers reflect native requirements and cloud-only checks", () => {
  const requiredFields = loadRequiredFields();
  const trainingSnapshot = label({ required: true });
  const optionalValidation = label({ required: false });
  const cloudBudget = label({ required: false }, { requiredWhenCloud: true });
  const cloudConsent = label({ required: false }, { requiredWhenCloud: true });
  const labels = [trainingSnapshot, optionalValidation, cloudBudget, cloudConsent];
  const root = { querySelectorAll(selector) { assert.equal(selector, "label"); return labels; } };

  requiredFields.sync(root);

  assert.equal(trainingSnapshot.getAttribute("data-required"), "");
  assert.equal(optionalValidation.getAttribute("data-required"), null);
  assert.equal(cloudBudget.getAttribute("data-required"), null);
  assert.equal(cloudConsent.control.hasAttribute("aria-required"), false);

  cloudBudget.closest = () => ({ elements: { execution: { value: "cloud" } } });
  cloudConsent.closest = cloudBudget.closest;
  requiredFields.sync(root);

  assert.equal(cloudBudget.getAttribute("data-required"), "");
  assert.equal(cloudConsent.getAttribute("data-required"), "");
  assert.equal(cloudConsent.control.hasAttribute("aria-required"), true);

  cloudBudget.closest = () => ({ elements: { execution: { value: "local" } } });
  cloudConsent.closest = cloudBudget.closest;
  requiredFields.sync(root);

  assert.equal(cloudBudget.getAttribute("data-required"), null);
  assert.equal(cloudConsent.getAttribute("data-required"), null);
  assert.equal(cloudConsent.control.hasAttribute("aria-required"), false);
});
