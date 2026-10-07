(() => {
  "use strict";

  const PREFIX = "amphilens:prediction-preferences:v1:";
  const OMITTED_TYPES = new Set([
    "button",
    "checkbox",
    "file",
    "hidden",
    "password",
    "radio",
    "reset",
    "submit",
  ]);
  const SENSITIVE_NAME = /token|secret|password|credential|consent|acknowledged|uploads_dataset/i;

  function storageKey(projectIdentity) {
    return `${PREFIX}${String(projectIdentity || "")}`;
  }

  function isPreferenceControl(control) {
    const type = String(control.type || "text").toLowerCase();
    return Boolean(control.name) && !OMITTED_TYPES.has(type) && !SENSITIVE_NAME.test(control.name);
  }

  function save(projectIdentity, form) {
    if (!projectIdentity || !form?.elements) return;
    try {
      const values = {};
      for (const control of Array.from(form.elements)) {
        if (isPreferenceControl(control)) values[control.name] = String(control.value ?? "");
      }
      window.localStorage.setItem(storageKey(projectIdentity), JSON.stringify(values));
    } catch {
      // Preferences are optional; restricted or full browser storage should not block prediction.
    }
  }

  function restore(projectIdentity, form) {
    if (!projectIdentity || !form?.elements) return;
    try {
      const encoded = window.localStorage.getItem(storageKey(projectIdentity));
      if (!encoded) return;
      const values = JSON.parse(encoded);
      if (!values || typeof values !== "object" || Array.isArray(values)) return;
      for (const control of Array.from(form.elements)) {
        if (!isPreferenceControl(control) || typeof values[control.name] !== "string") continue;
        if (String(control.tagName || "").toLowerCase() === "select") {
          const hasOption = Array.from(control.options || []).some(
            (option) => String(option.value) === values[control.name],
          );
          if (!hasOption) continue;
        }
        control.value = values[control.name];
      }
    } catch {
      // Ignore unavailable storage and stale preference data.
    }
  }

  window.AmphiLensPredictionPreferences = { save, restore };
})();
