(() => {
  "use strict";

  const PREDICTION_PREFIX = "amphilens:prediction-preferences:v1:";
  const WORKFLOW_PREFIX = "amphilens:workflow-preferences:v1:";
  const MODEL_PREFIX = "amphilens:model-preferences:v1:";
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
    return `${PREDICTION_PREFIX}${String(projectIdentity || "")}`;
  }

  function workflowStorageKey(projectIdentity, view) {
    return `${WORKFLOW_PREFIX}${encodeURIComponent(String(projectIdentity || ""))}:${encodeURIComponent(String(view || ""))}`;
  }

  function modelStorageKey(projectIdentity, workflow, modelIdentity) {
    return `${MODEL_PREFIX}${encodeURIComponent(String(projectIdentity || ""))}:${encodeURIComponent(String(workflow || ""))}:${encodeURIComponent(String(modelIdentity || ""))}`;
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

  function readObject(key) {
    try {
      const encoded = window.localStorage.getItem(key);
      if (!encoded) return null;
      const value = JSON.parse(encoded);
      return value && typeof value === "object" && !Array.isArray(value) ? value : null;
    } catch {
      return null;
    }
  }

  function preferenceKey(control) {
    return control.dataset?.preferenceKey || control.name;
  }

  function pageControls(container) {
    return Array.from(container?.querySelectorAll?.("input[name], select[name], textarea[name]") || [])
      .filter((control) => isPreferenceControl(control) && !control.dataset?.modelPreference);
  }

  function isTransientWorkflowControl(view, control) {
    return view === "predict" && control.name === "image_root";
  }

  function saveModel(projectIdentity, workflow, modelIdentity, settings) {
    if (!projectIdentity || !workflow || !modelIdentity || !settings) return;
    try {
      const values = {
        short_side_dimension: String(settings.short_side_dimension ?? ""),
        grayscale_enabled: Boolean(settings.grayscale_enabled),
        clahe_enabled: Boolean(settings.clahe_enabled),
      };
      window.localStorage.setItem(
        modelStorageKey(projectIdentity, workflow, modelIdentity),
        JSON.stringify(values),
      );
    } catch {
      // Preferences are optional; restricted or full browser storage should not block workflows.
    }
  }

  function restoreModel(projectIdentity, workflow, modelIdentity) {
    if (!projectIdentity || !workflow || !modelIdentity) return null;
    return readObject(modelStorageKey(projectIdentity, workflow, modelIdentity));
  }

  function resetModel(projectIdentity, workflow, modelIdentity) {
    if (!projectIdentity || !workflow || !modelIdentity) return;
    try {
      window.localStorage.removeItem(modelStorageKey(projectIdentity, workflow, modelIdentity));
    } catch {
      // A reset remains safe when browser storage is unavailable.
    }
  }

  function detailsState(container) {
    const state = {};
    for (const [index, detail] of Array.from(container?.querySelectorAll?.("details") || []).entries()) {
      const label = detail.querySelector("summary")?.textContent?.trim() || `detail-${index}`;
      state[label] = Boolean(detail.open);
    }
    return state;
  }

  function saveView(projectIdentity, view, container) {
    if (!projectIdentity || !view || !container) return;
    try {
      const values = {};
      for (const control of pageControls(container)) {
        if (isTransientWorkflowControl(view, control)) continue;
        values[preferenceKey(control)] = String(control.value ?? "");
      }
      const key = workflowStorageKey(projectIdentity, view);
      const previous = readObject(key);
      const previousValues = previous?.values && typeof previous.values === "object"
        ? previous.values
        : {};
      if (view === "predict") delete previousValues.image_root;
      const previousDetails = previous?.details && typeof previous.details === "object"
        ? previous.details
        : {};
      window.localStorage.setItem(
        key,
        JSON.stringify({
          values: { ...previousValues, ...values },
          details: { ...previousDetails, ...detailsState(container) },
        }),
      );
    } catch {
      // Preferences are optional; unavailable storage must not block the workflow.
    }
  }

  function applyViewState(container, state, keyPrefix = "", view = "") {
    if (!container || !state || typeof state !== "object") return;
    const values = state.values && typeof state.values === "object" ? state.values : state;
    for (const control of pageControls(container)) {
      if (isTransientWorkflowControl(view, control)) continue;
      const key = preferenceKey(control);
      if (keyPrefix && !key.startsWith(keyPrefix)) continue;
      const saved = typeof values[key] === "string" ? values[key] : values[control.name];
      if (typeof saved !== "string") continue;
      if (String(control.tagName || "").toLowerCase() === "select") {
        const hasOption = Array.from(control.options || []).some(
          (option) => String(option.value) === saved,
        );
        if (!hasOption) continue;
      }
      control.value = saved;
    }
    if (!keyPrefix && state.details && typeof state.details === "object") {
      for (const [index, detail] of Array.from(container.querySelectorAll("details")).entries()) {
        const label = detail.querySelector("summary")?.textContent?.trim() || `detail-${index}`;
        if (typeof state.details[label] === "boolean") detail.open = state.details[label];
      }
    }
  }

  function restoreView(projectIdentity, view, container, keyPrefix = "") {
    if (!projectIdentity || !view || !container) return;
    let state = readObject(workflowStorageKey(projectIdentity, view));
    if (!state && view === "predict") {
      // Migrate preferences saved by the earlier prediction-only helper.
      const legacy = readObject(storageKey(projectIdentity));
      if (legacy) {
        state = { values: legacy, details: {} };
        try {
          window.localStorage.setItem(
            workflowStorageKey(projectIdentity, view),
            JSON.stringify(state),
          );
        } catch {
          // Restore the legacy values even when migration cannot be saved.
        }
      }
    }
    applyViewState(container, state, keyPrefix, view);
  }

  window.AmphiLensPredictionPreferences = { save, restore, saveModel, restoreModel, resetModel };
  window.AmphiLensWorkflowState = { save: saveView, restore: restoreView, saveModel, restoreModel, resetModel };
})();
