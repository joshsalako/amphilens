(() => {
  "use strict";

  const STORAGE_KEY = "amphilens.active-view";
  const DEFAULT_VIEW = "home";
  const VIEWS = new Set(["home", "predict", "import", "training", "review", "cvat", "health"]);

  function load() {
    try {
      const saved = window.sessionStorage.getItem(STORAGE_KEY);
      return VIEWS.has(saved) ? saved : DEFAULT_VIEW;
    } catch {
      return DEFAULT_VIEW;
    }
  }

  function save(view) {
    if (!VIEWS.has(view)) return;
    try {
      window.sessionStorage.setItem(STORAGE_KEY, view);
    } catch {
      // Navigation still works when browser storage is unavailable.
    }
  }

  window.AmphiLensNavigationState = { load, save };
})();
