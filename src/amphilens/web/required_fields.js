(() => {
  "use strict";

  function sync(root) {
    root.querySelectorAll("label").forEach((label) => {
      const control = label.control || label.querySelector("input, select, textarea");
      const cloudOnly = label.hasAttribute("data-required-when-cloud");
      const cloud = label.closest("form")?.elements?.execution?.value === "cloud";
      const cloudRequired = cloudOnly && cloud;
      const required = Boolean(control?.required || cloudRequired);

      label.toggleAttribute("data-required", required);
      if (cloudOnly && control) {
        if (cloudRequired) control.setAttribute("aria-required", "true");
        else control.removeAttribute("aria-required");
      }
    });
  }

  function start(root = document) {
    sync(root);
    const observer = new MutationObserver(() => sync(root));
    observer.observe(root, {
      subtree: true,
      childList: true,
      attributes: true,
      attributeFilter: ["required"],
    });
    root.addEventListener("change", () => sync(root), true);
    root.addEventListener("input", () => sync(root), true);
    return observer;
  }

  window.AmphiLensRequiredFields = { sync, start };
})();
