(() => {
  "use strict";

  function normalizeLabel(value) {
    return String(value ?? "")
      .normalize("NFKC")
      .trim()
      .toLowerCase()
      .replace(/[\s_-]+/g, " ");
  }

  function matchProjectClass(sourceLabel, projectClasses) {
    const normalizedSource = normalizeLabel(sourceLabel);
    if (!normalizedSource) return "";
    const matches = (Array.isArray(projectClasses) ? projectClasses : [])
      .map((value) => String(value ?? ""))
      .filter((value) => value && normalizeLabel(value) === normalizedSource);
    return matches.length === 1 ? matches[0] : "";
  }

  function mappingPreferenceKey(serverUrl, projectId, sourceLabel) {
    const server = String(serverUrl ?? "").trim().replace(/\/+$/, "").toLowerCase();
    return `cvat_mapping:${[server, projectId, sourceLabel]
      .map((value) => encodeURIComponent(String(value ?? "")))
      .join(":")}`;
  }

  async function refreshSelectedProject(projects, selectedId, serverUrl, request) {
    const projectId = String(selectedId ?? "");
    if (!projectId) return Array.isArray(projects) ? projects : [];
    if (typeof request !== "function") {
      throw new TypeError("A request function is required to refresh the CVAT project.");
    }

    const body = { project_id: projectId };
    if (String(serverUrl ?? "").trim()) body.server_url = String(serverUrl).trim();
    const response = await request("/api/cvat/project", {
      method: "POST",
      body: JSON.stringify(body),
    });
    const freshProject = response?.project;
    const freshId = String(freshProject?.project_id ?? freshProject?.id ?? freshProject?.pk ?? "");
    if (!freshProject || freshId !== projectId) {
      throw new Error("CVAT did not return details for the selected project.");
    }

    const updated = Array.isArray(projects) ? projects.slice() : [];
    const index = updated.findIndex((project) =>
      String(project.project_id ?? project.id ?? project.pk) === projectId,
    );
    if (index < 0) updated.push(freshProject);
    else updated[index] = freshProject;
    return updated;
  }

  window.AmphiLensCvatProjectState = {
    matchProjectClass,
    mappingPreferenceKey,
    refreshSelectedProject,
  };
})();
