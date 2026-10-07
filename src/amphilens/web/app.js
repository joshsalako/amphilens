(() => {
  "use strict";

  const root = document.getElementById("view-root");
  const dialog = document.getElementById("project-dialog");
  const app = {
    view: "home",
    project: null,
    doctor: null,
    modalConnectivity: null,
    defaultProjectRoot: "",
    defaultModelId: "",
    models: [],
    hostedModels: [],
    modalGpus: [],
    jobs: new Map(),
    jobMeta: new Map(),
    activeJob: null,
    cvatProjects: null,
    cvatServerUrl: "",
    cloudEstimate: null,
    cloudJobs: [],
    cloudRefreshTimer: null,
    cloudRefreshRunning: false,
    bootstrapped: false,
  };

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const safeHref = (value) => {
    try {
      const parsed = new URL(String(value), window.location.href);
      return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "#";
    } catch { return "#"; }
  };
  const formatNumber = (value) => Number.isFinite(Number(value)) ? new Intl.NumberFormat().format(Number(value)) : "—";
  const basename = (value) => String(value || "").replace(/[\\/]+$/, "").split(/[\\/]/).pop() || value || "Unnamed project";
  const currentProject = () => app.project;

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    let payload;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) {
      const detail = [payload.detail, payload.error, payload.message].find((value) => typeof value === "string" && value.trim());
      throw new Error(detail ? detail.replace(/[\u0000-\u001f\u007f]/g, " ").slice(0, 300) : `Request failed (${response.status}). Try again, and check the local terminal if the problem continues.`);
    }
    return payload;
  }

  function toast(message, isError = false) {
    const region = document.getElementById("toast-region");
    const item = document.createElement("div");
    item.className = `toast${isError ? " error" : ""}`;
    item.setAttribute("role", isError ? "alert" : "status");
    item.textContent = message;
    region.append(item);
    window.setTimeout(() => item.remove(), 5200);
  }

  function setPage(view) {
    app.view = view;
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      if (button.matches("button.nav-link")) {
        if (active) button.setAttribute("aria-current", "page");
        else button.removeAttribute("aria-current");
      }
    });
    render();
    document.getElementById("main-content").focus({ preventScroll: true });
  }

  function activeLocalJob(job) {
    return !["completed", "failed", "canceled"].includes(String(job?.state || "").toLowerCase());
  }

  function activeCloudJob(job) {
    return !["finished", "verified", "incomplete", "failed", "canceled", "timed_out"].includes(String(job?.state || "").toLowerCase());
  }

  function progressDetails(job, cloud = false) {
    return cloud ? (job?.progress_details || job?.progress || {}) : (job?.progress || {});
  }

  function formatDuration(seconds) {
    const value = Math.max(0, Math.round(Number(seconds)));
    if (!Number.isFinite(value)) return "";
    if (value < 60) return `${value}s`;
    if (value < 3600) return `${Math.floor(value / 60)}m ${value % 60}s`;
    return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
  }

  function humanize(value) {
    const text = String(value || "").replace(/[_-]+/g, " ");
    return text ? text[0].toUpperCase() + text.slice(1) : "";
  }

  function displayDevice(value) {
    const text = String(value || "").trim();
    if (/^modal\s+/i.test(text)) return text.replace(/^modal\s+/i, "");
    if (/^cuda(?::\d+)?$/i.test(text)) return "CUDA GPU";
    if (/^mps(?::\d+)?$/i.test(text)) return "Apple MPS GPU";
    if (/^cpu$/i.test(text)) return "CPU";
    return text;
  }

  function activityProgress(job, cloud = false) {
    const details = progressDetails(job, cloud);
    const countPhase = ["prediction", "image_transfer", "saving_results"].includes(details.phase);
    const completed = countPhase ? Number(details.completed) : NaN;
    const failed = countPhase ? Number(details.failed || 0) : NaN;
    const remaining = countPhase ? Number(details.remaining) : NaN;
    const total = countPhase ? Number(details.total) : NaN;
    const counts = [
      Number.isFinite(completed) ? `${formatNumber(completed)} done` : "",
      Number.isFinite(failed) && failed > 0 ? `${formatNumber(failed)} failed` : "",
      Number.isFinite(remaining) ? `${formatNumber(remaining)} remaining` : "",
    ].filter(Boolean).join(" · ");
    const epoch = Number(details.epoch);
    const epochs = Number(details.epochs);
    const epochText = Number.isFinite(epoch) && Number.isFinite(epochs) && epochs > 0
      ? `Epoch ${formatNumber(epoch)} of ${formatNumber(epochs)}` : "";
    const metricEntries = Object.entries(details.metrics || {}).slice(0, 5);
    const metrics = metricEntries.map(([key, value]) => `${key.replace(/[_-]+/g, " ")} ${typeof value === "number" ? Number(value).toFixed(4).replace(/0+$/, "").replace(/\.$/, "") : value}`).join(" · ");
    const overallPercent = cloud ? (job.progress ?? details.progress) : (typeof job.progress === "number" ? job.progress : details.progress);
    let percent = overallPercent == null ? NaN : Number(overallPercent);
    if (!Number.isFinite(percent) && Number.isFinite(completed) && Number.isFinite(total) && total > 0) percent = completed / total;
    if (Number.isFinite(percent)) percent = Math.max(0, Math.min(100, percent <= 1 ? percent * 100 : percent));
    const rawPhasePercent = details.phase_progress;
    const phasePercent = rawPhasePercent == null || !Number.isFinite(Number(rawPhasePercent))
      ? null : Math.max(0, Math.min(100, Number(rawPhasePercent) <= 1 ? Number(rawPhasePercent) * 100 : Number(rawPhasePercent)));
    const eta = Number(details.eta_seconds);
    return { details, counts, epochText, metrics, percent, phasePercent, device: displayDevice(details.device || details.gpu), eta: Number.isFinite(eta) && eta >= 0 ? formatDuration(eta) : "" };
  }

  function activityTimeline(job, cloud = false) {
    return window.AmphiLensActivityView?.renderTimeline(job, cloud) || "";
  }

  function activityCard(job, { cloud = false, id = "", title = "Run", kind = "" } = {}) {
    const progress = activityProgress(job, cloud);
    const state = String(cloud ? job.state || job.phase || "saved" : job.state || "queued").replace(/[_-]+/g, " ");
    const active = cloud ? activeCloudJob(job) : activeLocalJob(job);
    const phase = humanize(progress.details.phase || job.phase || state);
    const message = progress.details.message || job.error || job.result?.message || job.message || (active ? "Working in the background" : state);
    const statusClass = /failed|incomplete|timed out/.test(state) ? "is-failed" : /finished|verified|completed/.test(state) ? "is-completed" : "";
    const shownPercent = progress.phasePercent ?? progress.percent;
    const bar = shownPercent == null
      ? (active ? `<div class="progress-track" aria-label="Progress"><div class="progress-fill indeterminate"></div></div>` : "")
      : `<div class="progress-track" role="progressbar" aria-label="${escapeHtml(title)} progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(shownPercent)}"><div class="progress-fill" style="width:${shownPercent}%"></div></div>`;
    const context = [cloud || app.jobMeta.get(id)?.execution === "modal" ? "Modal" : "This computer", progress.device, phase, progress.phasePercent != null ? `Step ${Math.round(progress.phasePercent)}%` : "", progress.epochText, progress.counts, progress.metrics, progress.eta ? `About ${progress.eta} left` : ""].filter(Boolean).join(" · ");
    const error = job.error ? `<p class="activity-error">${escapeHtml(job.error)}</p>` : "";
    const cancel = !cloud && active && kind === "predict" && app.jobMeta.get(id)?.execution === "modal"
      ? `<button class="button small" type="button" data-cancel-job="${escapeHtml(id)}">Cancel after current batch</button>` : "";
    return `<article class="activity-card" data-activity-id="${cloud ? "cloud" : "local"}:${escapeHtml(id)}"><div class="activity-card-head"><div><strong>${escapeHtml(title)}</strong><span class="activity-state ${statusClass}">${escapeHtml(state)}</span></div></div><p class="activity-message">${escapeHtml(message)}</p><p class="activity-context">${escapeHtml(context)}</p>${bar}${error}${cancel}${activityTimeline(job, cloud)}</article>`;
  }

  function renderActivity() {
    const panel = document.getElementById("activity-content");
    const count = document.querySelector("[data-activity-count]");
    if (!panel) return;
    const openRuns = new Set([...panel.querySelectorAll(".activity-details[open]")].map((details) => details.closest("[data-activity-id]")?.dataset.activityId).filter(Boolean));
    const scrollTop = panel.querySelector(".activity-list")?.scrollTop || 0;
    const local = [...app.jobs.entries()].slice(-12).reverse().map(([id, job]) => {
      const meta = app.jobMeta.get(id) || {};
      return activityCard(job, { id, title: meta.title || (meta.kind === "training" ? "Model training" : meta.kind === "predict" ? "Wildlife prediction" : "Workspace task"), kind: meta.kind || "" });
    });
    const cloud = app.cloudJobs.slice(0, 12).map((job) => activityCard(job, { cloud: true, id: job.run_id, title: cloudJobTitle(job), kind: "training" }));
    const running = [...app.jobs.values()].filter(activeLocalJob).length + app.cloudJobs.filter(activeCloudJob).length;
    if (count) { count.textContent = running ? String(running) : ""; count.hidden = !running; }
    const toggle = document.querySelector("[data-activity-toggle]");
    if (toggle) toggle.classList.toggle("has-running", running > 0);
    panel.innerHTML = `<div class="activity-heading"><div><p class="eyebrow">Run monitor</p><h2>Activity</h2><p>${running ? `${running} run${running === 1 ? " is" : "s are"} active. Updates appear here while you work.` : "Recent prediction and training runs appear here."}</p></div><button class="icon-button" type="button" data-activity-close aria-label="Close activity panel">×</button></div>${local.length || cloud.length ? `<div class="activity-list">${local.join("")}${cloud.join("")}</div>` : `<p class="activity-empty">No runs yet. Start a prediction or training run to follow its progress here.</p>`}`;
    for (const details of panel.querySelectorAll(".activity-details")) {
      if (openRuns.has(details.closest("[data-activity-id]")?.dataset.activityId)) details.open = true;
    }
    const list = panel.querySelector(".activity-list");
    if (list) list.scrollTop = scrollTop;
    window.AmphiLensActivityView?.scrollOutputToBottom(panel);
  }

  async function refreshBootstrap() {
    root.setAttribute("aria-busy", "true");
    try {
      const data = await api("/api/bootstrap", { method: "GET", headers: {} });
      app.project = data.project || null;
      app.doctor = data.doctor || {};
      if (app.modalConnectivity !== null) {
        app.doctor.modal_connectivity = app.modalConnectivity;
      }
      app.defaultProjectRoot = data.default_project_root || "";
      app.defaultModelId = data.default_model_id || "";
      app.models = app.project?.models || data.models || [];
      app.hostedModels = data.hosted_models || app.project?.hosted_models || [];
      app.modalGpus = data.modal_gpus || [];
      app.bootstrapped = true;
      updateProjectHeader();
      populateModelPreset();
      render();
      if (app.project) loadCloudJobs();
    } catch (error) {
      root.innerHTML = `<section class="surface empty-state"><div class="empty-illustration" aria-hidden="true">!</div><h2>Workspace could not load</h2><p>${escapeHtml(error.message)}. Check that the local AmphiLens service is running, then try again.</p><button class="button primary" type="button" data-retry-bootstrap>Try again</button></section>`;
    } finally {
      root.setAttribute("aria-busy", "false");
    }
  }

  function updateProjectHeader() {
    document.getElementById("current-project").textContent = app.project ? app.project.name || basename(app.project.path) : "No project open";
    document.getElementById("close-project-wrap").hidden = !app.project;
    if (!app.project && app.cloudRefreshTimer) {
      window.clearInterval(app.cloudRefreshTimer);
      app.cloudRefreshTimer = null;
    }
  }

  function populateModelPreset() {
    const select = document.getElementById("model-preset");
    if (!select) return;
    const list = app.models;
    const saved = select.value;
    select.innerHTML = list.length
      ? list.map((model) => `<option value="${escapeHtml(model.model_id)}">${escapeHtml(model.name || model.architecture || model.model_id)}</option>`).join("")
      : '<option value="">No model presets available</option>';
    select.disabled = list.length === 0;
    select.value = saved || app.defaultModelId || list[0]?.model_id || "";
  }

  function pageHead(eyebrow, title, description, action = "") {
    return `<header class="view-head"><div class="view-head-copy"><p class="eyebrow">${escapeHtml(eyebrow)}</p><h1>${escapeHtml(title)}</h1><p class="lead">${escapeHtml(description)}</p></div>${action}</header>`;
  }

  function pathPickerControl(id, name, kind, placeholder, options = {}) {
    const label = options.label || (kind === "directory" ? "a folder" : "a file");
    return `<div class="path-picker-control"><input id="${escapeHtml(id)}" name="${escapeHtml(name)}" value="${escapeHtml(options.value || "")}" placeholder="${escapeHtml(placeholder)}"${options.required ? " required" : ""}><button class="button path-picker-button" type="button" data-path-picker="${escapeHtml(kind)}" data-path-target="${escapeHtml(id)}" aria-label="Browse to select ${escapeHtml(label)}">Browse</button></div>`;
  }

  function projectRequired() {
    if (app.project) return false;
    root.innerHTML = `${pageHead("Project needed", "Start with a project", "A project keeps your image folders, datasets and model runs together.")}
      <section class="surface empty-state" style="margin-top:0"><div class="empty-illustration" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M4 9.5h9l2.3 2.7H28v12.3H4V9.5Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M4 13.5h24" stroke="currentColor" stroke-width="1.8"/></svg></div><h2>Open or create a project</h2><p>Once a project is open, you can scan images with a pretrained model right away. Labels and CVAT are only needed when you choose to improve a model.</p><div class="project-actions"><button class="button primary" type="button" data-open-project data-project-tab="create">Create project</button><button class="button" type="button" data-open-project>Open existing project</button></div></section>`;
    return true;
  }

  function renderHome() {
    if (!app.project) {
      root.innerHTML = `${pageHead("Field workspace", "Your wildlife field desk", "Scan camera-trap images with a pretrained model, or build a labeled dataset when you are ready.")}
      <section class="surface empty-state"><div class="empty-illustration" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><circle cx="16" cy="16" r="12" stroke="currentColor" stroke-width="1.7"/><circle cx="16" cy="16" r="7" stroke="currentColor" stroke-width="1.7"/><circle cx="16" cy="16" r="2.3" fill="currentColor"/></svg></div><h2>Set up your first project</h2><p>Choose a project folder and the image folder you want to scan. You can make predictions before collecting any annotations.</p><div class="project-actions"><button class="button primary" type="button" data-open-project data-project-tab="create">Create project</button><button class="button" type="button" data-open-project>Open existing project</button></div></section>`;
      return;
    }
    const project = app.project;
    const datasets = project.datasets || [];
    const checkpoints = project.checkpoints || [];
    const images = project.counts?.images ?? project.image_count;
    const imageRoot = project.image_roots?.[0] || "";
    root.innerHTML = `${pageHead("Field workspace", "Good to have you back", `${project.name || basename(project.path)} is ready. Start with a scan, or continue improving the project with labeled examples.`, `<button class="button" type="button" data-open-project>Switch project</button>`)}
      <section class="surface hero-panel">
        <div class="hero-copy"><p class="eyebrow">Prediction-only workflow</p><h2>Find wildlife in your images</h2><p>Use a pretrained model or one of your saved checkpoints. No labels or CVAT setup needed.</p><div class="hero-actions"><button class="button primary" type="button" data-view="predict">Run detection</button><button class="button subtle" type="button" data-view="import">Import labeled images</button></div></div>
        <div class="hero-art" aria-hidden="true"><span class="hero-sun"></span><svg class="marsh-lines" viewBox="0 0 380 260" fill="none"><path d="M2 211c62-45 118-49 179-20 67 32 119 31 197-5M12 230c66-41 117-43 174-17 72 32 130 27 187-3M39 250c51-29 93-32 143-11 65 27 125 24 184-1M54 194c51-36 93-42 145-18 65 30 112 31 172 6M84 174c37-25 77-28 119-10 55 24 106 24 157 5" stroke="currentColor" stroke-width="1.2"/><path d="M226 260V114m0 76c-26-25-43-30-59-29m59 3c17-32 35-43 53-45m-99 71c-15-10-28-14-43-13m134-33c13-15 26-21 42-23" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M167 161c16-5 31 4 35 15-14 3-29-1-35-15Zm59-20c6-17 19-24 34-23-3 15-15 25-34 23Zm-99 36c10-9 23-9 33-2-8 9-20 11-33 2Z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg><span class="hero-reed r1"></span><span class="hero-reed r2"></span><span class="hero-reed r3"></span></div>
      </section>
      <div class="content-grid">
        <section class="surface panel"><div class="panel-heading"><div><h2>Choose your next step</h2><p>Each workflow keeps your original images safe.</p></div></div><div class="quick-list">
          <button class="quick-row" type="button" data-view="predict"><span class="quick-icon" aria-hidden="true">⌕</span><span class="quick-text"><strong>Find wildlife</strong><span>Scan an image folder and save predictions and reports.</span></span><span class="quick-chevron" aria-hidden="true">›</span></button>
          <button class="quick-row" type="button" data-view="import"><span class="quick-icon" aria-hidden="true">⇧</span><span class="quick-text"><strong>Import labeled images</strong><span>Start from an archive or an existing annotated CVAT project.</span></span><span class="quick-chevron" aria-hidden="true">›</span></button>
          <button class="quick-row" type="button" data-view="review"><span class="quick-icon" aria-hidden="true">◉</span><span class="quick-text"><strong>Choose images to review</strong><span>Select a smaller, informative queue for annotation.</span></span><span class="quick-chevron" aria-hidden="true">›</span></button>
        </div></section>
        <aside class="surface panel"><div class="panel-heading"><div><h2>This project</h2><p>Saved locally on this computer</p></div></div><div class="current-project-line"><span class="project-glyph" aria-hidden="true">▤</span><div><strong>${escapeHtml(project.name || basename(project.path))}</strong><span class="project-path">${escapeHtml(project.path)}</span></div></div><div class="small-stat-list"><div class="small-stat"><span>Images indexed</span><strong>${formatNumber(images)}</strong></div><div class="small-stat"><span>Dataset snapshots</span><strong>${formatNumber(project.counts?.datasets ?? datasets.length)}</strong></div><div class="small-stat"><span>Model runs</span><strong>${formatNumber(project.counts?.runs)}</strong></div></div>${imageRoot ? `<p class="project-path">Image folder: ${escapeHtml(imageRoot)}</p>` : ""}<button class="button panel-action" type="button" data-open-project>Manage project</button></aside>
      </div>
      <p class="section-title">Recent work</p><section class="surface panel"><div class="panel-heading"><div><h2>Datasets &amp; models</h2><p>Choose one when you are ready to train a project-specific model.</p></div></div>${datasets.length || checkpoints.length ? `<div class="small-stat-list">${datasets.slice(0, 3).map((item) => `<div class="small-stat"><span>${escapeHtml(item.name || basename(item.path))} · ${formatNumber(item.image_count)} images</span><strong>Dataset</strong></div>`).join("")}${checkpoints.slice(0, 3).map((item) => `<div class="small-stat"><span>${escapeHtml(item.name || item.model_id || basename(item.path))}</span><strong>${escapeHtml(item.architecture || "Model")}</strong></div>`).join("")}</div>` : `<p class="empty-inline">No labeled dataset or project checkpoint yet. You can still run predictions with a pretrained model.</p>`}</section>`;
  }

  function modelOptions() {
    const saved = app.project?.checkpoints || [];
    const presets = app.models;
    const hosted = app.hostedModels || [];
    const preferred = app.project?.model_preset || app.defaultModelId;
    const presetOptions = presets.map((item) => `<option value="preset:${escapeHtml(item.model_id)}"${item.model_id === preferred ? " selected" : ""}>${escapeHtml(item.name || item.model_id)}</option>`).join("");
    const hostedOptions = hosted.map((item) => `<option value="hosted:${escapeHtml(item.model_id)}">${escapeHtml(item.name || item.model_id)}</option>`).join("");
    return `${presetOptions ? `<optgroup label="General pretrained models">${presetOptions}</optgroup>` : ""}${hostedOptions ? `<optgroup label="AmphiLens fine-tuned models">${hostedOptions}</optgroup>` : ""}${saved.length ? `<optgroup label="Project checkpoints">${saved.map((item) => `<option value="checkpoint:${escapeHtml(item.path)}">${escapeHtml(item.name || item.model_id || basename(item.path))}</option>`).join("")}</optgroup>` : ""}`;
  }

  function hostedModelById(modelId) {
    return (app.hostedModels || []).find((item) => item.model_id === modelId) || null;
  }

  function hostedModelDetails(model) {
    if (!model) return "";
    const dimensions = model.preprocessing?.compatibility_mode === "shortest-side"
      ? "640 px short side, rounded up to a multiple of 32"
      : "640 px maximum side, rounded up to a multiple of 32";
    return `<p class="hosted-model-facts"><strong>${escapeHtml(model.architecture)}</strong><span>${escapeHtml(model.size)}</span><span>Inference size ${formatNumber(model.inference_image_size)} px</span><span>${escapeHtml(dimensions)}</span></p><div class="notice hosted-access-note"><span class="notice-mark" aria-hidden="true">i</span><p>This public Hugging Face model can be downloaded directly by Modal for cloud prediction. Modal uses the model card revision recorded by AmphiLens and verifies the checkpoint SHA-256. Local prediction downloads it to this computer.</p></div>`;
  }

  function renderPredictionModelSelection() {
    const select = document.getElementById("prediction-model");
    const details = document.getElementById("prediction-hosted-details");
    const mapping = document.getElementById("prediction-class-mapping");
    if (!select || !details || !mapping) return;
    const [, kind, selected] = String(select.value || "").match(/^(preset|checkpoint|hosted):(.*)$/s) || [];
    const model = kind === "hosted" ? hostedModelById(selected) : null;
    details.hidden = !model;
    details.innerHTML = model ? hostedModelDetails(model) : "";
    mapping.hidden = !model;
    if (!model) {
      mapping.innerHTML = "";
      return;
    }
    const projectClasses = app.project?.classes || [];
    const rows = (model.source_class_order || []).map((source, index) => {
      const exact = projectClasses.includes(source);
      const targetOptions = projectClasses.map((name) => `<option value="${escapeHtml(name)}"${name === source ? " selected" : ""}>${escapeHtml(name)}</option>`).join("");
      return `<div class="cvat-label-row"><label for="hosted-class-map-${index}">${escapeHtml(source)}${exact ? `<span class="class-match">Exact match</span>` : ""}</label><select id="hosted-class-map-${index}" name="hosted_mapping_${index}" data-class-map-source="${escapeHtml(source)}" required><option value="">Choose a project class or Ignore</option>${targetOptions}<option value="__ignore__">Ignore detections</option></select></div>`;
    }).join("");
    mapping.innerHTML = `<div><span class="field-title">Map model labels to project classes</span><p class="field-help">Exact label matches are preselected. Map every other label or choose Ignore.</p></div><div class="cvat-label-list">${rows}</div>`;
  }

  function predictionPreferenceScope() {
    return app.project?.path || app.project?.name || "";
  }

  function savePredictionPreferences(form) {
    if (!form || !predictionPreferenceScope()) return;
    window.AmphiLensPredictionPreferences?.save(predictionPreferenceScope(), form);
  }

  function restorePredictionPreferences(form) {
    if (!form || !predictionPreferenceScope()) return;
    window.AmphiLensPredictionPreferences?.restore(predictionPreferenceScope(), form);
  }

  function syncPredictionExecutionControls(form) {
    if (!form) return;
    const cloud = form.elements.execution?.value === "modal";
    form.querySelector("[data-modal-prediction-settings]")?.toggleAttribute("hidden", !cloud);
    form.querySelector("[data-local-prediction-settings]")?.toggleAttribute("hidden", cloud);
  }

  function renderPredict() {
    if (projectRequired()) return;
    const roots = app.project.image_roots || [];
    const currentRoot = roots[0] || "";
    root.innerHTML = `${pageHead("Find wildlife", "Run detection", "Scan an image folder and create a predictions file with reports. This workflow works without annotations or CVAT.")}
      <div class="form-layout"><section class="surface form-panel"><div class="panel-heading"><div><h2>Choose images and model</h2><p>Original image files are never changed.</p></div></div>
        <form class="form-stack" data-form="predict">
          <div class="field"><label for="prediction-root">Image folder</label>${pathPickerControl("prediction-root", "image_root", "directory", "/path/to/camera-trap-images", { value: currentRoot, required: true, label: "an image folder" })}<span class="field-help">Choose a folder visible to the computer running AmphiLens.</span></div>
          <div class="field"><label for="prediction-model">Model</label><select id="prediction-model" name="model_choice">${modelOptions()}</select></div>
          <div id="prediction-hosted-details" class="hosted-model-panel" hidden></div>
          <div id="prediction-class-mapping" class="hosted-class-mapping" hidden></div>
          <div class="field"><label for="prediction-execution">Where to run</label><select id="prediction-execution" name="execution"><option value="local">This computer</option><option value="modal">Modal cloud GPU</option></select><span class="field-help">Cloud prediction transfers images in small batches. Images remain in this project on your computer.</span></div>
          <details class="advanced-settings"><summary>Detection settings</summary><div class="advanced-body">
            <div class="field"><label for="confidence">Confidence threshold</label><div class="range-field"><input id="confidence" name="confidence" type="range" min="0.05" max="0.95" step="0.05" value="0.25"><output class="range-value" for="confidence">0.25</output></div><span class="field-help">Higher values keep only more confident detections.</span></div>
            <div class="field"><label for="prediction-output">Save results in <span class="optional">optional</span></label>${pathPickerControl("prediction-output", "output_dir", "directory", "Use the project runs folder", { label: "a results folder" })}</div>
            <div class="field" data-local-prediction-settings><label for="prediction-device">Local device</label><select id="prediction-device" name="device"><option value="auto">Auto</option><option value="cpu">CPU</option>${app.doctor?.cuda_available ? `<option value="cuda">CUDA GPU</option>` : ""}${app.doctor?.mps_available ? `<option value="mps">Apple MPS GPU</option>` : ""}</select><span class="field-help">Auto uses CUDA or Apple MPS when available, then CPU.</span></div>
            <div class="field"><label for="prediction-batch-size">Batch size</label><select id="prediction-batch-size" name="batch_size"><option value="auto" selected>Auto</option>${Array.from({ length: 32 }, (_, index) => `<option value="${index + 1}">${index + 1}</option>`).join("")}</select><span class="field-help">Auto starts at up to 8 images on GPU and reduces the batch if memory is low. CPU always processes one image at a time.</span></div>
            <div class="modal-prediction-settings" data-modal-prediction-settings hidden><div class="field"><label for="prediction-gpu">Modal GPU</label><select id="prediction-gpu" name="gpu">${app.modalGpus.map((gpu) => `<option value="${escapeHtml(gpu.name)}"${gpu.name === "L4" ? " selected" : ""}>${escapeHtml(gpu.name)} · $${Number(gpu.usd_per_hour).toFixed(3)}/GPU hour</option>`).join("")}</select><span class="field-help">One remote GPU worker processes one image batch at a time.</span></div><div class="field"><label for="prediction-max-cost">Spending limit (USD)</label><input id="prediction-max-cost" name="max_cost_usd" type="number" min="0.01" step="0.01" value="5.00" required><span class="field-help">The worker stops scheduling batches when its runtime-based estimate reaches this amount.</span></div></div>
          </div></details>
          <div class="form-actions"><button class="button primary" type="submit">Run detection</button><span class="muted" style="font-size:.75rem">You can leave this page while it runs.</span></div>
        </form><div id="job-slot" aria-live="polite"></div>
      </section><aside class="surface side-note"><h3>What you will get</h3><p>Predictions include image names, detected classes, confidence scores and bounding boxes. AmphiLens can also save a run summary and visual evidence when available.</p><div class="note-rule"></div><h3>Choosing a model</h3><p>Use general pretrained weights to begin, an AmphiLens fine-tuned model for its trained classes, or a project checkpoint you have already trained.</p></aside></div>`;
    const form = root.querySelector('form[data-form="predict"]');
    restorePredictionPreferences(form);
    renderPredictionModelSelection();
    restorePredictionPreferences(form);
    const confidenceOutput = form?.querySelector('output[for="confidence"]');
    if (confidenceOutput && form.elements.confidence) {
      confidenceOutput.value = Number(form.elements.confidence.value).toFixed(2);
    }
    syncPredictionExecutionControls(form);
  }

  function renderImport() {
    if (projectRequired()) return;
    root.innerHTML = `${pageHead("Label & improve", "Import labeled images", "Bring in your first reviewed examples from a local archive or an existing CVAT project.")}
      <div class="content-grid">
        <section class="surface form-panel"><div class="panel-heading"><div><h2>Local annotation archive</h2><p>Use an export that includes both annotations and image files.</p></div></div>
          <form class="form-stack" data-form="import">
            <div class="field"><label for="dataset-archive">Archive path</label>${pathPickerControl("dataset-archive", "archive", "file", "/path/to/annotations.zip", { required: true, label: "an annotation archive" })}<span class="field-help">The archive must be accessible from the computer running AmphiLens.</span></div>
            <details class="advanced-settings"><summary>Map archive labels to project classes</summary><div class="advanced-body"><div class="field"><label for="class-mapping">Class mapping <span class="optional">optional</span></label><textarea id="class-mapping" name="class_mapping" placeholder="source label = project class"></textarea><span class="field-help">Add one verified mapping per line only when names differ.</span></div></div></details>
            <div class="notice"><span class="notice-mark" aria-hidden="true">i</span><p>AmphiLens checks the archive before creating an immutable dataset snapshot. Your source archive stays untouched.</p></div>
            <div class="form-actions"><button class="button primary" type="submit">Import archive</button></div>
          </form><div id="job-slot" aria-live="polite"></div>
        </section>
        <aside class="surface panel"><div class="panel-heading"><div><h2>Already using CVAT?</h2><p>Import a complete project, including its tasks and images.</p></div></div><p class="empty-inline">Connect to your CVAT server to select the initial annotated project and review its labels before importing.</p><div class="field" style="margin-top:13px"><label for="initial-cvat-server">Server URL <span class="optional">optional</span></label><input id="initial-cvat-server" name="server_url" inputmode="url" value="${escapeHtml(app.cvatServerUrl)}" placeholder="https://cvat.example.org"></div><button class="button panel-action" type="button" data-connect-cvat>Connect to CVAT</button><div id="cvat-projects-slot" aria-live="polite">${renderCvatProjects()}</div><div class="note-rule"></div><p class="field-help">CVAT credentials are not stored in your project.</p></aside>
      </div>`;
  }

  function renderCvatProjects() {
    if (app.cvatProjects === null) return "";
    if (!app.cvatProjects.length) return `<div class="notice warning" style="margin-top:13px"><span class="notice-mark" aria-hidden="true">!</span><p>No CVAT projects were found. Check the server and credentials, then try again.</p></div>`;
    const firstProject = app.cvatProjects[0];
    return `<form id="initial-cvat-form" class="form-stack cvat-import-form" data-form="initial-cvat" style="margin-top:14px"><div class="field"><label for="initial-cvat-project">CVAT project</label><select id="initial-cvat-project" name="project_id" required>${app.cvatProjects.map((project) => { const id = project.project_id ?? project.id ?? project.pk; const name = project.name || project.title || `Project ${id ?? ""}`; return `<option value="${escapeHtml(id)}">${escapeHtml(name)}</option>`; }).join("")}</select><span class="field-help">The complete project will be imported, including all tasks and image files.</span></div><div id="initial-cvat-summary">${renderCvatProjectSummary(firstProject)}</div><button class="button primary" type="submit" ${cvatProjectIssue(firstProject) ? "disabled" : ""}>Import CVAT project</button><div class="form-error" data-error-for="initial-cvat-form" role="alert"></div><div id="cvat-import-job-slot" aria-live="polite"></div></form>`;
  }

  function cvatLabelName(label) {
    return typeof label === "string" ? label : label?.name || label?.label || String(label?.id ?? "");
  }

  function renderCvatProjectSummary(project) {
    if (!project) return "";
    const labels = (Array.isArray(project.labels) ? project.labels : []).map(cvatLabelName).filter(Boolean);
    const classes = (app.project?.classes || []).map((item) => typeof item === "string" ? item : item?.name).filter(Boolean);
    const tasks = Array.isArray(project.tasks) ? project.tasks : [];
    const taskCount = Number(project.task_count ?? tasks.length);
    const labelMap = labels.map((label, index) => {
      if (classes.includes(label)) return `<div class="cvat-label-row"><span>${escapeHtml(label)}</span><span class="class-match">Matches project class</span></div>`;
      return `<div class="cvat-label-row"><label for="initial-cvat-map-${index}">${escapeHtml(label)}</label><select id="initial-cvat-map-${index}" name="mapping_${index}" data-cvat-map-source="${escapeHtml(label)}" required><option value="">Choose matching project class</option>${classes.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("")}</select></div>`;
    }).join("");
    const issue = cvatProjectIssue(project);
    return `<div class="cvat-project-summary"><div class="small-stat"><span>CVAT tasks</span><strong>${formatNumber(taskCount)}</strong></div><div class="field"><span class="field-title">Project labels</span>${labels.length ? `<div class="cvat-label-list">${labelMap}</div>` : `<div class="notice warning"><span class="notice-mark" aria-hidden="true">!</span><p>No labels were returned for this CVAT project.</p></div>`}</div>${issue ? `<div class="notice error"><span class="notice-mark" aria-hidden="true">!</span><p>${escapeHtml(issue)} Select a different project or review its contents in CVAT.</p></div>` : ""}${tasks.length ? `<details class="advanced-settings"><summary>Review task list (${tasks.length})</summary><ul class="task-list">${tasks.map((task) => `<li><span>${escapeHtml(task.name || `Task ${task.task_id ?? ""}`)}</span><span>${formatNumber(task.size)} images · ${escapeHtml(task.status || "")}</span></li>`).join("")}</ul></details>` : ""}</div>`;
  }

  function cvatProjectIssue(project) {
    const tasks = Array.isArray(project?.tasks) ? project.tasks : [];
    const labels = (Array.isArray(project?.labels) ? project.labels : []).map(cvatLabelName).filter(Boolean);
    const taskCount = Number(project?.task_count ?? tasks.length);
    if (taskCount < 1) return "This project has no tasks to import.";
    if (!labels.length) return "This project has no labels to map to your project classes.";
    if (tasks.length && tasks.every((task) => Number(task.size) === 0)) return "These tasks do not contain image files.";
    return "";
  }

  function trainingImageSize(form) {
    if (form.elements.training_source?.value !== "amphilens-pretrained") return 640;
    return hostedModelById(form.elements.hosted_model_id?.value)?.inference_image_size || 640;
  }

  function syncTrainingSourceControls(form) {
    if (!form) return;
    const source = form.elements.training_source?.value || "general-pretrained";
    const hostedWrap = form.querySelector("[data-training-hosted]");
    const checkpointWrap = form.querySelector("[data-training-checkpoint]");
    const hostedSelect = form.elements.hosted_model_id;
    const checkpointSelect = form.elements.checkpoint;
    const hostedDetails = form.querySelector("[data-training-hosted-details]");
    if (hostedWrap) hostedWrap.hidden = source !== "amphilens-pretrained";
    if (checkpointWrap) checkpointWrap.hidden = source !== "project-checkpoint";
    if (hostedSelect) hostedSelect.required = source === "amphilens-pretrained";
    if (checkpointSelect) checkpointSelect.required = source === "project-checkpoint";
    const model = hostedSelect ? hostedModelById(hostedSelect.value) : null;
    if (hostedDetails) {
      hostedDetails.hidden = source !== "amphilens-pretrained" || !model;
      hostedDetails.innerHTML = source === "amphilens-pretrained" && model ? hostedModelDetails(model) : "";
    }
  }

  function renderTraining() {
    if (projectRequired()) return;
    const datasets = app.project.datasets || [];
    const checkpoints = app.project.checkpoints || [];
    const workflow = datasets.length ? `<div class="form-layout"><section class="surface form-panel"><div class="panel-heading"><div><h2>Training setup</h2><p>Each run saves its configuration and checkpoint lineage in this project.</p></div></div>
      <form class="form-stack" data-form="training">
        <div class="field"><label for="training-snapshot">Labeled dataset snapshot</label><select id="training-snapshot" name="snapshot_path" required><option value="">Choose a dataset</option>${datasets.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name || basename(item.path))} · ${formatNumber(item.image_count)} images</option>`).join("")}</select></div>
        <div class="field"><label for="training-source">Training source</label><select id="training-source" name="training_source"><option value="general-pretrained" selected>General pretrained weights</option><option value="amphilens-pretrained">AmphiLens pretrained model</option><option value="project-checkpoint"${checkpoints.length ? "" : " disabled"}>Existing project checkpoint</option></select><span class="field-help">General pretrained weights are the default and use this project’s selected architecture.</span></div>
        <div class="field" data-training-hosted hidden><label for="training-hosted-model">AmphiLens model</label><select id="training-hosted-model" name="hosted_model_id"><option value="">Choose a model</option>${(app.hostedModels || []).map((item) => `<option value="${escapeHtml(item.model_id)}">${escapeHtml(item.name || item.model_id)}</option>`).join("")}</select><span class="field-help">The model supplies the architecture, size, and input preprocessing for this run.</span></div>
        <div class="hosted-model-panel" data-training-hosted-details hidden></div>
        <div class="field" data-training-checkpoint hidden><label for="training-checkpoint">Project checkpoint</label><select id="training-checkpoint" name="checkpoint"><option value="">Choose a saved checkpoint</option>${checkpoints.map((item) => `<option value="${escapeHtml(item.path)}">${escapeHtml(item.name || item.model_id || basename(item.path))}</option>`).join("")}</select></div>
        <div class="field"><label for="training-execution">Where should it run?</label><select id="training-execution" name="execution"><option value="local">On this computer</option><option value="cloud">Modal cloud GPU</option></select></div>
        <details class="advanced-settings"><summary>Training settings</summary><div class="advanced-body"><div class="field-row"><div class="field"><label for="training-epochs">Epochs</label><input id="training-epochs" name="epochs" type="number" min="1" max="1000" value="50"></div><div class="field"><label for="training-batch">Batch size</label><input id="training-batch" name="batch_size" type="number" min="1" max="256" value="8"></div></div><div class="field"><label for="training-output">Save run in <span class="optional">optional</span></label>${pathPickerControl("training-output", "output_dir", "directory", "Use the project runs folder", { label: "a run folder" })}</div><div class="field"><label for="training-device">Local device</label><select id="training-device" name="device"><option value="auto">Choose automatically</option><option value="cpu">CPU</option><option value="cuda">CUDA GPU</option></select></div></div></details>
        <div class="cloud-settings hidden" data-cloud-settings>
          <div class="notice warning"><span class="notice-mark" aria-hidden="true">!</span><p>Cloud training uploads a prepared copy of the selected dataset and checkpoint. Review the estimate and approve both the upload and cost before submitting.</p></div>
          <div class="field-row" style="margin-top:15px"><div class="field"><label for="training-gpu">GPU</label><select id="training-gpu" name="gpu"><option value="T4">T4</option><option value="A10G">A10G</option><option value="A100">A100</option></select></div><div class="field"><label for="training-max-cost">Maximum budget · USD</label><input id="training-max-cost" name="max_cost_usd" type="number" min="0.01" step="0.01" placeholder="Enter your limit"></div></div>
          <button class="button" type="button" data-cloud-estimate>Request estimate</button><div id="cloud-estimate-slot" aria-live="polite">${renderCloudEstimate()}</div>
          <label class="consent-row"><input type="checkbox" name="uploads_dataset"><span>I approve uploading the prepared dataset and selected checkpoint for this run.</span></label>
          <label class="consent-row"><input type="checkbox" name="acknowledged"><span>I understand the estimate is a planning range, not a guaranteed billing cap.</span></label>
        </div>
        <div class="form-actions"><button class="button primary" type="submit">Train model</button></div>
      </form><div id="job-slot" aria-live="polite"></div>
    </section><aside class="surface side-note"><h3>Before you start</h3><p>Training is available only from a validated labeled snapshot. Empty annotations can be valid for reviewed images with no target wildlife.</p><div class="note-rule"></div><p>Evaluation is reported as not evaluated when no holdout dataset is supplied.</p></aside></div>
      <section class="surface panel cloud-jobs-panel"><div class="panel-heading"><div><h2>Cloud jobs</h2><p>Active Modal training progress refreshes automatically while AmphiLens is open.</p></div><button class="button small" type="button" data-refresh-cloud-jobs>Refresh jobs</button></div><div id="cloud-jobs-slot" aria-live="polite"><p class="empty-inline">Loading saved cloud jobs…</p></div></section>` : `<section class="surface empty-state" style="margin-top:0"><div class="empty-illustration" aria-hidden="true">↗</div><h2>No labeled dataset yet</h2><p>Import a reviewed archive or CVAT project first. You can still use Find wildlife with a pretrained model while you gather annotations.</p><div class="project-actions"><button class="button primary" type="button" data-view="import">Import labeled images</button><button class="button" type="button" data-view="predict">Find wildlife</button></div></section>`;
    root.innerHTML = `${pageHead("Label & improve", "Train a model", "Fine-tune a model using a validated labeled dataset snapshot. Prediction-only projects do not need to train.")}${workflow}`;
    syncTrainingSourceControls(root.querySelector('form[data-form="training"]'));
    if (datasets.length) loadCloudJobs();
  }

  function cloudEstimateSignature(form) {
    return JSON.stringify({
      snapshot_path: form.elements.snapshot_path?.value || "",
      gpu: form.elements.gpu?.value || "",
      epochs: numberOrUndefined(form.elements.epochs?.value),
      max_cost_usd: numberOrUndefined(form.elements.max_cost_usd?.value),
      training_source: form.elements.training_source?.value || "general-pretrained",
      hosted_model_id: form.elements.training_source?.value === "amphilens-pretrained"
        ? form.elements.hosted_model_id?.value || ""
        : "",
      image_size: trainingImageSize(form),
    });
  }

  function renderCloudEstimate() {
    if (!app.cloudEstimate) return `<p class="job-description">Request an estimate for the selected dataset, GPU and budget before submitting.</p>`;
    const estimate = app.cloudEstimate;
    const cost = (value) => `$${Number(value).toFixed(2)} USD`;
    const duration = (seconds) => {
      const minutes = Math.max(1, Math.round(Number(seconds) / 60));
      return minutes >= 60 ? `${Math.floor(minutes / 60)} h ${minutes % 60} min` : `${minutes} min`;
    };
    const fields = [
      ["Estimated cost range", estimate.low_usd != null && estimate.high_usd != null ? `${cost(estimate.low_usd)}–${cost(estimate.high_usd)}` : null],
      ["Training time", estimate.runtime_low_seconds != null && estimate.runtime_high_seconds != null ? `${duration(estimate.runtime_low_seconds)}–${duration(estimate.runtime_high_seconds)}` : null],
      ["Dataset upload time", estimate.upload_time_low_seconds != null && estimate.upload_time_high_seconds != null ? `${duration(estimate.upload_time_low_seconds)}–${duration(estimate.upload_time_high_seconds)}` : null],
      ["Images in snapshot", estimate.image_count != null ? formatNumber(estimate.image_count) : null],
      ["GPU", estimate.gpu || null],
    ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    return `<div class="estimate-card"><p>${escapeHtml(estimate.disclaimer || "Planning range based on recent provider rates and an unverified runtime estimate.")}</p>${fields.length ? `<div class="estimate-values">${fields.map(([label, value]) => `<div class="estimate-value"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>` : ""}${estimate.time_limit_seconds != null ? `<p style="margin-top:9px">Budget-derived run time limit: ${escapeHtml(duration(estimate.time_limit_seconds))}. Provider billing is not capped by this estimate.</p>` : ""}${estimate.rates_stale ? `<p style="margin-top:9px">Provider rates may be out of date. Checked ${escapeHtml(estimate.rate_checked_at || "previously")}.</p>` : ""}</div>`;
  }

  async function requestCloudEstimate(button) {
    const form = button.closest("form");
    const slot = document.getElementById("cloud-estimate-slot");
    if (!form || !form.elements.snapshot_path.value) { toast("Choose a labeled dataset first.", true); return; }
    form.elements.uploads_dataset.checked = false;
    form.elements.acknowledged.checked = false;
    const payload = {
      snapshot_path: form.elements.snapshot_path.value,
      gpu: form.elements.gpu.value,
      epochs: numberOrUndefined(form.elements.epochs.value),
      max_cost_usd: numberOrUndefined(form.elements.max_cost_usd.value),
      training_source: form.elements.training_source?.value || "general-pretrained",
      hosted_model_id: form.elements.training_source?.value === "amphilens-pretrained"
        ? form.elements.hosted_model_id?.value || undefined
        : undefined,
      image_size: trainingImageSize(form),
    };
    const requestedSignature = cloudEstimateSignature(form);
    if (!payload.max_cost_usd || payload.max_cost_usd <= 0) { toast("Enter a positive maximum budget.", true); return; }
    button.disabled = true;
    button.textContent = "Estimating…";
    if (slot) slot.innerHTML = `<p class="job-description">Building a cost estimate…</p>`;
    try {
      const result = await api("/api/cloud/estimate", { method: "POST", body: JSON.stringify(payload) });
      app.cloudEstimate = result.estimate || null;
      app.cloudEstimateSignature = requestedSignature;
      if (!app.cloudEstimate) throw new Error("The estimate service did not return an estimate.");
      if (slot) slot.innerHTML = renderCloudEstimate();
    } catch (error) {
      app.cloudEstimate = null;
      app.cloudEstimateSignature = null;
      if (slot) slot.innerHTML = `<div class="notice error"><span class="notice-mark" aria-hidden="true">!</span><p>${escapeHtml(error.message)}</p></div>`;
    } finally {
      button.disabled = false;
      button.textContent = "Request estimate";
    }
  }

  function cloudJobTitle(job) {
    return job.name || job.run_id || job.id || job.job_id || "Cloud training job";
  }

  function cloudJobStatus(job) {
    return job.state || job.status || job.phase || "saved";
  }

  function renderCloudJobs() {
    const slot = document.getElementById("cloud-jobs-slot");
    if (!slot) return;
    const rows = app.cloudJobs.map((job) => {
      const runId = job.run_id || job.id || job.job_id;
      const state = cloudJobStatus(job);
      const status = humanize(state);
      const phase = humanize(job.phase || "");
      const terminal = /complete|success|finish|verified/i.test(`${state} ${job.phase || ""}`);
      const busy = /running|queued|pending/i.test(String(state));
      const verified = state === "verified" || job.phase === "artifacts-verified-and-registered";
      const detail = job.gpu || job.training_config?.gpu || "";
      const consent = job.consent && typeof job.consent === "object" ? [job.consent.uploads_dataset ? "dataset upload approved" : "", job.consent.acknowledged ? "estimate acknowledged" : ""].filter(Boolean).join(" · ") : "";
      const error = job.error ? `<p class="job-detail">${escapeHtml(job.error)}</p>` : "";
      const dashboard = job.dashboard_url ? `<a class="button small" href="${escapeHtml(safeHref(job.dashboard_url))}" target="_blank" rel="noopener noreferrer">Open provider job</a>` : "";
      const deadline = job.deadline_at ? new Date(job.deadline_at).toLocaleString() : "";
      return `<div class="cloud-job"><div class="cloud-job-copy"><strong>${escapeHtml(cloudJobTitle(job))}</strong><span>${escapeHtml(status)}${phase ? ` · ${escapeHtml(phase)}` : ""}${detail ? ` · ${escapeHtml(detail)}` : ""}${deadline ? ` · Deadline ${escapeHtml(deadline)}` : ""}</span>${job.progress != null ? `<span>Progress: ${escapeHtml(typeof job.progress === "object" ? job.progress.message || job.progress.percent || "In progress" : job.progress)}</span>` : ""}${consent ? `<span>${escapeHtml(consent)}</span>` : ""}${error}</div><div class="cloud-job-actions"><button class="button small" type="button" data-cloud-action="refresh" data-run-id="${escapeHtml(runId)}">Refresh</button>${dashboard}${busy ? `<button class="button small" type="button" data-cloud-action="cancel" data-run-id="${escapeHtml(runId)}">Cancel</button>` : ""}${terminal && !verified ? `<button class="button small" type="button" data-cloud-action="collect" data-run-id="${escapeHtml(runId)}">Collect checkpoint</button>` : ""}${job.cleanup_error || (terminal && job.cleanup_succeeded === false) ? `<button class="button small" type="button" data-cloud-action="cleanup" data-run-id="${escapeHtml(runId)}">Retry cleanup</button>` : ""}</div></div>`;
    }).join("");
    slot.innerHTML = `${rows || `<p class="empty-inline">No saved cloud jobs for this project yet.</p>`}${renderCloudActionResult()}`;
  }

  function renderCloudActionResult() {
    const response = app.cloudActionResult;
    if (!response) return "";
    const result = response.result || {};
    const downloads = Array.isArray(result.downloads) ? result.downloads : [];
    const paths = result.paths && typeof result.paths === "object" ? Object.entries(result.paths) : [];
    return `<div class="cloud-collection-result"><strong>${escapeHtml(result.message || response.job?.state || "Cloud job updated")}</strong>${paths.length ? `<ul class="path-list">${paths.map(([label, value]) => `<li><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></li>`).join("")}</ul>` : ""}${downloads.length ? `<div class="download-list">${downloads.map((item) => `<a class="download-link" href="${escapeHtml(safeHref(item.url))}" download="${escapeHtml(item.filename || "")}">${escapeHtml(item.label || item.filename || "Download file")}</a>`).join("")}</div>` : ""}</div>`;
  }

  async function loadCloudJobs() {
    try {
      const result = await api("/api/cloud/jobs", { method: "GET", headers: {} });
      app.cloudJobs = Array.isArray(result.jobs) ? result.jobs : [];
      renderCloudJobs();
      renderActivity();
      if (app.cloudJobs.some(activeCloudJob) && !app.cloudRefreshTimer) {
        app.cloudRefreshTimer = window.setInterval(refreshActiveCloudJobs, 3000);
        window.setTimeout(refreshActiveCloudJobs, 0);
      } else if (!app.cloudJobs.some(activeCloudJob) && app.cloudRefreshTimer) {
        window.clearInterval(app.cloudRefreshTimer);
        app.cloudRefreshTimer = null;
      }
    } catch (error) {
      const slot = document.getElementById("cloud-jobs-slot");
      if (slot) slot.innerHTML = `<div class="notice warning"><span class="notice-mark" aria-hidden="true">!</span><p>Saved cloud jobs could not be loaded: ${escapeHtml(error.message)}</p></div>${renderCloudActionResult()}`;
    }
  }

  async function refreshActiveCloudJobs() {
    if (app.cloudRefreshRunning || !app.cloudJobs.some(activeCloudJob)) return;
    app.cloudRefreshRunning = true;
    try {
      const current = app.cloudJobs.filter(activeCloudJob);
      for (const job of current) {
        const runId = job.run_id || job.id || job.job_id;
        if (!runId) continue;
        const result = await api(`/api/cloud/jobs/${encodeURIComponent(runId)}/refresh`, { method: "POST", body: "{}" });
        if (result.job) app.cloudJobs = [result.job, ...app.cloudJobs.filter((item) => (item.run_id || item.id || item.job_id) !== runId)];
      }
      renderCloudJobs();
      renderActivity();
      if (!app.cloudJobs.some(activeCloudJob) && app.cloudRefreshTimer) {
        window.clearInterval(app.cloudRefreshTimer);
        app.cloudRefreshTimer = null;
      }
    } catch {
      // Keep the most recent remote progress visible and retry on the next interval.
    } finally {
      app.cloudRefreshRunning = false;
    }
  }

  async function cloudJobAction(button) {
    const runId = button.dataset.runId;
    const action = button.dataset.cloudAction;
    if (!runId) { toast("This cloud job has no run ID.", true); return; }
    button.disabled = true;
    try {
      const result = await api(`/api/cloud/jobs/${encodeURIComponent(runId)}/${action}`, { method: "POST", body: "{}" });
      const job = result.job || {};
      const message = job.message || (action === "collect" ? "Checkpoint collection finished." : action === "cleanup" ? "Cleanup retry finished." : action === "cancel" ? "Cancellation requested." : "Cloud job refreshed.");
      app.cloudActionResult = result;
      if (job.run_id) app.cloudJobs = [job, ...app.cloudJobs.filter((item) => item.run_id !== job.run_id)];
      toast(message);
      await loadCloudJobs();
    } catch (error) {
      toast(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function renderReview() {
    if (projectRequired()) return;
    root.innerHTML = `${pageHead("Label & improve", "Choose images to review", "Select a smaller, informative queue for human annotation. This workflow is separate from prediction-only scans.")}
      <div class="form-layout"><section class="surface form-panel"><div class="panel-heading"><div><h2>Queue selection inputs</h2><p>Use artifacts produced by your analysis run. AmphiLens will not create missing evidence for you.</p></div></div>
        <div class="notice warning"><span class="notice-mark" aria-hidden="true">!</span><p><strong>Three inputs are required.</strong> Add the predictions CSV, calibration JSON, and feature JSON. If calibration or features are missing, create them with the analysis workflow before selecting a queue.</p></div>
        <form class="form-stack" data-form="review" style="margin-top:17px">
          <div class="field"><label for="review-predictions">Predictions CSV</label>${pathPickerControl("review-predictions", "predictions", "file", "/path/to/predictions.csv", { required: true, label: "a predictions CSV" })}</div>
          <div class="field"><label for="review-calibration">Calibration JSON</label>${pathPickerControl("review-calibration", "calibration", "file", "/path/to/calibration.json", { required: true, label: "a calibration JSON file" })}</div>
          <div class="field"><label for="review-features">Feature JSON</label>${pathPickerControl("review-features", "features", "file", "/path/to/features.json", { required: true, label: "a feature JSON file" })}</div>
          <div class="field"><label for="review-output">Save queue as</label>${pathPickerControl("review-output", "output", "save-file", "/path/to/selection_queue.csv", { required: true, label: "a queue output file" })}</div>
          <details class="advanced-settings"><summary>Sampling settings</summary><div class="advanced-body"><div class="field-row"><div class="field"><label for="review-budget">Annotation budget</label><input id="review-budget" name="budget" type="number" min="1" value="100"></div><div class="field"><label for="review-seed">Repeatable seed</label><input id="review-seed" name="seed" type="number" value="42"></div></div><div class="field"><label for="review-multiplier">Pool multiplier</label><input id="review-multiplier" name="pool_multiplier" type="number" min="1" value="5"></div></div></details>
          <div class="form-actions"><button class="button primary" type="submit">Select annotation queue</button></div>
        </form><div id="job-slot" aria-live="polite"></div>
      </section><aside class="surface side-note"><h3>What happens next</h3><p>Queue selection does not change your model. When the queue is ready, send it to a managed CVAT cycle, have every image reviewed, then continue the cycle to validate and merge the new snapshot.</p><a class="note-link" href="#cvat" data-view="cvat">Go to CVAT cycle</a></aside></div>`;
  }

  function renderCvat() {
    if (projectRequired()) return;
    root.innerHTML = `${pageHead("Label & improve", "Annotation cycle", "Create a CVAT task from a selected image queue, review annotation progress, then merge completed work into a new dataset snapshot.")}
      <div class="form-layout"><section class="surface form-panel"><div class="panel-heading"><div><h2>Managed cycle</h2><p>Repeating the same start action for a cycle is safe; changed selections are rejected.</p></div></div>
        <form class="form-stack" data-form="cvat">
          <div class="field"><label for="cvat-action">What would you like to do?</label><select id="cvat-action" name="action"><option value="start">Send a queue to CVAT</option><option value="refresh">Refresh annotation status</option><option value="continue">Continue a completed cycle</option></select></div>
          <div class="field"><label for="cvat-cycle">Cycle name</label><input id="cvat-cycle" name="cycle" placeholder="cycle-01" required><span class="field-help">Use the same name to refresh status or continue this cycle.</span></div>
          <div class="field" data-cvat-queue><label for="cvat-queue">Selection queue CSV</label>${pathPickerControl("cvat-queue", "queue", "file", "/path/to/selection_queue.csv", { required: true, label: "a selection queue CSV" })}<span class="field-help">Required when sending a new queue.</span></div>
          <details class="advanced-settings"><summary>CVAT server</summary><div class="advanced-body"><div class="field"><label for="cvat-server">Server URL <span class="optional">optional</span></label><input id="cvat-server" name="server_url" inputmode="url" placeholder="https://cvat.example.org"><span class="field-help">Credentials must be configured locally before connecting.</span></div></div></details>
          <div class="notice"><span class="notice-mark" aria-hidden="true">i</span><p>Every image in a CVAT task must be reviewed before continuing. Empty annotations are valid for a reviewed image with no target wildlife.</p></div>
          <div class="form-actions"><button class="button primary" type="submit">Start cycle</button><a class="button" href="#review" data-view="review">Choose images first</a></div>
        </form><div id="job-slot" aria-live="polite"></div>
      </section><aside class="surface side-note"><h3>Cycle steps</h3><p>Send the queue, annotate and save every image in CVAT, refresh job status, then continue the cycle to validate the export and create a new immutable snapshot.</p><div class="note-rule"></div><p>A cycle cannot continue until all CVAT jobs are complete.</p></aside></div>`;
  }

  function doctorChecks(doctor) {
    const safeCheck = (check) => {
      const name = check?.name || check?.label || check?.key || "System check";
      if (/token|secret|password/i.test(String(name))) return null;
      return {
        name: safeDoctorText(name),
        status: safeDoctorText(check?.status || check?.state || check?.result || "info"),
        message: safeDoctorText(check?.message || check?.detail || check?.description || ""),
      };
    };
    if (Array.isArray(doctor?.checks)) return doctor.checks.map(safeCheck).filter(Boolean);
    if (Array.isArray(doctor?.items)) return doctor.items.map(safeCheck).filter(Boolean);
    if (doctor && typeof doctor === "object") return Object.entries(doctor).map(([name, value]) => {
      if (name === "summary" || name === "ok" || name === "status" || name === "message") return null;
      if (/token|secret|password|credential.value/i.test(name)) return null;
      if (value && typeof value === "object") {
        const safeValue = Object.fromEntries(Object.entries(value).filter(([key]) => !/token|secret|password|credential.value/i.test(key)));
        return safeCheck({ name, ...safeValue });
      }
      if (name === "modal_connectivity" && typeof value === "string") {
        return safeCheck({ name, status: value });
      }
      return safeCheck({ name, status: typeof value === "boolean" ? (value ? "ready" : "attention") : "info", message: typeof value === "string" ? value : "" });
    }).filter(Boolean);
    return [];
  }

  function safeDoctorText(value) {
    return String(value ?? "").replace(/\b(token|secret|password)\b\s*[:=]\s*[^\s,;]+/gi, "$1: hidden").slice(0, 600);
  }

  function statusTone(status) {
    const text = String(status || "unknown").toLowerCase();
    if (/ready|ok|available|connected|pass|healthy|success/.test(text)) return "good";
    if (/error|fail|missing|unavailable|invalid|blocked/.test(text)) return "bad";
    if (/warn|optional|not configured|attention/.test(text)) return "warn";
    return "";
  }

  function renderHealth() {
    const checks = doctorChecks(app.doctor);
    root.innerHTML = `${pageHead("System health", "Check your setup", "Review local tools and optional services used by AmphiLens. Prediction-only work does not require CVAT or cloud credentials.", `<button class="button" type="button" data-health-refresh>Refresh check</button>`)}
      <div class="content-grid"><section class="surface panel"><div class="panel-heading"><div><h2>Local environment</h2><p>Tools AmphiLens can use on this computer.</p></div></div>${checks.length ? `<div class="health-list">${checks.map((check) => { const status = check.status || "info"; const name = check.name || "System check"; const message = check.message || ""; return `<div class="health-row"><span class="health-indicator ${statusTone(status)}" aria-hidden="true"></span><div class="health-copy"><strong>${escapeHtml(String(name).replace(/[_-]+/g, " "))}</strong><span>${escapeHtml(message)}</span></div><span class="health-status">${escapeHtml(status)}</span></div>`; }).join("")}</div>` : `<div class="notice warning"><span class="notice-mark" aria-hidden="true">!</span><p>Health details are not available yet. Refresh the check to ask the local service for the latest status.</p></div>`}</section>
      <aside class="surface panel"><div class="panel-heading"><div><h2>Optional connections</h2><p>Only needed for the workflows you choose.</p></div></div><div class="health-list"><div class="health-row"><span class="health-indicator" aria-hidden="true"></span><div class="health-copy"><strong>CVAT</strong><span>Needed to send annotation queues or import a CVAT project. Local archive import remains available.</span></div><span class="health-status">Optional</span></div><div class="health-row"><span class="health-indicator" aria-hidden="true"></span><div class="health-copy"><strong>Cloud GPU</strong><span>Needed for Modal training or prediction. Each cloud prediction asks before transferring images and requires a spending limit.</span></div><span class="health-status">Optional</span></div></div><details class="advanced-settings credential-settings"><summary>Configure Modal credentials</summary><form class="form-stack" data-form="cloud-credentials" style="margin-top:13px"><div class="field"><label for="modal-token-id">Token ID</label><input id="modal-token-id" name="token_id" type="password" autocomplete="new-password" required></div><div class="field"><label for="modal-token-secret">Token secret</label><input id="modal-token-secret" name="token_secret" type="password" autocomplete="new-password" required></div><span class="field-help">Credentials are sent only to the local AmphiLens service and are never shown back here.</span><button class="button" type="submit">Save credentials</button></form></details><button class="button panel-action" type="button" data-check-cloud>Check cloud connection</button><div id="cloud-status" aria-live="polite"></div></aside></div>`;
  }

  function render() {
    if (!app.bootstrapped) return;
    const pages = { home: renderHome, predict: renderPredict, import: renderImport, training: renderTraining, review: renderReview, cvat: renderCvat, health: renderHealth };
    (pages[app.view] || renderHome)();
    renderJobSlot();
    renderActivity();
    document.querySelectorAll("[data-view]").forEach((button) => {
      const active = button.dataset.view === app.view;
      button.classList.toggle("is-active", active);
      if (button.matches("button.nav-link")) active ? button.setAttribute("aria-current", "page") : button.removeAttribute("aria-current");
    });
  }

  function openProjectDialog(defaultTab = "open") {
    const openForm = document.getElementById("open-project-form");
    const createForm = document.getElementById("create-project-form");
    const path = document.getElementById("project-path-input");
    if (!path.value && app.defaultProjectRoot) path.value = app.defaultProjectRoot;
    setProjectTab(defaultTab);
    if (!dialog.open) dialog.showModal();
    window.setTimeout(() => (defaultTab === "create" ? createForm.querySelector("input") : openForm.querySelector("input"))?.focus(), 0);
  }

  function setProjectTab(tab) {
    document.querySelectorAll("[data-project-tab]").forEach((button) => {
      const selected = button.dataset.projectTab === tab;
      button.classList.toggle("is-selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    document.querySelectorAll("[data-project-form]").forEach((form) => { form.hidden = form.dataset.projectForm !== tab; });
    document.getElementById("project-dialog-title").textContent = tab === "create" ? "Create a project" : "Open a project";
  }

  function showError(form, message) {
    const target = document.querySelector(`[data-error-for="${form.id}"]`);
    if (target) target.textContent = message;
  }

  function formData(form) {
    return Object.fromEntries(new FormData(form).entries());
  }

  async function chooseLocalPath(button) {
    const input = document.getElementById(button.dataset.pathTarget);
    if (!input) return;
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = "Opening…";
    try {
      const result = await api("/api/path-picker", {
        method: "POST",
        body: JSON.stringify({ kind: button.dataset.pathPicker }),
      });
      if (!result.cancelled && typeof result.path === "string") {
        input.value = result.path;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
        input.focus();
      }
    } catch (error) {
      toast(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = originalText;
    }
  }

  function parseClassMapping(raw) {
    const mapping = {};
    String(raw || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean).forEach((line) => {
      const split = line.includes("=") ? line.indexOf("=") : line.indexOf(":");
      if (split < 1) throw new Error("Use one mapping per line in the form source label = project class.");
      const from = line.slice(0, split).trim();
      const to = line.slice(split + 1).trim();
      if (!from || !to) throw new Error("Each class mapping needs both a source label and a project class.");
      mapping[from] = to;
    });
    return mapping;
  }

  function numberOrUndefined(value) {
    if (value === "" || value === undefined || value === null) return undefined;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  async function handleSubmit(event) {
    const form = event.target.closest("form[data-form], form[data-project-form]");
    if (!form) return;
    event.preventDefault();
    const submit = form.querySelector('button[type="submit"]');
    if (submit) { submit.disabled = true; submit.dataset.originalText = submit.textContent; submit.textContent = "Working…"; }
    const errorTarget = document.querySelector(`[data-error-for="${form.id}"]`);
    if (errorTarget) errorTarget.textContent = "";
    try {
      if (form.dataset.projectForm === "open") {
        const result = await api("/api/projects/open", { method: "POST", body: JSON.stringify(formData(form)) });
        app.project = result.project || null;
        app.models = app.project?.models || app.models;
        updateProjectHeader();
        populateModelPreset();
        dialog.close();
        setPage("home");
        toast("Project opened.");
        return;
      }
      if (form.dataset.projectForm === "create") {
        const values = formData(form);
        values.classes = String(values.classes || "").split(",").map((item) => item.trim()).filter(Boolean);
        if (!values.classes.length) throw new Error("Add at least one wildlife class.");
        values.short_side_dimension = numberOrUndefined(values.short_side_dimension);
        values.grayscale = form.elements.grayscale.checked;
        values.clahe = form.elements.clahe.checked;
        if (!values.image_root?.trim()) throw new Error("Choose an image folder for this project.");
        values.model_preset = values.model_preset || undefined;
        const result = await api("/api/projects/create", { method: "POST", body: JSON.stringify(values) });
        app.project = result.project || null;
        updateProjectHeader();
        dialog.close();
        setPage("home");
        toast("Project created.");
        return;
      }
      const values = formData(form);
      let endpoint, payload;
      if (form.dataset.form === "cloud-credentials") {
        try {
          const response = await api("/api/cloud/credentials", { method: "POST", body: JSON.stringify({ token_id: values.token_id, token_secret: values.token_secret }) });
          form.reset();
          toast(response.credentials?.configured ? "Modal credentials saved." : "Credential request completed.");
        } finally {
          form.elements.token_id.value = "";
          form.elements.token_secret.value = "";
        }
        return;
      } else if (form.dataset.form === "initial-cvat") {
        const mapping = {};
        form.querySelectorAll("[data-cvat-map-source]").forEach((select) => {
          if (!select.value) throw new Error(`Choose the matching project class for “${select.dataset.cvatMapSource}”.`);
          mapping[select.dataset.cvatMapSource] = select.value;
        });
        endpoint = "/api/datasets/import-cvat";
        payload = { project_id: values.project_id, ...(Object.keys(mapping).length ? { class_mapping: mapping } : {}), ...(app.cvatServerUrl ? { server_url: app.cvatServerUrl } : {}) };
      } else if (form.dataset.form === "predict") {
        endpoint = "/api/predictions";
        const [, kind, selection] = String(values.model_choice || "").match(/^(preset|checkpoint|hosted):(.*)$/s) || [];
        const classMapping = {};
        if (kind === "hosted") {
          form.querySelectorAll("[data-class-map-source]").forEach((select) => {
            if (!select.value) throw new Error(`Map “${select.dataset.classMapSource}” to a project class or choose Ignore.`);
            classMapping[select.dataset.classMapSource] = select.value === "__ignore__" ? null : select.value;
          });
        }
        payload = {
          image_root: values.image_root || undefined,
          checkpoint: kind === "checkpoint" ? selection : undefined,
          model_preset: kind === "preset" ? selection : undefined,
          hosted_model_id: kind === "hosted" ? selection : undefined,
          class_mapping: kind === "hosted" ? classMapping : undefined,
          output_dir: values.output_dir || undefined,
          confidence: numberOrUndefined(values.confidence),
          device: values.device || undefined,
          execution: values.execution || "local",
          batch_size: values.batch_size === "auto" ? "auto" : numberOrUndefined(values.batch_size),
          gpu: values.gpu || "L4",
          max_cost_usd: numberOrUndefined(values.max_cost_usd) ?? 5,
        };
        if (payload.execution === "modal") {
          const preview = await api("/api/predictions/preflight", { method: "POST", body: JSON.stringify(payload) });
          const bytes = Number(preview.total_bytes || 0);
          const formattedBytes = bytes < 1024 ? `${bytes} bytes` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : bytes < 1024 ** 3 ? `${(bytes / 1024 ** 2).toFixed(2)} MB` : `${(bytes / 1024 ** 3).toFixed(2)} GB`;
          const estimate = preview.estimated_cost_usd == null
            ? "No timing history is available for this model and GPU."
            : `Estimated cost from prior timing: $${Number(preview.estimated_cost_usd).toFixed(4)}.`;
          const checkpointNotice = kind === "checkpoint" ? " The selected local checkpoint will also be uploaded once and SHA-256 verified." : "";
          const approved = window.confirm(`Modal cloud prediction\n\nImages to transfer: ${formatNumber(preview.image_count)} (${formattedBytes})\nGPU: ${preview.gpu}\n${estimate}\nSpending limit: $${Number(payload.max_cost_usd).toFixed(2)}.${checkpointNotice}\n\nImages are uploaded in bounded batches and deleted from the Modal volume after each batch. The estimate and execution timeout are not guaranteed billing caps. Continue?`);
          if (!approved) return;
          payload.acknowledged = true;
          payload.uploads_dataset = true;
          payload.expected_image_count = preview.image_count;
          payload.expected_total_bytes = preview.total_bytes;
        }
      } else if (form.dataset.form === "import") {
        endpoint = "/api/datasets/import";
        const mapping = parseClassMapping(values.class_mapping);
        payload = { archive: values.archive, ...(Object.keys(mapping).length ? { class_mapping: mapping } : {}) };
      } else if (form.dataset.form === "training") {
        const common = {
          snapshot_path: values.snapshot_path,
          training_source: values.training_source || "general-pretrained",
          hosted_model_id: values.training_source === "amphilens-pretrained" ? values.hosted_model_id || undefined : undefined,
          checkpoint: values.training_source === "project-checkpoint" ? values.checkpoint || undefined : undefined,
          epochs: numberOrUndefined(values.epochs),
          batch_size: numberOrUndefined(values.batch_size),
        };
        if (values.execution === "cloud") {
          if (!app.cloudEstimate || app.cloudEstimateSignature !== cloudEstimateSignature(form)) throw new Error("Request a fresh estimate for these training settings before submitting.");
          if (!form.elements.uploads_dataset.checked || !form.elements.acknowledged.checked) throw new Error("Approve the dataset upload and cost estimate to submit cloud training.");
          const estimatedUsd = app.cloudEstimate.high_usd;
          if (estimatedUsd === undefined || estimatedUsd === null) throw new Error("The returned estimate has no numeric estimated_usd value for the training request.");
          endpoint = "/api/cloud/training";
          payload = { ...common, image_size: trainingImageSize(form), gpu: values.gpu, max_cost_usd: numberOrUndefined(values.max_cost_usd), estimated_usd: Number(estimatedUsd), acknowledged: true, uploads_dataset: true };
        } else {
          endpoint = "/api/training";
          payload = { ...common, output_dir: values.output_dir || undefined, device: values.device || undefined };
        }
      } else if (form.dataset.form === "review") {
        endpoint = "/api/active-learning/select";
        payload = { predictions: values.predictions, calibration: values.calibration, features: values.features, output: values.output, budget: numberOrUndefined(values.budget), seed: numberOrUndefined(values.seed), pool_multiplier: numberOrUndefined(values.pool_multiplier) };
      } else if (form.dataset.form === "cvat") {
        endpoint = "/api/cvat/cycle";
        app.managedCvatUrl = values.server_url || app.managedCvatUrl || "";
        payload = { action: values.action, cycle: values.cycle, ...(values.queue ? { queue: values.queue } : {}), ...(values.server_url ? { server_url: values.server_url } : {}) };
      } else {
        return;
      }
      const result = await api(endpoint, { method: "POST", body: JSON.stringify(payload) });
      if (form.dataset.form === "training" && values.execution === "cloud") {
        app.cloudEstimate = null;
        app.cloudEstimateSignature = null;
        form.elements.uploads_dataset.checked = false;
        form.elements.acknowledged.checked = false;
        const slot = document.getElementById("cloud-estimate-slot");
        if (slot) slot.innerHTML = renderCloudEstimate();
        if (result.job && result.job.run_id) app.cloudJobs = [result.job, ...app.cloudJobs.filter((item) => (item.run_id || item.id) !== result.job.run_id)];
        await loadCloudJobs();
        renderActivity();
        toast(result.message || "Cloud training job submitted.");
        return;
      }
      if (result.job_id) {
        app.activeJob = { id: result.job_id, kind: form.dataset.form, execution: values.execution };
        app.jobMeta.set(result.job_id, {
          kind: form.dataset.form,
          execution: values.execution || "local",
          title: form.dataset.form === "predict" ? "Wildlife prediction" : form.dataset.form === "training" ? "Model training" : "Workspace task",
        });
        app.jobs.set(result.job_id, { ...result });
        renderJobSlot();
        renderActivity();
        pollJob(result.job_id);
      } else if (result.job) {
        const id = result.job.job_id || form.dataset.form;
        app.jobs.set(id, result.job);
        app.jobMeta.set(id, { kind: form.dataset.form, title: form.dataset.form === "training" ? "Model training" : form.dataset.form === "predict" ? "Wildlife prediction" : "Workspace task" });
        app.activeJob = { id, kind: form.dataset.form };
        renderJobSlot(result.job);
        renderActivity();
        toast(result.job.message || "Request started.");
      } else {
        toast(result.message || "Request completed.");
        if ((form.dataset.form === "cvat" && values.action === "refresh") || form.dataset.form === "initial-cvat") renderJobSlot(result);
      }
    } catch (error) {
      if (form.dataset.projectForm || form.id === "initial-cvat-form") showError(form, error.message);
      else toast(error.message, true);
    } finally {
      if (submit) { submit.disabled = false; submit.textContent = submit.dataset.originalText || submit.textContent; }
    }
  }

  function jobPanel(job) {
    const state = String(job.state || "queued").toLowerCase();
    const completed = state === "completed";
    const failed = state === "failed";
    const canceled = state === "canceled";
    const activity = activityProgress(job);
    const rawProgress = activity.phasePercent ?? activity.percent;
    const hasProgress = Number.isFinite(rawProgress);
    const progress = hasProgress ? Math.max(0, Math.min(100, rawProgress < 1 ? rawProgress * 100 : rawProgress)) : null;
    const result = job.result || {};
    const details = activity.details;
    const downloads = Array.isArray(result.downloads) ? result.downloads : [];
    const paths = result.paths && typeof result.paths === "object" ? Object.entries(result.paths) : [];
    const progressText = typeof job.progress === "object" ? job.progress.message || job.progress.label || "" : "";
    const progressContext = [activity.device, humanize(details.phase), activity.phasePercent != null ? `Step ${Math.round(activity.phasePercent)}%` : "", activity.epochText, activity.counts, activity.metrics, activity.eta ? `About ${activity.eta} left` : ""].filter(Boolean).join(" · ");
    return `<section class="job-card" aria-label="Job status"><div class="job-head"><div class="job-title"><span class="status-dot" aria-hidden="true"></span>${failed ? "This run needs attention" : canceled ? "Prediction canceled" : completed ? "Run complete" : "Working on your request"}</div><span class="job-state ${completed ? "is-completed" : failed ? "is-failed" : ""}"><span class="status-dot" aria-hidden="true"></span>${escapeHtml(state)}</span></div>
      ${result.message ? `<p class="job-description">${escapeHtml(result.message)}</p>` : progressText ? `<p class="job-description">${escapeHtml(progressText)}</p>` : `<p class="job-description">${failed ? "The job could not be completed." : completed ? "Your files are ready in the project." : "You can continue using AmphiLens while this runs."}</p>`}
      ${progressContext ? `<p class="job-description">${escapeHtml(progressContext)}</p>` : ""}
      ${!completed && !failed && !canceled ? `<div class="progress-track" role="progressbar" aria-label="Job progress" aria-valuemin="0" aria-valuemax="100"${hasProgress ? ` aria-valuenow="${Math.round(progress)}"` : ""}><div class="progress-fill${hasProgress ? "" : " indeterminate"}" ${hasProgress ? `style="width:${progress}%"` : ""}></div></div>` : ""}
      ${!completed && !failed && !canceled && app.activeJob?.kind === "predict" && app.jobMeta.get(app.activeJob.id)?.execution === "modal" ? `<button class="button small" type="button" data-cancel-job="${escapeHtml(job.job_id || "")}">Cancel after current batch</button>` : ""}
      ${job.error ? `<p class="job-detail">${escapeHtml(job.error)}</p>` : ""}
      ${activityTimeline(job)}
      ${paths.length ? `<ul class="path-list">${paths.map(([label, value]) => `<li><span>${escapeHtml(String(label).replace(/[_-]+/g, " "))}</span><span>${escapeHtml(value)}</span></li>`).join("")}</ul>` : ""}
      ${downloads.length ? `<div class="download-list">${downloads.map((item) => `<a class="download-link" href="${escapeHtml(safeHref(item.url))}" download="${escapeHtml(item.filename || "")}">${escapeHtml(item.label || item.filename || "Download file")}</a>`).join("")}</div>` : ""}
      ${app.activeJob?.kind === "cvat" && (job.result?.task_url || job.result?.cvat_url || job.result?.server_url || app.managedCvatUrl) ? `<div class="download-list"><a class="download-link" href="${escapeHtml(safeHref(job.result?.task_url || job.result?.cvat_url || job.result?.server_url || app.managedCvatUrl))}" target="_blank" rel="noopener noreferrer">Open CVAT</a></div>` : ""}
      ${job.status_offline ? `<button class="button small" type="button" data-poll-job-id="${escapeHtml(job.job_id || "")}">Check status again</button>` : ""}
    </section>`;
  }

  function renderJobSlot(explicitJob = null) {
    const slot = document.getElementById(app.activeJob?.kind === "initial-cvat" ? "cvat-import-job-slot" : "job-slot");
    if (!slot) return;
    const belongsToView = app.activeJob && (app.activeJob.kind === app.view || (app.activeJob.kind === "initial-cvat" && app.view === "import"));
    if (!belongsToView) { slot.innerHTML = ""; return; }
    const historyWasOpen = Boolean(slot.querySelector(".activity-details")?.open);
    const job = explicitJob || (app.activeJob ? app.jobs.get(app.activeJob.id) : null);
    slot.innerHTML = job ? jobPanel(job) : "";
    if (historyWasOpen) {
      const history = slot.querySelector(".activity-details");
      if (history) history.open = true;
      window.AmphiLensActivityView?.scrollOutputToBottom(slot);
    }
  }

  async function pollJob(jobId) {
    for (;;) {
      await new Promise((resolve) => window.setTimeout(resolve, 1400));
      try {
        const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "GET", headers: {} });
        app.jobs.set(jobId, job);
        if (app.activeJob?.id === jobId) renderJobSlot();
        renderActivity();
        if (["completed", "failed", "canceled"].includes(String(job.state).toLowerCase())) {
          if (job.state === "completed") await refreshProjectDetails(jobId);
          return;
        }
      } catch (error) {
        const job = app.jobs.get(jobId) || { job_id: jobId, state: "running" };
        if (!job.state) job.state = "running";
        job.status_offline = true;
        job.error = `Could not refresh status: ${error.message}`;
        app.jobs.set(jobId, job);
        if (app.activeJob?.id === jobId) renderJobSlot();
        renderActivity();
        return;
      }
    }
  }

  async function refreshProjectDetails(jobId = null) {
    try {
      const data = await api("/api/bootstrap", { method: "GET", headers: {} });
      app.project = data.project || app.project;
      app.doctor = data.doctor || app.doctor;
      updateProjectHeader();
      if (jobId && app.activeJob?.id === jobId) {
        render();
      }
    } catch { /* the completed job details remain visible */ }
  }

  async function refreshCloud() {
    let slot = document.getElementById("cloud-status");
    if (slot) slot.innerHTML = `<p class="job-description">Checking the optional cloud connection…</p>`;
    try {
      const data = await api("/api/environment/check-cloud", { method: "POST", body: "{}" });
      const report = data.doctor || {};
      app.modalConnectivity = report.modal_connectivity ?? "not checked";
      app.doctor = { ...(app.doctor || {}), ...report };
      render();
      slot = document.getElementById("cloud-status");
      if (slot) {
        const checks = doctorChecks(report);
        const message = checks.map((item) => [item.name || item.label, item.message || item.status].filter(Boolean).join(": ")).filter(Boolean).join(" · ") || report.message || "Cloud check complete.";
        slot.innerHTML = `<div class="notice ${statusTone(report.modal_connectivity) === "bad" ? "error" : ""}" style="margin-top:12px"><span class="notice-mark" aria-hidden="true">i</span><p>${escapeHtml(message)}</p></div>`;
      }
    } catch (error) {
      if (slot) slot.innerHTML = `<div class="notice error" style="margin-top:12px"><span class="notice-mark" aria-hidden="true">!</span><p>${escapeHtml(error.message)}</p></div>`;
    }
  }

  document.addEventListener("submit", handleSubmit);
  document.addEventListener("click", async (event) => {
    const activityPanel = document.getElementById("activity-panel");
    const activityToggle = event.target.closest("[data-activity-toggle]");
    if (activityToggle) {
      const opening = activityPanel.hidden;
      activityPanel.hidden = !opening;
      activityToggle.setAttribute("aria-expanded", String(opening));
      if (opening) renderActivity();
      return;
    }
    if (event.target.closest("[data-activity-close]")) {
      activityPanel.hidden = true;
      document.querySelector("[data-activity-toggle]")?.setAttribute("aria-expanded", "false");
      return;
    }
    const historySummary = event.target.closest(".activity-details > summary");
    if (historySummary) {
      const history = historySummary.closest(".activity-details");
      window.requestAnimationFrame(() => {
        window.AmphiLensActivityView?.scrollOutputToBottom(history);
      });
    }
    if (!event.target.closest(".activity-wrap") && activityPanel && !activityPanel.hidden) {
      activityPanel.hidden = true;
      document.querySelector("[data-activity-toggle]")?.setAttribute("aria-expanded", "false");
    }
    const pathButton = event.target.closest("[data-path-picker]");
    if (pathButton) {
      event.preventDefault();
      await chooseLocalPath(pathButton);
      return;
    }
    const viewButton = event.target.closest("[data-view]");
    if (viewButton) {
      event.preventDefault();
      setPage(viewButton.dataset.view);
      return;
    }
    if (event.target.closest("[data-open-project]")) {
      const button = event.target.closest("[data-open-project]");
      openProjectDialog(button.dataset.projectTab || "open");
      return;
    }
    if (event.target.closest("[data-close-dialog]")) { dialog.close(); return; }
    const tab = event.target.closest("[data-project-tab]");
    if (tab) { setProjectTab(tab.dataset.projectTab); return; }
    if (event.target.closest("[data-close-project]")) {
      try {
        const result = await api("/api/projects/close", { method: "POST", body: "{}" });
        app.project = result.project || null;
        updateProjectHeader();
        dialog.close();
        setPage("home");
        toast("Project closed.");
      } catch (error) { toast(error.message, true); }
      return;
    }
    if (event.target.closest("[data-retry-bootstrap]")) { refreshBootstrap(); return; }
    if (event.target.closest("[data-health-refresh]")) { await refreshBootstrap(); setPage("health"); return; }
    if (event.target.closest("[data-check-cloud]")) { refreshCloud(); return; }
    if (event.target.closest("[data-connect-cvat]")) {
      const input = document.getElementById("initial-cvat-server");
      app.cvatServerUrl = input?.value.trim() || "";
      const button = event.target.closest("[data-connect-cvat]");
      const slot = document.getElementById("cvat-projects-slot");
      button.disabled = true;
      button.textContent = "Connecting…";
      try {
        const result = await api("/api/cvat/projects", { method: "POST", body: JSON.stringify(app.cvatServerUrl ? { server_url: app.cvatServerUrl } : {}) });
        app.cvatProjects = Array.isArray(result.projects) ? result.projects : [];
        if (slot) slot.innerHTML = renderCvatProjects();
        if (!app.cvatProjects.length) toast("No CVAT projects were found.");
      } catch (error) {
        app.cvatProjects = null;
        if (slot) slot.innerHTML = `<div class="notice error" style="margin-top:13px"><span class="notice-mark" aria-hidden="true">!</span><p>${escapeHtml(error.message)}</p></div>`;
      } finally {
        button.disabled = false;
        button.textContent = "Connect to CVAT";
      }
      return;
    }
    if (event.target.closest("[data-cloud-estimate]")) { requestCloudEstimate(event.target.closest("[data-cloud-estimate]")); return; }
    if (event.target.closest("[data-refresh-cloud-jobs]")) { loadCloudJobs(); return; }
    if (event.target.closest("[data-cloud-action]")) { cloudJobAction(event.target.closest("[data-cloud-action]")); return; }
    if (event.target.closest("[data-cancel-job]")) {
      const button = event.target.closest("[data-cancel-job]");
      button.disabled = true;
      button.textContent = "Stopping after active batch…";
      try {
        const result = await api(`/api/jobs/${encodeURIComponent(button.dataset.cancelJob)}/cancel`, { method: "POST", body: "{}" });
        const job = app.jobs.get(button.dataset.cancelJob);
        if (job) job.cancel_requested = result.cancel_requested;
        toast("Cancellation requested. The active batch will be cleaned up before stopping.");
      } catch (error) {
        toast(error.message, true);
        button.disabled = false;
        button.textContent = "Cancel after current batch";
      }
      return;
    }
    if (event.target.closest("[data-poll-job-id]")) {
      const button = event.target.closest("[data-poll-job-id]");
      const jobId = button.dataset.pollJobId;
      const job = app.jobs.get(jobId);
      if (job) {
        job.status_offline = false;
        job.error = "";
        renderJobSlot();
        pollJob(jobId);
      }
      return;
    }
  });

  document.addEventListener("input", (event) => {
    if (event.target.id === "confidence") {
      const output = document.querySelector('output[for="confidence"]');
      if (output) output.value = Number(event.target.value).toFixed(2);
    }
    savePredictionPreferences(event.target.closest('form[data-form="predict"]'));
  });
  document.addEventListener("change", (event) => {
    if (event.target.id === "cvat-action") {
      const queue = document.querySelector("[data-cvat-queue]");
      const start = event.target.value === "start";
      if (queue) {
        queue.hidden = !start;
        const input = queue.querySelector("input");
        if (input) input.required = start;
      }
      const submit = event.target.form?.querySelector('button[type="submit"]');
      if (submit) submit.textContent = start ? "Start cycle" : event.target.value === "refresh" ? "Refresh status" : "Continue cycle";
    }
    if (event.target.id === "training-execution") {
      document.querySelector("[data-cloud-settings]")?.classList.toggle("hidden", event.target.value !== "cloud");
      document.querySelector('[data-form="training"] .form-actions button[type="submit"]')?.replaceChildren(document.createTextNode(event.target.value === "cloud" ? "Submit cloud training" : "Train model"));
    }
    if (event.target.id === "prediction-model") renderPredictionModelSelection();
    if (event.target.id === "prediction-execution") {
      syncPredictionExecutionControls(event.target.form);
    }
    if (event.target.id === "training-source" || event.target.id === "training-hosted-model") {
      const form = event.target.closest('form[data-form="training"]');
      syncTrainingSourceControls(form);
      if (form) {
        form.elements.uploads_dataset.checked = false;
        form.elements.acknowledged.checked = false;
        app.cloudEstimate = null;
        app.cloudEstimateSignature = null;
        const slot = document.getElementById("cloud-estimate-slot");
        if (slot) slot.innerHTML = renderCloudEstimate();
      }
    }
    if (event.target.id === "initial-cvat-project") {
      const selected = app.cvatProjects?.find((project) => String(project.project_id ?? project.id ?? project.pk) === event.target.value);
      const summary = document.getElementById("initial-cvat-summary");
      if (summary) summary.innerHTML = renderCvatProjectSummary(selected);
      const importButton = document.querySelector("#initial-cvat-form button[type=submit]");
      if (importButton) importButton.disabled = Boolean(cvatProjectIssue(selected));
    }
    const trainingForm = event.target.closest('form[data-form="training"]');
    if (trainingForm && ["snapshot_path", "gpu", "epochs", "max_cost_usd", "checkpoint", "training_source", "hosted_model_id"].includes(event.target.name)) {
      trainingForm.elements.uploads_dataset.checked = false;
      trainingForm.elements.acknowledged.checked = false;
      if (["snapshot_path", "gpu", "epochs", "max_cost_usd", "training_source", "hosted_model_id"].includes(event.target.name)) {
        app.cloudEstimate = null;
        app.cloudEstimateSignature = null;
        const slot = document.getElementById("cloud-estimate-slot");
        if (slot) slot.innerHTML = renderCloudEstimate();
      }
    }
    savePredictionPreferences(event.target.closest('form[data-form="predict"]'));
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      const panel = document.getElementById("activity-panel");
      if (panel && !panel.hidden) {
        panel.hidden = true;
        document.querySelector("[data-activity-toggle]")?.setAttribute("aria-expanded", "false");
        document.querySelector("[data-activity-toggle]")?.focus();
      }
    }
  });

  refreshBootstrap();
})();
