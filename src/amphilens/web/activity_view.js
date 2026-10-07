(() => {
  "use strict";

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);

  function isActive(job, cloud) {
    const state = String(job?.state || "").toLowerCase();
    return cloud
      ? !["finished", "verified", "incomplete", "failed", "canceled", "timed_out"].includes(state)
      : !["completed", "failed", "canceled"].includes(state);
  }

  function eventRow(event) {
    const stamp = Number(event.updated_at);
    const date = Number.isFinite(stamp) ? new Date(stamp * 1000) : new Date(event.updated_at || NaN);
    const time = Number.isNaN(date.getTime())
      ? ""
      : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const message = String(event.message || event.error || event.phase || "Progress updated");
    const details = [];
    const epoch = Number(event.epoch);
    const epochs = Number(event.epochs);
    if (Number.isFinite(epoch) && Number.isFinite(epochs) && epochs > 0 && !/\bepoch\s+\d+\s+(of|\/)\s+\d+/i.test(message)) {
      details.push(`epoch ${epoch}/${epochs}`);
    }
    const phaseProgress = event.phase_progress == null ? NaN : Number(event.phase_progress);
    if (Number.isFinite(phaseProgress) && !/\b\d+(\.\d+)?\s*%/.test(message)) {
      const percent = Math.max(0, Math.min(100, phaseProgress <= 1 ? phaseProgress * 100 : phaseProgress));
      details.push(`${Math.round(percent)}%`);
    }
    const completed = Number(event.completed);
    const total = Number(event.total);
    if (event.phase !== "model_download" && Number.isFinite(completed) && Number.isFinite(total) && !/\b\d+\s+of\s+\d+\b|\b\d+\/\d+\b/i.test(message)) {
      details.push(`${completed}/${total}`);
    }
    const metrics = Object.entries(event.metrics || {}).slice(0, 4).map(([key, value]) => {
      const formatted = typeof value === "number" && Number.isFinite(value)
        ? value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "")
        : String(value);
      return `${key.replace(/[_-]+/g, " ")} ${formatted}`;
    });
    if (metrics.length) details.push(metrics.join(" · "));
    return `<li><time>${escapeHtml(time)}</time><span>${escapeHtml(message)}${details.length ? ` · ${escapeHtml(details.join(" · "))}` : ""}</span></li>`;
  }

  function renderTimeline(job, cloud = false) {
    const events = Array.isArray(job?.progress_events) ? job.progress_events.slice(-24) : [];
    const tail = cloud ? job?.log_tail : job?.progress?.log_tail;
    if (!events.length && !tail) return "";
    const recent = events.slice(-3).map(eventRow).join("");
    const rows = events.map(eventRow).join("");
    const history = rows ? `<ol class="activity-timeline">${rows}</ol>` : "";
    const output = tail
      ? `<pre class="activity-log">${escapeHtml(String(tail).slice(-8000))}</pre>`
      : "";
    const summary = isActive(job, cloud) ? "Full run history and output" : "Run history and output";
    return `${recent ? `<section class="activity-live" aria-label="Latest updates" aria-live="polite"><p class="activity-live-label">Latest updates</p><ol class="activity-timeline">${recent}</ol></section>` : ""}<details class="activity-details"><summary>${summary}</summary>${history}${output}</details>`;
  }

  window.AmphiLensActivityView = { renderTimeline };
})();
