const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const sourcePath = path.join(__dirname, "..", "src", "amphilens", "web", "activity_view.js");
const window = {};
vm.runInNewContext(fs.readFileSync(sourcePath, "utf8"), { window });
const activityView = window.AmphiLensActivityView;

test("active activity shows recent progress outside expandable history", () => {
  const html = activityView.renderTimeline({
    state: "running",
    progress: { phase: "prediction", message: "Predicted 10 of 50 images", completed: 10, total: 50 },
    progress_events: [
      { phase: "model_download", message: "Downloading model weights" },
      { phase: "prediction", message: "Predicted 2 of 50 images", completed: 2, total: 50 },
      { phase: "prediction", message: "Predicted 6 of 50 images", completed: 6, total: 50 },
      { phase: "prediction", message: "Predicted 10 of 50 images", completed: 10, total: 50 },
    ],
  });
  const visibleUpdates = html.slice(0, html.indexOf("<details"));

  assert.match(visibleUpdates, /Latest updates/);
  assert.match(visibleUpdates, /Predicted 6 of 50 images/);
  assert.match(visibleUpdates, /Predicted 10 of 50 images/);
  assert.doesNotMatch(visibleUpdates, /01\.jpg|02\.jpg/);
});

test("completed activity retains recent updates and expandable history", () => {
  const html = activityView.renderTimeline({
    state: "completed",
    progress_events: [{
      phase: "training",
      message: "Epoch 5 of 5",
      epoch: 5,
      epochs: 5,
      metrics: { train_loss: 0.25 },
    }],
  });

  assert.match(html, /Epoch 5 of 5/);
  assert.match(html, /train loss 0\.25/);
  assert.match(html, /Run history and output/);
  assert.match(html, /<details/);
});

test("model download events show download progress without treating bytes as images", () => {
  const html = activityView.renderTimeline({
    state: "running",
    progress_events: [{
      phase: "model_download",
      message: "Downloading AmphiLens model",
      phase_progress: 0.6,
      completed: 600,
      total: 1000,
    }],
  });
  const visibleUpdates = html.slice(0, html.indexOf("<details"));

  assert.match(visibleUpdates, /Downloading AmphiLens model/);
  assert.match(visibleUpdates, /60%/);
  assert.doesNotMatch(visibleUpdates, /600\/1000/);
});

test("run history and log output scroll to their newest content", () => {
  const timeline = { scrollHeight: 840, scrollTop: 0, closest: () => ({ open: true }) };
  const log = { scrollHeight: 420, scrollTop: 0, closest: () => ({ open: true }) };
  const closedTimeline = { scrollHeight: 600, scrollTop: 0, closest: () => ({ open: false }) };
  const container = {
    querySelectorAll(selector) {
      assert.equal(selector, ".activity-timeline, .activity-log");
      return [timeline, log, closedTimeline];
    },
  };

  activityView.scrollOutputToBottom(container);

  assert.equal(timeline.scrollTop, 840);
  assert.equal(log.scrollTop, 420);
  assert.equal(closedTimeline.scrollTop, 0);
});
