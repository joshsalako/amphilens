const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const webPath = path.join(__dirname, "..", "src", "amphilens", "web");

function loadCvatState() {
  const window = { localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} } };
  vm.runInNewContext(fs.readFileSync(path.join(webPath, "cvat_project_state.js"), "utf8"), { window });
  return window.AmphiLensCvatProjectState;
}

test("CVAT labels match a unique project class despite case and separator differences", () => {
  const cvatState = loadCvatState();

  assert.equal(
    cvatState?.matchProjectClass("Small_Mammal", ["other amphibian", "small mammal"]),
    "small mammal",
  );
  assert.equal(cvatState?.matchProjectClass?.("Small-Mammal", ["small mammal", "small_mammal"]), "");
});

test("CVAT mapping preferences differ between CVAT projects and servers", () => {
  const cvatState = loadCvatState();

  assert.notEqual(
    cvatState?.mappingPreferenceKey("http://localhost:8080", "17", "Small_Mammal"),
    cvatState?.mappingPreferenceKey("http://localhost:8080", "26", "Small_Mammal"),
  );
  assert.notEqual(
    cvatState?.mappingPreferenceKey("http://localhost:8080", "17", "Small_Mammal"),
    cvatState?.mappingPreferenceKey("https://cvat.example.org", "17", "Small_Mammal"),
  );
});

test("switching CVAT projects replaces cached labels with the selected project's fresh details", async () => {
  const cvatState = loadCvatState();
  const projects = [
    { project_id: "17", name: "WLT-UP", labels: ["stale label"], tasks: [] },
    { project_id: "26", name: "2026", labels: ["frog"], tasks: [] },
  ];
  let request;

  const updated = await cvatState?.refreshSelectedProject(
    projects,
    "17",
    "http://localhost:8080",
    async (path, options) => {
      request = { path, options };
      return {
        project: {
          project_id: "17",
          name: "WLT-UP",
          labels: ["Small_Mammal", "Other_Amphibian"],
          task_count: 3,
          tasks: [{ task_id: "31", name: "train", size: 12, status: "completed" }],
        },
      };
    },
  );

  assert.ok(updated, "refreshSelectedProject should return the updated project list");
  assert.equal(request.path, "/api/cvat/project");
  assert.deepEqual(JSON.parse(request.options.body), {
    project_id: "17",
    server_url: "http://localhost:8080",
  });
  assert.deepEqual(updated[0].labels, ["Small_Mammal", "Other_Amphibian"]);
  assert.deepEqual(updated[1].labels, ["frog"]);
});
