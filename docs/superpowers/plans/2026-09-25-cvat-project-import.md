# Direct CVAT Project Import for Initial Datasets

## Objective

Allow users to select an existing CVAT project through the CVAT API and import
its complete annotated dataset, including images, without manually downloading
a ZIP. Preserve the existing local-archive fallback and managed active-learning
cycle behavior.

## Implementation tasks

1. Extend the pinned CVAT SDK transport with project listing, project metadata,
   task metadata, and project-level dataset export.
2. Add a reusable initial CVAT import service that downloads a project export,
   validates it through `DatasetImporter`, records provenance, and creates an
   immutable snapshot without duplicate or destructive writes.
3. Add `ProjectStore.import_cvat_project(...)` and public exports for the new
   transport models and service.
4. Add CLI commands for listing CVAT projects and importing a selected project.
5. Replace the archive-only Streamlit initial-import form with local-archive and
   CVAT-project modes, including label preview and explicit class mapping.
6. Add focused tests for transport behavior, provenance, validation failures,
   duplicate imports, CLI wiring, and UI helpers.
7. Update the README, training guide, deployment guide, technical overview,
   release checklist, and changelog.

## Decisions

- Initial import scope is the entire selected CVAT project.
- Project exports include images and are stored locally in the immutable
  snapshot.
- Authentication uses `CVAT_URL` and `CVAT_TOKEN`; tokens are never persisted.
- Task-level initial import and local-image reuse are out of scope for v1.
- The pinned compatibility profile remains `cvat-sdk==2.76.0` and
  `cvat-cli==2.76.0`.

## Acceptance criteria

- The UI lists accessible CVAT projects and imports a selected project with one
  import action after class confirmation.
- The CLI can list projects and import one by project ID.
- Project labels and task IDs are visible and preserved as provenance.
- The downloaded export is validated by the existing importer, including image,
  class, dimension, duplicate, and bounding-box checks.
- Repeating an unchanged import does not overwrite or duplicate a snapshot.
- Existing active-learning CVAT task workflows remain green.
