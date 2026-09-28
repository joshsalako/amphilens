# AmphiLens project storage and loading

## Goal

Keep generated AmphiLens projects outside the source checkout while making it
easy to create, open, and migrate projects through the local browser app.

## Tasks

### Task 1: Project-location contract

Add a tested project-location contract: Downloads-based project defaults,
   source-checkout safety, manifest validation, and verified relocation.

### Task 2: Streamlit project lifecycle

Add Streamlit create/open/active-project behavior with validated path-entry
fields. Remove repeated repository-relative defaults.

### Task 3: CLI project locations

Add CLI location commands and preserve explicit path-based headless workflows.

### Task 4: Documentation and verification

Update documentation, add the generated project ignore rule, run the full
suite, and commit the completed slices.

## Decisions

- New projects default to `~/Downloads/AmphiLens/projects`.
- Custom locations are entered as paths so the UI works in local, Docker, and
  headless environments without desktop GUI dependencies.
- New projects inside the source checkout are rejected.
- Existing repository-local projects can be migrated by copy, verification,
  and explicit confirmation before removing the original.
- The app remembers only the last active project in a platform-standard,
  versioned user-state file and validates it after refresh.
- **Close active project** clears the session and remembered path; no global
  recent-project catalog is added in v1.
- External source-image paths remain unchanged when a project folder moves.
- Project configuration remains in `manifest.json`; workflow overrides are
  temporary and recorded in run/checkpoint metadata.

## Verification

- Focused location and UI helper tests pass.
- The dependency-light full test suite passes.
- `git check-ignore` confirms the generated `amphilens-project/` folder is
  ignored without adding the user's existing files to Git.
