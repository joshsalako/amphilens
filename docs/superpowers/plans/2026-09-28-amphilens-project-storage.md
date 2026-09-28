# AmphiLens project storage and loading

## Goal

Keep generated AmphiLens projects outside the source checkout while making it
easy to create, open, and migrate projects through the local browser app.

## Tasks

### Task 1: Project-location contract

Add a tested project-location contract: platform user-project defaults,
   source-checkout safety, manifest validation, and verified relocation.

### Task 2: Streamlit project lifecycle

Add Streamlit create/open/active-project behavior and a local folder picker
   with a path-entry fallback. Remove repeated repository-relative defaults.

### Task 3: CLI project locations

Add CLI location commands and preserve explicit path-based headless workflows.

### Task 4: Documentation and verification

Update documentation, add the generated project ignore rule, run the full
suite, and commit the completed slices.

## Decisions

- New projects default to the platform-appropriate AmphiLens user data folder.
- Custom locations use a native local folder dialog when available, with a
  portable path input fallback.
- New projects inside the source checkout are rejected.
- Existing repository-local projects can be migrated by copy, verification,
  and explicit confirmation before removing the original.
- Project loading is explicit; no global recent-project catalog is added in v1.
- External source-image paths remain unchanged when a project folder moves.

## Verification

- Focused location and UI helper tests pass.
- The dependency-light full test suite passes.
- `git check-ignore` confirms the generated `amphilens-project/` folder is
  ignored without adding the user's existing files to Git.
