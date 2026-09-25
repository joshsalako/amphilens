# AmphiLens release checklist

Use this checklist for a tagged release. A release must describe its tested
scope honestly; optional detector runtimes and remote execution are not
considered verified merely because the package builds.

## Repository and provenance

- [ ] Confirm the version in `pyproject.toml`, `CHANGELOG.md`, and the tag.
- [ ] Confirm `wtl-detection` and all user datasets, annotations, weights,
  credentials, caches, and generated artifacts are absent from the release.
- [ ] Record the scientific reference commit, model-card sources, licenses,
  and any training-data restrictions.
- [ ] Review `SECURITY.md`, privacy boundaries, and dependency licenses.

## Automated checks

- [ ] Run the full dependency-light test suite on every supported Python
  version.
- [ ] Run Ruff and `git diff --check`.
- [ ] Build both wheel and source distribution.
- [ ] Install each artifact in a clean environment and run `amphilens doctor`.
- [ ] Exercise a fixture prediction run, report generation, and CVAT round trip.
- [ ] Exercise CVAT XML, COCO, and YOLO initial-dataset import, including a
  reviewed negative image and an immutable merge.
- [ ] Exercise the preprocessing fixture: grayscale, aspect-ratio-preserving
  max-dimension resizing, no upscaling, resize-before-CLAHE, and box remapping.
- [ ] If detector extras changed, run backend contract tests with each
  available runtime and record the exact versions and device.
- [ ] If training changed, run a short CPU smoke test and a representative GPU
  smoke test where applicable.

## Scientific and operational review

- [ ] Confirm calibration evidence, class order, preprocessing, thresholds,
  image size, seed, and checkpoint provenance are present in every reusable
  checkpoint manifest.
- [ ] Confirm holdout metrics are reported as measured or explicitly marked
  `not evaluated`.
- [ ] Confirm active-learning selections are reproducible from their saved
  configuration and calibration artifacts.
- [ ] Review Docker image pins and Compose volume/privacy boundaries.
- [ ] Publish release notes with tested hardware tiers, known limitations, and
  upgrade or migration notes.

## Publication

- [ ] Build and install the package from TestPyPI in a clean Python 3.11
  environment. Use a fresh upload-scoped token supplied through environment
  variables only; never store it in the repository, shell history, logs, or
  package metadata.

- [ ] Verify PyPI package-name availability and the AmphiLens name/trademark
  status before public publication.
- [ ] Tag the verified commit and push the tag using the intended GitHub
  identity.
- [ ] Upload artifacts only after the clean-install and scientific review
  gates above pass.
