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
- [x] Run Ruff and `git diff --check`.
- [x] Run `uv lock --check` against the committed `uv.lock`.
- [x] Build both wheel and source distribution.
- [x] Install the published prerelease in a clean Python 3.11 environment and
  run `amphilens doctor`.
- [ ] Exercise a fixture prediction run, report generation, and CVAT round trip.
- [x] Exercise the managed CVAT fake-SDK contract; the live CVAT Community
  `v2.76.0` server contract remains pending.
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

- [x] Build and install `amphilens==0.1.0a1` from TestPyPI in a clean Python
  3.11 environment. The fresh upload-scoped token was supplied through
  `TWINE_USERNAME`/`TWINE_PASSWORD` process variables loaded from a gitignored
  env file and was not committed, logged, or written to package metadata.

- [ ] Verify PyPI package-name availability and the AmphiLens name/trademark
  status before public publication.
- [ ] Tag the verified commit and push the tag using the intended GitHub
  identity.
- [x] Upload artifacts only after the package build, `twine check`, and clean
  TestPyPI install checks passed. Scientific holdout/GPU/live-CVAT checks remain
  outside this prerelease verification.

Published prerelease: [AmphiLens 0.1.0a1 on TestPyPI](https://test.pypi.org/project/amphilens/0.1.0a1/).
