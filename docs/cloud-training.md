# Cloud GPU training

AmphiLens can run one training job in your own Modal account while keeping the
project folder and result registry on your computer. The cloud path is an
optional provider adapter; local training remains available.

## Setup and credentials

Install the cloud extra in addition to the usual training and UI dependencies:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra ui --extra inference --extra training --extra cloud
```

Create a Modal API token in your Modal account, then save it in the masked
fields on the cloud training screen or with the CLI. Neither path puts token
values in shell history or command arguments:

```bash
amphilens cloud login
amphilens cloud diagnose
amphilens doctor --check-cloud
```

The CLI reads both values interactively and stores them in a user-scoped
`credentials.json` file with owner-only permissions. `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` can be set in the process environment or in a `.env` file
in the working directory or a parent directory; process environment values
take precedence, followed by `.env`, then the saved file. Only these two `.env`
values are used, and they are never copied into the project or remote payload.
`amphilens cloud logout` removes the saved file. Environment and `.env` values
must be cleared separately.

Modal requires a valid payment method for GPU use. AmphiLens creates a detached
ephemeral invocation in your account; it does not deploy a persistent
multi-user service.

## Submit a job

In the browser app, open **Train model**, choose **Modal cloud GPU**, select a
dataset snapshot and model settings, choose a GPU and budget, review the cost
range, then check the upload and cost consent box before submitting.

The equivalent CLI flow is:

```bash
amphilens cloud estimate /path/to/project --gpu L4 --max-cost-usd 5
amphilens cloud train /path/to/project \
  --snapshot /path/to/project/datasets/snapshot-123 \
  --gpu L4 --max-cost-usd 5
amphilens cloud status /path/to/project cloud-0123456789ab
amphilens cloud collect /path/to/project cloud-0123456789ab
```

The CLI prints the estimate and requires confirmation before upload. A job is
keyed by the dataset contents, effective configuration, checkpoint hash,
source digest, GPU, and budget-derived timeout. Repeating the same request
reuses its saved job. A changed GPU or budget makes a different request.

AmphiLens prepares a copy from the immutable dataset snapshot, applies the
selected preprocessing, removes image metadata by re-encoding the images, and
packages images, YOLO labels, and a path-rewritten `dataset.yaml`. The local
snapshot and source image folders are not changed. A selected base checkpoint
is uploaded only with the same explicit consent. Its manifest is reduced to
the compatibility fields needed to load it; local paths and unrelated
training metadata are omitted. Image filenames remain in the uploaded data.

The prepared dataset and base checkpoint are written to a named Modal Volume;
the selected training settings and job identity are sent as invocation
arguments. The worker checks the payload hash, source digest, class order,
model configuration, and absolute deadline before training. It uses the same
AmphiLens detector adapters and finalizer as local training. On success, it
returns `best.pt`, `last.pt`, `metrics.json`, and `checkpoint.json` with hashes.
The local app checks the returned job identity, file sizes, SHA-256 values,
model compatibility, and checkpoint manifest before registering the checkpoint
in the project. Training does not create validation splits and records
`evaluation: not evaluated`. Since no validation score selects a best epoch,
the cloud `best.pt` artifact aliases the final `last.pt` weights; the manifest
records `checkpoint_selection: last-no-validation`.

## Status, cancel, and cleanup

The saved cloud job list is shown on the **Train model** page, including status,
progress, environment, errors, logs, and the Modal dashboard link. Use **Refresh
status**, **Cancel job**, and **Download and register checkpoint** as needed.
The CLI supports the same actions:

```bash
amphilens cloud status PROJECT RUN_ID
amphilens cloud cancel PROJECT RUN_ID
amphilens cloud collect PROJECT RUN_ID
amphilens cloud cleanup PROJECT RUN_ID
```

After collection, AmphiLens removes the remote job directory. If provider
cleanup fails, the saved job record shows the failure and cleanup can be
retried. Modal documents that deleted Volume files can remain billable for up
to four days, depending on storage conditions. See [Modal Volumes](https://modal.com/docs/guide/volumes).

## Cost and runtime estimates

Estimates use Modal's published GPU, CPU, memory, and Volume rates, checked on
2026-09-29. The GPU rate table is visible in
[Modal pricing](https://modal.com/pricing). The current estimator assumes 8 CPU
cores and 32 GiB RAM, adds a 1.75 multiplier to the high estimate for regional
rate variation, estimates upload time at 10–50 MiB/s, and derives a function
timeout from the selected budget, capped at 24 hours.

Runtime is a planning heuristic, not a measured AmphiLens benchmark. It starts
from an unverified example of 2,000 images × 100 epochs taking 90–120 minutes
on an L4, then scales by image count, epochs, and the square of image size.
Actual speed depends on model, image dimensions, preprocessing, data loading,
and GPU availability. Check current rates before submitting if prices may have
changed.

The budget derives an absolute server-side deadline and a Modal function
timeout; it is not a guaranteed bill cap. Startup and queue time, GPU
preemption, provider accounting, and interruptions can change the final charge.
The worker disables application-level retries and limits the job to one
container. Modal GPU functions are preemptible; inspect job status after an
interruption before deciding whether to submit again. See [Modal GPU
preemption](https://modal.com/docs/guide/preemption).

## Privacy and retention

Cloud training sends the selected prepared images, labels, effective model
settings, and any explicitly selected base weights to Modal. Original source
paths and the complete local dataset manifest are not packed. The project
folder, API token, and result registration stay local. Modal receives image
filenames and image content, which may still identify a site or contain
sensitive wildlife data. Review your organization's rules and Modal's privacy
and retention terms before enabling uploads. A payment method is required by
Modal for GPU use. See [Modal GPU requirements](https://modal.com/docs/guide/gpu).

Credential checks do not upload data. `cloud train` and the browser submit
button display a consent step before the payload is sent. No cloud training
starts during `doctor`, `cloud estimate`, or `cloud diagnose`.

## Live-provider verification

The automated suite exercises the provider-neutral service and a fake Modal
SDK. It does not prove detached-call survival after the client exits, Modal's
real timeout and cancellation behavior, Volume upload/read/delete semantics,
GPU package compatibility, or the billed cost. On 2026-09-29, a live attempt
used three synthetic labeled images, one YOLO26-L epoch at 64px, a requested
T4, a $0.02–$0.04 estimate, and a 600-second derived timeout. The Modal
credentials resolved directly from `.env` and authenticated. The image first
failed because NumPy 2.5.3 requires Python 3.12 while the container uses Python
3.11; the pin is now 2.3.5, and the full image built successfully. Modal then
rejected GPU function creation with “Please add a payment method to use L4 GPU
functions.” No GPU container or training call was started. All failed job
prefixes were removed, and the detached apps stopped with zero tasks. Successful
live GPU training, artifact download, and billed cost remain unverified. A
payment method is required before the GPU run can proceed.

Before release, run a consented smoke job with a non-sensitive snapshot of about
20 labelled images, two epochs, a T4 GPU, and a budget that derives a
600-second function timeout. Record the estimate, wall-clock duration, and
reported charge here after observing them. Verify that the job uploads and
reaches `finished`, downloads all four artifacts, registers a checkpoint whose
local manifest hash matches `best.pt`, removes the uploaded job directory, and
leaves no running container in the Modal dashboard. Do not treat CI or a
successful package build as a substitute for this check.
