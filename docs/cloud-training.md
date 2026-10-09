# Cloud GPU training and batch prediction

AmphiLens can run training and batch prediction locally or through Modal, Azure
Machine Learning, and Google Vertex AI. Cloud jobs use your configured account
and existing storage resources; the project folder and result registry stay on
your computer. Cloud SDKs are optional dependencies.

## Setup and credentials

Install the cloud extra in addition to the usual training and browser app dependencies:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra web --extra inference --extra training --extra cloud
```

The `cloud` extra installs Modal. Install `cloud-azure` or `cloud-gcp` instead
when using those providers:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra web --extra inference --extra training --extra cloud-azure
# or use --extra cloud-gcp
```

### Azure Machine Learning

Sign in on the computer running AmphiLens and configure the existing Azure ML
workspace in **Check your setup**:

```bash
az login
```

Enter the subscription, resource group, workspace, region, and the client ID of
the user-assigned managed identity configured on that workspace. The signed-in
user needs permission to submit Azure ML jobs and access the workspace's
`workspaceblobstore`. The workspace identity needs permission to read and write
job objects in that storage container; allow deletion too so staged files can
be cleaned up. Azure ML serverless jobs use the selected VM type for each run.
The app's connection check verifies the local identity, workspace identity, and
storage access without uploading a project dataset. See the [Azure ML serverless
compute guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-serverless-compute?view=azureml-api-2)
and [Azure ML authentication guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2).

### Google Vertex AI

Configure Application Default Credentials on the computer running AmphiLens,
then enter an existing project, region, staging bucket, and least-privilege job
service account in **Check your setup**:

```bash
gcloud auth application-default login
```

Enable the Vertex AI and Cloud Storage APIs. The signed-in user needs permission
to submit custom jobs, impersonate the selected job service account, and read,
write, and delete objects in the staging bucket. The job service account also
needs read, write, and delete access to that bucket. The app's connection check
verifies Vertex job-list access and that the bucket is visible; it does not
submit a GPU job. See Google's [local ADC setup](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment),
[custom training service account guide](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/custom-service-account),
and [Vertex AI Python SDK reference](https://docs.cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob).

For both providers, AmphiLens creates temporary per-run GPU jobs and staged
objects. It does not create a workspace, project, bucket, or persistent GPU
cluster. Cloud estimates use the provider's public price catalog for the chosen
region; an unknown rate blocks submission. Estimates are advisory and are not
billing caps. Review the upload consent and spending limit before each job.

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

In the browser app, open **Train a model**, choose a data mode and its snapshot
roles, then choose **Modal cloud GPU**, model settings, GPU, and budget. Review
the estimate and confirm the upload and cost consent before submitting.

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

AmphiLens prepares role-specific copies from the immutable dataset snapshots,
applies the selected preprocessing, removes image metadata by re-encoding the
images, and packages images, YOLO labels, and a path-rewritten `dataset.yaml`.
Automatic splitting is deterministic and keeps source groups together unless
the user explicitly allows image-level fallback. The local snapshots and source
image folders are not changed. A selected base checkpoint
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
in the project. Validation data is used for early stopping in each of the two
freeze/unfreeze phases. If a test partition or test snapshot was selected, it
is used only for final evaluation. Otherwise, the run records
`evaluation: not evaluated` and skips final evaluation. The returned
`best.pt` is selected using validation performance.

## Status, cancel, and cleanup

The saved cloud job list is shown on the **Train a model** page, including status,
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
on an L4, then scales by training image count, the total epoch budget across
both phases, and the square of image size.
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
GPU package compatibility, or the billed cost. On 2026-09-29, a live smoke test
used three synthetic labeled images, one YOLO26-L epoch at 64px, a requested
T4, a $0.02–$0.04 estimate, and a 600-second derived timeout. Credentials
resolved from `.env`; the Modal SDK 1.6.0 authenticated and dispatched the job.
The worker reported a Tesla T4 and completed training without validation. The
dataset payload uploaded, all four outputs (`best.pt`, `last.pt`, `metrics.json`,
and `checkpoint.json`) returned with matching hashes and sizes, and the best
checkpoint was registered locally. CPU inference using that returned checkpoint
produced a local prediction image. The remote job directory was removed and the
Modal app stopped with no active app listed. The live image probe also confirmed
that OpenCV and YOLO import successfully in the pinned Python 3.11 image.

The smoke test verifies transfer and lifecycle behavior with synthetic data; it
does not measure model quality on wildlife data or confirm the provider's final
billed charge. The $0.02–$0.04 figure is an estimate, not a billing cap. Before
production use, run a consented job with a representative non-sensitive
snapshot, then compare the estimate with Modal's recorded usage and inspect the
registered model's behavior on a held-out dataset.
