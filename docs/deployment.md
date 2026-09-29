# Deployment

## 1. Choose where AmphiLens will run

AmphiLens is local-first. Use one of these options:

1. **Local Python** for the normal browser workflow.
2. **Docker Compose** for a repeatable local machine setup.
3. **A future remote worker** for institutional or cloud GPUs.

The first release does not provide a hosted multi-user service.

## 2. Run locally with Python

The supported release profile is Python 3.11. From your AmphiLens checkout:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra ui --extra inference --extra training
uv run --locked amphilens doctor
uv run --locked amphilens app
```

The committed `uv.lock` keeps the tested dependency resolution reproducible.
The core package does not install CUDA automatically because PyTorch wheels and
drivers depend on the host operating system.

## 3. Store projects outside the source checkout

The source checkout is for AmphiLens code. Project folders contain generated
datasets, annotations, checkpoints, predictions, and reports. New projects
therefore default to `~/Downloads/AmphiLens/projects`, and the app rejects new
locations inside the Git checkout.

In the browser app, use **Create project** to enter a project path or **Open
project** to enter a folder containing `manifest.json`. The app remembers the
last active project in a small user-state file and restores it after a browser
refresh when the project still validates. **Close active project** clears the
remembered path.

The state file contains only the project path. It does not contain CVAT tokens,
images, model weights, or project configuration. Its platform locations are:

1. macOS: `~/Library/Application Support/AmphiLens/state.json`;
2. Linux: `~/.config/amphilens/state.json` or `$XDG_CONFIG_HOME/amphilens/state.json`;
3. Windows: `%APPDATA%/AmphiLens/state.json`.

For Docker, mount a host directory at `/projects` and select `/projects` in the
app. This keeps project data on the host rather than inside a disposable
container layer.

The CLI exposes the same policy:

```bash
amphilens project default-location
amphilens project move /path/to/old-project /path/to/projects/study
amphilens project move /path/to/old-project /path/to/projects/study \
  --remove-source
```

`--remove-source` is explicit. Relocation verifies the copied file hashes and
manifest before removing the old folder. Source image folders remain where
they are.

## 4. Decide whether you need a GPU

- CPU inference is supported for small and moderate collections.
- A CUDA GPU is recommended for large image pools and fine-tuning.
- Use `amphilens doctor` before starting a project.
- Do not treat a GPU as proof that a model is suitable for a new wildlife
  domain; evaluate the model on representative data.

## 5. Connect to CVAT

Install the managed CVAT extra when you want the one-button workflow:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra ui --extra inference --extra training --extra cvat
```

Set the CVAT server and token in the process environment:

```bash
export CVAT_URL="http://localhost:8080"
read -r -s CVAT_TOKEN
export CVAT_TOKEN
uv run --locked amphilens app
```

The supported managed profile uses CVAT Community `v2.76.0`,
`cvat-sdk==2.76.0`, and `cvat-cli==2.76.0`. The same connection supports two
workflows:

1. **Import initial annotations:** list existing CVAT projects, select one,
   and download its complete project export with images through the API.
2. **Active learning:** press **Send to CVAT** to create a task for a selected
   review queue, then return to AmphiLens after annotation.

The token is not written to project files, manifests, logs, CSV files, URLs, or
Git. AmphiLens never changes or deletes an existing CVAT project during initial
import.

If images must remain entirely local, use the portable CVAT or YOLO ZIP
exchange instead.

## 6. Run with Docker Compose

The CPU service runs the local Streamlit app and mounts project and model
folders from the host:

```bash
mkdir -p projects models
docker compose up --build amphilens
```

Open [http://localhost:8501](http://localhost:8501). The mounted folders stay
on the host; the Compose file does not upload them to a remote service.

For a configured NVIDIA host, run the CUDA foundation:

```bash
docker compose --profile cuda run --rm amphilens-cuda doctor
```

The Docker images are foundations. Pin and test the CUDA/PyTorch combination
for the target machine before production training.

## 7. Train on a Modal cloud GPU

Cloud training is an optional single-user workflow using the current user's
Modal account. It needs the cloud extra, credentials, network access, and a
valid Modal payment method:

```bash
uv sync --locked --python 3.11 \
  --extra cli --extra ui --extra inference --extra training --extra cloud
amphilens cloud login
amphilens cloud diagnose
```

The app uploads a prepared copy of the selected labelled snapshot and any
selected base checkpoint only after the user accepts the upload and cost
consent. Local projects and result registration stay on the client. Use the
same **Train model** workflow and choose **Modal cloud GPU**, or run
`amphilens cloud estimate` and `amphilens cloud train`. Cloud jobs run in a
detached ephemeral Modal invocation with bounded concurrency and a
budget-derived time limit. The local job record survives app restarts.

Install the `cvat` extra separately when CVAT workflows are also needed. Read
the [cloud training guide](cloud-training.md) for estimates, payload contents,
credential priority, status, cancellation, and cleanup. Institutional SSH,
Slurm, and hosted multi-user execution are still future work.
