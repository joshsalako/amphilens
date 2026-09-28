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
therefore default to the platform user-data directory, such as
`~/Library/Application Support/AmphiLens/projects` on macOS, and the app
rejects new locations inside the Git checkout.

In the browser app, choose **Create project** to select a location or **Open
project** to load a folder containing `manifest.json`. The app remembers the
active project for the current session and uses it across all workflows.

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

## 7. Future remote execution

Remote execution will use the same project and artifact contracts through SSH,
Slurm, Docker workers, and provider-neutral cloud workers. Hosted deployment
will require separate APIs, queues, object storage, authentication, and
multi-user isolation; those components are not part of local v1.
