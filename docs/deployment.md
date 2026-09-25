# AmphiLens deployment

## Local Python mode

Local mode is the primary v1 deployment. It keeps camera-trap images, annotations, checkpoints, and predictions on the user's machine or institutional workstation.

```bash
python -m pip install -e ".[cli,ui,inference]"
amphilens doctor
amphilens app
```

Use a CUDA-enabled Python environment for large-pool inference and training. The core package does not install CUDA automatically because PyTorch wheels depend on the host operating system and driver.

The supported v1 release profile is Python 3.11. The `inference` extra includes
Pillow, NumPy, OpenCV, PyTorch, torchvision, and Ultralytics. The `training`
extra adds PyYAML. OpenCV is needed when CLAHE is enabled. Run `amphilens doctor`
before the first project to see which optional runtime pieces are available.

For managed CVAT compatibility, use the pinned CVAT Community `v2.76.0`
profile with `cvat-sdk==2.76.0` and `cvat-cli==2.76.0`. Portable ZIP exchange
does not require a CVAT server or CVAT credentials.

Managed CVAT reads `CVAT_URL` and `CVAT_TOKEN` from the process environment.
The token is not accepted in project files or command-line output. The app
uploads selected local images to the configured server only after the user
presses **Send to CVAT**. Use the portable exchange when images must remain
entirely local.

## Docker Compose reference deployment

The CPU service runs the local Streamlit UI with explicit project and model volumes:

```bash
mkdir -p projects models
docker compose up --build amphilens
```

Open `http://localhost:8501`. The Compose file does not expose image data to a remote service; the mounted directories remain local to the host.

The CUDA service is an execution foundation for a configured NVIDIA host:

```bash
docker compose --profile cuda run --rm amphilens-cuda doctor
```

Training images should be pinned to a tested CUDA/PyTorch combination before use in production. Do not treat a successful container build as evidence that a model is scientifically suitable for a new domain.

## Future remote execution

Remote execution will submit a serializable `JobSpec` containing project reference, model reference, configuration, code version, and input references. Workers return a `JobStatus` and an `ArtifactBundle` containing logs, predictions, checkpoints, metrics, and hashes.

The planned order is local process → SSH/Slurm → Docker worker → provider-neutral cloud worker. Object storage, queues, metadata databases, authentication, and multi-user isolation belong to the hosted phase, not the local v1 contract.
