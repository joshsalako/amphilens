# AmphiLens

AmphiLens is a local browser app for finding amphibians and other wildlife in
camera-trap images. It helps a conservation or ecology team:

- scan a large image collection with a detection model;
- find the images that are most useful to label;
- improve the model with a small amount of human annotation; and
- download predictions, reports, and review images.

It is designed to work with sensitive wildlife data on the user's own
computer. A GPU is helpful for large collections and model training, but is
not required for basic inference.

## How it works

```mermaid
flowchart LR
    A[Camera-trap images] --> B[AmphiLens browser app]
    B --> C[Base model finds likely animals]
    C --> D[AmphiLens selects useful images to review]
    D --> E[Human checks selected images in CVAT]
    E --> F[AmphiLens improves the model]
    F --> G[Predictions and reports]
    C --> G
```

The selection step is the active-learning method described in the research
paper. It reduces the amount of annotation needed when the new images are
similar to the model's training domain.

## Run the app

AmphiLens currently runs locally from a Python environment. Copy these
commands into a terminal.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cli,ui,inference]"
amphilens app
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[cli,ui,inference]"
amphilens app
```

The app runs on your computer. Open [http://localhost:8501](http://localhost:8501)
if the browser does not open automatically.

## Use the browser app

1. Check the environment and available hardware.
2. Create a project and choose the folder containing the camera-trap images.
3. Choose the wildlife classes, model, maximum image dimension, grayscale, and
   optional CLAHE settings.
4. If you already have labels, import a CVAT-for-Images or YOLO ZIP. The
   archive must include its images.
5. Train the initial model, or run the selected base model directly.
6. Review the active-learning queue and annotate the selected images in CVAT.
7. Import the completed annotations and train the next cycle.
8. Download the predictions CSV, report, and visual evidence.

AmphiLens keeps source images unchanged. By default it downsizes only images
larger than 640 pixels on their longest side, preserves aspect ratio, converts
to three-channel grayscale, and leaves CLAHE off for generic projects. Every
choice is saved with the project and run.

## What you get

- A CSV containing image names, detected classes, confidence scores, and
  bounding boxes.
- A human-readable report and optional images with detection boxes drawn on
  them.
- A reproducible record of the model, settings, source images, and outputs.
- Reusable model checkpoints when fine-tuning is enabled.

## Hardware

CPU inference works for small or moderate collections. A CUDA GPU is strongly
recommended for fine-tuning and large collections. The app's **Environment**
page and `amphilens doctor` show the installed libraries, GPU status, and free
disk space before a run.

## Learn more

- [Research paper: Annotation-Efficient Object Detection of Endangered Western Leopard Toads](https://openreview.net/pdf?id=0YnE65NGna)
- [Technical overview](docs/technical-overview.md)
- [Local and Docker deployment](docs/deployment.md)
- [Fine-tuning notes](docs/training.md)
- [Contributor guide](CONTRIBUTING.md)

## Privacy and scope

In local mode, AmphiLens does not upload images. The project is broader than
Western Leopard Toads: it can support amphibians and other organisms when a
compatible model and labelled examples are available.

## License

AmphiLens is released under the [Apache License 2.0](LICENSE).
