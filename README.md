# AmphiLens

<p align="center">
  <img src="src/amphilens/web/wlt.JPG" alt="Hand-painted western leopard toad" width="220">
</p>

AmphiLens is a local browser app for finding wildlife in camera-trap images and managing image review,
model training, and results.

## Choose a workflow

```mermaid
flowchart TD
    A[Camera-trap images] --> B{Choose a workflow}
    B -->|Quick start| C[Use a pretrained model]
    C --> D[Run detection]
    D --> E[Review or download results]
    B -->|Have labelled images| F[Fine-tune with labelled images]
    F --> G[Run detection]
    G --> E
    B -->|Improve a model| H[Active learning]
    H --> I[Run detection and select a small sample]
    I --> J[Label the sample]
    J --> K[Train or update the model]
    K --> L[Run detection again]
    L --> E
```

## Links

- [AmphiLens on GitHub](https://github.com/joshsalako/amphilens)
- [Pretrained models on Hugging Face](https://huggingface.co/josh-salako/amphilens)
- [Detection, fine-tuning, and active learning guide](docs/training.md)
- [Project paper](https://openreview.net/pdf?id=0YnE65NGna)

## Install and run

1. Download and unzip the project from [AmphiLens on GitHub](https://github.com/joshsalako/amphilens).
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). See the
   [uv GitHub repository](https://github.com/astral-sh/uv) for more information.
3. Open a terminal in the unzipped `amphilens` folder and run:

   ```bash
   uv python install 3.11
   uv sync --locked --python 3.11 --extra cli --extra web --extra inference --extra training
   ```

   uv manages Python 3.11 and installs the app's packages in a local `.venv` folder, so you do not
   need to install pip separately. See [uv's Python installation guide](https://docs.astral.sh/uv/guides/install-python/).

4. Start the app:

   ```bash
   uv run --locked amphilens app
   ```

5. Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser. Keep the terminal open
   while you use the app.

## Use AmphiLens

Choose **Create project** in the app and select the folder containing your images. AmphiLens keeps
project files and results separate from your original images. By default, projects are saved in
`~/Downloads/AmphiLens/projects`.

Your images stay on your computer unless you choose to send a review queue to CVAT or upload labelled
data for cloud training. A CPU can run predictions; a CUDA GPU is recommended for training and large
collections.

For CVAT support, add `--extra cvat` to the `uv sync` command above. For cloud training, add
`--extra cloud` and see the [cloud training guide](docs/cloud-training.md).

## License

The application code is released under [AGPL-3.0-only](LICENSE).
