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

- <a href="https://huggingface.co/josh-salako/amphilens" target="_blank" rel="noopener noreferrer">Pretrained models on Hugging Face</a>
- <a href="docs/training.md" target="_blank" rel="noopener noreferrer">Detection, fine-tuning, and active learning guide</a>
- <a href="https://openreview.net/pdf?id=0YnE65NGna" target="_blank" rel="noopener noreferrer">Project paper</a>

## Install and run

1. Clone this repo or download and unzip it.
2. Install <a href="https://docs.astral.sh/uv/getting-started/installation/" target="_blank" rel="noopener noreferrer">uv</a>. See the
   <a href="https://github.com/astral-sh/uv" target="_blank" rel="noopener noreferrer">uv GitHub repository</a> for more information.
3. Linux only: install Zenity for the file picker. macOS and Windows need no extra install.

   - Ubuntu, Debian, or Mint: `sudo apt install zenity`
   - Fedora: `sudo dnf install zenity`
   - Arch or Manjaro: `sudo pacman -S zenity`
   - openSUSE: `sudo zypper install zenity`

4. Open a terminal in the unzipped `amphilens` folder and run:

   ```bash
   uv python install 3.11
   uv sync --locked --python 3.11 --extra cli --extra web --extra inference --extra training
   ```

   uv manages Python 3.11 and installs the app's packages in a local `.venv` folder, so you do not
   need to install pip separately. See <a href="https://docs.astral.sh/uv/guides/install-python/" target="_blank" rel="noopener noreferrer">uv's Python installation guide</a>.

`uv sync` creates `.venv`. For CVAT or Modal, copy the example:

```bash
cp .env.example .env
```

Set the variables for the integrations you use:

- CVAT: Set `CVAT_URL` to your CVAT app URL (for a local server, `http://localhost:8080`). Create an API token in CVAT user settings and set it as `CVAT_TOKEN`.
- Modal: Create a token in the Modal dashboard. Set its ID as `MODAL_TOKEN_ID` and its secret as `MODAL_TOKEN_SECRET`.

`.env` is ignored by Git; do not commit credentials.

5. Start the app:

   ```bash
   uv run --locked amphilens app
   ```

6. Open <a href="http://127.0.0.1:8501" target="_blank" rel="noopener noreferrer">http://127.0.0.1:8501</a> in your browser. Keep the terminal open
   while you use the app.

## Use AmphiLens

Choose **Create project** in the app and select the folder containing your images. AmphiLens keeps
project files and results separate from your original images. By default, projects are saved in
`~/Downloads/AmphiLens/projects`.

Your images stay on your computer unless you choose to send a review queue to CVAT or upload labelled
data for cloud training. A CPU can run predictions; a CUDA GPU is recommended for training and large
collections.

### Optional integrations

Add `--extra cvat` and/or `--extra cloud` to the `uv sync` command above for the integrations you use.

- **CVAT** is for annotating and reviewing images. The `cvat` extra installs AmphiLens's Python client, not the CVAT server. To run CVAT locally, follow <a href="https://docs.cvat.ai/docs/administration/community/basics/installation/" target="_blank" rel="noopener noreferrer">CVAT's Docker installation guide</a>, then start it from the CVAT directory with `docker compose up -d`. Open `http://localhost:8080` to create your account and API token.
- **Modal** runs training and prediction on cloud GPUs when you do not have suitable local hardware. Add `--extra cloud` and see the <a href="docs/cloud-training.md" target="_blank" rel="noopener noreferrer">cloud training guide</a>.

In AmphiLens, send an image queue to CVAT, annotate and save the images there, then return to AmphiLens to continue the cycle. Choose Modal cloud GPU to train or run predictions remotely. AmphiLens coordinates the handoff and tracks the cloud job; annotation is done in CVAT.

## License

The application code is released under <a href="LICENSE" target="_blank" rel="noopener noreferrer">AGPL-3.0-only</a>.
