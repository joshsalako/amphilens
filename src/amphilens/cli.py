"""Typer CLI for AmphiLens."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import typer
except ImportError:  # pragma: no cover - exercised only in minimal installs
    typer = None

from .core import ProjectManifest, ProjectStore, iter_images
from .doctor import run_doctor


def _require_typer():
    if typer is None:
        raise RuntimeError("The CLI requires the 'cli' extra: pip install 'amphilens[cli]'")


if typer is not None:
    app = typer.Typer(help="Reproducible wildlife camera-trap detection.")

    @app.command()
    def doctor(path: str = "."):
        """Report local CPU, disk, ML dependency, and CUDA status."""
        typer.echo(json.dumps(run_doctor(path).to_dict(), indent=2))

    @app.command("project-create")
    def project_create(
        project_dir: Path,
        image_root: list[Path] = typer.Option(..., "--image-root"),
        class_name: list[str] = typer.Option(..., "--class-name"),
        name: str = typer.Option("amphilens-project", "--name"),
    ):
        """Create a portable AmphiLens project directory."""
        manifest = ProjectManifest.create(name, image_root, class_name)
        ProjectStore(project_dir).create(manifest)
        typer.echo(f"Created project at {project_dir.resolve()}")

    @app.command()
    def app_ui():
        """Launch the local browser application."""
        ui_path = Path(__file__).with_name("ui.py")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("The UI requires the 'ui' extra: pip install 'amphilens[ui]'") from exc

    @app.command()
    def images(project_dir: Path):
        """List image files recorded by a project manifest."""
        manifest = ProjectStore(project_dir).load_manifest()
        for path in iter_images(manifest.image_roots):
            typer.echo(path)

    def main():
        app()
else:

    def main():
        _require_typer()

