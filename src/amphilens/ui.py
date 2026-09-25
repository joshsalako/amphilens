"""Small local Streamlit shell over the AmphiLens project contracts."""

from __future__ import annotations

from pathlib import Path


def main():
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("The UI requires the 'ui' extra: pip install 'amphilens[ui]'") from exc

    from .doctor import run_doctor

    st.set_page_config(page_title="AmphiLens", page_icon="🐸", layout="wide")
    st.title("AmphiLens")
    st.caption("Reproducible amphibian and wildlife camera-trap detection")
    st.info(
        "This local-first shell keeps image paths and model artifacts on your machine. "
        "Create a project with the CLI, then use this interface to inspect its environment."
    )
    report = run_doctor(".")
    st.subheader("Environment")
    st.json(report.to_dict())
    st.subheader("Next steps")
    st.markdown(
        "1. Create a project with `amphilens project-create`.\n"
        "2. Register or select a compatible checkpoint.\n"
        "3. Run prediction or export a Hybrid PPAL review queue to CVAT."
    )


if __name__ == "__main__":
    main()

