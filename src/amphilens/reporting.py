"""Human- and machine-readable reports for prediction artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def write_report(predictions_csv: str | Path, output_dir: str | Path) -> dict[str, Path]:
    source = Path(predictions_csv).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    classes = Counter(row.get("class_name", "") for row in rows if row.get("class_name"))
    images = {row.get("image_path", "") for row in rows if row.get("image_path")}
    confidences = [float(row["confidence"]) for row in rows if row.get("confidence") not in {None, ""}]
    models = sorted({row.get("model_id", "") for row in rows if row.get("model_id")})
    runs = sorted({row.get("run_id", "") for row in rows if row.get("run_id")})
    summary = {
        "predictions_csv": str(source),
        "image_count": len(images),
        "detection_count": len(rows),
        "class_counts": dict(sorted(classes.items())),
        "model_ids": models,
        "run_ids": runs,
        "confidence": {
            "min": min(confidences) if confidences else None,
            "max": max(confidences) if confidences else None,
            "mean": sum(confidences) / len(confidences) if confidences else None,
        },
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    markdown_path = destination / "report.md"
    class_lines = "\n".join(f"- `{name}`: {count}" for name, count in sorted(classes.items())) or "- None"
    markdown_path.write_text(
        "# AmphiLens prediction report\n\n"
        f"- Source CSV: `{source}`\n"
        f"- Images: **{len(images)}**\n"
        f"- Detections: **{len(rows)}**\n"
        f"- Models: `{', '.join(models) or 'unknown'}`\n"
        f"- Runs: `{', '.join(runs) or 'unknown'}`\n\n"
        "## Class counts\n\n"
        f"{class_lines}\n"
    )
    return {"summary_json": summary_path, "markdown": markdown_path}

