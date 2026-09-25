import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from amphilens.core import ProjectManifest, ProjectStore, ValidationError
from amphilens.dataset import DatasetImporter, DatasetMerger
from amphilens.preprocessing import PreprocessingConfig


def _image_bytes(size=(40, 30), color=(0, 0, 0)) -> bytes:
    from io import BytesIO

    stream = BytesIO()
    Image.new("RGB", size, color=color).save(stream, format="PNG")
    return stream.getvalue()


def _zip(path: Path, files: dict[str, bytes | str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value.encode() if isinstance(value, str) else value)
    return path


def test_imports_cvat_xml_zip_and_preserves_reviewed_negative(tmp_path: Path):
    archive = _zip(
        tmp_path / "cvat.zip",
        {
            "images/toad.png": _image_bytes(),
            "images/negative.png": _image_bytes(color=(20, 20, 20)),
            "annotations.xml": """
                <annotations>
                  <meta><task><labels>
                    <label><name>toad</name></label>
                  </labels></task></meta>
                  <image name="images/toad.png" width="40" height="30">
                    <box label="toad" xtl="4" ytl="5" xbr="20" ybr="25"/>
                  </image>
                  <image name="images/negative.png" width="40" height="30"/>
                </annotations>
            """,
        },
    )

    snapshot = DatasetImporter().import_archive(archive, tmp_path / "snapshot", classes=["toad"])

    assert len(snapshot.manifest.images) == 2
    assert sum(len(image.annotations) for image in snapshot.manifest.images) == 1
    negative = next(
        image for image in snapshot.manifest.images if image.source_path.endswith("negative.png")
    )
    assert negative.reviewed is True
    assert negative.annotations == []
    assert snapshot.manifest.source_format == "cvat-xml"
    assert (snapshot.root / "source" / "cvat.zip").is_file()


def test_imports_coco_zip_and_maps_classes_explicitly(tmp_path: Path):
    coco = {
        "images": [{"id": 7, "file_name": "images/camera.png", "width": 40, "height": 30}],
        "categories": [{"id": 3, "name": "Western_Leopard_Toad"}],
        "annotations": [{"id": 9, "image_id": 7, "category_id": 3, "bbox": [4, 5, 16, 20]}],
    }
    archive = _zip(
        tmp_path / "coco.zip",
        {"images/camera.png": _image_bytes(), "annotations.json": json.dumps(coco)},
    )

    snapshot = DatasetImporter().import_archive(
        archive,
        tmp_path / "snapshot",
        classes=["toad"],
        class_mapping={"Western_Leopard_Toad": "toad"},
    )

    annotation = snapshot.manifest.images[0].annotations[0]
    assert annotation.class_name == "toad"
    assert annotation.bbox_xyxy == [4.0, 5.0, 20.0, 25.0]


def test_imports_yolo_zip_and_retains_empty_label_as_negative(tmp_path: Path):
    archive = _zip(
        tmp_path / "yolo.zip",
        {
            "images/a.png": _image_bytes(),
            "images/b.png": _image_bytes(color=(10, 10, 10)),
            "labels/a.txt": "0 0.3 0.5 0.4 0.666667\n",
            "labels/b.txt": "",
            "classes.txt": "toad\n",
        },
    )

    snapshot = DatasetImporter().import_archive(archive, tmp_path / "snapshot", classes=["toad"])

    assert [len(image.annotations) for image in snapshot.manifest.images] == [1, 0]
    assert all(image.reviewed for image in snapshot.manifest.images)
    assert snapshot.manifest.images[0].annotations[0].bbox_xyxy == pytest.approx(
        [4.0, 5.0, 20.0, 25.0]
    )


def test_import_rejects_annotation_only_archives_and_invalid_boxes(tmp_path: Path):
    annotation_only = _zip(tmp_path / "only.zip", {"annotations.xml": "<annotations/>"})
    with pytest.raises(ValidationError, match="image files"):
        DatasetImporter().import_archive(annotation_only, tmp_path / "only-out", classes=["toad"])

    invalid = _zip(
        tmp_path / "invalid.zip",
        {
            "images/a.png": _image_bytes(),
            "annotations.json": json.dumps(
                {
                    "images": [{"id": 1, "file_name": "images/a.png", "width": 40, "height": 30}],
                    "categories": [{"id": 1, "name": "toad"}],
                    "annotations": [
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [39, 2, 4, 4]}
                    ],
                }
            ),
        },
    )
    with pytest.raises(ValidationError, match="outside image bounds"):
        DatasetImporter().import_archive(invalid, tmp_path / "invalid-out", classes=["toad"])


def test_import_rejects_class_mismatch_corrupt_image_and_duplicate_hash(tmp_path: Path):
    mismatch = _zip(
        tmp_path / "mismatch.zip",
        {
            "images/a.png": _image_bytes(),
            "classes.txt": "frog\n",
            "labels/a.txt": "0 0.5 0.5 0.2 0.2\n",
        },
    )
    with pytest.raises(ValidationError, match="class mapping"):
        DatasetImporter().import_archive(mismatch, tmp_path / "mismatch-out", classes=["toad"])

    corrupt = _zip(
        tmp_path / "corrupt.zip", {"images/a.png": b"not an image", "classes.txt": "toad\n"}
    )
    with pytest.raises(ValidationError, match="cannot be read"):
        DatasetImporter().import_archive(corrupt, tmp_path / "corrupt-out", classes=["toad"])

    duplicate = _zip(
        tmp_path / "duplicate.zip",
        {"images/a.png": _image_bytes(), "images/b.png": _image_bytes(), "classes.txt": "toad\n"},
    )
    with pytest.raises(ValidationError, match="duplicate image content"):
        DatasetImporter().import_archive(duplicate, tmp_path / "duplicate-out", classes=["toad"])


def test_dataset_merge_creates_new_snapshot_without_mutating_parent(tmp_path: Path):
    first = _zip(
        tmp_path / "first.zip",
        {"images/a.png": _image_bytes(), "classes.txt": "toad\n", "labels/a.txt": ""},
    )
    second = _zip(
        tmp_path / "second.zip",
        {
            "images/b.png": _image_bytes(color=(4, 4, 4)),
            "classes.txt": "toad\n",
            "labels/b.txt": "",
        },
    )
    importer = DatasetImporter()
    parent = importer.import_archive(first, tmp_path / "parent", classes=["toad"])
    incoming = importer.import_archive(second, tmp_path / "incoming", classes=["toad"])

    merged = DatasetMerger().merge(parent, incoming, tmp_path / "merged")

    assert merged.manifest.parent_snapshot == parent.manifest.snapshot_id
    assert len(parent.manifest.images) == 1
    assert len(merged.manifest.images) == 2
    assert (parent.root / "manifest.json").is_file()
    assert parent.root != merged.root


def test_project_store_imports_snapshot_using_manifest_classes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    project = ProjectStore(tmp_path / "project")
    project.create(ProjectManifest.create("study", [source], ["toad"]))
    archive = _zip(
        tmp_path / "initial.zip",
        {"images/a.png": _image_bytes(), "classes.txt": "toad\n", "labels/a.txt": ""},
    )

    snapshot = project.import_dataset(archive)

    assert snapshot.root.is_relative_to(project.root / "datasets")
    assert snapshot.manifest.classes == ["toad"]


def test_snapshot_prepares_yolo_training_data_with_shared_preprocessing(tmp_path: Path):
    archive = _zip(
        tmp_path / "initial.zip",
        {
            "images/a.png": _image_bytes(size=(80, 40)),
            "classes.txt": "toad\n",
            "labels/a.txt": "0 0.5 0.5 0.5 0.5\n",
        },
    )
    snapshot = DatasetImporter().import_archive(archive, tmp_path / "snapshot", classes=["toad"])

    dataset_yaml = snapshot.to_yolo_dataset(
        tmp_path / "prepared",
        preprocessing=PreprocessingConfig(max_dimension=40),
    )

    assert dataset_yaml.is_file()
    with Image.open(dataset_yaml.parent / "images" / "a.png") as image:
        assert image.size == (40, 20)
        assert image.getpixel((0, 0))[0] == image.getpixel((0, 0))[1]
    values = (dataset_yaml.parent / "labels" / "a.txt").read_text().split()
    assert values[0] == "0"
    assert [float(value) for value in values[1:]] == pytest.approx([0.5, 0.5, 0.5, 0.5])
