from __future__ import annotations

from subprocess import CompletedProcess

import pytest

pytest.importorskip("starlette")
from starlette.testclient import TestClient

import amphilens.webapp as webapp


def test_local_path_picker_returns_the_selected_path(client, monkeypatch, tmp_path):
    selected = tmp_path / "images"
    selected.mkdir()
    monkeypatch.setattr(webapp, "_native_path_picker", lambda kind: str(selected), raising=False)

    response = client.post("/api/path-picker", json={"kind": "directory"})

    assert response.status_code == 200
    assert response.json() == {"path": str(selected), "cancelled": False}


def test_local_path_picker_rejects_remote_clients(client):
    remote_client = TestClient(client.app, client=("192.0.2.10", 50000))

    response = remote_client.post("/api/path-picker", json={"kind": "directory"})

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "File and folder selection is available only from this computer."
    )


def test_local_path_picker_treats_cancel_as_a_normal_result(client, monkeypatch):
    monkeypatch.setattr(webapp, "_native_path_picker", lambda kind: None, raising=False)

    response = client.post("/api/path-picker", json={"kind": "file"})

    assert response.status_code == 200
    assert response.json() == {"path": None, "cancelled": True}


def test_macos_picker_uses_the_native_folder_dialog(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "darwin")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="/Users/example/images/\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("directory")

    assert selected == "/Users/example/images"
    assert calls[0][0][0] == "osascript"
    assert "choose folder" in calls[0][0][-1]


def test_macos_picker_treats_cancel_as_no_selection(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "darwin")

    def run(command, **kwargs):
        return CompletedProcess(command, 1, stdout="", stderr="User canceled. (-128)")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    assert webapp._native_path_picker("file") is None


def test_windows_picker_uses_a_sta_windows_forms_dialog(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "win32")
    monkeypatch.setattr(webapp.shutil, "which", lambda name: "powershell.exe")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="C:\\Users\\example\\images\r\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("directory")

    assert selected == "C:\\Users\\example\\images"
    command, kwargs = calls[0]
    assert command[:3] == ["powershell.exe", "-NoProfile", "-STA"]
    assert "System.Windows.Forms.FolderBrowserDialog" in command[-1]
    assert kwargs["timeout"] == 120


@pytest.mark.parametrize(
    ("picker", "expected_args"),
    [
        ("zenity", ["--file-selection", "--title=AmphiLens", "--save", "--confirm-overwrite"]),
        ("kdialog", ["--getsavefilename", "selection_queue.csv"]),
    ],
)
def test_linux_picker_uses_an_available_native_dialog(monkeypatch, picker, expected_args):
    monkeypatch.setattr(webapp.sys, "platform", "linux")
    monkeypatch.setattr(
        webapp.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == picker else None,
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="/tmp/selection_queue.csv\n", stderr="")

    monkeypatch.setattr(webapp.subprocess, "run", run)

    selected = webapp._native_path_picker("save-file")

    assert selected == "/tmp/selection_queue.csv"
    command, _ = calls[0]
    assert command[0] == f"/usr/bin/{picker}"
    assert all(argument in command for argument in expected_args)


def test_linux_picker_explains_when_no_native_dialog_is_installed(monkeypatch):
    monkeypatch.setattr(webapp.sys, "platform", "linux")
    monkeypatch.setattr(webapp.shutil, "which", lambda name: None)

    with pytest.raises(webapp.LocalPathPickerError, match="install zenity or kdialog"):
        webapp._native_path_picker("directory")
