import pytest

from amphilens.doctor import DoctorReport, run_doctor


@pytest.mark.parametrize(
    ("installed", "expected_devices"),
    [
        ({"torch_installed": True, "ultralytics_installed": True}, ["auto", "cpu"]),
        (
            {
                "torch_installed": True,
                "ultralytics_installed": True,
                "cuda_available": True,
            },
            ["auto", "cpu", "cuda"],
        ),
        (
            {
                "torch_installed": True,
                "ultralytics_installed": True,
                "mps_available": True,
            },
            ["auto", "cpu", "mps"],
        ),
        ({}, []),
    ],
    ids=["cpu", "cuda", "mps", "cloud-only"],
)
def test_doctor_reports_only_usable_local_devices_for_prediction_and_training(
    installed, expected_devices
):
    report = DoctorReport(
        **{
            "python": "3.11.0",
            "platform": "test-platform",
            "cpu_count": 1,
            "free_disk_gb": 1.0,
            "torch_installed": False,
            "cuda_available": False,
            "mps_available": False,
            "ultralytics_installed": False,
            "browser_app_installed": True,
            "yaml_installed": True,
            **installed,
        }
    )

    capabilities = report.to_dict()

    assert capabilities["local_devices"] == expected_devices
    assert capabilities["local_training_devices"] == expected_devices
    assert capabilities["local_inference_available"] is bool(expected_devices)
    assert capabilities["local_training_available"] is bool(expected_devices)


def test_doctor_keeps_inference_available_when_local_training_extra_is_missing():
    report = DoctorReport(
        python="3.11.0",
        platform="test-platform",
        cpu_count=1,
        free_disk_gb=1.0,
        torch_installed=True,
        cuda_available=False,
        mps_available=False,
        ultralytics_installed=True,
        browser_app_installed=True,
        yaml_installed=False,
    )

    capabilities = report.to_dict()

    assert capabilities["local_inference_available"] is True
    assert capabilities["local_training_available"] is False
    assert capabilities["local_devices"] == ["auto", "cpu"]
    assert capabilities["local_training_devices"] == []


def test_doctor_reports_runtime_and_disk_information(tmp_path):
    report = run_doctor(tmp_path)

    assert report.python
    assert report.platform
    assert report.cpu_count > 0
    assert report.free_disk_gb >= 0
    assert report.cvat_exchange is True
    assert isinstance(report.pillow_installed, bool)
    assert isinstance(report.opencv_installed, bool)
    assert isinstance(report.cvat_sdk_installed, bool)
    assert isinstance(report.mps_available, bool)
    assert report.modal_connectivity == "not checked"
    assert isinstance(report.modal_token_present, bool)
    assert isinstance(report.browser_app_installed, bool)
