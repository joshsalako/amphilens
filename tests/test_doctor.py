from amphilens.doctor import run_doctor


def test_doctor_reports_runtime_and_disk_information(tmp_path):
    report = run_doctor(tmp_path)

    assert report.python
    assert report.platform
    assert report.cpu_count > 0
    assert report.free_disk_gb >= 0
    assert report.cvat_exchange is True

