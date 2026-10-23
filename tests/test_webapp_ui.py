from __future__ import annotations

import pytest

pytest.importorskip("starlette")


def test_static_frontend_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "AmphiLens" in page.text
    assert 'data-path-picker="directory" data-path-target="open-path"' in page.text
    assert 'data-path-picker="directory" data-path-target="project-path-input"' in page.text
    assert 'name="model_preset"' not in page.text
    assert 'name="short_side_dimension"' not in page.text
    assert 'name="grayscale"' not in page.text
    assert 'name="clahe"' not in page.text
    assert 'src="/static/prediction_state.js"' in page.text
    assert 'src="/static/cvat_project_state.js"' in page.text
    assert 'src="/static/activity_view.js"' in page.text
    assert 'src="/static/navigation_state.js"' in page.text
    assert 'src="/static/required_fields.js"' in page.text
    assert '<span class="nav-label">Choose images to review</span>' in page.text
    assert client.get("/static/prediction_state.js").status_code == 200
    assert client.get("/static/cvat_project_state.js").status_code == 200
    assert client.get("/static/activity_view.js").status_code == 200
    assert client.get("/static/navigation_state.js").status_code == 200
    assert client.get("/static/required_fields.js").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert 'api("/api/path-picker"' in client.get("/static/app.js").text
    script = client.get("/static/app.js").text
    assert 'name="run_name" type="text" maxlength="120"' in script
    assert "run_name: values.run_name" in script
    assert "Prediction run:" in script
    assert "Collect detected images" in script
    assert "/collect-images" in script
    assert "Extract date and time" in script
    assert "/ocr-timestamps" in script
    assert 'data-ocr-timestamps="' in script
    assert 'detectionCount <= 0 || timestampActive ? "disabled" : ""' in script
    assert 'detectionCount <= 0 ? "disabled" : ""' in script
    assert "There are no detected images to collect or process." in script
    assert 'name="training_source"' in script
    assert "data-required-when-cloud" in script
    assert "window.AmphiLensRequiredFields.start(document)" in script
    assert 'name="hosted_model_id"' in script
    assert 'name="model_preset"' in script
    assert "preprocessing: selectedPreprocessing(form)" in script
    assert "resetModelSettings" in script
    assert 'app.cvatServerUrl = data.cvat_server_url || app.cvatServerUrl || "";' in script
    assert 'value="${escapeHtml(app.cvatServerUrl)}"' in script
    assert "__ignore__" in script
    assert 'localDeviceOptionsMarkup(app.doctor?.local_devices)' in script
    assert 'localDeviceOptionsMarkup(app.doctor?.local_training_devices)' in script
    assert 'app.doctor?.local_inference_available === true' in script
    assert 'app.doctor?.local_training_available === true' in script
    assert 'localInferenceAvailable ? "" : " disabled"' in script
    assert 'localInferenceAvailable ? "" : " selected"' in script
    assert 'localTrainingAvailable ? "" : " selected"' in script
    assert 'execution.value = "cloud"' in script
    assert "model card revision recorded by AmphiLens" not in script
    assert "CVAT credentials are not stored in your project." not in script
    assert "Credentials must be configured locally before connecting." not in script
    assert "Credentials are sent only to the local AmphiLens service" not in script
    assert "private Hugging Face repository" not in script
    stylesheet = client.get("/static/app.css")
    assert stylesheet.status_code == 200
    assert 'label[data-required]::after { content: "*";' in stylesheet.text
    assert (
        ".nav-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; }"
        in stylesheet.text
    )
    assert client.get("/static/favicon.svg").status_code == 200


def test_matched_cvat_project_labels_follow_review_task_list_layout(client):
    stylesheet = client.get("/static/app.css")
    script = client.get("/static/app.js")

    assert stylesheet.status_code == 200
    assert script.status_code == 200
    assert ('class="cvat-label-row cvat-label-row-matched"') in script.text
    assert (
        ".cvat-label-row-matched { display: flex; justify-content: space-between; gap: 10px; }"
    ) in stylesheet.text
    assert (
        ".cvat-label-row-matched > :first-child { flex: 1; min-width: 0; white-space: nowrap; }"
    ) in stylesheet.text
    assert (
        ".cvat-label-row-matched > .class-match "
        "{ flex: none; white-space: nowrap; text-align: right; }"
    ) in stylesheet.text
    mobile_matched_row = (
        ".cvat-label-row-matched { flex-direction: column; align-items: flex-start; gap: 5px; }"
    )
    assert mobile_matched_row in stylesheet.text
