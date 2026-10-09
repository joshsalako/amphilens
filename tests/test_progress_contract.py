from __future__ import annotations

import os
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from amphilens.cloud.modal_transport import ModalTransport
from amphilens.cloud.models import CloudConsent, CloudJobRecord
from amphilens.cloud.prediction import ModalPredictionDetector
from amphilens.cloud.service import CloudTrainingService
from amphilens.cloud.worker import _TailBuffer
from amphilens.core import DetectionRecord, InferenceConfig, atomic_write_json
from amphilens.models.backends import _trainer_metrics
from amphilens.runs import run_resumable_inference
from amphilens.training import TrainingConfig, train_and_register
from amphilens.webapp import JobManager, _modal_prediction_timing_samples


class ProgressDetector:
    model_id = "progress-model"

    def predict(self, image_paths, config):
        for path in image_paths:
            if path.name == "01.jpg":
                raise RuntimeError("fixture image error")
            yield DetectionRecord(
                image_path=str(path),
                image_id=path.name,
                class_id=0,
                class_name="toad",
                confidence=0.9,
                bbox_xyxy=[1, 2, 11, 22],
                image_width=20,
                image_height=40,
                model_id=self.model_id,
                run_id=config.run_id,
            )


class TrainingProgressDetector:
    model_id = "training-progress-model"
    architecture = "yolo"
    classes = ["toad"]

    def train(self, dataset_yaml, output_dir, config, resume_from=None, progress_callback=None):
        del dataset_yaml, resume_from
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(b"checkpoint")
        if progress_callback:
            progress_callback(
                {
                    "epoch": 1,
                    "epochs": int(config.get("epochs", 1)),
                    "progress": 0.5,
                    "metrics": {"train_loss": 0.25},
                }
            )
        return checkpoint


class ProgressModalTransport:
    def upload_prediction_batch(self, job_key, batch_id, sources):
        return [
            {
                "image_id": "local-1",
                "remote_path": f"prediction-jobs/{job_key}/{batch_id}/images/1.jpg",
                "sha256": "0" * 64,
            }
            for _ in sources
        ]

    def predict_batch(
        self,
        payload,
        *,
        cancellation_requested=None,
        progress_callback=None,
        job_id_callback=None,
    ):
        del payload, cancellation_requested
        if job_id_callback:
            job_id_callback("progress-call")
        if progress_callback:
            progress_callback(
                {"phase": "model_download", "message": "Downloading model", "progress": 0.5}
            )
            progress_callback(
                {"phase": "prediction", "message": "Predicting current batch", "progress": None}
            )
        return {"state": "finished", "records": [], "elapsed_seconds": 1}

    def cleanup_prediction_batch(self, job_key, batch_id):
        del job_key, batch_id


class ProgressContractTests(unittest.TestCase):
    def test_resumable_inference_reports_counts_after_each_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = []
            for index in range(3):
                image = root / f"{index:02}.jpg"
                image.write_bytes(b"image")
                images.append(image)
            events = []

            summary = run_resumable_inference(
                ProgressDetector(),
                images,
                InferenceConfig(model_id="progress-model", batch_size=1, run_id="progress-run"),
                root / "artifacts",
                progress_callback=events.append,
            )

            self.assertEqual(summary.completed_images, 2)
            self.assertEqual(events[-1]["completed"], 2)
            self.assertEqual(events[-1]["failed"], 1)
            self.assertEqual(events[-1]["remaining"], 0)
            self.assertEqual(events[-1]["progress"], 1.0)
            prediction_events = [event for event in events if event["phase"] == "prediction"]
            self.assertEqual([event["completed"] for event in prediction_events], [0, 1, 1, 2])

    def test_job_status_keeps_bounded_redacted_progress_events(self):
        manager = JobManager(max_workers=1, max_pending=2, max_history=4)
        started = threading.Event()
        release = threading.Event()

        def operation(report):
            report(
                {
                    "phase": "training",
                    "message": "Epoch 1 of 2",
                    "epoch": 1,
                    "epochs": 2,
                    "metrics": {"status": "MODAL_TOKEN_SECRET=secret-value"},
                    "log_tail": "MODAL_TOKEN_SECRET=secret-value\n" + ("x" * 9000),
                }
            )
            started.set()
            release.wait()
            return {"message": "done"}

        try:
            with patch.dict(os.environ, {"MODAL_TOKEN_SECRET": "secret-value"}):
                job_id = manager.submit(operation, with_progress=True)
                self.assertTrue(
                    started.wait(timeout=10),
                    f"background job did not start: {manager.snapshot(job_id)}",
                )
                status = manager.snapshot(job_id)
                self.assertEqual(status["progress"]["phase"], "training")
                self.assertEqual(len(status["progress"]["log_tail"]), 8000)
                self.assertNotIn("secret-value", status["progress"]["log_tail"])
                self.assertNotIn("secret-value", status["progress"]["metrics"]["status"])
                self.assertEqual(status["progress_events"][-1]["epoch"], 1)
        finally:
            release.set()
            manager.shutdown()

    def test_training_reports_phase_epoch_metrics_and_finalization(self):
        with TemporaryDirectory() as directory:
            events = []
            train_and_register(
                TrainingProgressDetector(),
                dataset_yaml=Path(directory) / "dataset.yaml",
                output_dir=Path(directory) / "training",
                config=TrainingConfig(epochs=2),
                preprocessing={"name": "none"},
                progress_callback=events.append,
            )

            self.assertEqual(events[0]["phase"], "training")
            self.assertEqual(events[1]["epoch"], 1)
            self.assertEqual(events[1]["epochs"], 2)
            self.assertEqual(events[1]["metrics"], {"train_loss": 0.25})
            self.assertEqual(events[-1]["phase"], "finalizing")

    def test_ultralytics_training_metrics_are_normalized_for_the_activity_view(self):
        trainer = SimpleNamespace(
            metrics={"metrics/precision(B)": 0.75, "fitness": None},
            loss_names=["box_loss", "cls_loss"],
            tloss=[0.2, 0.1],
        )

        self.assertEqual(
            _trainer_metrics(trainer),
            {"metrics/precision(B)": 0.75, "train/box_loss": 0.2, "train/cls_loss": 0.1},
        )

    def test_modal_eta_history_matches_both_model_and_gpu(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            for name, model, gpu, elapsed, provider in (
                ("match", "model-a", "L4", 20, None),
                ("wrong-model", "model-b", "L4", 10, "modal"),
                ("wrong-gpu", "model-a", "A10G", 5, "modal"),
                ("wrong-provider", "model-a", "L4", 50, "vertex_ai"),
            ):
                run = runs / name
                run.mkdir(parents=True)
                sample = {
                    "model_id": model,
                    "gpu": gpu,
                    "elapsed_seconds": elapsed,
                    "completed_images": 10,
                }
                if provider is not None:
                    sample["provider"] = provider
                atomic_write_json(run / "cloud-cost.json", sample)

            samples = _modal_prediction_timing_samples(SimpleNamespace(root=root), "model-a", "L4")

            self.assertEqual(samples, [2.0])

    def test_remote_output_tail_drops_terminal_progress_bars_and_stays_bounded(self):
        tail = _TailBuffer(32)
        tail.write("Starting training\n")
        tail.write(" 42%|████▏     | 21/50 [00:10<00:14, 2.0it/s]\r")
        tail.write("Epoch 21: train loss 0.14\n")
        tail.write("z" * 100)

        self.assertNotIn("42%", tail.getvalue())
        self.assertEqual(len(tail.getvalue()), 32)
        self.assertTrue(tail.getvalue().endswith("z" * 32))

    def test_modal_prediction_exposes_upload_remote_setup_and_completed_batch(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "image.jpg"
            image.write_bytes(b"image")
            events = []
            detector = ModalPredictionDetector(
                ProgressModalTransport(),
                job_key="progress-job",
                gpu="L4",
                model_spec={"source": "hosted", "model_id": "model-a"},
                timeout_seconds=60,
                max_cost_usd=1.0,
                output_dir=Path(directory) / "run",
                seconds_per_image=9.5,
                progress_callback=events.append,
                image_count=1,
            )

            list(detector.predict([image], InferenceConfig(model_id="model-a", run_id="run-1")))

            self.assertEqual(events[0]["phase"], "image_transfer")
            self.assertEqual(events[0]["remaining"], 1)
            self.assertIn("model_download", [event["phase"] for event in events])
            download_event = next(event for event in events if event["phase"] == "model_download")
            self.assertAlmostEqual(download_event["eta_seconds"], 9.5)
            self.assertEqual(events[-1]["completed"], 1)
            self.assertEqual(events[-1]["remaining"], 0)

    def test_modal_transport_forwards_progress_while_remote_batch_runs(self):
        class RunContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeApp:
            def run(self, **kwargs):
                del kwargs
                return RunContext()

        class FakeCall:
            attempts = 0

            def get(self, timeout):
                del timeout
                self.attempts += 1
                if self.attempts == 1:
                    raise TimeoutError()
                return {"state": "finished", "records": []}

        call = FakeCall()

        class RemoteMethod:
            def spawn(self, payload):
                del payload
                return call

        class FakeEngine:
            predict_batch = RemoteMethod()

        class EngineFactory:
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return FakeEngine()

        factory = EngineFactory()

        class PredictionEngine:
            @classmethod
            def with_options(cls, **kwargs):
                del kwargs
                return factory

        class AppModule:
            pass

        AppModule.app = FakeApp()
        AppModule.PredictionEngine = PredictionEngine

        transport = object.__new__(ModalTransport)
        transport._load_modal = lambda: None
        transport._get_client = lambda: object()
        transport._load_app_module = lambda: AppModule
        transport._prediction_progress = lambda job_key: {
            "phase": "model_download",
            "message": f"Downloading for {job_key}",
            "phase_progress": 0.6,
        }
        events = []

        result = transport.predict_batch(
            {
                "job_key": "job-1",
                "batch_id": "batch-1",
                "gpu": "L4",
                "timeout_seconds": 30,
                "model_spec": {},
            },
            progress_callback=events.append,
        )

        self.assertEqual(result["state"], "finished")
        self.assertTrue(any(event["phase"] == "model_download" for event in events))
        self.assertEqual(factory.kwargs["job_key"], "job-1")
        self.assertEqual(factory.kwargs["gpu_type"], "L4")

    def test_cloud_refresh_keeps_live_epoch_and_log_details(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = CloudJobRecord(
                run_id="cloud-progress-run",
                job_key="job-1",
                state="submitted",
                created_at="2026-10-07T00:00:00+00:00",
                updated_at="2026-10-07T00:00:00+00:00",
                project_name="field-study",
                snapshot_id="snapshot-1",
                effective_fingerprint="fingerprint",
                effective_configuration={},
                training_config={"epochs": 2},
                consent=CloudConsent(
                    acknowledged=True,
                    uploads_dataset=True,
                    estimated_usd=0.1,
                    max_cost_usd=1.0,
                ),
                estimate={},
                gpu="L4",
                timeout_seconds=60,
                call_id="call-1",
            )
            record_path = root / "runs" / record.run_id / "cloud-job.json"
            record_path.parent.mkdir(parents=True)
            atomic_write_json(record_path, record.to_dict())

            class FakeTransport:
                def poll(self, call_id, *, remote_prefix):
                    del call_id, remote_prefix
                    return {
                        "state": "running",
                        "progress": 0.5,
                        "progress_details": {
                            "phase": "training",
                            "message": "Epoch 1 of 2",
                            "epoch": 1,
                            "epochs": 2,
                            "metrics": {"train_loss": 0.3},
                            "log_tail": "training output",
                        },
                    }

            service = CloudTrainingService(SimpleNamespace(root=root), FakeTransport())

            updated = service.refresh(record.run_id)

            self.assertEqual(updated.progress_details["epoch"], 1)
            self.assertEqual(updated.log_tail, "training output")
            self.assertEqual(updated.progress_events[-1]["metrics"]["train_loss"], 0.3)

            persisted = service.get_job(record.run_id)
            self.assertEqual(persisted.progress_details["epoch"], 1)
            self.assertEqual(persisted.log_tail, "training output")
            self.assertEqual(persisted.progress_events[-1]["metrics"]["train_loss"], 0.3)


if __name__ == "__main__":
    unittest.main()
