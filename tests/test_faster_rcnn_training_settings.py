from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from amphilens.core import ValidationError
from amphilens.models import faster_rcnn_training


def _loader_options():
    helper = getattr(faster_rcnn_training, "_faster_rcnn_loader_options", None)
    assert helper is not None, "Faster R-CNN loader options helper is missing"
    return helper


def _amp_enabled():
    helper = getattr(faster_rcnn_training, "_faster_rcnn_amp_enabled", None)
    assert helper is not None, "Faster R-CNN AMP selection helper is missing"
    return helper


def test_faster_rcnn_loader_defaults_to_eight_workers_and_cuda_prefetch():
    options = _loader_options()(
        {"batch_size": 4},
        device=SimpleNamespace(type="cuda"),
    )

    assert options["batch_size"] == 4
    assert options["num_workers"] == 8
    assert options["pin_memory"] is True
    assert options["persistent_workers"] is True
    assert options["prefetch_factor"] == 2


def test_faster_rcnn_loader_omits_worker_only_options_when_workers_are_disabled():
    options = _loader_options()(
        {"batch_size": 2, "num_workers": 0},
        device=SimpleNamespace(type="cuda"),
    )

    assert options["num_workers"] == 0
    assert options["pin_memory"] is True
    assert "persistent_workers" not in options
    assert "prefetch_factor" not in options


def test_faster_rcnn_loader_rejects_negative_worker_count():
    with pytest.raises(ValidationError, match="num_workers must be non-negative"):
        _loader_options()(
            {"num_workers": -1},
            device=SimpleNamespace(type="cuda"),
        )


def test_faster_rcnn_amp_defaults_to_cuda_and_can_be_disabled():
    cuda = SimpleNamespace(type="cuda")
    cpu = SimpleNamespace(type="cpu")

    assert _amp_enabled()({}, cuda) is True
    assert _amp_enabled()({"amp": False}, cuda) is False
    assert _amp_enabled()({}, cpu) is False


def test_faster_rcnn_amp_forward_and_optimizer_use_autocast_and_scaled_step():
    events = []

    class FakeLoss:
        def __init__(self, value):
            self.value = value

        def __radd__(self, other):
            return FakeLoss(other + self.value)

        def backward(self):
            events.append("backward")

    class FakeTorch:
        float16 = object()

        @staticmethod
        @contextmanager
        def autocast(device_type, *, dtype):
            events.append(("autocast-enter", device_type, dtype))
            yield
            events.append("autocast-exit")

    class FakeModel:
        def __call__(self, images, targets):
            assert events[0][0] == "autocast-enter"
            events.append(("model", images, targets))
            return {"classification": FakeLoss(1.25)}

    class FakeOptimizer:
        def zero_grad(self):
            events.append("zero-grad")

        def step(self):
            events.append("optimizer-step")

    class FakeScaler:
        def scale(self, loss):
            events.append(("scale", loss.value))
            return loss

        def step(self, optimizer):
            events.append("scaled-step")
            optimizer.step()

        def update(self):
            events.append("scaler-update")

    helper = getattr(faster_rcnn_training, "_faster_rcnn_training_step", None)
    assert helper is not None, "Faster R-CNN training step helper is missing"
    _, loss = helper(
        FakeTorch,
        FakeModel(),
        ["image"],
        ["target"],
        optimizer=FakeOptimizer(),
        scaler=FakeScaler(),
        training=True,
        amp_enabled=True,
    )

    assert loss.value == 1.25
    assert events == [
        ("autocast-enter", "cuda", FakeTorch.float16),
        ("model", ["image"], ["target"]),
        "autocast-exit",
        "zero-grad",
        ("scale", 1.25),
        "backward",
        "scaled-step",
        "optimizer-step",
        "scaler-update",
    ]


def test_faster_rcnn_grad_scaler_supports_legacy_torch_cuda_amp_api():
    calls = []
    legacy_scaler = object()

    class LegacyCudaAmp:
        @staticmethod
        def GradScaler(*, enabled):
            calls.append(enabled)
            return legacy_scaler

    fake_torch = SimpleNamespace(
        amp=SimpleNamespace(),
        cuda=SimpleNamespace(amp=LegacyCudaAmp),
    )
    helper = getattr(faster_rcnn_training, "_faster_rcnn_grad_scaler", None)
    assert helper is not None, "Faster R-CNN GradScaler compatibility helper is missing"

    assert helper(fake_torch, amp_enabled=True) is legacy_scaler
    assert calls == [True]
