import importlib.util
from pathlib import Path

import numpy as np
import pickle
import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "training"))
    spec = importlib.util.spec_from_file_location("train_distill", Path(__file__).resolve().parents[1] / "training/train_distill.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_device", lambda: torch.device("cpu"))
    return module


def test_teacher_free_loss_is_cross_entropy(trainer):
    logits = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.randint(8, (2, 3))
    ce, kl = trainer.loss_terms(logits, None, targets, 2)
    expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.testing.assert_close(ce, expected)
    assert kl.item() == 0
    ce.backward()
    assert logits.grad.abs().sum() > 0


def test_teacher_free_training_resumes_and_quantizes_without_teacher(trainer, tmp_path, monkeypatch):
    meta = {"vocab_size": 128, "context_length": 8, "stoi": {str(i): i for i in range(128)}}
    (tmp_path / "meta.pkl").write_bytes(pickle.dumps(meta))
    for split in ("train", "val"):
        np.arange(128, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    settings = {"DATASET_DIR": str(tmp_path), "CKPT_DIR": str(tmp_path / "float"),
                "TEACHER_CKPT": str(tmp_path / "nonexistent.pt"), "DISTILL_ALPHA": "0",
                "STUDENT_LAYERS": "1", "STUDENT_HEADS": "1", "STUDENT_KV_HEADS": "1", "STUDENT_WIDTH": "64",
                "BATCH_SIZE": "1", "BLOCK_SIZE": "4", "GRAD_ACCUM_STEPS": "1",
                "VAL_BATCHES": "1", "MAX_STEPS": "1", "EVAL_INTERVAL": "1"}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    trainer.main()
    monkeypatch.setenv("MAX_STEPS", "2")
    trainer.main()
    saved = torch.load(tmp_path / "float/student_latest.pt", weights_only=False)
    assert saved["step"] == 2
    assert saved["meta"]["distill_alpha"] == 0
    monkeypatch.setenv("INIT_FROM", str(tmp_path / "float/student_best.pt"))
    monkeypatch.setenv("CKPT_DIR", str(tmp_path / "qat"))
    monkeypatch.setenv("QAT", "1")
    trainer.main()
    saved = torch.load(tmp_path / "qat/student_best.pt", weights_only=False)
    assert saved["meta"]["qat"]
    assert saved["meta"]["distill_alpha"] == 0
