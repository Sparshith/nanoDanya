import torch

from nanochat.gpt import GPT, GPTConfig
from training.student import StudentGPT, fake_quant_int4


def test_student_matches_teacher_math_when_weights_are_tied():
    config = GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=1, n_kv_head=1, n_embd=64)
    student = StudentGPT(config)
    teacher = GPT(config)
    state = student.state_dict()
    teacher.load_state_dict({**state, "lm_head.weight": state["transformer.wte.weight"]})
    tokens = torch.randint(config.vocab_size, (2, 9))
    with torch.no_grad():
        assert torch.allclose(student(tokens), teacher(tokens), atol=1e-6)


def test_int4_qat_quantizes_tied_weights_and_keeps_gradients():
    config = GPTConfig(sequence_len=8, vocab_size=128, n_layer=1, n_head=1, n_kv_head=1, n_embd=64)
    student = StudentGPT(config, qat=True)
    assert "lm_head.weight" not in student.state_dict()
    assert sum(p.numel() for p in StudentGPT(GPTConfig(
        sequence_len=8, vocab_size=14750, n_layer=6, n_head=6, n_kv_head=6, n_embd=384
    )).parameters()) == 16_280_832
    quantized = fake_quant_int4(student.transformer.wte.weight)
    scale = student.transformer.wte.weight.detach().abs().amax(dim=1, keepdim=True) / 7
    levels = quantized.detach() / scale
    assert torch.allclose(levels, levels.round(), atol=1e-5)
    assert levels.abs().max() <= 7 + 1e-5
    student(torch.randint(config.vocab_size, (2, 5))).square().mean().backward()
    assert student.transformer.wte.weight.grad.abs().sum() > 0
