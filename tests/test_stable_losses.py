"""Numerical regressions at saturation and near teacher/student equality."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stable_losses import hard_target_cross_entropy, soft_target_cross_entropy


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_hard_labels_match_fp64_values_and_gradients(reduction):
    torch.manual_seed(271)
    z = torch.randn(3, 7, 14, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(14, (3, 7))
    targets[1, :3] = -100
    actual = hard_target_cross_entropy(z, targets, reduction=reduction)
    expected = F.cross_entropy(z.flatten(0, 1), targets.flatten(), reduction=reduction)
    if reduction == "none":
        expected = expected.view_as(targets)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, z, weight, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, z, weight)[0]
    torch.testing.assert_close(ga, ge)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_saturated_hard_labels_keep_loss_target_gradient_and_zero_sum(dtype):
    gap = 12. if dtype == torch.float16 else 20.
    z = torch.tensor([[gap] + [0.] * 13], dtype=dtype, requires_grad=True)
    loss = hard_target_cross_entropy(z, torch.tensor([0]), reduction="sum")
    loss.backward()
    assert loss.dtype == torch.float32
    assert loss.item() > 0
    assert z.grad[0, 0] < 0
    assert (z.grad[0, 1:] >= 0).all()
    reference = torch.tensor(13., dtype=torch.float64).log().sub(gap).exp()
    torch.testing.assert_close(loss.double(), reference.log1p(), rtol=2e-6, atol=0)
    # FP16 cannot represent these per-class gradients; BF16 and FP32 can.
    if dtype != torch.float16:
        torch.testing.assert_close(z.grad.float().sum(), torch.tensor(0.), atol=2e-10, rtol=0)


def test_high_confidence_incorrect_target_ties_and_custom_ignore():
    z = torch.tensor([[1000., -1000., 0.], [0., 0., 0.], [20., 0., 0.]], requires_grad=True)
    targets = torch.tensor([1, 0, -7])
    result = hard_target_cross_entropy(z, targets, ignore_index=-7)
    expected = F.cross_entropy(z, targets, ignore_index=-7, reduction="none")
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert torch.count_nonzero(z.grad[2]) == 0
    torch.testing.assert_close(z.grad.sum(-1), torch.zeros(3))


@pytest.mark.parametrize("rows", [0, 5])
def test_empty_or_ignored_sum_is_zero_with_zero_gradients(rows):
    z = torch.randn(rows, 14, requires_grad=True)
    loss = hard_target_cross_entropy(z, torch.full((rows,), -100), reduction="sum")
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(z.grad) == 0


def test_hard_labels_first_order_gradcheck():
    z = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: hard_target_cross_entropy(x, torch.tensor([0, 1, 2, 3, -100])), (z,))


def test_all_ignored_mean_matches_pytorch_nan_value_and_zero_gradient():
    z = torch.randn(3, 14, requires_grad=True)
    result = hard_target_cross_entropy(z, torch.full((3,), -100), reduction="mean")
    assert torch.isnan(result)
    result.backward()
    assert torch.count_nonzero(z.grad) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_equal_teacher_student_has_exact_zero_student_gradient(dtype):
    teacher = torch.tensor([[2., 1., 0.], [20., 0., -100.]], dtype=dtype, requires_grad=True)
    student = teacher.detach().clone().requires_grad_()
    loss = soft_target_cross_entropy(student, teacher, reduction="sum")
    loss.backward()
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None
    assert torch.isfinite(loss)


def test_near_teacher_gradient_matches_fp64_instead_of_probability_cancellation():
    teacher = torch.tensor([[2., 1., 0.]])
    student = torch.nextafter(teacher, torch.tensor([[-torch.inf, torch.inf, torch.inf]])).requires_grad_()
    actual = soft_target_cross_entropy(student, teacher, reduction="sum")
    actual.backward()
    reference = student.detach().double().requires_grad_()
    expected = -(teacher.double().softmax(-1) * reference.log_softmax(-1)).sum()
    expected.backward()
    torch.testing.assert_close(student.grad.double(), reference.grad, rtol=2e-5, atol=1e-15)
    assert student.grad.abs().sum() > 0
    assert student.grad.sum().abs() < 1e-14


def test_near_confident_teacher_keeps_tiny_balanced_student_gradient():
    teacher = torch.tensor([[20., 0.]])
    student = torch.tensor([[20., .001]], requires_grad=True)
    soft_target_cross_entropy(student, teacher, reduction="sum").backward()
    other = (student.detach().double().softmax(-1) - teacher.double().softmax(-1))[0, 1]
    expected = torch.stack((-other, other)).unsqueeze(0)
    torch.testing.assert_close(student.grad.double(), expected, rtol=3e-6, atol=0)
    assert student.grad[0, 0] < 0 and student.grad.sum() == 0


def test_constant_logit_offset_preserves_zero_student_gradient():
    teacher = torch.tensor([[2., 1., 0.], [20., 0., -100.]])
    student = (teacher + 128.).requires_grad_()
    soft_target_cross_entropy(student, teacher, reduction="sum").backward()
    assert torch.count_nonzero(student.grad) == 0


def test_bfloat16_student_accepts_float32_teacher_without_training_the_teacher():
    teacher = torch.tensor([[2., 1., 0.]], requires_grad=True)
    student = teacher.detach().bfloat16().requires_grad_()
    loss = soft_target_cross_entropy(student, teacher, reduction="sum")
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None


@pytest.mark.parametrize("gap", [20., 40., 80.])
def test_saturated_hard_label_gradient_survives_until_exponent_underflow(gap):
    z = torch.tensor([[0., gap, 0.]], requires_grad=True)
    loss = hard_target_cross_entropy(z, torch.tensor([1]), reduction="sum")
    loss.backward()
    expected = (torch.tensor(-gap, dtype=torch.float64).exp() * 2).log1p()
    torch.testing.assert_close(loss.double(), expected, rtol=2e-6, atol=0)
    assert z.grad[0, 1] < 0
    assert z.grad.sum() == 0


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_soft_labels_values_and_gradients_match_fp64(reduction):
    torch.manual_seed(199)
    z = torch.randn(2, 8, 14, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(z)
    actual = soft_target_cross_entropy(z, teacher, reduction=reduction)
    expected = -(teacher.softmax(-1) * z.log_softmax(-1)).sum(-1)
    if reduction != "none":
        expected = expected.sum() if reduction == "sum" else expected.mean()
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), z, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), z)[0]
    torch.testing.assert_close(ga, ge)


def test_soft_labels_first_order_gradcheck():
    z = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(z)
    assert torch.autograd.gradcheck(lambda x: soft_target_cross_entropy(x, teacher), (z,))


def test_distillation_extreme_logits_and_offsets_are_finite():
    teacher = torch.tensor([[1000., -1000., 0.], [0., -1000., 0.]])
    z = torch.tensor([[-1000., 1000., 0.], [0., 1000., 0.]], requires_grad=True)
    result = soft_target_cross_entropy(z, teacher, reduction="sum")
    result.backward()
    assert torch.isfinite(result)
    assert torch.isfinite(z.grad).all()
    expected = z.detach().double().requires_grad_()
    (-(teacher.double().softmax(-1) * expected.log_softmax(-1)).sum()).backward()
    torch.testing.assert_close(z.grad.double(), expected.grad)


@pytest.mark.parametrize("loss", [hard_target_cross_entropy, soft_target_cross_entropy])
def test_loss_and_backward_compile_without_graph_breaks(loss):
    z = torch.tensor([[20.] + [0.] * 13, [2.] + [1.] * 13], requires_grad=True)
    target = torch.tensor([0, -100]) if loss is hard_target_cross_entropy else z.detach().clone()
    eager = loss(z, target, reduction="sum")
    expected = torch.autograd.grad(eager, z)[0]
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True, dynamic=True)
    result = compiled(z, target, reduction="sum")
    actual = torch.autograd.grad(result, z)[0]
    torch.testing.assert_close(result, eager)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_cuda_bfloat16_compiled_saturation_and_equal_teacher():
    z = torch.tensor([[20.] + [0.] * 13], dtype=torch.bfloat16, device="cuda", requires_grad=True)
    hard = torch.compile(hard_target_cross_entropy, fullgraph=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = hard(z, torch.tensor([0], device="cuda"), reduction="sum")
    loss.backward()
    assert loss.dtype == torch.float32 and loss > 0 and z.grad[0, 0] < 0
    z.grad = None
    soft = torch.compile(soft_target_cross_entropy, fullgraph=True)
    soft(z, z.detach(), reduction="sum").backward()
    assert torch.count_nonzero(z.grad) == 0
    teacher = torch.tensor([[2., 1., 0.]], device="cuda")
    student = torch.nextafter(teacher, torch.tensor([[-torch.inf, torch.inf, torch.inf]], device="cuda")).requires_grad_()
    soft(student, teacher, reduction="sum").backward()
    reference = student.detach().double().requires_grad_()
    (-(teacher.double().softmax(-1) * reference.log_softmax(-1)).sum()).backward()
    torch.testing.assert_close(student.grad.double(), reference.grad, rtol=2e-5, atol=1e-15)
