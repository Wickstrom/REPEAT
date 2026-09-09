import pytest
import torch
import torch.nn as nn

from repeat_xai.repeat import REPEAT
from repeat_xai.thresholds import (
    THRESHOLD_METHODS,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
)

SHAPE = (32, 32)


class ConstantEncoder(nn.Module):
    """Returns the same one-hot representation for every input, making all
    similarity scores exactly 1 and the importance maps exactly constant."""

    def forward(self, x):
        representation = torch.zeros(x.shape[0], 8, device=x.device)
        representation[:, 0] = 1.0
        return representation


class GaussianKernelSimilarity(nn.Module):
    """Similarity scores in (0, 1], giving non-negative importance maps."""

    def forward(self, a, b):
        return torch.exp(-((a - b) ** 2).sum(dim=1))


def make_encoder(seed=0, out_dim=16):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * SHAPE[0] * SHAPE[1], out_dim))


def make_image(seed=1):
    torch.manual_seed(seed)
    return torch.rand(1, 3, *SHAPE)


def make_repeat(**kwargs):
    return REPEAT(
        make_image(),
        make_encoder(),
        num_repeats=kwargs.pop("num_repeats", 3),
        batch_size=kwargs.pop("batch_size", 8),
        num_batches=kwargs.pop("num_batches", 2),
        **kwargs,
    )


def test_output_shapes():
    repeat = make_repeat()
    repeat.forward()
    assert repeat.probability_of_importance.shape == SHAPE
    assert repeat.uncertainty.shape == SHAPE


def test_probability_and_uncertainty_are_finite():
    repeat = make_repeat(similarity_measure=GaussianKernelSimilarity())
    repeat.forward()
    assert torch.isfinite(repeat.probability_of_importance).all()
    assert torch.isfinite(repeat.uncertainty).all()


def test_probability_is_in_unit_interval():
    repeat = make_repeat(similarity_measure=GaussianKernelSimilarity())
    repeat.forward()
    assert (repeat.probability_of_importance >= 0).all()
    assert (repeat.probability_of_importance <= 1).all()


def test_uncertainty_is_bernoulli_variance():
    repeat = make_repeat(similarity_measure=GaussianKernelSimilarity())
    repeat.forward()
    assert torch.allclose(
        repeat.uncertainty,
        repeat.probability_of_importance * (1 - repeat.probability_of_importance),
    )
    assert repeat.uncertainty.max() <= 0.25


def test_deterministic_given_seed():
    torch.manual_seed(42)
    repeat_a = make_repeat()
    repeat_a.forward()

    torch.manual_seed(42)
    repeat_b = make_repeat()
    repeat_b.forward()

    assert torch.equal(repeat_a.probability_of_importance, repeat_b.probability_of_importance)
    assert torch.equal(repeat_a.uncertainty, repeat_b.uncertainty)


def test_forward_resets_accumulators():
    torch.manual_seed(42)
    repeat = make_repeat()
    rng_state = torch.get_rng_state()

    repeat.forward()
    first = repeat.probability_of_importance.clone()

    # Corrupt the accumulators, then rerun from the same RNG state. If
    # forward() did not reset its accumulators, the corruption would leak
    # into the result.
    repeat.probability_of_importance.fill_(7.0)
    torch.set_rng_state(rng_state)
    repeat.forward()

    assert torch.equal(repeat.probability_of_importance, first)
    assert torch.equal(
        repeat.uncertainty,
        first * (1 - first),
    )


def test_no_autograd_graph_is_tracked():
    image = make_image()
    image.requires_grad_(True)
    repeat = REPEAT(image, make_encoder(), num_repeats=2, batch_size=4, num_batches=2)
    repeat.forward()
    assert not repeat.probability_of_importance.requires_grad
    assert not repeat.uncertainty.requires_grad


def test_input_image_must_be_batched():
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        REPEAT(torch.rand(3, *SHAPE), make_encoder())
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        REPEAT(torch.rand(2, 3, *SHAPE), make_encoder())


def test_constant_encoder_gives_certain_importance():
    repeat = REPEAT(
        make_image(),
        ConstantEncoder(),
        num_repeats=3,
        batch_size=16,
        num_batches=2,
    )
    repeat.forward()
    assert torch.allclose(repeat.probability_of_importance, torch.ones(SHAPE))
    assert torch.allclose(repeat.uncertainty, torch.zeros(SHAPE))


@pytest.mark.parametrize("explanation_threshold", ["mean", "otsu", "triangle", "li"])
def test_all_threshold_methods_run(explanation_threshold):
    repeat = make_repeat(explanation_threshold=explanation_threshold)
    repeat.forward()
    assert repeat.probability_of_importance.shape == SHAPE
    assert torch.isfinite(repeat.probability_of_importance).all()
    assert torch.isfinite(repeat.uncertainty).all()


def test_masking_kwargs_are_passed_to_relax():
    torch.manual_seed(0)
    repeat = make_repeat()
    repeat.forward(num_cells=3, probability_of_drop=0.3)
    assert torch.isfinite(repeat.probability_of_importance).all()


def test_custom_similarity_measure():
    repeat = make_repeat(similarity_measure=GaussianKernelSimilarity())
    repeat.forward()
    assert torch.isfinite(repeat.probability_of_importance).all()


def test_callable_threshold():
    repeat = make_repeat(explanation_threshold=lambda importance: importance.median())
    repeat.forward()
    assert torch.isfinite(repeat.probability_of_importance).all()


def test_unknown_threshold_name_raises():
    with pytest.raises(ValueError, match="otsu"):
        make_repeat(explanation_threshold="median")


def test_threshold_methods_are_mapped_correctly():
    assert THRESHOLD_METHODS["mean"] is threshold_mean
    assert THRESHOLD_METHODS["otsu"] is threshold_otsu
    assert THRESHOLD_METHODS["triangle"] is threshold_triangle
    assert THRESHOLD_METHODS["li"] is threshold_li


def test_package_exposes_repeat_class():
    from repeat_xai import REPEAT as TopLevelREPEAT

    assert TopLevelREPEAT is REPEAT
