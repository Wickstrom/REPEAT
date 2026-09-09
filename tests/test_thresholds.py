import pytest
import torch

pytest.importorskip("skimage")

from skimage import data  # noqa: E402
from skimage.filters import (  # noqa: E402
    threshold_li as skimage_threshold_li,
)
from skimage.filters import (  # noqa: E402
    threshold_mean as skimage_threshold_mean,
)
from skimage.filters import (  # noqa: E402
    threshold_otsu as skimage_threshold_otsu,
)
from skimage.filters import (  # noqa: E402
    threshold_triangle as skimage_threshold_triangle,
)

from repeat_xai.thresholds import (  # noqa: E402
    THRESHOLD_METHODS,
    get_threshold_method,
    threshold_li,
    threshold_mean,
    threshold_otsu,
    threshold_triangle,
)


def make_importance_maps():
    torch.manual_seed(0)

    uniform = torch.rand(64, 64)

    low_mode = 0.1 + 0.2 * torch.rand(64, 32)
    high_mode = 0.7 + 0.2 * torch.rand(64, 32)
    bimodal = torch.cat([low_mode, high_mode], dim=1)

    skewed = torch.rand(64, 64) ** 5

    camera = torch.from_numpy(data.camera()).float() / 255.0

    gradient_x = torch.linspace(0, 1, 64).repeat(64, 1)
    gradient = gradient_x * torch.linspace(0.2, 1, 64).reshape(64, 1)

    return {
        "uniform": uniform,
        "bimodal": bimodal,
        "skewed": skewed,
        "camera": camera,
        "gradient": gradient,
    }


IMPORTANCE_MAPS = make_importance_maps()

THRESHOLD_PAIRS = {
    "mean": (threshold_mean, skimage_threshold_mean),
    "otsu": (threshold_otsu, skimage_threshold_otsu),
    "triangle": (threshold_triangle, skimage_threshold_triangle),
    "li": (threshold_li, skimage_threshold_li),
}


def test_importance_maps_have_expected_properties():
    assert IMPORTANCE_MAPS["bimodal"].min() < 0.3 < IMPORTANCE_MAPS["bimodal"].max()
    assert (IMPORTANCE_MAPS["skewed"] < 0.1).float().mean() > 0.5


@pytest.mark.parametrize("map_name", IMPORTANCE_MAPS.keys())
@pytest.mark.parametrize("method_name", THRESHOLD_PAIRS.keys())
def test_thresholds_match_scikit_image(map_name, method_name):
    importance_map = IMPORTANCE_MAPS[map_name]
    our_threshold, skimage_threshold = THRESHOLD_PAIRS[method_name]

    ours = our_threshold(importance_map).item()
    reference = float(skimage_threshold(importance_map.numpy()))

    # Histogram-based methods return bin centers, so allow a difference of at
    # most one bin width.
    bin_width = (importance_map.max() - importance_map.min()).item() / 256
    tolerance = max(bin_width, 1e-6)

    assert ours == pytest.approx(reference, abs=tolerance)


@pytest.mark.parametrize("nbins", [64, 128, 256])
def test_otsu_and_triangle_nbins_match_scikit_image(nbins):
    importance_map = IMPORTANCE_MAPS["camera"]

    ours_otsu = threshold_otsu(importance_map, nbins=nbins).item()
    reference_otsu = float(skimage_threshold_otsu(importance_map.numpy(), nbins=nbins))

    ours_triangle = threshold_triangle(importance_map, nbins=nbins).item()
    reference_triangle = float(skimage_threshold_triangle(importance_map.numpy(), nbins=nbins))

    bin_width = (importance_map.max() - importance_map.min()).item() / nbins

    assert ours_otsu == pytest.approx(reference_otsu, abs=bin_width)
    assert ours_triangle == pytest.approx(reference_triangle, abs=bin_width)


@pytest.mark.parametrize(
    ("method_name", "our_threshold", "skimage_threshold"),
    [
        ("mean", threshold_mean, skimage_threshold_mean),
        ("otsu", threshold_otsu, skimage_threshold_otsu),
        ("triangle", threshold_triangle, skimage_threshold_triangle),
        ("li", threshold_li, skimage_threshold_li),
    ],
)
def test_constant_input_returns_the_constant(method_name, our_threshold, skimage_threshold):
    constant_map = torch.full((8, 8), 0.42)

    assert our_threshold(constant_map).item() == pytest.approx(0.42)
    assert float(skimage_threshold(constant_map.numpy())) == pytest.approx(0.42)


def test_li_tolerance_and_initial_guess_match_scikit_image():
    importance_map = IMPORTANCE_MAPS["camera"]

    ours = threshold_li(importance_map, tolerance=1e-4, initial_guess=0.5).item()
    reference = float(
        skimage_threshold_li(importance_map.numpy(), tolerance=1e-4, initial_guess=0.5)
    )

    assert ours == pytest.approx(reference, abs=1e-3)


def test_li_initial_guess_out_of_range_raises():
    importance_map = IMPORTANCE_MAPS["camera"]

    with pytest.raises(ValueError, match="within the range"):
        threshold_li(importance_map, initial_guess=42.0)


def test_thresholds_work_on_positive_and_negative_values():
    importance_map = IMPORTANCE_MAPS["uniform"] * 2 - 1

    for threshold_method in THRESHOLD_METHODS.values():
        assert torch.isfinite(threshold_method(importance_map))


def test_get_threshold_method_resolves_names():
    assert get_threshold_method("mean") is threshold_mean
    assert get_threshold_method("otsu") is threshold_otsu
    assert get_threshold_method("triangle") is threshold_triangle
    assert get_threshold_method("li") is threshold_li


def test_get_threshold_method_resolves_callables():
    def custom_threshold(importance):
        return importance.median()

    assert get_threshold_method(custom_threshold) is custom_threshold


def test_get_threshold_method_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown explanation_threshold"):
        get_threshold_method("median")
