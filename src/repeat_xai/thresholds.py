"""Histogram-based thresholding methods used to separate important from
unimportant pixels in REPEAT.

The methods are pure-torch ports of the corresponding scikit-image
implementations (`skimage.filters.threshold_mean`, `threshold_otsu`,
`threshold_triangle` and `threshold_li`), so they work on tensors on any
device without numpy round-trips, and without requiring scikit-image as a
runtime dependency.
"""

from collections.abc import Callable

import torch

__all__ = [
    "THRESHOLD_METHODS",
    "get_threshold_method",
    "threshold_li",
    "threshold_mean",
    "threshold_otsu",
    "threshold_triangle",
]


def _image_histogram(explanation: torch.Tensor, nbins: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the histogram of the input values over the range
    [explanation.min(), explanation.max()], mimicking
    ``skimage.exposure.histogram(..., source_range="image")``. The rightmost
    bin is closed, so values equal to the maximum are included in the last
    bin.
    """

    values = explanation.reshape(-1)
    min_value = values.min().item()
    max_value = values.max().item()

    counts = torch.histc(values.float(), bins=nbins, min=min_value, max=max_value).double()

    bin_edges = torch.linspace(min_value, max_value, nbins + 1, dtype=torch.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return counts, bin_centers


def threshold_mean(explanation: torch.Tensor) -> torch.Tensor:
    """
    Threshold value based on the mean of the input values.

    Parameters
    ----------
    explanation
        Tensor of importance scores.

    Returns
    -------
    torch.Tensor
        Threshold value. All pixels with an importance score higher than or
        equal to this value are considered important.

    References
    ----------
    .. [1] Glasbey, C. "An analysis of histogram-based thresholding
           algorithms," CVGIP: Graphical Models and Image Processing,
           vol. 55, pp. 532-537, 1993.
           :DOI:`10.1006/cgip.1993.1040`
    """

    return explanation.mean()


def threshold_otsu(explanation: torch.Tensor, nbins: int = 256) -> torch.Tensor:
    """
    Threshold value based on Otsu's method. The threshold is chosen to
    maximize the between-class variance of important and unimportant pixels.

    Port of ``skimage.filters.threshold_otsu``.

    Parameters
    ----------
    explanation
        Tensor of importance scores.
    nbins
        Number of bins used to calculate the histogram.

    Returns
    -------
    torch.Tensor
        Threshold value. All pixels with an importance score higher than or
        equal to this value are considered important.

    References
    ----------
    .. [1] Otsu, N., "A Threshold Selection Method from Gray-Level
           Histograms," IEEE Transactions on Systems, Man, and Cybernetics,
           vol. 9, no. 1, pp. 62-66, 1979.
           :DOI:`10.1109/TSMC.1979.4310076`
    """

    values = explanation.reshape(-1)

    if torch.all(values == values[0]):
        return values[0]

    counts, bin_centers = _image_histogram(explanation, nbins)

    # Class weights for all possible thresholds.
    weight1 = counts.cumsum(0)
    weight2 = counts.flip(0).cumsum(0).flip(0)

    # Class means for all possible thresholds.
    mean1 = (counts * bin_centers).cumsum(0) / weight1
    mean2 = ((counts * bin_centers).flip(0).cumsum(0) / weight2.flip(0)).flip(0)

    # Between-class variance. The last bin is excluded so that class 1 and
    # class 2 variables align.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]).square()

    idx = variance12.argmax()

    return bin_centers[idx]


def threshold_triangle(explanation: torch.Tensor, nbins: int = 256) -> torch.Tensor:
    """
    Threshold value based on the triangle algorithm. A line is constructed
    between the histogram peak and the far end of the histogram tail, and the
    threshold is the bin furthest from this line.

    Port of ``skimage.filters.threshold_triangle``.

    Parameters
    ----------
    explanation
        Tensor of importance scores.
    nbins
        Number of bins used to calculate the histogram.

    Returns
    -------
    torch.Tensor
        Threshold value. All pixels with an importance score higher than or
        equal to this value are considered important.

    References
    ----------
    .. [1] Zack, G. W., Rogers, W. E. and Latt, S. A., 1977, "Automatic
           Measurement of Sister Chromatid Exchange Frequency," Journal of
           Histochemistry and Cytochemistry 25 (7), pp. 741-753.
           :DOI:`10.1177/25.7.70454`
    .. [2] ImageJ AutoThresholder code,
           http://fiji.sc/wiki/index.php/Auto_Threshold
    """

    hist, bin_centers = _image_histogram(explanation, nbins)

    arg_peak_height = hist.argmax().item()
    peak_height = hist[arg_peak_height]

    nonzero_bins = hist.nonzero().flatten()
    arg_low_level = nonzero_bins[0].item()
    arg_high_level = nonzero_bins[-1].item()

    if arg_low_level == arg_high_level:
        # The image has constant intensity.
        return explanation.reshape(-1)[0]

    # Flip if the left tail is shorter than the right tail.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        hist = hist.flip(0)
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # Set up the coordinate system of the tail.
    width = arg_peak_height - arg_low_level
    x1 = torch.arange(width, dtype=torch.float64, device=hist.device)
    y1 = hist[arg_low_level : arg_low_level + width]

    # Normalize so that the length maximization below becomes a dot product.
    norm = (peak_height.square() + width**2).sqrt()
    peak_height = peak_height / norm
    width_normalized = width / norm

    # Maximize the length. The ImageJ implementation includes an additional
    # constant when calculating the length, but here we omit it as it does not
    # affect the location of the maximum.
    length = peak_height * x1 - width_normalized * y1
    arg_level = length.argmax().item() + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return bin_centers[arg_level]


def threshold_li(
    explanation: torch.Tensor,
    tolerance: float | None = None,
    initial_guess: float | Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Threshold value based on Li's iterative minimum cross entropy method.

    Port of ``skimage.filters.threshold_li``. Non-finite values are ignored,
    and the values are shifted to be positive (the iteration requires
    logarithms of class means).

    Parameters
    ----------
    explanation
        Tensor of importance scores.
    tolerance
        Finish the computation when the change in the threshold between
        iterations is smaller than this value. Defaults to half the smallest
        difference between unique values in the input.
    initial_guess
        Initial estimate for the iterative optimization. Defaults to the
        mean of the input values. Can also be a scalar, or a callable that
        maps the input values to a scalar.

    Returns
    -------
    torch.Tensor
        Threshold value. All pixels with an importance score higher than or
        equal to this value are considered important.

    References
    ----------
    .. [1] Li C.H. and Lee C.K. (1993) "Minimum Cross Entropy Thresholding"
           Pattern Recognition, 26(4): 617-625.
           :DOI:`10.1016/0031-3203(93)90115-D`
    .. [2] Li C.H. and Tam P.K.S. (1998) "An Iterative Algorithm for Minimum
           Cross Entropy Thresholding" Pattern Recognition Letters, 18(8):
           771-776. :DOI:`10.1016/S0167-8655(98)00057-9`
    .. [3] Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
           Techniques and Quantitative Performance Evaluation" Journal of
           Electronic Imaging, 13(1): 146-165.
           :DOI:`10.1117/1.1631315`
    """

    values = explanation.reshape(-1).double()

    # Remove NaN, and make sure the input has more than one value.
    values = values[~values.isnan()]
    if values.numel() == 0:
        return values.new_full((), float("nan"))
    if torch.all(values == values[0]):
        return values[0]

    values = values[values.isfinite()]
    if values.numel() == 0:
        return values.new_zeros(())

    # Li's algorithm requires a positive input (because of log(mean)).
    image_min = values.min()
    values = values - image_min

    if tolerance is None:
        tolerance = torch.diff(torch.unique(values)).min().item() / 2

    # Initial estimate for the iteration.
    if initial_guess is None:
        t_next = values.mean()
    elif callable(initial_guess):
        t_next = initial_guess(values)
    else:
        t_next = torch.as_tensor(initial_guess, dtype=values.dtype) - image_min
        if not 0 < t_next < values.max():
            raise ValueError(
                "The initial guess for threshold_li must be within the range of the"
                f" input. Got {initial_guess} for input min {image_min.item()} and max"
                f" {(values.max() + image_min).item()}."
            )

    # The initial value for t_curr must differ from t_next by at least the
    # tolerance. Since the input is positive, this is ensured by setting it
    # to a large enough negative number.
    t_curr = -2 * tolerance

    # Stop the iterations when the change in the threshold is below the
    # tolerance, or if the background mode has only one value left, since
    # log(0) is not defined.
    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next

        foreground = values > t_curr
        mean_foreground = values[foreground].mean()
        mean_background = values[~foreground].mean()

        if mean_background == 0:
            break

        t_next = (mean_background - mean_foreground) / (
            mean_background.log() - mean_foreground.log()
        )

    return t_next + image_min


THRESHOLD_METHODS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": threshold_mean,
    "otsu": threshold_otsu,
    "triangle": threshold_triangle,
    "li": threshold_li,
}
"""Available histogram-based thresholding methods, by name."""


def get_threshold_method(
    explanation_threshold: str | Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Resolves a thresholding method specification into a callable.

    Parameters
    ----------
    explanation_threshold
        Either the name of a built-in method ('mean', 'otsu', 'triangle' or
        'li'), or a callable that maps a tensor of importance scores to a
        threshold value.

    Returns
    -------
    Callable
        Function mapping a tensor of importance scores to a threshold value.

    Raises
    ------
    ValueError
        If a string is given and it does not match any built-in method.
    """

    if callable(explanation_threshold):
        return explanation_threshold

    if explanation_threshold not in THRESHOLD_METHODS:
        raise ValueError(
            f"Unknown explanation_threshold {explanation_threshold!r}. Expected one of"
            f" {sorted(THRESHOLD_METHODS)} or a callable."
        )

    return THRESHOLD_METHODS[explanation_threshold]
