# REPEAT: Improving Uncertainty Estimation in Representation Learning Explainability

[![PyPI version](https://img.shields.io/pypi/v/repeat-xai.svg)](https://pypi.org/project/repeat-xai/)
[![Python versions](https://img.shields.io/pypi/pyversions/repeat-xai.svg)](https://pypi.org/project/repeat-xai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/Wickstrom/REPEAT/actions/workflows/ci.yml/badge.svg)](https://github.com/Wickstrom/REPEAT/actions/workflows/ci.yml)

[![arXiv](https://img.shields.io/badge/arXiv-2412.08513-b31b1b.svg)](https://arxiv.org/abs/2412.08513)
[![AAAI](https://img.shields.io/badge/AAAI-2025-007ec7.svg)](https://doi.org/10.1609/aaai.v39i8.32900)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wickstrom/REPEAT/blob/main/notebooks/getting_started_with_repeat.ipynb)

<p align="center">
  <img width="600" src="https://raw.githubusercontent.com/Wickstrom/REPEAT/main/overview-figure.png">
</p>

This repository contains code for REPEAT, a framework for uncertainty estimation in representation learning explainability. REPEAT builds on [RELAX](https://github.com/Wickstrom/RELAX), and treats each pixel in an image as a Bernoulli random variable that is either important or unimportant to the representation of the image. From these Bernoulli random variables we can directly estimate the probability of a pixel being important, and the associated uncertainty, thus enabling users to ascertain certainty in pixel importance.

When should you use REPEAT? If your output is a vector representation, you have no label information, and you need to know how certain it is that a pixel is important.

More information can be found in the paper: **REPEAT: Improving Uncertainty Estimation in Representation Learning Explainability** [Wickstrøm et al., 2025, AAAI](https://doi.org/10.1609/aaai.v39i8.32900), [arXiv](https://arxiv.org/abs/2412.08513).

## Installation

REPEAT can be installed using pip:

```setup
pip install repeat-xai
```

This also installs the required dependencies (`torch` and `relax-xai`). To install the latest version from source:

```setup
pip install git+https://github.com/Wickstrom/REPEAT.git
```

## Toy example

Here is a very simple example showing the basic structure for how to use REPEAT. The input is assumed to be organized in (channel, height, width)-format, and preprocessed using the Imagenet normalization (for encoders pretrained on Imagenet). The `imagenet_image_transforms` function also reshapes the image into a square image ((224 x 224) by default) and places the image on the desired device.

```python
import torch
import torchvision
import torch.nn as nn
from repeat_xai import REPEAT
from relax_xai.utils import imagenet_image_transforms

x = torch.rand(3, 313, 210)  # Generate some random data.
x = imagenet_image_transforms(device="cpu", new_shape_of_image=224)(
    x
)  # Resize image and apply Imagenet normalization.

resnet18 = torchvision.models.resnet18(weights="DEFAULT")
encoder = nn.Sequential(
    *list(resnet18.children())[:-1], nn.Flatten()
)  # Remove classification head and only keep encoder part.

repeat = REPEAT(x, encoder)  # Initialize REPEAT (encoder is moved to the
# image device and set to evaluation mode).
repeat.forward()  # Run REPEAT (gradient tracking is disabled internally).

# The probability of each pixel being important for the representation, and
# the associated uncertainty, can be accessed in
# repeat.probability_of_importance and repeat.uncertainty. The uncertainty is
# the Bernoulli variance p(1-p): low for pixels that are consistently
# important or unimportant, and high for pixels that fluctuate.
```

## Getting started with REPEAT

The `notebooks`-folder contains a notebook called `getting_started_with_repeat`, where you can test out REPEAT. We recommend to use Google Colab with GPU-support enabled to speed up computation: [open the notebook in Colab](https://colab.research.google.com/github/Wickstrom/REPEAT/blob/main/notebooks/getting_started_with_repeat.ipynb).

## Important hyperparameters

There are several hyperparameters that can affect the performance of REPEAT. Masking-related hyperparameters are passed as keyword arguments to `repeat.forward(...)`.

- **`num_repeats`:** The number of repeated base R-XAI computations (K in the paper). Each repeat generates one Bernoulli sample per pixel, so more repeats give better estimates of the probability of importance, at a linear computational cost. The default is 10, following the paper.
- **`explanation_threshold`:** The method used to threshold each importance map into important and unimportant pixels. Built-in options are `'mean'` (the default, which was found to work best in the paper), `'otsu'`, `'triangle'` and `'li'`, all implemented in pure torch. Any callable mapping a tensor of importance scores to a threshold value can also be used.
- **`batch_size` and `num_batches`:** The number of masks used in each of the repeated RELAX runs is governed by these two parameters. The total number of encoder evaluations is `num_repeats * batch_size * num_batches`, so reducing these parameters will make REPEAT faster, but could decrease the quality of the explanations.
- **`num_cells` and `probability_of_drop`:** A mask in the underlying RELAX runs is generated following the same procedure as in [RISE](https://arxiv.org/abs/1806.07421). In this procedure, an image (`num_cells` x `num_cells`) smaller than the original image, with each pixel following a Bernoulli distribution with `probability_of_drop`, is randomly sampled. The default value for `num_cells` is 7 and `probability_of_drop` is 0.5. See the [RELAX repository](https://github.com/Wickstrom/RELAX) for more details.

## Citation

If you find REPEAT interesting and use it in your research, cite it using the BibTeX annotation below (also available as [CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{wickstrom2025repeat,
  author    = {Wickstr\o{}m, Kristoffer K. and Br{\"u}sch, Thea and Kampffmeyer, Michael C. and Jenssen, Robert},
  title     = {{REPEAT: Improving Uncertainty Estimation in Representation Learning Explainability}},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  volume    = {39},
  number    = {8},
  pages     = {8341--8350},
  doi       = {10.1609/aaai.v39i8.32900}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and the release process.

## License

REPEAT is released under the [MIT License](LICENSE).
