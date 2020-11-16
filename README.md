This code implements probabilistic decoding of image-based spatial
transcriptomics via a novel re-parametrized Gaussian Mixture Model (GMM)
implemented using stochastic variational inference in [pyro](https://pyro.ai/).

To get started, please explore the Jupyter Notebook
[example_iss_mousebrain.ipynb](example_iss_mousebrain.ipynb), which provides an
example of how the code can be used to decode a
ISS mouse brain dataset.

The code has been tested with python 3.6.12 and its requirements can be
fulfilled by running `python3 -m pip install -r requirements.txt`
