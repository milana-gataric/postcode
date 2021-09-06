## PoSTcode

This is an implementation of the **P**r**o**babilistic Image-based **S**patial **T**ranscriptomics De**code**r (PoSTcode), which is based on a re-parametrised matrix-variate Gaussian mixture model,
implemented using stochastic variational inference in [pyro](https://pyro.ai/). 
<!The method implemented here is described in the paper ["PoSTcode: Probabilistic image-based spatial transcriptomics decoder"]().
>

To get started, please explore the Jupyter Notebook
[example_iss_mousebrain.ipynb](notebooks/example_iss_mousebrain.ipynb), which provides an
example of how the code can be used to decode a
ISS mouse brain dataset.

The code has been tested with python 3.6.12 and its requirements can be
fulfilled by running
```
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```
