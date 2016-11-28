# FORTE: Fast Ordinal Triplets Embedding

## Installation
It is best to work in a `virtualenv`. It is assumed that you have the standard `scipy` stack, including `numpy` and `matplotlib` installed.

From the top level directory:
```python
python setup.py install
python setup.py build_ext --inplace
```

You will also need to install [`blackbox`](https://github.com/lalitkumarj/BlackBox) which serves as the logger.

## TODO: ##
- Think over updates
- Lalit: PGD 
- Lalit: PGD debiased
- Lalit: Fix naming conventions (too many gets?)
- Blake: document naming convention, code convention
- Ari: Hinge loss objectives file
- All: Decide on using utils score functions
- All of us - demo run file? - STARTED

- Blake: Factored Gradient for LogisticLoss using SGD and then Factored Gradient - DONE
- Blake + Lalit: CK (maybe wait on this) - DONE
- Lalit: setup blackbox in setup.py DONE
- Lalit: Nuclear Norm PGD - DONE
- Blake: procrustes - DONE
- Lalit - blackbox appropriate things DONE
- Blake: Fix utils - DONE
