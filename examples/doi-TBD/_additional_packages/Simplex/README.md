# Simplex 
A python package that implements the geometric object of a [simplex](https://en.wikipedia.org/wiki/Simplex).

## Usage 
Here is an example on how to get the mapping from barycentric coordinates (``concentrations``) to cartesian coordinates (``coords``) for a ``n_dim = 2`` for motifs with 12 atoms in their first coordination shell. ``generators`` return the unique set of concentration of the sorted possible concentrations, ``count`` is their multiplicities. This example can be found in the ``examples/`` folder. 

```python
import simplex as sp 

triangle = sp.Simplex(n_dim=2, edge_length=1, nneigh=12)
concentrations, coords = triangle.get_mapping()
generators, counts = triangle.get_generators(concentrations)
```

## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user simplex
```

If you want to install the lastest git commit, please replace ``simplex`` by ``git+https://github.com/killiansheriff/simplex.git``.

## Contact
If any questions, feel free to contact me (ksheriff at mit dot edu).

## References & Citing 
If you use this repository in your work, please cite:

```
@article{TOBEUPDATED,
  title={TOBEUPDATED},
  author={Sheriff, Killian and Cao, Yifan and Freitas, Rodrigo},
  journal={arXiv preprint TOBEUPDATED},
  year={2024}
}
```