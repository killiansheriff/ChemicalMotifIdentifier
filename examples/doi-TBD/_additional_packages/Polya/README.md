# Polya 
A python implementation of Polya's enumeration theory and pattern inventory formula.

# Usage 

Here is an example on how to extract the cycle index polynomial (p_g) and number of distinct first coordination polyhedron (nms) for the fcc crystal structure with a number of ntypes = 3 chemical elements. Additonal graph geometries can be defined in ``polya/_src/graphs.py``. This example can be found in the ``examples/`` folder. 

```python
from polya import Polya
pl = Polya(graph_name="fcc")
p_g, nms = pl.get_gt(ntypes=3)
print(p_g)
```

## Installation
For a standalone Python package or Conda environment, please use:
```bash
pip install --user polya
```

If you want to install the lastest git commit, please replace ``polya`` by ``git+https://github.com/killiansheriff/polya.git``.

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