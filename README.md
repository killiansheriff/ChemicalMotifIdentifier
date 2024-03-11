# ChemicalMotifIdentifier

This repository contains the codes necessary to perform a chemical-motif characterization of short-range order, as described in our [Quantifying chemical short-range order in metallic alloys](https://arxiv.org/abs/2311.01545) paper and our [Chemical-motif characterization of short-range order using E(3)-equivariant graph neural networks](https://google.com) paper. 

This framework allows for correlating any per-atom property to their local chemical motif. It also allows for the determination of predictive short-range chemical fluctuations length scale. It is based on E(3)-equivariant graph neural networks. Our framework has 100% accuracy in the identification of *any* motif that could ever be found in an fcc, bcc, or hcp solid solution with up to 5 chemical elements.  

![](assets/figure_2.png)


## Instalation 

```bash
# To install the latest PyPi release
pip install chemicalmotifidentifier

# To install the latest git commit 
pip install htpps://github.com/killiansheriff/ChemicalMotifIdentifier.git
```

## Example of usage

```python 
from eca import ECA_MD

structure='fcc'
dump_files = glob.glob('*.dump')

eca = ECA_MD(crystal_structure=structure)
for i, dump_file in enumerate(dump_files):
    root = f'data/eca_id/dump_{i}/'
    df = eca.predict(root=root, dump_file=dump_file)
    kl = eca.get_kl(df)
    df.to_pickle(root+'df_microstates.pkl')
```

A jupyter notebook presenting a few test cases can be found in the ``examples/`` folder.

