# ChemicalMotifIdentifier
![PyPI Version](https://img.shields.io/pypi/v/chemicalmotifidentifier.svg) ![PyPI Downloads](https://static.pepy.tech/badge/chemicalmotifidentifier)

This repository contains the codes necessary to perform a chemical-motif characterization of short-range order, as described in our [Quantifying chemical short-range order in metallic alloys](https://arxiv.org/abs/2311.01545) paper and our [Chemical-motif characterization of short-range order using E(3)-equivariant graph neural networks](https://google.com) paper. 

This framework allows for correlating any per-atom property to their local chemical motif. It also allows for the determination of predictive short-range chemical fluctuations length scale. It is based on E(3)-equivariant graph neural networks. Our framework has 100% accuracy in the identification of *any* motif that could ever be found in an fcc, bcc, or hcp solid solution with up to 5 chemical elements.  

![](assets/figure_2.png)

## Instalation 

```bash
# To install the latest PyPi release
pip install chemicalmotifidentifier

# To install the latest git commit 
pip install git+https://github.com/killiansheriff/ChemicalMotifIdentifier.git
```

You will also need to install ``torch``, ``torch_scatter`` and ``torch_geometric``.

## Example of usage

A jupyter notebook presenting a few test cases can be found in the [examples/](examples/) folder.

## References & Citing
If you use this repository in your work, please cite:

```
@article{sheriff2023quantifying,
  title={Quantifying chemical short-range order in metallic alloys},
  author={Sheriff, Killian and Cao, Yifan and Smidt, Tess and Freitas, Rodrigo},
  journal={arXiv},
  year={2023},
  doi={10.48550/arXiv.2311.01545}
}
```

and 

```
@article{TBD
}
```
