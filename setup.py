import os

from setuptools import find_packages, setup

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chemicalmotifidentifier",
    version="0.0.5",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Killian Sheriff",
    author_email="ksheriff@mit.edu",
    description="Chemical Motif Identifier",
    url="https://github.com/killiansheriff/ChemicalMotifIdentifier",
    packages=["chemicalmotifidentifier"],
    install_requires=[
        "e3nn",
        "matplotlib",
        "networkx",
        "numpy",
        "ovito",
        "pandas",
        "scikit_learn",
        "scipy",
        "simplex",
        "torch==2.0.1",
        "torch_geometric",
        #"torch_scatter==2.1.2",
        "tqdm",
        "NshellFinder",
        "polyaenum",
        'nsimplex',
    ],
    classifiers=[],
)
