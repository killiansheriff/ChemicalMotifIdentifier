import os

from setuptools import setup, find_packages

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="chemicalmotifidentifier",
    version="0.0.8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Killian Sheriff",
    author_email="ksheriff@mit.edu",
    description="Chemical Motif Identifier",
    url="https://github.com/killiansheriff/ChemicalMotifIdentifier",
    packages=find_packages(),
    install_requires=[
        "e3nn",
        "matplotlib",
        "networkx",
        "numpy",
        "ovito",
        "pandas",
        "scikit_learn",
        "scipy",
        # "torch_geometric",
        "tqdm",
        "NshellFinder",
        "polyaenum",
        'nsimplex',
        # "torch==2.0.1",
        # "torch_scatter==2.1.2",
    ],
    classifiers=[],
    include_package_data=True,
)
