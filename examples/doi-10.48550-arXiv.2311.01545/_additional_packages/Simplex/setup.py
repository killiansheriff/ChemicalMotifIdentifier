import os

from setuptools import find_packages, setup

# Get description from README
root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(root, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="simplex",
    version="0.0.1",
    packages=find_packages(exclude=["tests.*", "tests", "figs", "examples", "media"]),
    author="Killian Sheriff",
    author_email="ksheriff@mit.edu",
    description="A python package that implements the geometric object of a simplex.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    keywords=[],
    url="https://github.com/killiansheriff/simplex",
    install_requires=[
        "numpy",
    ],
    include_package_data=True,
)
