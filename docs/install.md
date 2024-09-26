# Installation Guide

GNATSS requires Python v3.10 or v3.11 to run. It is not currently compatible
with Python 3.12. If you have a conda installation, we recommend creating a
fresh environment to install GNATSS in.

```
conda create -n gnatss -c conda-forge --yes python=3.10 ipykernel
conda activate gnatss
```

Once you have the appropriate python environment enabled, GNATSS may be
installed via pip:

```
pip install gnatss
```

In order to update GNATSS to the most current version, use the command:

```
pip install --upgrade gnatss
```
