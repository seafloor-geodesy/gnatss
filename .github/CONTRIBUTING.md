# Welcome to GNATSS docs contributing guide

Thank you for investing your time in contributing to our project!

Read our [Code of Conduct](../CODE_OF_CONDUCT.md) to keep our community
approachable and respectable.

## Quick development

The fastest way to start with development is to use nox. If you don't have nox,
you can use `pipx run nox` to run it without installing, or `pipx install nox`.
If you don't have pipx (pip for applications), then you can install with
`pip install pipx` (the only case were installing an application with regular
pip is reasonable). If you use macOS, then pipx and nox are both in brew, use
`brew install pipx nox`.

To use, run `nox`. This will lint and test using every installed version of
Python on your system, skipping ones that are not installed. You can also run
specific jobs:

```bash
nox -s lint  # Lint only
nox -s tests  # Python tests
nox -s build  # Make an SDist and wheel
```

Nox handles everything for you, including setting up an temporary virtual
environment for each run.

## Setting up a development environment manually

This way of developing assumes that you're using conda as your environment and
package management.

### Note

We recommend using the `libmamba` solver instead of the classic solver, since
the `conda create` and `conda install` step could take very long or fail. See
instructions
[here](https://conda.github.io/conda-libmamba-solver/getting-started/) for
installation and usage.

### Environment setup

1. Create conda environment

   ```bash
   conda create --yes -n gnatss python=3.10 # use python 3.10
   ```

2. Activate and install dependencies

   ```bash
   conda activate gnatss
   pip install -e ".[all]"
   ```

## Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks:

```bash
pip install pre-commit # or brew install pre-commit on macOS
pre-commit install # Will install a pre-commit hook into the git repo
```

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

## Testing

Use pytest to run the unit checks via `nox`. This will download the necessary
test data and run the tests:

```bash
nox -s tests
```

## Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=gnatss
```
