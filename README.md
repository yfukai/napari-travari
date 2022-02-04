# Napari Travali

```{=html}
<!--
[![PyPI](https://img.shields.io/pypi/v/napari-travali.svg)](https://pypi.org/project/napari-travali/)
[![Status](https://img.shields.io/pypi/status/napari-travali.svg)](https://pypi.org/project/napari-travali/)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-travali)](https://pypi.org/project/napari-travali)
-->
```

[![License](https://img.shields.io/pypi/l/napari-travali)](https://opensource.org/licenses/MIT)

[![Read the documentation at
<https://napari-travali.readthedocs.io/>](https://img.shields.io/readthedocs/napari-travali/latest.svg?label=Read%20the%20Docs)](https://napari-travali.readthedocs.io/)
[![Tests](https://github.com/yfukai/napari-travali/workflows/Tests/badge.svg)](https://github.com/yfukai/napari-travali/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/yfukai/napari-travali/branch/main/graph/badge.svg)](https://codecov.io/gh/yfukai/napari-travali)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Napari TRAck VALIdator

## Features

- Enables manual validation of cell tracking results.

## Installation

Currently at this early stage, you can install Napari Travali only fron
GitHub You can install _Napari Travali_ via G

```console
$ pip install git+https://github.com/yfukai/napari-travali
```

## Usage

### Input data specification

Currently we accept segmented / tracked data in the following
format.

- file format : `zarr`

- contents :
  - `Dataset` named `image` ... 5-dim dataset with dimensional order `TCZYX`
  - `Group` named `labels`
    - `Dataset` (default name `original` or `original.travali`, let name: `(dataset_name)`)
      ... 4-dim dataset with dimensional order `TZYX`
      - `attr` named `target_Ts`
  - `Group` named `df_segments`
    - `Dataset` name: `(dataset_name)`
      ... 2-dim dataset with columns
    - `attr` named `finalized_segment_ids` ... (optional) list of finalized segment ids
    - `attr` named `candidate_segment_ids` ... (optional) list of candidate segment ids
  - `Group` named `df_divisions`
    - `Dataset` name: `(dataset_name)`
      ... 2-dim dataset with columns

### Input data specification

- When reading file, `original.travali` is read if exits. If not, `original` is read.
- When writing file,
  - If the read dataset is suffixed with `.travali`, the same name is used.
  - If the read dataset is not suffixed with `.travali`, `(original name).travali` is used.

Please see the [Command-line
Reference](https://napari-travali.readthedocs.io/en/latest/usage.html)
for details.

## Contributing

Contributions are very welcome. To learn more, see the [Contributor
Guide](CONTRIBUTING.rst).

## License

Distributed under the terms of the [MIT
license](https://opensource.org/licenses/MIT), _Napari Travali_ is free
and open source software.

## Issues

If you encounter any problems, please [file an
issue](https://github.com/yfukai/napari-travali/issues) along with a
detailed description.

## Credits

This project was generated from
[\@cjolowicz](https://github.com/cjolowicz)\'s [Hypermodern Python
Cookiecutter](https://github.com/cjolowicz/cookiecutter-hypermodern-python)
template.
