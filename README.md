# Napari Travali

<!--
[![PyPI](https://img.shields.io/pypi/v/napari-travali.svg)](https://pypi.org/project/napari-travali/)
[![Status](https://img.shields.io/pypi/status/napari-travali.svg)](https://pypi.org/project/napari-travali/)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-travali)](https://pypi.org/project/napari-travali)
-->

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

Currently at this early stage, you can install Napari Travali only from
GitHub

```console
$ pip install git+https://github.com/yfukai/napari-travali
```

## Usage
### How to run
```console
python -m napari_travali /path/to/zarr --presist # to for small-scaled dataset
```

### Input data specification

Currently we accept segmented / tracked data in the following
format.

- file format : `zarr` (hopefully `HDF5` in future)

- contents :
  - `Dataset` named `image` ... 5-dim dataset with dimensional order `TCZYX`
  - `Group` named `labels`
    - `Dataset` (default name `original` or `original.travali`, let name: `(dataset_name)`)
      ... 4-dim dataset with dimensional order `TZYX`
      - `attr` named `target_Ts`
  - `Group` named `df_segments`
    - `Dataset` name: `(dataset_name)`
      ... 2-dim dataset with columns (frame, label, segment_id, bbox_y0, bbox_y1, bbox_x0, bbox_x1)
    - `attr` named `finalized_segment_ids` ... (optional) list of finalized segment ids
    - `attr` named `candidate_segment_ids` ... (optional) list of candidate segment ids
  - `Group` named `df_divisions`
    - `Dataset` name: `(dataset_name)`
      ... 2-dim dataset with columns (parent_segment_id, frame_child1, label_child1, frame_child2, label_child2)

### Label loading logic

- When reading file, `original.travali` is read if exits. If not, `original` is read.
- When writing file,
  - If the read dataset is suffixed with `.travali`, the same name is used.
  - If the read dataset is not suffixed with `.travali`, `(original name).travali` is used.

Please see the [Command-line
Reference](https://napari-travali.readthedocs.io/en/latest/usage.html)
for details.

### Modes and editing commands

- `All label`
  - Clicking a label ... select the label and to `Label selected`
- `Label selected`
  - Typing `Escape` ... cancel editing and to `All label`
  - Typing `Enter` ... finalize editing and to `All label`
  - Typing `r` ... Redrawing the segmentation. Transition to `Label redraw`
  - Typing `s` ... Switching the segmented area. Transition to `Label switch`
  - Typing `d` ... Modifying the daughter areas. Transition to `Daughter switch` or `Daughter draw` (asks before transition)
  - Typing `t` ... Marking termination
- `Label redraw`
  - Typing `Escape` ... cancel redraw and make transition to `Label selected`
  - Typing `Enter` ... finalize redraw and make transition to `Label selected`
- `Label switch`
  - Typing `Escape` ... cancel switch and make transition to `Label selected`
  - Clicking a label ... select the label to switch and make transition to `Label selected`
- `Daughter switch`
  - Typing `Escape` ... cancel choosing daughters and make transition to `Label selected`
  - Clicking a label ... select the label as a daughter and make transition to either of `Daughter switch` or `Daughter draw` (asks before transition) or `Label selected` (if you already selected two daughters)
- `Daughter draw`
  - Typing `Escape` ... cancel choosing daughters and to `Label selected`
  - Typing `Enter` ... finalize drawing a daughter and make transition to either of `Daughter switch` or `Daughter draw` (asks before transition) or `Label selected` (if you already selected two daughters)

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
