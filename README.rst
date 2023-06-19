Napari Travali
==============

.. raw:: html

   <!--
   [![PyPI](https://img.shields.io/pypi/v/napari-travali.svg)](https://pypi.org/project/napari-travali/)
   [![Status](https://img.shields.io/pypi/status/napari-travali.svg)](https://pypi.org/project/napari-travali/)
   [![Python Version](https://img.shields.io/pypi/pyversions/napari-travali)](https://pypi.org/project/napari-travali)
   -->

|License|

|Read the documentation at https://napari-travali.readthedocs.io/|
|Tests| |Codecov|

|pre-commit| |Black|

Napari TRAck VALIdator

Features
--------

-  Enables manual validation of cell tracking results.

Installation
------------

Currently at this early stage, you can install Napari Travali only fron
GitHub You can install *Napari Travali* via G

.. code:: console

   $ pip install git+https://github.com/yfukai/napari-travali

Usage
-----

Input data specification
~~~~~~~~~~~~~~~~~~~~~~~~

<!--
Currently we accept segmented / tracked data in the following format.

-  file format : ``zarr``
-  contents :
   -  ``Dataset`` named ``image`` … 5-dim dataset with dimensional order ``TCZYX``
   -  ``Group`` named ``labels``
      -  ``Dataset`` named ``original`` … 4-dim dataset with dimensional order ``TZYX``
         -  ``attr`` named ``finalized_track_ids``
         -  ``attr`` named ``candidate_track_ids``
         -  ``attr`` named ``target_Ts``
      -  ``Dataset`` named ``df_tracks``
      -  ``Dataset`` named ``df_divisions``
-->

Please see the `Command-line
Reference <https://napari-travali.readthedocs.io/en/latest/usage.html>`__
for details.

Contributing
------------

Contributions are very welcome. To learn more, see the `Contributor
Guide <CONTRIBUTING.rst>`__.

License
-------

Distributed under the terms of the `MIT
license <https://opensource.org/licenses/MIT>`__, *Napari Travali* is
free and open source software.

Issues
------

If you encounter any problems, please `file an
issue <https://github.com/yfukai/napari-travali/issues>`__ along with a
detailed description.

Credits
-------

This project was generated from
`@cjolowicz <https://github.com/cjolowicz>`__'s `Hypermodern Python
Cookiecutter <https://github.com/cjolowicz/cookiecutter-hypermodern-python>`__
template.

.. |License| image:: https://img.shields.io/pypi/l/napari-travali
   :target: https://opensource.org/licenses/MIT
.. |Read the documentation at https://napari-travali.readthedocs.io/| image:: https://img.shields.io/readthedocs/napari-travali/latest.svg?label=Read%20the%20Docs
   :target: https://napari-travali.readthedocs.io/
.. |Tests| image:: https://github.com/yfukai/napari-travali/workflows/Tests/badge.svg
   :target: https://github.com/yfukai/napari-travali/actions?workflow=Tests
.. |Codecov| image:: https://codecov.io/gh/yfukai/napari-travali/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yfukai/napari-travali
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
