# napari-pillow-image-processing (npil)

[![License](https://img.shields.io/pypi/l/napari-pillow-image-processing.svg?color=green)](https://github.com/haesleinhuepf/napari-pillow-image-processing/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-pillow-image-processing.svg?color=green)](https://pypi.org/project/napari-pillow-image-processing)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-pillow-image-processing.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-pillow-image-processing/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-pillow-image-processing/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-pillow-image-processing/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-pillow-image-processing)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-pillow-image-processing)](https://napari-hub.org/plugins/napari-pillow-image-processing)

Process images using pillow from within napari. In the current implementation, only single-channel `uint` 8-bit images 
are supported. Multi-channel and RGB images should be split in individual images and then processed per channel.
Furthermore, 3D image stacks and time-lapse data sets are processed slice-by-slice 2D.

You find all implemented filters visualized in the [demo notebook](docs/demo.ipynb)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-pillow-image-processing` via [pip]

    pip install git+https://github.com/haesleinhuepf/napari-pillow-image-processing.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-pillow-image-processing" is free and open source software

Some documentation snippets have been copied over from the [pillow documentation](https://pillow.readthedocs.io/).
Hence, please respect the [pillow license](thirdparty_licenses.txt)

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/haesleinhuepf/napari-pillow-image-processing/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
