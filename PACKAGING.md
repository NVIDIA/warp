# Release Instructions

## Versioning
-------------

Versions take the format X.Y.Z, similar to [Python itself](https://devguide.python.org/developer-workflow/development-cycle/#devcycle):

- Increments in X are reserved for major reworks of the project causing disruptive incompatibility (or reaching the 1.0 milestone).
- Increments in Y are for regular releases with a new set of features.
- Increments in Z are for bug fixes. In principle there are no new features. Can be omitted if 0 or not relevant.

This is similar to [Semantic Versioning](https://semver.org/) but less strict around backward compatibility.
Like with Python, some breaking changes can be present between minor versions if well documented and gradually introduced.

Note that prior to 0.11.0 this schema was not strictly adhered to.


## Repositories
---------------

Development happens internally on a GitLab repository (part of the Omniverse group), while releases are made public on GitHub.

This document uses the following Git remote names:

* **omniverse**: `git remote add omniverse https://gitlab-master.nvidia.com/omniverse/warp.git`
* **github**: `git remote add github https://github.com/NVIDIA/warp.git`


## GitLab Release Branch
------------------------

1) Search & replace the current version string.

   We want to keep the Omniverse extensions's version in sync with the library so update the strings found in the `exts` folder as well.

   Be sure *not* to update previous strings in `CHANGELOG.md`.

2) Update `CHANGELOG.md` from Git history (since the last release branch). Only list user-facing changes.

   The changelogs from the Omniverse extensions found in `exts` are kept in sync with the one from the library, so update them all at the same time and list any change made to the extensions.

3) Commit and push to `master`.

4) For new X.Y versions, create a release branch (note .Z maintenance versions remain on the same branch):

   `git checkout -b release-X.Y [<start-point>]`

   If branching from an older revision or reusing a branch, make sure to cherry-pick the version and changelog update.

5) Make any release-specific changes (e.g. disable/remove features not ready yet).

6) Check that the last revision on the release branch passes TeamCity tests:

   https://teamcity.nvidia.com/project/Omniverse_Warp?mode=builds

   Fix issues until all tests pass. Cherry-pick fixes for `master` where applicable.


## GitLab Public Branch
-----------------------

1) Manually trigger the `publish_bin` config on the `release-X.Y` branch:

    https://teamcity.nvidia.com/buildConfiguration/Omniverse_Warp_Publishing_PublishSource?mode=builds

2) Download artifacts .zip - it should contain a `.whl` file for each supported platform.

3) Extract it to a clean folder and run tests for at least one platform:

    - Run `python -m pip install warp_lang-<version>-<platform-tag>.whl`
    - Run `python -m warp.tests`

    Check that the correct version number gets printed.

4) If tests fail, make fixes on `release-X.Y` and where necessary cherry-pick to `master` before repeating from step (1).

5) If all tests passed:

   * `git push github master:main`
   * `git push github release-X.Y`

6) Tag the release with `vX.Y.Z` on both `omniverse/release-X.Y` and `github/release-X.Y`.

   It is safest to push *just* the new tag using `git push <remote> vX.Y.Z`.

   In case of a mistake, tags can be moved using `git push <remote> vX.Y.Z -f`.


## Creating a GitHub Release Package
------------------------------------

Create a new release on [GitHub](https://github.com/NVIDIA/warp) with a tag of `vX.Y.Z` and upload the .whl artifacts as attachments.


## Upload a PyPI Release
------------------------

First time:

* Create a [PyPI](https://pypi.org/) account.
* [Create a Token](https://pypi.org/manage/account/#api-tokens) for uploading to the `warp-lang` project (store it somewhere safe).
* Get an admin (mmacklin@nvidia.com) to give you write access to the project.

Per release:

Run `python -m twine upload *` from the unzipped .whl artifacts folder (on Windows make sure to use `cmd` shell; Git Bash doesn't work).

* username: `__token__`
* password: `(your token string from PyPI)`


## Publishing the Omniverse Extensions
--------------------------------------

1) Ensure that the version strings and `CHANGELOG.md` files in the `exts` folder are in sync with the ones from the library.

2) Manually trigger the `publish_ext` config on the `release-X.Y` branch:

    https://teamcity.nvidia.com/buildConfiguration/Omniverse_Warp_Publishing_Publish

3) Download artifacts .zip.

4) Extract it to a clean folder and check the extensions inside of Kit:

    - Run `omni.create.sh --ext-folder /path/to/artifacts/exts --enable omni.warp-X.Y.Z --enable omni.warp.core-X.Y.Z`
    - Ensure that the example scenes are working as expected
    - Run test suites for both extensions

4) If tests fail, make fixes on `release-X.Y` and where necessary cherry-pick to `master` before repeating from step (2).

5) If all tests passed:

   * `kit --ext-folder /path/to/artifacts/exts --publish omni-warp.core-X.Y.Z`
   * `kit --ext-folder /path/to/artifacts/exts --publish omni-warp-X.Y.Z`

6) Ensure that the release is tagged with `vX.Y.Z` on both `omniverse/release-X.Y` and `github/release-X.Y`.


## Automated processes
----------------------

The following is just for your information. These steps should run automatically by CI/CD pipelines, but can be replicated manually if needed:

### Building the documentation

The contents of https://nvidia.github.io/warp/ is generated by a GitHub pipeline which runs `python build_docs.py` (prerequisites: `pip install sphinx sphinx_copybutton black furo`).

### Building pip wheels

The TeamCity `publish_bin` configuration combines artifacts from each platform build, moving the contents of `warp/bin` to platform- and architecture-specific
subfolders; e.g. `warp/bin/linux-x86_64` and `warp/bin/linux-aarch64` both contain `warp.so` and `warp-clang.so` files.

Pip wheels are then built using:

```bash
python -m build --wheel -C--build-option=-Pwindows-x86_64
python -m build --wheel -C--build-option=-Plinux-x86_64
python -m build --wheel -C--build-option=-Plinux-aarch64
python -m build --wheel -C--build-option=-Pmacos-universal
```

Selecting the correct library files for each wheel happens in [`setup.py`](setup.py).
