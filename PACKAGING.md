# Release Instructions

##  Updating the Github repository
----------------------

1) Search/Replace the current version string (be sure not to update previous string in CHANGELOG.md)
   
2) Update CHANGELOG.md from history
   
3) Ensure docs have been built with `python build_docs.py`

4) Ensure that all changes are committed to `master` (including _static build files)

5) Checkout `public` branch

6) Merge `master` to `public` using the following:

    `git merge --no-commit --allow-unrelated-histories --strategy-option theirs master`

7) Review the staged changes and ensure that no sensitive files are present (benchmarks, Gitlab config files, large assets, etc)

8) Commit changes to `public`

9)  Push changes to Gitlab branch `public`

10) Push changes to Github branch `main` - NOTE: the external facing Github release branch is called `main` *not* `public`


## Creating a Github Release Package
----------------------

1) Ensure both `master` and `public` branches are up to date as above, and pushed to Gitlab

2) Ensure all build configurations passed:

    https://teamcity.nvidia.com/project/Sandbox_mmacklin_Warp_Building?mode=builds

3) Manually trigger the binary deployment config:

    https://teamcity.nvidia.com/buildConfiguration/Sandbox_mmacklin_Warp_Publishing_PublishSource?mode=builds

4) Download artifacts .zip

5) Test release .zip by extracting to a clean folder and doing:

    a) Run `cd warp`

    b) Run `pip install .`

    c) Run `cd examples`

    d) Run `python example_mesh.py, etc`

    e) Run `python -m warp.tests`
    
5) Create a new release on Github with tag of `v0.2.0` or equivalent and upload release .zip as an attachment


## Upload a PyPi build
----------------------

First time:

* Create a PyPi account
* Create a Token (write it down somewhere safe)
* Get an admin (mmacklin@nvidia.com) to give you write access to the project

Release Steps:

1) Download artifacts as above and unzip

2) Run `cd warp`

3) Run
    ```bash
    python -m build --wheel -C--build-option=-Pwindows-x86_64 &&
    python -m build --wheel -C--build-option=-Plinux-x86_64 &&
    python -m build --wheel -C--build-option=-Plinux-aarch64 &&
    python -m build --wheel -C--build-option=-Pmacos-universal
    ```

4) Run `python -m twine upload dist/*`

    * user: `__token__`
    * pass: `(your token string from pypi)`

