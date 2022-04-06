Release Instructions
====================


Preparing the public Branch
---------------------------
---------------------------

1) Ensure docs have been built with `python build_docs.py`

2) Ensure that all changes are committed to `master` (including _static build files)

3) Checkout `public` branch

4) Merge `master` to `public` using the following:

    `git merge --no-commit --allow-unrelated-histories --strategy-option theirs master`

5) Review the staged changes and ensure that no sensitive files are present (benchmarks, Gitlab config files, large assets, etc)

6) Commit changes to `public`

7) Push changes to Gitlab branch `public`

8) Push changes to Github branch `main` - NOTE: the external facing Github release branch is called `main` *not* `public`


Preparing a Binary Release
---------------------------
---------------------------

1) Ensure both `master` and `public` branches are up to date as above, and pushed to Gitlab

2) Ensure all build configurations passed:

    https://teamcity.nvidia.com/project/Sandbox_mmacklin_Warp_Building?mode=builds

3) Manually trigger the binary deployment config:

    https://teamcity.nvidia.com/buildConfiguration/Sandbox_mmacklin_Warp_Publishing_PublishSource?mode=builds

4) Download artifacts .zip

5) Test release .zip by extracting to a clean folder and doing:

    a) Run `pip install .`

    b) Run `cd examples`

    c) Run `python example_mesh.py, etc`

    d) Run `python -m warp.tests`
    
5) Create a new release on Github with tag of `v0.1.25` or equivalent and upload release .zip as an attachment




