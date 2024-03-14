# Contributing to NVIDIA Warp

The following guidelines are for NVIDIA developers working on Warp using the internal GitLab repository and TeamCity CI/CD.

## Submitter Guidelines

The first step in developing for Warp is to create a fork of the Warp repo here: [NVIDIA Warp repository](https://gitlab-master.nvidia.com/omniverse/warp).

Features should be developed on a branch with the following naming scheme:

    user/feature-name

For example:

    mmacklin/cuda-bvh-optimizations

When you're ready to submit your changes, please follow these steps to create a Merge Request (MR):

1. **Create MR**: Submit your MR against the Omniverse Warp repo. Ensure your MR has a descriptive title that clearly states the purpose of the changes.

2. **Add a Detailed Description**: Your MR should include a brief description covering:
   - Summary of changes.
   - Areas affected by the changes.
   - The problem being solved.
   - Any limitations or non-handled areas in the changes.
   - A link to the JIRA Or GitHub issue it is addressing.

3. **Pre-Review Checklist**: The following should be checked before assigning reviews:
   - Unit / regression tests are written.
   - Docs have been updated.
   - Run `flake8` or `black` to auto-format changes (GitLab pipeline will indicate if there are Flake8 errors).
   - The GitLab CI/CD pipeline for the merge request is successful.

4. **Assign Reviewers**: Select one or more reviewers from the owners list below to review your changes.
Use the **Assignees** field to indicate reviewers who must _all_ approve the MR before it can be merged.
Additional reviewers whose approvals are not required can be listed in the **Reviewers** field.

5. **Address Reviewer Comments**: Respond to all reviewer feedback. Be open to revising your approach based on their suggestions.
Once you have addressed a comment then reply to notify reviewers.
_Do not_ resolve the thread yourself, this makes it harder for the reviewer to verify what has been changed.
If a reviewer has already approved the MR, you may self-resolve any of their outstanding threads in the interest of convenience.

6. **Final Steps for Merging**: Before your MR can be merged, ensure that:
   - All reviewer comments are resolved.
   - All mandatory reviewers (in the **Assignees** field) have approved the MR.
   - TeamCity builds are passing without issues.

## Reviewer Guidelines

As a reviewer, your role is crucial in maintaining the quality of the NVIDIA Warp library. Here's what to look for in an MR:

1. **Bug and Regression Checks**: If the MR addresses any bugs or regressions, verify that new unit tests are added to prevent future regressions.

2. **Code Style and Conventions**: The code should generally adhere to PEP8 standards. However, if the surrounding code deviates from these standards, prioritize existing conventions. Avoid introducing new styles, layouts, or terminology for existing concepts.

3. **Documentation**: Check for appropriate documentation of new features. This includes docstrings and updates to the User Manual. Note that documentation is auto-generated for each MR, so contributors should not submit built documentation files.

4. **Review Thoroughly**: Take your time with the review.
   - Consider if there's a simpler or better solution, ask clarifying questions or add comments if the intention is not clear.
   - Consider the impact on the user experience, ease of use, intuitiveness, and consistency.
   - Beware of breaking changes, even if the API does not change, does it break semantics existing users may be relying on?

   Once you are satisfied with a thread resolution you should mark it as resolved. All threads must be resolved for the MR to be merged.

## Feature Owners

If you're contributing to a specific area of NVIDIA Warp, please consult the relevant feature owners:

- **Public API**: MilesM + relevant area owner from below
  
- **Code Generation**: NicolasC, MilesM

- **Platform Support (macOS, Tegra)**: NicolasC
- **CI/CD**: EricS, NicolasC, ZachC
- **CUDA, MGPU**: LukaszW, EricS
- **Kit Extensions**: ChristopherC
- **Torch/dlpack Interop**: LukaszW, ZachC
- **warp.sim**: EricH, MilesM
- **warp.fem**: GillesD
- **warp.optim**: GillesD, JonathanL
- **NanoVDB**: GregK
- **Testing/Packaging/Deployment**: EricS, NicolasC, LukaszW

Thank you for your contributions to making NVIDIA Warp a great tool for developers!
