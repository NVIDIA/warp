# Contributing to NVIDIA Warp

Contributions and PRs from the community are welcome. Rather than requiring a
formal Contributor License Agreement (CLA), we use the
[Developer Certificate of Origin](https://developercertificate.org/) to
ensure contributors have the right to submit their contributions to this project.
Please ensure that all commits have a
[sign-off](https://git-scm.com/docs/git-commit#Documentation/git-commit.txt--s)
added with an email address that matches the commit author
to agree to the DCO terms for each particular contribution.

```text
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

See the [Contribution Guide](https://nvidia.github.io/warp/modules/contribution_guide.html) for
more information about how to contribute to Warp.

## Forking and Branch Naming

The first step in developing for Warp is to create a fork of the Warp repository.

- GitHub community developers can fork the [GitHub Warp repository](https://github.com/NVIDIA/warp).
- NVIDIA developers can fork the [GitLab Warp repository](https://gitlab-master.nvidia.com/omniverse/warp).

Features should be developed on a branch with the following naming scheme:

    user/feature-name

For example:

    mmacklin/cuda-bvh-optimizations

## Opening a Merge Request

The following guidelines were originally written for NVIDIA developers
working on Warp using the internal GitLab repository. Developers working
on GitHub should generally follow this process, replacing the GitLab-specific
components with their GitHub counterparts.

When you're ready to submit your changes, please follow these steps to create a Merge Request (MR):

1. **Create MR**: Submit your MR against the Warp repo.
Ensure your MR has a descriptive title that clearly states the purpose of the changes.

2. **Add a Detailed Description**: Your MR should include a brief description covering:
   - Summary of changes.
   - Areas affected by the changes.
   - The problem being solved.
   - Any limitations or non-handled areas in the changes.
   - A link to the JIRA Or GitHub issue it is addressing.

3. **Pre-Review Checklist**: The following should be checked before assigning reviews:
   - Unit / regression tests are written.
   - Docs have been updated.
   - Use `ruff check` and `ruff format --check` to check for code quality issues.
     The GitLab pipeline will fail if there are issues.
     Exclusions may be used as appropriate, e.g. `# noqa: F841` or `#fmt: skip`.
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
