<!-- Labels: apply every existing project label whose description fits this MR. Do not create new labels. -->
<!-- Use this template to bring a GitHub pull request into GitLab. For your own changes, use Default.md. -->

> :warning: **Do not merge this MR until a maintainer has temporarily turned off `Squash commits`.**
>
> This MR carries a GitHub contributor's commits unchanged. A squash merge replaces them with a new commit authored by whoever opened this MR, which erases the contributor from the history. The pull request's head commit also never reaches `main`, so GitHub never marks the pull request as merged. It stays open, and the only way left to resolve it is to close it by hand with a comment saying it was merged. A closed pull request looks rejected, which misleads anyone tracking contributions on GitHub. GitLab's **Rebase** button keeps the author but rewrites the commits, so it has the same effect on GitHub.
>
> This project requires squashing (**Settings > Merge requests > Squash commits when merging** is set to **Require**). Change it to **Allow** and merge with `Squash commits` cleared. GitLab reads this setting when the merge happens, so set it back to **Require** only after this MR shows as merged. With auto-merge, that means after the pipeline finishes.

## GitHub pull request

- Pull request: https://github.com/NVIDIA/warp/pull/<NUMBER>
- Contributor: `<GITHUB_HANDLE>`
- Pull request head commit: `<FULL_SHA>` <!-- Update after every refetch. -->
- Closes GH-<NUM> <!-- Or "Part of GH-<NUM>". Delete this line if there is no issue. -->

## Summary

<!-- One or two sentences on what the pull request does. The contributor already explained the
     motivation, design, and validation on GitHub, so do not repeat them here. Add only what a
     GitLab reviewer needs that the pull request does not say, such as a design change made
     during GitHub review that the pull request description does not mention. -->

## Maintainer commits

<!-- Ideally none. With fast-forward merges, every commit listed here lands on `main`.
     List each commit added on top of the contributor's commits, with a line on why it
     is not part of the GitHub pull request. -->

None.

## GitHub review activity

<!-- One row per review thread on the GitHub pull request, so readers can see how
     thoroughly it was reviewed and by whom. Include review-bot threads only when
     they led to a change or were deliberately declined.
     Resolution: ✅ addressed, ❌ not addressed, 🟡 deferred to future work (link the follow-up issue). -->

| Comment                                                               | Reviewer          | Summary | Resolution |
| --------------------------------------------------------------------- | ----------------- | ------- | ---------- |
| [link](https://github.com/NVIDIA/warp/pull/<NUMBER>#discussion_r<ID>) | `<GITHUB_HANDLE>` |         | ✅         |

## Checklist

- [ ] I fetched the branch from the GitHub pull request (for example, `gh pr checkout <NUMBER>`) and pushed it to a personal GitLab fork (for example, `git push origin`) without rewriting the contributor's commits. Any maintainer commits sit on top of them and are listed under Maintainer commits.
- [ ] The pull request is up to date with `main`. If it was behind, it was rebased **on the GitHub pull request** (by the contributor, or with **Update branch > Update with rebase**), then fetched again and force-pushed to the fork. Never rebase only the fork's copy: it would no longer match the pull request, and GitHub would leave the pull request open instead of marking it as merged.
- [ ] The contributor squashed their commits on GitHub to as few as possible, ideally one. Maintainers do not squash them on the contributor's behalf.
- [ ] Every contributor commit has a DCO `Signed-off-by` line.
- [ ] The last contributor commit on this branch is the pull request head commit listed above. Right before merging, check that the pull request on GitHub still has that head commit, with no new pushes since the last fetch.
- [ ] Optional, at the maintainer's discretion: a complete GitHub CI pipeline passed on the pull request head commit (`/ok to test <FULL_SHA>`, see [pull request testing](https://docs.gha-runners.nvidia.com/platform/onboarding/pull-request-testing/)). The GitLab pipeline runs on this MR either way.

/label ~"github pr"
