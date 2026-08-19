# Project-wide Codex operating rules

These rules apply to every Codex task performed anywhere in this repository.

## GitHub completion workflow

- The private source of truth is `https://github.com/Dwan-123456/micarray-6plus1-voice-direction`.
- `CHANGELOG.md` is the mandatory project-wide change record for L1, L2, L3, L4, Development Test UI, audio recording/data management, runtime contracts, tests, and assets. Every project-changing task must add a specific dated entry before commit; unchanged components must be stated explicitly. A change is not complete and must not be pushed as completed if the changelog is missing or inaccurate.
- When a user-requested change to project code, configuration, documentation, tests, models, or curated test assets is genuinely complete, run the relevant verification, inspect the final Git diff/status, create a clear Git commit, and push it to the tracked GitHub branch before reporting completion.
- Push matching Git LFS objects whenever an LFS-managed file changes, then verify the remote branch/commit. If authentication, networking, tests, or repository state prevents a safe push, do not claim that the task is fully uploaded: preserve the local commit when safe and clearly tell the user what remains.
- Do not upload incomplete or failing work as a completed change. The user's latest explicit instruction to delay or skip an upload overrides the normal automatic push workflow.
- Use semantic versions for releases. Do not move, replace, or rewrite an already published version tag.

## Data boundaries

- Never commit `.venv/`, `data/`, runtime recordings, scratch recordings, catalogs, logs, caches, `.partial` files, secrets, tokens, passwords, or local proxy settings.
- Curated test audio belongs under `tests/fixtures/audio/` and binary audio/model/array assets must follow `.gitattributes` and Git LFS rules.
- Before every commit, check that ignored local data and credentials are not staged.

## Destructive-action boundaries

- Never delete the GitHub repository. Never delete remote branches or tags, clear remote history, or force-push unless the user identifies the exact target and explicitly confirms that specific high-risk action at the time; repository deletion is prohibited even with a general cleanup request.
- Any local project file or directory that must be removed must be sent to the Windows Recycle Bin, not permanently deleted. If Recycle Bin recovery is unavailable or the item is too large, stop and ask the user instead of deleting it.
- After a local removal, report the original path, what was moved to the Recycle Bin, and how it can be restored. Git-tracked removals must also remain recoverable from Git history.

## Completion report

- Report the changelog entry, verification performed, commit hash and message, pushed branch, affected release tag if any, Git LFS changes, and whether the working tree is clean.
- Remind the user about GitHub only when a push could not be completed or when the user explicitly requested review before upload; otherwise complete the authorized push as part of finishing the change.
