# Documentation index

This project separates three kinds of documentation, and they must never be
blurred together (see `docs/replication/deviations.md` and
`docs/decisions/README.md` for why this separation exists):

- **`paper/`** -- what the reference paper (Ohashi et al.) specifies, and
  nothing else. If it's not in the paper, it isn't in `paper/`.
- **`replication/`** -- how *this project* adapts and replicates that
  methodology: our dataset, our implementation decisions where the paper is
  silent, and every deliberate deviation.
- **`decisions/`** -- a running log of structural/process decisions about
  the repository itself (not the science).

Also in this directory:

- `git_worktree_audit.md` -- prior Git/worktree hygiene audit.
- `repository_architecture_audit.md` -- the audit that this migration was
  based on.

Start with the root [`README.md`](../README.md) for a project overview,
installation, and how to run training/evaluation.
