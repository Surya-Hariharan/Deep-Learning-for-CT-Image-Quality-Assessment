# Git Worktree & Repository Hygiene Audit

Date: 2026-09-12

## 1. Current Git state

- **Repository root:** `C:/Users/surya/OneDrive/Desktop/Deep-Learning-for-CT-Image-Quality-Assessment`
- **`.git` dir / common dir:** `.git` (a normal, single-worktree repository — not a linked worktree, not a submodule)
- **Current branch:** `main`
- **HEAD:** `730bb2c` (`refactor(data): normalize dataset directory names`) — attached, not detached
- **Upstream:** `main` tracks `origin/main`, and is up to date with it (`## main...origin/main`, no ahead/behind marker)
- **Remote:** `origin` → `https://github.com/Surya-Hariharan/Deep-Learning-for-CT-Image-Quality-Assessment.git` (fetch + push)
- **Worktrees:** exactly one — the primary worktree at the repo root, on `main` at `730bb2c`. No linked/stale worktrees registered.
- **Working tree status:** **clean**. `git status --short` shows nothing; `git diff --stat` and `git diff --name-status` are empty (no staged or unstaged changes); `git ls-files --others --exclude-standard` returns nothing (no untracked files that aren't ignored).
- **Recent history:** linear, no merge commits, no obvious force-push artifacts; last commit is the dataset-directory-rename refactor mentioned in the task.

## 2. Worktree problems

**None found.** Specifically checked and ruled out:

| Problem checked | Evidence | Verdict |
|---|---|---|
| Extra/stale worktrees | `git worktree list --verbose` shows only the main worktree | Not present |
| Detached HEAD | `git branch --show-current` → `main`; `git branch -vv` shows `* main ... [origin/main]` | Not present |
| Wrong current branch | On `main`, matches `origin/HEAD` | Not present |
| Nested `.git` directories | `find . -name ".git" -not -path "./.git"` → no results | Not present |
| Git lock files | `.git/*.lock` → none found | Not present |
| Unexpected staged files | `git diff --stat` (cached) empty, `git status --short` empty | Not present |
| Unexpected unstaged modifications | `git diff --name-status` empty | Not present |
| Duplicate repository roots | Only one `.git` dir; `git rev-parse --show-toplevel`/`--git-dir`/`--git-common-dir` all consistent | Not present |
| IDE/OS metadata files | No `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db` present anywhere in the tree | Not present |

**Severity: N/A — no problems to fix.** The repository is already in a clean, professional state from a Git/worktree perspective.

## 3. File-state classification

### Tracked files (27 total, all small — largest is a 2.1 MB notebook)

All 27 tracked files are legitimate, small, version-control-appropriate artifacts:

- `.gitignore`
- `data/train/train.json`, `data/test/test.json` — small (24 KB / 10 KB) JSON label files (filename → quality score), explicitly intended to be kept per the `.gitignore` header comments
- `notebooks/01_eda_ldct_iqac.ipynb` (2.1 MB), `notebooks/02_ohashi_resnet50_ldct_iqac.ipynb` (484 KB) — research/training record notebooks
- `scripts/*.py`, `src/ct_iqa/**/*.py`, `tests/*.py` — source code, all a few KB each
- `weights/README.md` — documentation stub (not weights themselves)

**Classification: KEEP** for all 27 files. Nothing tracked looks like it should be removed from version control.

### Ignored (correctly) — currently untracked, on disk, matched by `.gitignore`

| Path | Contents | Matched by | Classification |
|---|---|---|---|
| `data/train/image/` | `.tif` CT images (LDCT-IQAC, patient-derived) | `*.tif` | **IGNORE** — correct, must never be tracked |
| `data/test/images/` | `.tiff` CT images | `*.tiff` | **IGNORE** — correct |
| `weights/pretrained/` | `RadImageNet-ResNet50_notop.h5`, `radimagenet_resnet50_backbone.pt` | `*.h5`, `*.pt` | **IGNORE** — correct, large pretrained weights |
| `scripts/__pycache__/`, `src/ct_iqa/**/__pycache__/`, `tests/__pycache__/` | Python bytecode cache | `__pycache__/` | **IGNORE** — correct |

No `.ipynb_checkpoints`, no stray temp files, no `__pycache__` currently tracked.

### Untracked and NOT ignored

**None.** `git ls-files --others --exclude-standard` returned zero results, so there are no files needing a TRACK decision and nothing currently falling through the `.gitignore` net.

### REMOVE / INVESTIGATE

**None identified.** No stray lock files, swap files, or ambiguous artifacts were found anywhere in the tree.

## 4. Risk assessment

**No data-loss risk identified.** In detail:

- The working tree is clean, so there is no uncommitted work that any repair step could destroy.
- There is exactly one worktree, so there is no risk of accidentally deleting a worktree that holds unique uncommitted changes.
- The dataset directories (`data/train/image/`, `data/test/images/`) and the pretrained weights (`weights/pretrained/`) are untracked-but-present on disk and correctly ignored — they were never at risk of being swept up by a `git clean`, and this audit does not recommend running one.
- The one thing worth flagging (not a risk, a staleness note): the `.gitignore` header comments and some path-based rules still refer to the pre-rename directory names `data/training/image/`, `data/testing/images/`, `data/testing/**`, and the allow-list line `!data/testing/test.json`, even though the last commit (`730bb2c`) renamed these to `data/train/` and `data/test/`. This is currently **harmless** because the extension-based rules (`*.tif`, `*.tiff`, `*.h5`, `*.pt`, etc.) independently catch every file inside the renamed directories regardless of path — confirmed via `git status --ignored` showing `data/train/image/` and `data/test/images/` as ignored even though no path rule matches those exact new paths. But the path-based rules and comments no longer describe reality, which is a latent documentation/maintainability risk (a future contributor reading `.gitignore` could be misled about which path-based rule is "the one" protecting the dataset).

## 5. Proposed repair plan

Because no worktree problems, tracking errors, or untracked-but-ignorable files exist, **no destructive or tracking-changing commands are needed.** The only recommended action is a documentation/clarity fix to `.gitignore`, which is additive and non-destructive:

```bash
# OPTIONAL, non-destructive clarity fix — update .gitignore path rules/comments
# to match the post-rename directory names (data/train/, data/test/) instead of
# the old (data/training/, data/testing/) names. This changes no ignore
# behavior in practice (extension rules already cover these paths) but makes
# the file's intent match current reality.
#
# No `git rm --cached`, no worktree removal, and no cleanup commands are
# required — there is nothing to stage, untrack, or delete.
```

No other commands are proposed for Phase 2.
