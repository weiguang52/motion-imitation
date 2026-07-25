# Repository instructions

These instructions apply to the entire `motion-imitation` repository.

## Repository and remote

- The repository root is `/home/wd/Agent_memory_patch/IK_Retargeting`.
- The GitHub remote is `https://github.com/weiguang52/motion-imitation.git`.
- The primary branch is `main`.
- Use the repository-local author identity already configured in `.git/config`.
- Do not commit the parent workspace `/home/wd/Agent_memory_patch`; run Git commands from this repository root.

## Repository structure

- `GVHMR/` is vendored source code, not a Git submodule.
- The original GVHMR Git metadata is preserved locally at `GVHMR/.git-upstream/` and is ignored. Do not rename it back to `GVHMR/.git` before staging, because Git would treat all of GVHMR as an embedded repository instead of tracking its source files.
- `GVHMR/third-party/DPVO` is the intentional submodule. Keep the root `.gitmodules` entry and the pinned gitlink unless the task explicitly updates DPVO.
- Keep third-party LICENSE files and upstream attribution intact.
- `hmr4d` is an intentional symbolic link to `GVHMR/hmr4d`.

## Files that must not be committed

Never use `git add -f` for ignored model or output paths.

- `inputs/checkpoints/`
- `GVHMR/inputs/`
- `data/smpl/**/*.pkl`
- `output/`
- `outputs/`
- `raw_motion_npy/`
- `data/output/`
- `GVHMR/.git-upstream/`
- `.vscode/`, Python caches, temporary files and downloaded root ZIP archives

Reasons:

- HMR2, ViTPose and other checkpoints exceed GitHub's 100MB file limit.
- SMPL and SMPL-X assets have separate licences and must not be redistributed.
- Generated videos, images, NPY/PKL results and caches create noisy or very large commits.

When new generated directories or model formats appear, update `.gitignore` before staging them. Do not delete local ignored assets merely to make Git status clean.

## Before committing

1. Read `git status --short --branch` and inspect all changed paths.
2. Review `git diff` and `git diff --cached`.
3. Scan new text files for credentials, tokens, private keys, passwords and machine-specific absolute paths.
4. Confirm no staged regular file is close to GitHub's 100MB limit:

   ```bash
   git ls-files -z | xargs -0 -r stat -c '%s %n' | sort -nr | head -30
   ```

5. Confirm ignored assets remain ignored:

   ```bash
   git check-ignore -v \
     inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt \
     data/smpl/SMPL_NEUTRAL.pkl \
     raw_motion_npy
   ```

6. Preserve unrelated user changes. Do not reset, checkout or rewrite them.

Legacy third-party files contain existing trailing whitespace, so a repository-wide `git diff --check` may report old content during an initial import. For later commits, check the files changed by the current task and do not mechanically rewrite vendored code.

## Validation

For primary-person selection or tracker changes, run:

```bash
python -m unittest GVHMR/tools/unitest/test_person_selector.py -v
python -m py_compile \
  GVHMR/hmr4d/utils/preproc/person_selector.py \
  GVHMR/hmr4d/utils/preproc/tracker.py \
  GVHMR/tools/unitest/test_person_selector.py
```

For server changes, at minimum compile the changed Python files. Full GVHMR validation requires a compatible CUDA environment and all ignored checkpoints; state clearly when end-to-end inference could not be run.

## Commit and push workflow

- Make focused commits with clear imperative or descriptive messages.
- Push normal updates with `git push origin main`.
- Never force-push, delete remote branches or rewrite published `main` history unless the user explicitly requests it.
- Before pushing, use `git ls-remote --symref origin HEAD refs/heads/main` when the remote state is uncertain.
- After pushing, compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/main`.
- Report the final commit hash and any files intentionally excluded from Git.
- Network access or GitHub authentication may require user approval; request it only when the push or remote verification is ready.

## Documentation maintenance

- Keep the root `README.md` in sync when commands, environment variables, endpoints, output formats, model paths or required assets change.
- Document new ignored assets in both `.gitignore` and the README's “未上传到 GitHub 的文件” section.
- Do not claim full end-to-end verification unless it was actually run with CUDA and the required model files.
