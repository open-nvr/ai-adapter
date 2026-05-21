# Releasing `opennvr-adapter-sdk`

This is the runbook for cutting a new release of the SDK to PyPI. The release is "push a tag" — everything else is automated by [`.github/workflows/publish-sdk.yml`](../.github/workflows/publish-sdk.yml).

## One-time setup (maintainer-only, do this once per project)

The publish workflow uses [PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/) (OIDC) so we never store a PyPI API token in the repo or in GitHub secrets. PyPI is configured to trust a specific workflow on a specific repo, and GitHub Actions mints a short-lived OIDC token at job runtime that PyPI accepts in lieu of a token.

You need to do this once for PyPI **and** once for TestPyPI:

1. Create the project (or claim the name) on the target index:
   - PyPI: <https://pypi.org/manage/account/publishing/>
   - TestPyPI: <https://test.pypi.org/manage/account/publishing/>

2. Add a "pending publisher" with:
   - PyPI project name: `opennvr-adapter-sdk`
   - Owner: `open-nvr`
   - Repository name: `ai-adapter`
   - Workflow name: `publish-sdk.yml`
   - Environment name: `pypi` for PyPI, `testpypi` for TestPyPI

3. In the GitHub repo, create two GitHub Environments matching the names above (`Settings → Environments → New environment`):
   - `pypi` — **add required reviewers**. Tag-push is the entire authorization story for a real PyPI release; an environment-required-reviewer check is the second factor that stops a compromised laptop or a fat-fingered `git push --tags` from shipping a release nobody intended.
   - `testpypi` — no reviewers needed; TestPyPI fires on every `arch-rev` / `main` push.

4. **Add branch protection to `arch-rev` and `main`.** Both branches trigger TestPyPI publishes — without protection, anyone with repo-write can push directly and poison the TestPyPI dogfooding feed. Require PR review + status checks before merge. Strongly recommended; the trusted-publisher trust boundary doesn't gate TestPyPI behind reviewers (we don't want a review prompt on every dogfood push), so branch protection is what holds the line there.

5. (Optional but recommended) Add a tag protection rule for `sdk-v*` tags so only maintainers can push them.

> **About the owner/repo strings**: PyPI's trusted-publisher matcher is **case-sensitive**. `open-nvr/ai-adapter` and `Open-NVR/ai-adapter` are different. Copy the exact slug from your repo URL — the value to the right of `https://github.com/`.

After this, no further PyPI-side configuration is needed for any release. The OIDC trust is sticky.

### "What if v1.0.0 was already uploaded manually?"

The "pending publisher" flow above only works if the PyPI project name has no existing releases. If someone (a previous maintainer, you in a panic, etc.) already uploaded a version using a classic API token, you can't use the *pending* publisher form — you have to add a publisher under an existing project:

1. Sign in to PyPI as a maintainer of `opennvr-adapter-sdk`.
2. Go to `https://pypi.org/manage/project/opennvr-adapter-sdk/settings/publishing/`.
3. Add a "GitHub Publisher" with the same `Owner` / `Repository` / `Workflow` / `Environment` fields documented in step 2 above.

After that, future releases via this workflow will work the same way. The old API-token-based uploads aren't affected — they remain published — but the *next* release uses OIDC. We recommend revoking the legacy API token after the first trusted-publisher release succeeds.

## Cutting a release

The branch model: every feature PR targets `arch-rev`. When `arch-rev` is ready to release, a single "release PR" promotes `arch-rev` → `main`. Tag from `main`. The workflow publishes to TestPyPI on every push to `arch-rev` AND `main` (so the exact SHA you tag has been validated against TestPyPI no matter which branch the work came from), and to PyPI on the tag push.

1. Land all the work you want in the release on `arch-rev` via feature PRs. Then open a release PR from `arch-rev` → `main`. **Use "Create a merge commit" or "Rebase and merge"** on the release PR, **not "Squash and merge"** — squash rewrites the commit and the resulting `main` HEAD has never been on `arch-rev` (and was last TestPyPI-validated under a different SHA). Merge or rebase preserves identity.

2. Bump the version. Two places, both must match (CI verifies). Both are the single source of truth — there is no generated-from-the-other relationship, you bump both by hand:
   - `opennvr_adapter_sdk/__init__.py` — `__version__ = "X.Y.Z"`
   - `opennvr_adapter_sdk/pyproject.toml` — `version = "X.Y.Z"`

3. Update `opennvr_adapter_sdk/CHANGELOG.md`:
   - Add a new `## [X.Y.Z] — YYYY-MM` heading with the changes.
   - Update the link reference at the bottom: `[X.Y.Z]: https://github.com/open-nvr/ai-adapter/releases/tag/sdk-vX.Y.Z`

4. Open a PR from `arch-rev` → `main` titled "SDK release vX.Y.Z." The PR run builds the sdist + wheel and runs `twine check`, so a packaging regression fails the PR.

5. Merge the PR.

6. From `main`, tag the release commit and push the tag. Sanity-check first that the SHA you're tagging is the same SHA that was on `arch-rev` (`git log --oneline main arch-rev` should show the release commit in both).

   ```bash
   git checkout main
   git pull
   # First release uses the current declared version, 1.0.0.
   git tag -a sdk-v1.0.0 -m "opennvr-adapter-sdk v1.0.0"
   git push origin sdk-v1.0.0
   ```

7. Watch the `publish-sdk` workflow run. On a release tag you'll see four jobs:
   - **`build`** — sdist + wheel + `twine check` + `__init__.py` vs `pyproject.toml` version-match check.
   - **`smoke-test`** — matrix on Python 3.10/3.11/3.12: installs the just-built wheel in a fresh interpreter AND installs the sdist in a separate fresh venv, then imports both. All three matrix slots must pass.
   - **`verify-tag-matches-version`** — the pushed tag (e.g. `sdk-v1.0.0`) must match the package's `__version__`.
   - **`publish-pypi`** — uploads sdist + wheel to PyPI under the `pypi` environment with PEP 740 attestations. If you configured required reviewers on that environment, you'll get the approval prompt here before upload.

8. After the workflow goes green, create a GitHub Release pointing at the tag with the CHANGELOG entry pasted in.

9. Smoke-test the published package in a fresh venv:

   ```bash
   python -m venv /tmp/sdk-smoke && source /tmp/sdk-smoke/bin/activate
   pip install opennvr-adapter-sdk==1.0.0
   python -c "import opennvr_adapter_sdk; print(opennvr_adapter_sdk.__version__)"
   ```

## Versioning rules

The SDK is semver, and the major version is locked to the AI Adapter Contract major version:

- `1.x.y` targets contract v1. No breaking changes within `1.*`.
- `2.x.y` would target contract v2. New major contract → new major SDK.

Within a major, what counts as breaking:
- Removing or renaming any symbol re-exported at the package root (`AdapterApp`, `AdapterService`, `BodyShape`, `BODY_BYTES_KEY`, `ServiceError`, contract types).
- Changing an abstract method's signature on `AdapterService`.
- Removing/changing the meaning of an `AdapterApp` constructor parameter.

Non-breaking (minor bump):
- New optional `AdapterApp` parameters with defaults.
- New abstract methods on `AdapterService` that have a default implementation.
- New contract types re-exported.

Bugfix-only (patch bump):
- Wire format unchanged, behavior fixes, new tests, doc-only changes.

## Pre-release dogfooding

Pushes to `arch-rev` automatically publish to **TestPyPI** with `skip-existing: true`. To dogfood a pre-release build (e.g. inside one of the reference adapters or a downstream project):

```bash
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  opennvr-adapter-sdk
```

The extra-index-url is so transitive deps (`fastapi`, `pydantic`, etc.) come from real PyPI — TestPyPI doesn't mirror them.

If you need to test a build that *isn't* on `arch-rev` yet (a feature branch you haven't merged), bump the version locally to a `.devN` suffix and run `python -m build` from `opennvr_adapter_sdk/`, then `pip install dist/*.whl` directly. No PyPI involved.

## What to do when the workflow fails

| Failure | What it means | Fix |
|---|---|---|
| `twine check` fails | README.md has invalid markdown for PyPI's renderer, or metadata is malformed | Reproduce locally: `cd opennvr_adapter_sdk && python -m build && twine check dist/*`. Fix what twine reports, re-tag. |
| `version drift` step fails | `__init__.py` and `pyproject.toml` versions disagree | Bump both, re-tag |
| `verify-tag-matches-version` fails | The tag you pushed doesn't match `__version__` | Delete the bad tag (`git push origin :sdk-vX.Y.Z`), push the right one |
| `publish-pypi` fails with "trusted publisher" error | The PyPI-side trusted-publisher config drifted from this workflow | Re-check the workflow name / environment / owner / repo in PyPI publishing settings |
| `publish-pypi` fails with "file already exists" | This version was already published. PyPI versions are immutable | Bump to the next patch, re-tag |

## Yanking a bad release

If you publish a release with a serious bug, you can yank it from PyPI (`Manage release → Yank`). Yanked versions still install if pinned exactly (`==1.2.3`) but won't be selected by resolvers. Bump to the next patch and ship a fix; don't try to re-upload the same version.

### Partial-upload recovery

If `publish-pypi` partially succeeds (e.g. the sdist uploaded but the wheel failed and the job errored), PyPI now holds a half-broken release: `pip install opennvr-adapter-sdk==X.Y.Z` will resolve, but only the sdist is available. You **cannot** re-push the missing wheel for the same version — PyPI versions are immutable per artifact. Recovery is:

1. Yank `X.Y.Z` on PyPI.
2. Bump to `X.Y.(Z+1)`.
3. Land the version bump on `arch-rev`, then `main`, then tag the new SHA.
4. Note the yank in `CHANGELOG.md` against the new version so users searching pin history see what happened.

The same applies if a release goes out and we discover a security or correctness bug within minutes: yank, bump, re-tag.
