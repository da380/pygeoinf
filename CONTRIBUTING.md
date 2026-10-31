# Contributing

## Development setup

Install the package with its development dependencies, then install the git
hooks:

```bash
poetry install --with dev --all-extras
poetry run pre-commit install
```

On `git commit`, `pre-commit` runs Ruff (linting with `--fix`, then
formatting) and `nbstripout`, and refuses commits made directly on `main`.

The test suite is not a hook. CI runs it on Linux, Windows and macOS on every
pull request to `main` and on `main` itself, but not on pushes to other
branches; run whatever you touched locally before pushing (`poetry run pytest
tests`, say). To run CI on a branch before opening a pull request, start the
workflow by hand (`gh workflow run ci.yml --ref <branch>`, or "Run workflow" on
the Actions tab), which runs the slow set as well.

The same hooks run in CI (the `Lint (pre-commit)` job reads the very same
`.pre-commit-config.yaml`), so anything that passes locally passes there. To
check the whole repository by hand rather than waiting for a commit:

```bash
poetry run pre-commit run --all-files
```

### Notebooks

The tutorials are the scripts under `tutorials/`; the notebooks under
`tutorials/notebooks/` are generated from them by
`tutorials/make_notebooks.py` and are never edited by hand. After editing a
tutorial, regenerate:

```bash
poetry run python tutorials/make_notebooks.py
```

The test suite checks that the notebooks are current. They are committed as
source, never as a record of a particular run: `nbstripout` removes cell
outputs and execution counts, so that merely opening a notebook does not
dirty the working tree, and committing outputs would add megabytes of
base64 images and make every diff unreadable.

### Ruff versions

`ruff` appears twice: as a dev dependency in `pyproject.toml` and as the `rev`
of the `ruff-pre-commit` hook. The two must be bumped together. If they drift
apart, the hook and `poetry run ruff` disagree about formatting and reformat
the same files back and forth between commits.

## Release Process

This document outlines the steps to publish a new version of `pygeoinf` to PyPI.

### Prerequisites

- All code changes for the new version have been merged into the `main` branch.
- The `main` branch is stable and all tests are passing.

### Step-by-Step Guide

#### 1. Create a Version Bump Pull Request

The first step is to update the version number in the `pyproject.toml` file. This should be done in a separate branch and pull request.

1.  **Create a new branch** from `main`:
    ```bash
    git checkout main
    git pull origin main
    git checkout -b release/vX.Y.Z  # Replace X.Y.Z with the new version
    ```

2.  **Update the version number** in `pyproject.toml`:
    ```toml
    [project]
    name = "pygeoinf"
    version = "X.Y.Z"  # Change this line
    # ...
    ```

3.  **Commit and push** the change:
    ```bash
    git add pyproject.toml
    git commit -m "Bump version to X.Y.Z"
    git push origin release/vX.Y.Z
    ```

4.  **Open a Pull Request** on GitHub from your `release/vX.Y.Z` branch to `main`. Title it "Release vX.Y.Z".

#### 2. Review and Merge

- The pull request will trigger the CI workflow to run all tests.
- As the repository owner, you will need to review and approve this pull request.
- Once all checks have passed and the PR is approved, merge it into the `main` branch.

#### 3. Publish the Release on GitHub

Merging the PR does **not** publish the package. The final step is to create a "Release" on GitHub, which acts as the trigger for the publishing workflow.

1.  **Navigate to the repository homepage** on GitHub.
2.  **Click on "Releases"** in the right-hand sidebar.
3.  **Click "Draft a new release"**.
4.  **Create a new tag:** In the "Tag version" box, type the new version number (e.g., `vX.Y.Z`). Click "Create new tag: vX.Y.Z on publish".
5.  **Target the `main` branch:** Ensure the "Target" dropdown is set to `main`.
6.  **Add a release title:** Use the tag name as the title (e.g., `vX.Y.Z`).
7.  **(Recommended) Write release notes:** Summarize the key changes, new features, and bug fixes in this version.
8.  **Click "Publish release"**.

Once the release is published, the `publish.yml` GitHub Action will automatically run, build the package, and upload it to PyPI.


