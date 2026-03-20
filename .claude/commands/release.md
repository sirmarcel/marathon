Create a new release of marathon. Optional argument: version bump level (`micro`, `meso`, or `macro`).

## Steps

### 1. Pre-flight checks

- Verify you are on the `main` branch with a clean working tree
- If the working tree is dirty, do NOT proceed — ask the user what to do

### 2. Run full local verification

- Run `ruff check marathon/ && ruff format --check marathon/` — must pass
- Run the inline tests (every file with a `# -- test --` marker):
  ```
  python -c "import marathon.data.properties"
  python -c "import marathon.data.sample"
  python -c "import marathon.data.batching"
  python -c "import marathon.utils"
  python -c "import marathon.emit.pretty"
  python -c "import marathon.emit.properties"
  python -c "import marathon.evaluate.metrics"
  python -c "import marathon.extra.edge_to_edge.neighborlist"
  python -c "import marathon.grain.data_source.flatten_atoms"
  python -c "import marathon.grain.data_source.properties"
  ```
- Run `cd tests && python -m pytest -x . && cd ..` — must pass
- Run the examples:
  ```
  cd examples/inference && bash run.sh && cd ../..
  cd examples/calculator && bash run.sh && cd ../..
  cd examples/train_plain && bash run.sh && cd ../..
  cd examples/train_grain && bash run.sh && cd ../..
  ```
- Do NOT proceed if any of these fail

### 3. Review changes

- Find the latest git tag with `git describe --tags --abbrev=0` (or note if there are no tags yet)
- Run `git log <last-tag>..HEAD --oneline` to see all commits since the last release (or all commits if no prior tag)
- Read through the actual diffs (`git diff <last-tag>..HEAD` or individual commits) to understand what changed — commit messages in this repo are not structured, so you need to read the code

### 4. Determine version

This project uses **EffVer** (effort-based versioning), not semver. Version bumps signal the effort a user needs to adopt the new version:

- **MICRO** (e.g. 0.2.0 → 0.2.1): Little to no effort needed. Bug fixes, docs, internal refactors that don't change behaviour.
- **MESO** (e.g. 0.2.0 → 0.3.0): Some effort required. New features, changed defaults, deprecations, renamed functions.
- **MACRO** (e.g. 0.2.0 → 1.0.0): Significant effort. Large API redesigns, removed modules, major behavioural changes.

If a bump level was given ($ARGUMENTS), use it. Otherwise, based on your review of the changes, discuss with the user what the appropriate level should be. Confirm the new version with the user before proceeding.

### 5. Write changelog

- Write a brief changelog summarising the changes, grouped by category where appropriate (e.g. Added, Changed, Fixed, Removed, Internal)
- Present the changelog to the user for review and approval
- If `CHANGELOG.md` exists, add the new section above the previous release. If not, create it with a header and the first entry.

### 6. Update version and commit

- Update the `version` field in `pyproject.toml`
- Stage both `CHANGELOG.md` and `pyproject.toml`
- Ask the user if there are any other files that should be included
- Commit as `marathon vX.Y.Z`

### 7. Tag and push

- Push the commit
- Create a tag: `git tag vX.Y.Z`
- Push the tag: `git push --tags`
- This triggers the `release.yml` CI workflow which builds and publishes to PyPI

### 8. Create GitHub release

- Use `gh release create vX.Y.Z --title "vX.Y.Z" --notes "<changelog>"` to create a GitHub release with the changelog from step 5

## Notes

- PyPI package name is `marathon-train`, importable as `marathon`.
- The CI release workflow uses PyPI trusted publishing (OIDC, no tokens needed).
- EffVer reference: https://docs.jax.dev/en/latest/jep/25516-effver.html
