# Contributing

Thank you for contributing to `deep4downscaling`!

## Branching model

| Branch | Purpose |
| --- | --- |
| `main` | Stable, release-ready code |
| `devel` | Active development and integration |

All pull requests must target **`devel`**. Maintainers periodically merge `devel` into `main` and create tagged releases.

## For collaborators (with write access)

```bash
git clone https://github.com/SantanderMetGroup/deep4downscaling.git
cd deep4downscaling
git checkout devel
git pull origin devel
git checkout -b feature/my-change
# ... make changes ...
git push -u origin feature/my-change
```

Open a PR with base branch `devel`.

## For external contributors

1. Fork the repository on GitHub
2. Clone your fork and add upstream:

   ```bash
   git clone https://github.com/<your-username>/deep4downscaling.git
   cd deep4downscaling
   git remote add upstream https://github.com/SantanderMetGroup/deep4downscaling.git
   git fetch upstream
   git checkout -b feature/my-change upstream/devel
   ```

3. Push to your fork and open a PR against `SantanderMetGroup/deep4downscaling` → `devel`

## Documentation contributions

When adding or changing public API:

1. Write a **NumPy-style docstring** (see `deep4downscaling.trans` for examples)
2. Update the relevant **user guide** page if the change affects workflows or conventions
3. Add a **how-to** or extend a **tutorial notebook** for new end-to-end capabilities
4. Build docs locally to verify:

   ```bash
   pip install -e ".[docs]"
   mkdocs serve
   ```

### Documentation checklist for PRs

- [ ] Docstring added/updated for new public functions or classes
- [ ] User guide updated if behaviour or conventions change
- [ ] API reference renders correctly (`mkdocs build`)
- [ ] Notebook updated or added for new workflows (if applicable)

## Code style

- Match the style of surrounding code
- Use type hints where existing modules do
- Keep SPDX license header on new Python files: `# SPDX-License-Identifier: MIT`

## Questions

Open a [GitHub issue](https://github.com/SantanderMetGroup/deep4downscaling/issues) for questions, bug reports, or documentation improvements.
