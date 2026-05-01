# Contributing to Distributed Matrix Multiplication Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🍴 Fork & Clone

```bash
# Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/cuda-mpi-matmul-engine.git
cd cuda-mpi-matmul-engine

# Add upstream remote
git remote add upstream https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine.git
```

## 🌿 Branch Strategy

- `main` — stable, production-ready code
- `feature/<name>` — new features or enhancements
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation updates

```bash
git checkout -b feature/my-new-feature main
```

## 🔨 Development Workflow

1. **Plan first** — For complex features, open an issue to discuss the approach before coding
2. **Build** — Ensure your changes compile: `make all`
3. **Test** — Run benchmarks and verify correctness:
   ```bash
   make cpu && make mpi
   bash scripts/benchmark.sh
   ```
4. **Verify** — All implementations must pass correctness verification (PASS in CSV output)
5. **Edge cases** — Test with edge sizes: 1, 15, 17, 31, 33, 255, 257
6. **Commit** — Use conventional commits format:
   ```
   feat: add block-cyclic MPI decomposition
   fix: correct tile boundary check for N % TILE_SIZE != 0
   docs: update Docker setup instructions
   refactor: extract GPU kernel launch config
   perf: optimize shared memory bank conflict pattern
   ```
7. **Push & PR** — Push to your fork and open a Pull Request against `main`

## 📏 Code Style & Linting

### C / CUDA
- **Standard:** C99 for C files, C++14 for CUDA (`.cu`) files
- **Compiler flags:** `-O2 -Wall -Wextra -std=c99` — code must compile cleanly with no warnings
- **Naming:** `snake_case` for functions and variables, `UPPER_CASE` for macros/constants
- **Headers:** Every `.c`/`.cu` file must have a corresponding `.h` header with include guards
- **Comments:** Only comment *why*, not *what*. Code should be self-documenting.
- **Deterministic:** Never change `MATRIX_SEED` (42) or `VERIFY_TOLERANCE` (1e-3f) without discussion

### Python (Benchmark Scripts)
- Follow **PEP 8** conventions
- Use type annotations on function signatures
- Format with **black**, lint with **ruff**

### Shell Scripts
- Use `set -e` at the top of all bash scripts
- Quote all variable expansions: `"$VAR"`
- Use shellcheck to validate: `shellcheck scripts/*.sh`

## 🐛 Bug Reports

When reporting a bug, please include:

1. **Summary** — One-line description of the problem
2. **Steps to Reproduce** — Exact commands and inputs
3. **Expected Behavior** — What should happen
4. **Actual Behavior** — What actually happens (include error output)
5. **Environment** — OS, GCC version, CUDA version (if applicable), MPI version
6. **Logs** — Full terminal output, especially CSV lines and verification results
7. **Severity** — `critical` / `high` / `medium` / `low`

Use the [Bug Report template](https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine/issues/new?template=bug_report.yml).

## 💡 Feature Requests

When proposing a feature, please include:

1. **Problem Statement** — What problem does this solve?
2. **Proposed Solution** — Your idea for implementation
3. **Alternatives Considered** — Other approaches you've thought about
4. **Scope** — What's in and out of scope
5. **Risks** — Potential downsides or complications

Use the [Feature Request template](https://github.com/H0NEYP0T-466/cuda-mpi-matmul-engine/issues/new?template=feature_request.yml).

## 🧪 Testing Requirements

- All implementations must be verified against the CPU sequential reference
- New implementations must support all size presets (small, medium, large, xlarge)
- Edge cases must be tested: N = 1, 15, 17, 31, 33, 255, 257
- Benchmark results must be reproducible (deterministic seed = 42)

## 📝 Documentation

- Update `README.md` when adding new build targets or CLI flags
- Update `CLAUDE.md` when changing project architecture or conventions
- Add comments for non-obvious algorithmic choices
- Update `report/report.md` for significant performance changes

## 🔍 Code Review

All PRs require at least one review. Reviewers will check:
- [ ] Code compiles without warnings (`make all`)
- [ ] Correctness verification passes
- [ ] No hardcoded values or magic numbers
- [ ] Follows project naming conventions
- [ ] Documentation updated if needed

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
