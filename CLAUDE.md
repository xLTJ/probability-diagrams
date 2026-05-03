# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This is an **exam-use toolkit** for a probability & statistics course. The primary goal is to let the user focus entirely on problem-solving during an exam — not on syntax, imports, or implementation details. Every addition should serve that goal:

- Functions must be fast to call with minimal arguments
- Output should be human-readable and self-labelling (already printed, not just returned)
- Docstrings and inline examples are critical — they are the on-exam reference
- Discoverability matters: it should be obvious from a quick read *when* to reach for a function and *what to pass in*

When adding or modifying functions, prioritise clarity of the call signature and correctness of printed output over internal elegance.

## Running the toolkit

```bash
# Run main.py as a scratch file for quick calculations
python main.py

# Run compute.py standalone to see built-in examples
python compute.py

# Run plots.py standalone to see built-in plot examples
python plots.py

# Interactive session (preferred for exam/homework use)
ipython -i init.py
```

The virtual environment is in `.venv/` (Python 3.14). Activate with `source .venv/bin/activate` before running if needed.

## Architecture

Three-file structure with no tests or build system:

- **`compute.py`** — probability calculations. Each function prints a labelled result and returns the value. Takes `sigma2` (variance) as input for normal-related functions, not `std`. Bounds `a`/`b` on discrete distributions are **inclusive**; normal bounds are **exclusive**.
- **`plots.py`** — matplotlib visualisation helpers. `plot_pdf`/`plot_cdf_continuous` take a frozen `scipy.stats` distribution object as first argument. All other functions take raw values.
- **`main.py`** — scratch file; import both modules with `*` and call functions for the current problem.
- **`init.py`** — `ipython -i init.py` startup: suppresses `Out[]` echo, imports `numpy`, `scipy.stats`, and both toolkit modules, then drops into a live REPL.

## Key conventions

- All normal/gaussian functions across both modules take `sigma2` (variance), never `std`. The only place `std` appears is when constructing a `scipy.stats` dist object for `plot_pdf`/`plot_cdf_continuous`.
- Geometric distribution is parameterised as X = trial of first success (X ≥ 1), matching `scipy.stats.geom`.
- Docstrings follow a recipe format: signature on line 1, one-line description, then copy-pasteable examples. Keep this style when adding new functions.
- HTML API docs are pre-generated in `docs/` via pdoc.
