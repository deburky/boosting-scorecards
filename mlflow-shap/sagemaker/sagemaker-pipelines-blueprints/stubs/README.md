# Type Stubs for MLflow 3.0

This directory contains type stub files (`.pyi`) for MLflow 3.0 to help type checkers and linters understand the MLflow API.

## What's Included

- `mlflow/data/__init__.pyi` - Type stubs for `mlflow.data.from_pandas()` and related functions

## Why Are These Needed?

MLflow 3.0's type stubs are incomplete or not fully recognized by all linters (Pylint, Mypy, etc.). These stub files provide type hints for:
- `mlflow.data.from_pandas()` - Creating PandasDataset from DataFrame
- Other mlflow.data utilities

## How to Use

### For Pylint

Add to your `.pylintrc` or `pyproject.toml`:

```ini
[MASTER]
init-hook='import sys; sys.path.append("./stubs")'
```

Or in `pyproject.toml`:

```toml
[tool.pylint.MASTER]
init-hook = 'import sys; sys.path.append("./stubs")'
```

### For Mypy

Add to your `mypy.ini` or `pyproject.toml`:

```ini
[mypy]
mypy_path = ./stubs
```

Or in `pyproject.toml`:

```toml
[tool.mypy]
mypy_path = "./stubs"
```

### For VS Code

Add to your `.vscode/settings.json`:

```json
{
  "python.analysis.extraPaths": ["./stubs"],
  "python.analysis.stubPath": "./stubs"
}
```

## Structure

```
stubs/
├── README.md
└── mlflow/
    ├── __init__.pyi
    ├── py.typed
    └── data/
        └── __init__.pyi
```

## Updating Stubs

When MLflow updates its API, update the corresponding `.pyi` files to match the new signatures.

## Troubleshooting

### Stubs Not Being Recognized

If Pylint still shows "Module 'mlflow.data' has no 'from_pandas' member":

1. **Restart your IDE/Linter** - Type checkers cache module information
2. **Verify stub path** - Check that `stubs/mlflow/data/__init__.pyi` exists
3. **Check Pylint config** - Verify `.pylintrc` is in the project root
4. **Manual disable** - As a fallback, use `# pylint: disable=no-member` comments

### VS Code Specific

If using VS Code, you may need to:
1. Reload the window: `Cmd/Ctrl + Shift + P` → "Reload Window"
2. Or restart the Python language server: "Python: Restart Language Server"

### PyCharm Specific

1. File → Invalidate Caches → Invalidate and Restart
2. Or mark `stubs` as "Sources Root" in Project Structure

## Note

These stubs are for development/linting purposes only. They are not required at runtime as the actual MLflow library provides the implementations.

For immediate error suppression while the IDE picks up the stubs, the code includes `# pylint: disable=no-member` comments around the MLflow data calls.


