# Coding Constitution & Style Guide

## 1. Introduction

This document defines the coding conventions and style for this project. The goal is to ensure that the codebase is consistent, readable, and maintainable. Adherence to these guidelines is paramount.

## 2. File Structure

All Python scripts (`.py`) should be structured using section headers to clearly delineate logical parts of the file.

The mandatory structure is as follows:

```python
### ~~~ GLOBAL IMPORTS ~~~ ###
# (e.g., pandas, numpy, sklearn)

### ~~~ LOCAL IMPORTS ~~~ ###
# (Imports from within this project's `src` directory)

### ~~~ CUSTOM TYPES ~~~ ###
# (Type aliases using `typing.TypeAlias`)

### ~~~ STATE DEFINITIONS ~~~ ###
# (Global constants and module-level state)

### ~~~ FUNCTION DEFINITIONS ~~~ ###
# (Core logic)

### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    # (Entry point for executable scripts)
```

## 3. Imports

- **Grouping**: Imports must be grouped into `GLOBAL` (third-party libraries) and `LOCAL` (project-specific modules).
- **Sorting**: Within each group, imports should be sorted alphabetically.
- **Syntax**: Use the `from <module> import <object>` syntax.

## 4. Naming Conventions

- **Variables & Functions**: Use `snake_case` (e.g., `my_variable`, `calculate_results`).
- **Classes**: Use `PascalCase` (e.g., `DataLoader`, `SpatialBackend`).
- **Constants**: Use `UPPER_SNAKE_CASE` (e.g., `EARTH_RADIUS_KM`, `COLS_PASSENGER`).
- **Type Aliases**: Use `snake_case` with a `_t` suffix (e.g., `tensor_t`, `backend_t`).

## 5. Docstrings and Comments

- **Docstrings**: All public functions, methods, and classes must have a Google-style docstring.
  ```python
  def my_function(param1: int, param2: str) -> bool:
      """
      A brief one-line summary of the function.

      A more detailed explanation of what the function does, its purpose,
      and any relevant context.

      Args:
          param1: Description of the first parameter.
          param2: Description of the second parameter.

      Returns:
          A description of the return value.
      """
      # ...
  ```
- **Inline Comments**: Use `### comment ###` to clarify non-obvious implementation steps or to label logical blocks within a function.
- **Block Comments**: Use multi-line string literals (`"""..."""`) to explain complex algorithms, design choices, or justifications for why a certain approach was taken (e.g., why a feature was dropped).

## 6. Typing

- **Mandatory Typing**: All function and method signatures (arguments and return values) must include type hints.
- **Complex Types**: For complex, repeated type annotations (e.g., `np.ndarray`, `dict[str, int]`), define a type alias in the `### ~~~ CUSTOM TYPES ~~~ ###` section.
- **Clarity**: Use the `typing` module (`Optional`, `Tuple`, `TypeAlias`, `TypedDict`, etc.) to create precise and readable types.

## 7. Pandas Best Practices

- **Avoid In-place Operations**: Do not use `inplace=True`. Instead, reassign the DataFrame.
- **Prevent Warnings**: When modifying a subset of a DataFrame, always use `.copy()` to avoid `SettingWithCopyWarning`.
  ```python
  # Good
  df_subset = df[df['column'] > 10].copy()
  df_subset['new_col'] = 5

  # Bad
  df[df['column'] > 10]['new_col'] = 5
  ```
- **Vectorization**: Prioritize vectorized Pandas/NumPy operations over loops or `.apply()` where possible for performance. For unavoidable `.apply()` calls on large datasets, use `tqdm.pandas()` to show progress.
- **Method Chaining**: Chain methods for readability, with each method call on a new line.

## 8. Script Execution

- Any script intended to be run directly must have its entry point logic contained within a `main()` function.
- The call to `main()` must be protected by an `if __name__ == "__main__":` guard. This ensures that the script can be both executed directly and imported by other modules without running its main logic.
