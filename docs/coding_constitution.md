# Coding Constitution & Style Guide

## 1. Introduction

This document defines the coding conventions and style for this project. The goal is to ensure that the codebase is consistent, readable, and maintainable. Adherence to these guidelines is paramount for producing high-quality, collaborative code.

## 2. File Structure

All Python scripts (`.py`) should be structured using section headers to clearly delineate logical parts of the file. This convention enhances readability and makes it easier to navigate the code.

The mandatory structure is as follows:

```python
### ~~~ GLOBAL IMPORTS ~~~ ###
# (e.g., pandas, numpy, sklearn)

### ~~~ LOCAL IMPORTS ~~~ ###
# (Imports from within this project's `src` directory)

### ~~~ CUSTOM TYPES ~~~ ###
# (Type aliases using typing.TypeAlias, TypedDict, etc.)

### ~~~ STATE DEFINITIONS ~~~ ###
# (Global constants and module-level state)

### ~~~ FUNCTION DEFINITIONS ~~~ ###
# (Core logic)

### ~~~ SCRIPT EXECUTION ~~~ ###
if __name__ == "__main__":
    # (Entry point for executable scripts)
```

**Jupyter Notebooks (`.ipynb`)**: While Python scripts follow a strict structure, Jupyter Notebooks are used for exploration, visualization, and reporting. They should be well-documented with Markdown cells for headings, explanations, and analysis. The structure should be logical, but the formal section headers are not required.

## 3. Imports

- **Grouping**: Imports must be grouped into `GLOBAL` (third-party libraries) and `LOCAL` (project-specific modules). This separation makes it clear where dependencies are coming from.
- **Sorting**: Within each group, imports should ideally be sorted alphabetically. While not strictly enforced, it is highly recommended for consistency.
- **Syntax**: Both `from <module> import <object>` and `import <module> as <alias>` are acceptable. Use the syntax that provides the most clarity in the given context.

## 4. Naming Conventions

- **Variables & Functions**: Use `snake_case` (e.g., `my_variable`, `calculate_results`).
- **Classes**: Use `PascalCase` (e.g., `DataLoader`, `SpatialBackend`). For simple data structures, prefer using `@dataclass`.
- **Constants**: Use `UPPER_SNAKE_CASE` (e.g., `EARTH_RADIUS_KM`, `COLS_PASSENGER`).
- **Type Aliases**: Use `snake_case` with a `_t` suffix (e.g., `tensor_t`, `list_str_t`).

## 5. Docstrings and Comments

- **Docstrings**: All public functions, methods, and classes must have a Google-style docstring. This is crucial for auto-generating documentation and for providing a clear understanding of the code's purpose.
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
- **Block Comments**: Use multi-line string literals (`"""..."""`) to explain complex algorithms, design choices, or justifications for why a certain approach was taken.

## 6. Typing

- **Mandatory Typing**: All function and method signatures (arguments and return values) must include type hints. Variable declarations should also be typed. This is a core principle of the project to ensure code quality and catch errors early.
- **Complex Types**: For complex, repeated type annotations, define a type alias in the `### ~~~ CUSTOM TYPES ~~~ ###` section using `typing.TypeAlias` or `typing.TypedDict`.
  ```python
  from typing import TypeAlias, List
  tensor_t: TypeAlias = np.ndarray
  list_str_t: TypeAlias = List[str]
  ```
- **Clarity**: Use the `typing` module (`Optional`, `Tuple`, `TypeAlias`, `TypedDict`, etc.) to create precise and readable types.

## 7. Pandas Best Practices

- **Avoid In-place Operations**: Do not use `inplace=True`. Instead, reassign the DataFrame. This improves readability and prevents unexpected side effects.
- **Prevent Warnings**: When modifying a subset of a DataFrame, always use `.copy()` to avoid `SettingWithCopyWarning`.
  ```python
  # Good
  df_subset = df[df['column'] > 10].copy()
  df_subset['new_col'] = 5

  # Bad
  df[df['column'] > 10]['new_col'] = 5
  ```
- **Vectorization**: Prioritize vectorized Pandas/NumPy operations over loops or `.apply()` where possible for performance.
- **Progress Bars**: For unavoidable `.apply()` calls on large datasets, use `tqdm.pandas()` to show progress.
- **Method Chaining**: Chain methods for readability, with each method call on a new line.

## 8. Script Execution

- Any script intended to be run directly must have its entry point logic contained within a `main()` function.
- The call to `main()` must be protected by an `if __name__ == "__main__":` guard. This ensures that the script can be both executed directly and imported by other modules without running its main logic.

## 9. File Paths

- **Constants**: Define file paths as `UPPER_SNAKE_CASE` constants at the top of the script. This makes it easy to change paths and improves readability.
  ```python
  DATA_PATH = "dbs/cooked/data.npz"
  MODEL_CHECKPOINT_PATH = "src/models/best_model.keras"
  ```

## 10. Model Explainability

This project emphasizes the importance of model explainability. When building and evaluating models, we strive to understand their behavior.
- **Tools**: Use libraries like `SHAP` and `LIME` to analyze model predictions.
- **Reporting**: Generate and save explainability reports (e.g., summary plots, force plots) in the `reports/explainability` directory.

## 11. Encoding Strategies

- **Hierarchical Encoding**: For categorical variables with a hierarchical structure (e.g., continent -> country -> airport), consider using a hierarchical encoding scheme to preserve information. An example of this can be found in `src/data/pre_process.py`.
- **Clarity**: Choose encoding strategies that are appropriate for the data and the model, and document the rationale behind the choice.
