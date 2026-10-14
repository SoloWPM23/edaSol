# edaSol

[![PyPI version](https://img.shields.io/pypi/v/edasol.svg)](https://pypi.org/project/edasol/) 
[![Python versions](https://img.shields.io/pypi/pyversions/edasol.svg)](https://pypi.org/project/edasol/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Python library for the repetitive first pass of exploratory data analysis (EDA).

edaSol turns the checks you run on every new dataset into single function calls. It covers data types, missing values, duplicates, outliers, and distributions. Each function returns a plain pandas object or a matplotlib figure, so you keep working with the result instead of reformatting it.

## Features

- **Column summary** returns data types, null counts, null percentages, and unique counts in one table.
- **Data quality report** returns shape, memory use, duplicate count, and a missing-value breakdown in one dictionary.
- **IQR outlier detection** returns the index labels of outliers, with optional lower and upper bounds.
- **Categorical summary** returns the count, cardinality, and most frequent value per column.
- **Six plotting helpers** draw distributions, a correlation heatmap, a missing-value matrix, category bars, boxplots, and pairplots.

## Installation

```bash
pip install edasol
```

> **Naming:** PyPI normalizes the distribution name to lowercase `edasol`, while the import name keeps its original casing. Install `edasol`, then write `import edaSol`.

Install from source for development:

```bash
git clone https://github.com/SoloWPM23/edaSol.git
cd edaSol
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from edaSol import quick_summary, data_quality_report, plot_numerical_dist

df = pd.read_csv('your_data.csv')

# Summarize every column
summary = quick_summary(df)
print(summary)

# Inspect overall quality
report = data_quality_report(df)
print(f"Shape: {report['shape']}")
print(f"Duplicates: {report['duplicates']}")
print(f"Memory: {report['memory_usage']} MB")

# Plot numeric distributions
plot_numerical_dist(df)
```

## API Reference

### Core Functions

#### `quick_summary(df, columns=None)`

Returns a summary table with the data type, null count, null percentage, unique count, and a sample value for each column.

```python
from edaSol import quick_summary

summary = quick_summary(df)
summary = quick_summary(df, columns=['age', 'salary'])
```

#### `detect_outliers_iqr(df, column, return_bounds=False)`

Detects outliers in a numeric column with the interquartile range (IQR) method. Returns the index labels of the outlier rows, and the bounds when you ask for them.

```python
from edaSol import detect_outliers_iqr

outliers = detect_outliers_iqr(df, 'price')
outliers, lower, upper = detect_outliers_iqr(df, 'price', return_bounds=True)
```

#### `describe_categorical(df, columns=None)`

Returns the count, cardinality, top value, top frequency, and top percentage for each categorical column. With `columns=None` it picks the `object` and `category` columns.

```python
from edaSol import describe_categorical

cat_stats = describe_categorical(df)
```

#### `detect_duplicates(df, subset=None, keep='first')`

Returns the duplicate rows. Limit the comparison to specific columns with `subset`, and control which copy counts as the original with `keep`.

```python
from edaSol import detect_duplicates

duplicates = detect_duplicates(df)
duplicates = detect_duplicates(df, subset=['name', 'email'])
```

#### `data_quality_report(df)`

Returns one dictionary with the shape, memory usage, data type counts, missing-value table, duplicate count, numeric summary, and categorical summary.

```python
from edaSol import data_quality_report

report = data_quality_report(df)
# Keys: shape, memory_usage, dtypes, missing, duplicates,
#       numeric_summary, categorical_summary
```

### Visualization Functions

Every plotting function accepts `show`. Leave it at `True` to display the plot, or set it to `False` to get the figure object back for saving or further edits.

#### `plot_numerical_dist(df, columns=None, figsize=(12, 4), show=True)`

Draws a histogram with a KDE curve for each numeric column.

```python
from edaSol import plot_numerical_dist

plot_numerical_dist(df)
plot_numerical_dist(df, columns=['age', 'salary'])
```

#### `plot_correlation_heatmap(df, figsize=(10, 8), annot=True, mask_upper=True, show=True)`

Draws a correlation heatmap for the numeric columns. Set `mask_upper=False` to show the full matrix.

```python
from edaSol import plot_correlation_heatmap

plot_correlation_heatmap(df)
plot_correlation_heatmap(df, mask_upper=False)
```

#### `plot_missing_matrix(df, figsize=(12, 6), show=True)`

Draws the missing-value pattern as a matrix and prints the total share of missing cells.

```python
from edaSol import plot_missing_matrix

plot_missing_matrix(df)
```

#### `plot_categorical_dist(df, columns=None, figsize=(12, 4), top_n=10, show=True)`

Draws a bar plot of value counts for each categorical column. Use `top_n` to cap how many categories appear.

```python
from edaSol import plot_categorical_dist

plot_categorical_dist(df)
plot_categorical_dist(df, top_n=5)
```

#### `plot_boxplots(df, columns=None, figsize=(12, 4), show=True)`

Draws a boxplot for each numeric column to show spread and outliers.

```python
from edaSol import plot_boxplots

plot_boxplots(df)
plot_boxplots(df, columns=['age', 'income'])
```

#### `plot_pairplot(df, columns=None, hue=None, diag_kind='kde', show=True)`

Draws pairwise scatter plots for the numeric columns. With `columns=None` it uses the first five numeric columns. Pass `hue` to color the points by a category.

```python
from edaSol import plot_pairplot

plot_pairplot(df)
plot_pairplot(df, hue='category')
```

## Requirements

- Python 3.8 or newer
- pandas 1.5.0 or newer
- numpy 1.20.0 or newer
- matplotlib 3.5.0 or newer
- seaborn 0.11.0 or newer

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, coding standards, and pull request steps. Report bugs on the [issue tracker](https://github.com/SoloWPM23/edaSol/issues).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the release history.

## License

edaSol is released under the MIT License. See [LICENSE](LICENSE) for the full text.

## Author

Solo Manurung (solowandika490@gmail.com)
