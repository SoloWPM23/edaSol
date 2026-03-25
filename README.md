# edaSol

A Python library to accelerate Exploratory Data Analysis (EDA).

edaSol provides simple, intuitive functions to quickly understand your dataset's structure, quality, and distributions without writing repetitive boilerplate code.

## Features

- **Quick Data Summary** - Get data types, null counts, unique values at a glance
- **Data Quality Reports** - Comprehensive analysis of missing values, duplicates, and memory usage
- **Outlier Detection** - IQR-based outlier identification
- **Categorical Analysis** - Statistics for categorical columns
- **Visualization Suite** - Distribution plots, correlation heatmaps, boxplots, and more

## Installation

```bash
pip install edaSol
```

Or install from source:

```bash
git clone https://github.com/SoloWPM23/edaSol.git
cd edaSol
pip install -e .
```

## Quick Start

```python
import pandas as pd
from edaSol import quick_summary, data_quality_report, plot_numerical_dist

# Load your data
df = pd.read_csv('your_data.csv')

# Get a quick summary of all columns
summary = quick_summary(df)
print(summary)

# Generate a comprehensive quality report
report = data_quality_report(df)
print(f"Shape: {report['shape']}")
print(f"Duplicates: {report['duplicates']}")
print(f"Memory: {report['memory_usage']} MB")

# Visualize numeric distributions
plot_numerical_dist(df)
```

## API Reference

### Core Functions

#### `quick_summary(df, columns=None)`
Returns a summary DataFrame with key statistics for each column.

```python
from edaSol import quick_summary

summary = quick_summary(df)
# Returns: Data Type, Null Count, Null Percent, Unique Count, Sample Value
```

#### `detect_outliers_iqr(df, column, return_bounds=False)`
Detects outliers using the IQR (Interquartile Range) method.

```python
from edaSol import detect_outliers_iqr

# Get outlier indices
outliers = detect_outliers_iqr(df, 'price')

# Get outliers with bounds
outliers, lower, upper = detect_outliers_iqr(df, 'price', return_bounds=True)
```

#### `describe_categorical(df, columns=None)`
Returns descriptive statistics for categorical columns.

```python
from edaSol import describe_categorical

cat_stats = describe_categorical(df)
# Returns: Count, Unique, Top Value, Top Frequency, Top Percent
```

#### `detect_duplicates(df, subset=None, keep='first')`
Finds and returns duplicate rows.

```python
from edaSol import detect_duplicates

duplicates = detect_duplicates(df)
duplicates_by_cols = detect_duplicates(df, subset=['name', 'email'])
```

#### `data_quality_report(df)`
Generates a comprehensive data quality report.

```python
from edaSol import data_quality_report

report = data_quality_report(df)
# Returns dict with: shape, memory_usage, dtypes, missing, duplicates,
# numeric_summary, categorical_summary
```

### Visualization Functions

#### `plot_numerical_dist(df, columns=None, figsize=(12, 4), show=True)`
Creates histogram with KDE plots for numeric columns.

```python
from edaSol import plot_numerical_dist

plot_numerical_dist(df)
plot_numerical_dist(df, columns=['age', 'salary'])
```

#### `plot_correlation_heatmap(df, figsize=(10, 8), annot=True, mask_upper=True, show=True)`
Creates a correlation heatmap for numeric columns.

```python
from edaSol import plot_correlation_heatmap

plot_correlation_heatmap(df)
plot_correlation_heatmap(df, mask_upper=False)  # Show full matrix
```

#### `plot_missing_matrix(df, figsize=(12, 6), show=True)`
Visualizes missing values as a heatmap matrix.

```python
from edaSol import plot_missing_matrix

plot_missing_matrix(df)
```

#### `plot_categorical_dist(df, columns=None, figsize=(12, 4), top_n=10, show=True)`
Creates bar plots for categorical columns.

```python
from edaSol import plot_categorical_dist

plot_categorical_dist(df)
plot_categorical_dist(df, top_n=5)  # Show only top 5 categories
```

#### `plot_boxplots(df, columns=None, figsize=(12, 4), show=True)`
Creates boxplots to visualize distributions and outliers.

```python
from edaSol import plot_boxplots

plot_boxplots(df)
plot_boxplots(df, columns=['age', 'income'])
```

#### `plot_pairplot(df, columns=None, hue=None, diag_kind='kde', show=True)`
Creates pairwise scatter plots for numeric columns.

```python
from edaSol import plot_pairplot

plot_pairplot(df)
plot_pairplot(df, hue='category')  # Color by category
```

## Requirements

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.20.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## License

MIT License

## Author

Solo Manurung (solowandika490@gmail.com)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
