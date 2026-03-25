"""
edaSol - A Python library to accelerate Exploratory Data Analysis (EDA).

This library provides convenient functions for quick data analysis and visualization,
helping data scientists and analysts understand their datasets faster.

Modules
-------
core : Core analysis functions (summaries, outliers, duplicates, quality reports)
visuals : Visualization functions (distributions, correlations, missing values)

Examples
--------
>>> import pandas as pd
>>> from edaSol import quick_summary, plot_numerical_dist
>>> df = pd.read_csv('data.csv')
>>> quick_summary(df)
>>> plot_numerical_dist(df)
"""

from .core import (
    quick_summary,
    detect_outliers_iqr,
    describe_categorical,
    detect_duplicates,
    data_quality_report
)

from .visuals import (
    plot_numerical_dist,
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_categorical_dist,
    plot_boxplots,
    plot_pairplot
)

__version__ = "0.1.1"
__author__ = "Solo Manurung"

__all__ = [
    # Core functions
    'quick_summary',
    'detect_outliers_iqr',
    'describe_categorical',
    'detect_duplicates',
    'data_quality_report',
    # Visualization functions
    'plot_numerical_dist',
    'plot_correlation_heatmap',
    'plot_missing_matrix',
    'plot_categorical_dist',
    'plot_boxplots',
    'plot_pairplot',
]
