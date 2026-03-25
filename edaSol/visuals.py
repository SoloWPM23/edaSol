"""
Visualization functions for edaSol EDA library.
"""
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_numerical_dist(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True
) -> Optional[plt.Figure]: # type: ignore
    """
    Create histogram with KDE plots for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific numeric columns to plot. If None, all numeric columns are used.
    figsize : tuple of int, default (12, 4)
        Figure size as (width, height) per row.
    show : bool, default True
        If True, display the plot. If False, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import plot_numerical_dist
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
    >>> plot_numerical_dist(df)
    """
    if columns is None:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        num_cols = columns

    if len(num_cols) == 0:
        print("No numeric columns to plot.")
        return None

    n_cols = 2
    n_rows = (len(num_cols) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    # Flatten axes for consistent iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='steelblue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)

    # Hide empty subplots
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
    mask_upper: bool = True,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Create a correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    figsize : tuple of int, default (10, 8)
        Figure size as (width, height).
    annot : bool, default True
        If True, show correlation values on the heatmap.
    mask_upper : bool, default True
        If True, hide the upper triangle (redundant values).
    show : bool, default True
        If True, display the plot. If False, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import plot_correlation_heatmap
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>> plot_correlation_heatmap(df)
    """
    num_df = df.select_dtypes(include=['number'])
    if num_df.shape[1] < 2:
        print("Need at least 2 numeric columns for correlation.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    corr = num_df.corr()

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        center=0,
        vmin=-1,
        vmax=1
    )
    ax.set_title('Correlation Heatmap')

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_missing_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Visualize missing values as a heatmap matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    figsize : tuple of int, default (12, 6)
        Figure size as (width, height).
    show : bool, default True
        If True, display the plot. If False, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from edaSol import plot_missing_matrix
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [None, 2, 3]})
    >>> plot_missing_matrix(df)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # True = Missing (red), False = Present (light gray)
    sns.heatmap(
        df.isnull(),
        cbar_kws={'label': 'Missing'},
        cmap=['#E8E8E8', '#E74C3C'],
        ax=ax
    )
    ax.set_title('Missing Values Matrix')

    # Show column names but hide row indices for cleaner view
    ax.set_yticks([])

    # Add summary text
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    ax.set_xlabel(f'Total Missing: {total_missing} ({missing_pct:.1f}%)')

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_categorical_dist(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    top_n: int = 10,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Create bar plots for categorical columns showing value counts.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific categorical columns to plot. If None, all object/category
        columns are used.
    figsize : tuple of int, default (12, 4)
        Figure size as (width, height) per row.
    top_n : int, default 10
        Maximum number of categories to show per column.
    show : bool, default True
        If True, display the plot. If False, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import plot_categorical_dist
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green', 'blue']})
    >>> plot_categorical_dist(df)
    """
    if columns is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        cat_cols = columns

    if len(cat_cols) == 0:
        print("No categorical columns to plot.")
        return None

    n_cols = 2
    n_rows = (len(cat_cols) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    # Flatten axes for consistent iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(cat_cols):
        if i < len(axes):
            value_counts = df[col].value_counts().head(top_n)
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[i], color='steelblue')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel('Count')
            axes[i].set_ylabel('')

            # Add count labels
            for j, v in enumerate(value_counts.values):
                axes[i].text(v + 0.1, j, str(v), va='center', fontsize=9)

    # Hide empty subplots
    for j in range(len(cat_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Create boxplots to visualize distributions and outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific numeric columns to plot. If None, all numeric columns are used.
    figsize : tuple of int, default (12, 4)
        Figure size as (width, height) per row.
    show : bool, default True
        If True, display the plot. If False, return the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import plot_boxplots
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100, 4], 'B': [5, 6, 7, 8, 9]})
    >>> plot_boxplots(df)
    """
    if columns is None:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        num_cols = columns

    if len(num_cols) == 0:
        print("No numeric columns to plot.")
        return None

    n_cols = 2
    n_rows = (len(num_cols) + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))

    # Flatten axes for consistent iteration
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(num_cols):
        if i < len(axes):
            sns.boxplot(x=df[col].dropna(), ax=axes[i], color='steelblue')
            axes[i].set_title(f'Boxplot of {col}')
            axes[i].set_xlabel(col)

    # Hide empty subplots
    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if show:
        plt.show()
        return None
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue: Optional[str] = None,
    diag_kind: str = 'kde',
    show: bool = True
) -> Optional[sns.PairGrid]:
    """
    Create pairwise scatter plots for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific numeric columns to include. If None, all numeric columns
        are used (limited to first 5 for performance).
    hue : str, optional
        Column name for color encoding.
    diag_kind : {'kde', 'hist'}, default 'kde'
        Type of plot for diagonal subplots.
    show : bool, default True
        If True, display the plot. If False, return the PairGrid object.

    Returns
    -------
    seaborn.PairGrid or None
        PairGrid object if show=False, otherwise None.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import plot_pairplot
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})
    >>> plot_pairplot(df)
    """
    if columns is None:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Limit to first 5 columns for performance
        if len(num_cols) > 5:
            print(f"Limiting to first 5 numeric columns (out of {len(num_cols)}).")
            num_cols = num_cols[:5]
    else:
        num_cols = columns

    if len(num_cols) < 2:
        print("Need at least 2 numeric columns for pairplot.")
        return None

    plot_df = df[num_cols].copy()
    if hue is not None and hue in df.columns:
        plot_df[hue] = df[hue]

    g = sns.pairplot(
        plot_df,
        hue=hue,
        diag_kind=diag_kind,
        plot_kws={'alpha': 0.6},
        corner=True
    )
    g.fig.suptitle('Pairwise Relationships', y=1.02)

    if show:
        plt.show()
        return None
    return g
