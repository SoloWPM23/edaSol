"""
Core analysis functions for edaSol EDA library.
"""
from typing import List, Optional, Tuple, Union
import pandas as pd


def quick_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Return a summary DataFrame with key statistics for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.
    columns : list of str, optional
        Specific columns to include. If None, all columns are used.

    Returns
    -------
    pd.DataFrame
        Summary with: Data Type, Null Count, Null Percent, Unique Count, Sample Value.

    Raises
    ------
    TypeError
        If input is not a pandas DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import quick_summary
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', 'y', 'z']})
    >>> quick_summary(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if columns is not None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        df = df[columns]

    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Null Percent': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique Count': df.nunique(),
        'Sample Value': df.iloc[0] if len(df) > 0 else None
    })

    return summary


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    return_bounds: bool = False
) -> Union[List[int], Tuple[List[int], float, float]]:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Name of the numeric column to analyze.
    return_bounds : bool, default False
        If True, also return the lower and upper bounds.

    Returns
    -------
    list of int
        Index positions of outlier rows.
    tuple of (list, float, float)
        If return_bounds=True: (outlier_indices, lower_bound, upper_bound).

    Raises
    ------
    ValueError
        If the column is not found in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import detect_outliers_iqr
    >>> df = pd.DataFrame({'values': [1, 2, 3, 100, 4, 5]})
    >>> detect_outliers_iqr(df, 'values')
    [3]
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_indices = outliers.index.tolist()

    if return_bounds:
        return outlier_indices, lower_bound, upper_bound
    return outlier_indices


def describe_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Return descriptive statistics for categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Specific categorical columns to analyze. If None, all object/category
        columns are used.

    Returns
    -------
    pd.DataFrame
        Statistics with: Count, Unique, Top Value, Top Frequency, Top Percent.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import describe_categorical
    >>> df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})
    >>> describe_categorical(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if columns is None:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    else:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        cat_cols = columns

    if len(cat_cols) == 0:
        return pd.DataFrame(columns=['Count', 'Unique', 'Top Value', 'Top Frequency', 'Top Percent'])

    stats = []
    for col in cat_cols:
        value_counts = df[col].value_counts()
        top_value = value_counts.index[0] if len(value_counts) > 0 else None
        top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0

        stats.append({
            'Column': col,
            'Count': df[col].count(),
            'Unique': df[col].nunique(),
            'Top Value': top_value,
            'Top Frequency': top_freq,
            'Top Percent': round(top_freq / len(df) * 100, 2) if len(df) > 0 else 0
        })

    return pd.DataFrame(stats).set_index('Column')


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first'
) -> pd.DataFrame:
    """
    Find and return duplicate rows in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    subset : list of str, optional
        Columns to consider for identifying duplicates. If None, all columns
        are used.
    keep : {'first', 'last', False}, default 'first'
        - 'first': Mark duplicates except for the first occurrence.
        - 'last': Mark duplicates except for the last occurrence.
        - False: Mark all duplicates.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the duplicate rows.

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import detect_duplicates
    >>> df = pd.DataFrame({'A': [1, 1, 2], 'B': ['x', 'x', 'y']})
    >>> detect_duplicates(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if subset is not None:
        missing = [col for col in subset if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

    duplicates = df[df.duplicated(subset=subset, keep=keep)] # type: ignore
    return duplicates


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to analyze.

    Returns
    -------
    dict
        Dictionary containing:
        - 'shape': Tuple of (rows, columns)
        - 'memory_usage': Total memory usage in MB
        - 'dtypes': Count of each data type
        - 'missing': DataFrame with missing value statistics
        - 'duplicates': Number of duplicate rows
        - 'numeric_summary': Summary stats for numeric columns
        - 'categorical_summary': Summary stats for categorical columns

    Examples
    --------
    >>> import pandas as pd
    >>> from edaSol import data_quality_report
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', 'y', 'z']})
    >>> report = data_quality_report(df)
    >>> print(report['shape'])
    (3, 2)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Missing values analysis
    missing_df = pd.DataFrame({
        'Null Count': df.isnull().sum(),
        'Null Percent': (df.isnull().sum() / len(df) * 100).round(2),
        'Non-Null Count': df.notnull().sum()
    })
    missing_df = missing_df[missing_df['Null Count'] > 0].sort_values(
        'Null Count', ascending=False
    )

    # Data type counts
    dtype_counts = df.dtypes.value_counts().to_dict()
    dtype_counts = {str(k): v for k, v in dtype_counts.items()}

    # Numeric summary
    num_cols = df.select_dtypes(include=['number']).columns
    numeric_summary = df[num_cols].describe() if len(num_cols) > 0 else pd.DataFrame()

    # Categorical summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_summary = describe_categorical(df, list(cat_cols)) if len(cat_cols) > 0 else pd.DataFrame()

    report = {
        'shape': df.shape,
        'memory_usage': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'dtypes': dtype_counts,
        'missing': missing_df,
        'duplicates': df.duplicated().sum(),
        'numeric_summary': numeric_summary,
        'categorical_summary': categorical_summary
    }

    return report
