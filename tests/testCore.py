"""
Test untuk fungsi-fungsi analisis di edaSol.core.

Mencakup jalur normal dan kasus tepi: DataFrame kosong, kolom tidak ada,
input bukan DataFrame, dan kolom non-numerik.
"""
import pandas as pd
import pytest

from edaSol import (
    quick_summary,
    detect_outliers_iqr,
    describe_categorical,
    detect_duplicates,
    data_quality_report,
)


@pytest.fixture
def sampleDf():
    """DataFrame kecil dengan kolom numerik, kategorikal, dan nilai hilang."""
    return pd.DataFrame({
        'num': [1, 2, 3, 100, 4, 5],
        'cat': ['a', 'b', 'a', 'a', 'c', None],
    })


# --- quick_summary ---

def testQuickSummaryColumnsAndStats(sampleDf):
    summary = quick_summary(sampleDf)
    assert list(summary.columns) == [
        'Data Type', 'Null Count', 'Null Percent', 'Unique Count', 'Sample Value'
    ]
    # Kolom 'cat' punya satu nilai hilang dari 6 baris
    assert summary.loc['cat', 'Null Count'] == 1
    assert summary.loc['cat', 'Null Percent'] == pytest.approx(16.67, abs=0.01)


def testQuickSummaryEmptyDataframeNoDivisionError():
    # DataFrame tanpa baris tidak boleh menghasilkan error pembagian nol
    df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
    summary = quick_summary(df)
    assert summary.loc['a', 'Null Percent'] == 0.0
    assert summary.loc['a', 'Sample Value'] is None


def testQuickSummarySubsetColumns(sampleDf):
    summary = quick_summary(sampleDf, columns=['num'])
    assert list(summary.index) == ['num']


def testQuickSummaryMissingColumnRaises(sampleDf):
    with pytest.raises(ValueError):
        quick_summary(sampleDf, columns=['tidakAda'])


def testQuickSummaryNonDataframeRaises():
    with pytest.raises(TypeError):
        quick_summary([1, 2, 3])


# --- detect_outliers_iqr ---

def testDetectOutliersIqrFindsOutlier(sampleDf):
    outliers = detect_outliers_iqr(sampleDf, 'num')
    assert 3 in outliers  # nilai 100 di posisi index 3


def testDetectOutliersIqrReturnBounds(sampleDf):
    outliers, lower, upper = detect_outliers_iqr(sampleDf, 'num', return_bounds=True)
    assert lower < upper
    assert isinstance(outliers, list)


def testDetectOutliersIqrMissingColumnRaises(sampleDf):
    with pytest.raises(ValueError):
        detect_outliers_iqr(sampleDf, 'tidakAda')


def testDetectOutliersIqrNonNumericRaises(sampleDf):
    with pytest.raises(TypeError):
        detect_outliers_iqr(sampleDf, 'cat')


def testDetectOutliersIqrNonDataframeRaises():
    with pytest.raises(TypeError):
        detect_outliers_iqr([1, 2, 3], 'num')


# --- describe_categorical ---

def testDescribeCategoricalBasic(sampleDf):
    stats = describe_categorical(sampleDf)
    assert 'cat' in stats.index
    assert stats.loc['cat', 'Top Value'] == 'a'
    assert stats.loc['cat', 'Top Frequency'] == 3


def testDescribeCategoricalNoCategoricalReturnsEmpty():
    df = pd.DataFrame({'num': [1, 2, 3]})
    stats = describe_categorical(df)
    assert stats.empty


def testDescribeCategoricalMissingColumnRaises(sampleDf):
    with pytest.raises(ValueError):
        describe_categorical(sampleDf, columns=['tidakAda'])


# --- detect_duplicates ---

def testDetectDuplicatesFindsRows():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
    dups = detect_duplicates(df)
    assert len(dups) == 1


def testDetectDuplicatesSubset():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'z', 'y']})
    dups = detect_duplicates(df, subset=['a'])
    assert len(dups) == 1


def testDetectDuplicatesMissingColumnRaises():
    df = pd.DataFrame({'a': [1, 1, 2]})
    with pytest.raises(ValueError):
        detect_duplicates(df, subset=['tidakAda'])


def testDetectDuplicatesNonDataframeRaises():
    with pytest.raises(TypeError):
        detect_duplicates([1, 1, 2])


# --- data_quality_report ---

def testDataQualityReportKeys(sampleDf):
    report = data_quality_report(sampleDf)
    for key in ['shape', 'memory_usage', 'dtypes', 'missing',
                'duplicates', 'numeric_summary', 'categorical_summary']:
        assert key in report
    assert report['shape'] == (6, 2)


def testDataQualityReportEmptyDataframeNoDivisionError():
    df = pd.DataFrame({'a': pd.Series([], dtype='float64')})
    report = data_quality_report(df)
    assert report['shape'] == (0, 1)


def testDataQualityReportNonDataframeRaises():
    with pytest.raises(TypeError):
        data_quality_report("bukan dataframe")
