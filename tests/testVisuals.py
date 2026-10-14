"""
Test untuk fungsi-fungsi visualisasi di edaSol.visuals.

Backend matplotlib dipaksa ke 'Agg' (tanpa jendela GUI) agar test bisa jalan
headless di CI. Semua plot dipanggil dengan show=False supaya objek figure
dikembalikan untuk diperiksa, lalu ditutup agar tidak membebani memori.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import seaborn as sns

from edaSol import (
    plot_numerical_dist,
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_categorical_dist,
    plot_boxplots,
    plot_pairplot,
)


@pytest.fixture
def sampleDf():
    return pd.DataFrame({
        'num1': [1, 2, 3, 100, 4, 5],
        'num2': [10, 20, 30, 40, 50, 60],
        'cat': ['a', 'b', 'a', 'a', 'c', None],
    })


@pytest.fixture(autouse=True)
def closeFiguresAfterTest():
    # Tutup semua figure setelah tiap test untuk mencegah penumpukan di memori
    yield
    plt.close("all")


# --- plot_numerical_dist ---

def testPlotNumericalDistReturnsFigure(sampleDf):
    fig = plot_numerical_dist(sampleDf, show=False)
    assert fig is not None


def testPlotNumericalDistNoNumericReturnsNone():
    df = pd.DataFrame({'cat': ['a', 'b', 'c']})
    assert plot_numerical_dist(df, show=False) is None


def testPlotNumericalDistMissingColumnRaises(sampleDf):
    with pytest.raises(ValueError):
        plot_numerical_dist(sampleDf, columns=['tidakAda'], show=False)


def testPlotNumericalDistNonDataframeRaises():
    with pytest.raises(TypeError):
        plot_numerical_dist([1, 2, 3], show=False)


# --- plot_correlation_heatmap ---

def testPlotCorrelationHeatmapReturnsFigure(sampleDf):
    fig = plot_correlation_heatmap(sampleDf, show=False)
    assert fig is not None


def testPlotCorrelationHeatmapTooFewNumericReturnsNone():
    df = pd.DataFrame({'num1': [1, 2, 3], 'cat': ['a', 'b', 'c']})
    assert plot_correlation_heatmap(df, show=False) is None


def testPlotCorrelationHeatmapNonDataframeRaises():
    with pytest.raises(TypeError):
        plot_correlation_heatmap("bukan dataframe", show=False)


# --- plot_missing_matrix ---

def testPlotMissingMatrixReturnsFigure(sampleDf):
    fig = plot_missing_matrix(sampleDf, show=False)
    assert fig is not None


def testPlotMissingMatrixNonDataframeRaises():
    with pytest.raises(TypeError):
        plot_missing_matrix(42, show=False)


# --- plot_categorical_dist ---

def testPlotCategoricalDistReturnsFigure(sampleDf):
    fig = plot_categorical_dist(sampleDf, show=False)
    assert fig is not None


def testPlotCategoricalDistNoCategoricalReturnsNone():
    df = pd.DataFrame({'num': [1, 2, 3]})
    assert plot_categorical_dist(df, show=False) is None


def testPlotCategoricalDistMissingColumnRaises(sampleDf):
    with pytest.raises(ValueError):
        plot_categorical_dist(sampleDf, columns=['tidakAda'], show=False)


# --- plot_boxplots ---

def testPlotBoxplotsReturnsFigure(sampleDf):
    fig = plot_boxplots(sampleDf, show=False)
    assert fig is not None


def testPlotBoxplotsNoNumericReturnsNone():
    df = pd.DataFrame({'cat': ['a', 'b', 'c']})
    assert plot_boxplots(df, show=False) is None


# --- plot_pairplot ---

def testPlotPairplotReturnsGrid(sampleDf):
    grid = plot_pairplot(sampleDf, columns=['num1', 'num2'], show=False)
    assert isinstance(grid, sns.PairGrid)


def testPlotPairplotTooFewNumericReturnsNone():
    df = pd.DataFrame({'num1': [1, 2, 3], 'cat': ['a', 'b', 'c']})
    assert plot_pairplot(df, columns=['num1'], show=False) is None


def testPlotPairplotNonDataframeRaises():
    with pytest.raises(TypeError):
        plot_pairplot([1, 2, 3], show=False)
