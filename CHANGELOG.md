# Changelog

Semua perubahan penting pada edaSol dicatat di berkas ini.

Format mengikuti [Keep a Changelog](https://keepachangelog.com/id/1.1.0/),
dan proyek ini memakai [Semantic Versioning](https://semver.org/lang/id/).

## [Unreleased]

## [1.0.0] - 2026-10-15

### Added
- Berkas `LICENSE` (MIT).
- Suite test pytest di `tests/` untuk fungsi core dan visual, termasuk kasus tepi.
- Workflow CI `test.yml` yang menjalankan test pada Python 3.8 sampai 3.12.
- Marker PEP 561 `py.typed` supaya type hints terbaca oleh mypy di sisi pengguna.
- Metadata paket: `keywords`, classifier versi Python, dan URL Repository/Issues.
- Grup dependency opsional `dev` (`pytest`, `ruff`, `mypy`).
- Konfigurasi ruff di `pyproject.toml`.

### Changed
- Menandai rilis stabil pertama (Development Status: Production/Stable).
- `publish.yml` memakai PyPI Trusted Publishing (OIDC) dan menjalankan test sebagai gate sebelum upload, menggantikan autentikasi API token.
- Fungsi visual menutup figure (`plt.close`) setelah `plt.show()` untuk mencegah penumpukan di memori.

### Fixed
- `quick_summary` dan `data_quality_report` tidak lagi menghasilkan NaN/inf saat DataFrame kosong (guard pembagian nol).
- `detect_outliers_iqr` menolak input bukan DataFrame dan kolom non-numerik dengan pesan error yang jelas.
- Fungsi visual memvalidasi tipe input dan keberadaan kolom yang diminta.
- Workflow publish lama berisi key `run:` ganda yang membuat step tidak valid.

## [0.1.1] - 2026-03-26

### Fixed
- Ketidakcocokan versi dan case sensitivity nama paket di README.

## [0.1.0] - 2026-03-26

### Added
- Rilis awal: fungsi core (`quick_summary`, `detect_outliers_iqr`, `describe_categorical`, `detect_duplicates`, `data_quality_report`) dan fungsi visual (`plot_numerical_dist`, `plot_correlation_heatmap`, `plot_missing_matrix`, `plot_categorical_dist`, `plot_boxplots`, `plot_pairplot`).
