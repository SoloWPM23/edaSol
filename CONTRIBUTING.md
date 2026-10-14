# Berkontribusi ke edaSol

Terima kasih sudah tertarik berkontribusi. Dokumen ini menjelaskan cara menyiapkan lingkungan, standar kode, dan alur mengirim perubahan.

## Menyiapkan Lingkungan

```bash
git clone https://github.com/SoloWPM23/edaSol.git
cd edaSol
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -e ".[dev]"
```

Perintah di atas memasang edaSol dalam mode editable beserta dependency pengembangan (`pytest`, `ruff`, `mypy`).

## Menjalankan Test dan Linter

```bash
pytest          # jalankan seluruh test
ruff check .    # periksa gaya kode
mypy edaSol     # periksa type hints
```

Semua test wajib lulus sebelum perubahan dikirim. Tambahkan test untuk setiap fungsi baru atau perbaikan bug, termasuk kasus tepi (DataFrame kosong, kolom tidak ada, tipe input salah).

## Standar Kode

- Nama file, fungsi, dan class memakai lowerCamelCase.
- Nama variabel, fungsi, dan class ditulis dalam Bahasa Inggris.
- Komentar kode dan dokumentasi naratif ditulis dalam Bahasa Indonesia.
- Docstring memakai gaya NumPy dengan bagian Parameters, Returns, Raises, dan Examples.
- Jangan memakai emoji di kode, komentar, maupun commit message.

## Mengirim Perubahan

1. Buat branch dari `main`.
2. Lakukan perubahan yang fokus dan terbatas pada satu tujuan.
3. Perbarui `CHANGELOG.md` di bagian `[Unreleased]`.
4. Pastikan `pytest` dan `ruff check` lulus.
5. Buka Pull Request ke `main` dengan deskripsi singkat tentang apa dan mengapa.

## Melaporkan Bug

Buka issue di https://github.com/SoloWPM23/edaSol/issues dengan langkah reproduksi, perilaku yang diharapkan, perilaku yang terjadi, serta versi Python dan edaSol.
