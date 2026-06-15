# FiRec

TIFF画像から照射野と光照射野を解析し、位置情報とサイズ比較を記録するためのPythonアプリケーションです。

## Project Structure

```text
src/firec/
  core/       画像解析ロジック
  gui/        PySide6 GUI
  storage/    SQLite保存とCSV出力
tests/        自動テスト
sample/       解析サンプル画像
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python -m firec
```

## Test

```bash
pytest
```
