# FiRec

TIFF画像から照射野と光照射野を解析し、位置情報とサイズ比較を記録するためのPythonアプリケーションです。

## 操作マニュアル

- [Markdown版](docs/user-manual.md)
- [PDF版](docs/FiRec-user-manual.pdf)

最新版はMarkdown版です。FiRecフォルダをVS Codeで開き、Markdown PDF拡張で `docs/user-manual.md` を出力できます。用紙・余白・`docs/manual.css` の指定は `.vscode/settings.json` に設定しています。既存PDFは今回の編集では更新していません。

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
