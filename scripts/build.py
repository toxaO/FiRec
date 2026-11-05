"""Build the Windows distribution on Windows with PyInstaller."""

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["windows"], required=True)
    parser.add_argument("--clean", action="store_true", help="Clear PyInstaller caches before building")
    args = parser.parse_args()
    if platform.system() != "Windows":
        parser.error("Windows builds must run on Windows.")
    if importlib.util.find_spec("PyInstaller") is None:
        parser.error("Install build dependencies: python -m pip install -r requirements.txt")

    output = ROOT / "dist" / args.platform
    bundle = output / "FiRec"
    if any(path.suffix.lower() in {".sqlite", ".sqlite3"} for path in bundle.rglob("*") if path.is_file()):
        parser.error("The existing FiRec bundle contains a database. Move it outside the bundle before building.")

    work = ROOT / "build" / args.platform
    work.mkdir(parents=True, exist_ok=True)
    source = ROOT / "src"
    assets = source / "firec" / "gui" / "assets"
    command = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--onedir", "--windowed", "--name", "FiRec",
        "--paths", str(source),
        "--distpath", str(output),
        "--workpath", str(work / "work"),
        "--specpath", str(work),
        "--icon", str(assets / "firec.ico"),
    ]
    for asset in sorted(assets.rglob("*")):
        if asset.is_file() and asset.suffix.lower() in {".png", ".svg", ".ico"}:
            destination = asset.parent.relative_to(source).as_posix()
            command.extend(("--add-data", f"{asset}:{destination}"))
    if args.clean:
        command.append("--clean")
    command.append(str(source / "firec" / "__main__.py"))

    env = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(ROOT / "build" / "pyinstaller-cache"))
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    shutil.copy2(ROOT / "docs" / "FiRec_manual.pdf", output / "FiRec_manual.pdf")
    print(f"Build complete: {bundle / 'FiRec.exe'}")


if __name__ == "__main__":
    main()
