import csv
import sqlite3
from datetime import datetime
from pathlib import Path

from firec.core.analysis import AnalysisResult


ANALYSES_COLUMNS = [
    "created_at",
    "image_path",
    "origin",
    "dpi",
    "laser_center_x_px",
    "laser_center_y_px",
    "radiation_center_x_px",
    "radiation_center_y_px",
    "light_center_x_px",
    "light_center_y_px",
    "radiation_edge_length_x_px",
    "radiation_edge_length_y_px",
    "radiation_area_px2",
    "light_area_px2",
]


def connect_database(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    initialize_database(connection)
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    _drop_outdated_analyses_table(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            origin TEXT NOT NULL,
            dpi REAL NOT NULL DEFAULT 0,
            laser_center_x_px REAL NOT NULL,
            laser_center_y_px REAL NOT NULL,
            radiation_center_x_px REAL NOT NULL,
            radiation_center_y_px REAL NOT NULL,
            light_center_x_px REAL NOT NULL,
            light_center_y_px REAL NOT NULL,
            radiation_edge_length_x_px REAL NOT NULL,
            radiation_edge_length_y_px REAL NOT NULL,
            radiation_area_px2 REAL NOT NULL,
            light_area_px2 REAL NOT NULL
        )
        """
    )
    connection.commit()


def save_analysis(
    connection: sqlite3.Connection,
    image_path: str | Path,
    result: AnalysisResult,
    origin: str,
    dpi: float,
) -> None:
    connection.execute(
        """
        INSERT INTO analyses (
            created_at,
            image_path,
            origin,
            dpi,
            laser_center_x_px,
            laser_center_y_px,
            radiation_center_x_px,
            radiation_center_y_px,
            light_center_x_px,
            light_center_y_px,
            radiation_edge_length_x_px,
            radiation_edge_length_y_px,
            radiation_area_px2,
            light_area_px2
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            Path(image_path).name,
            origin,
            _round1(dpi),
            _round1(result.laser_center.x if result.laser_center is not None else 0.0),
            _round1(result.laser_center.y if result.laser_center is not None else 0.0),
            _round1(result.radiation_field.center.x),
            _round1(result.radiation_field.center.y),
            _round1(result.light_field.center.x),
            _round1(result.light_field.center.y),
            _round1(result.radiation_field.area_length_x),
            _round1(result.radiation_field.area_length_y),
            _round1(result.radiation_field.area),
            _round1(result.light_field.area),
        ),
    )
    connection.commit()


def fetch_analysis_rows(connection: sqlite3.Connection) -> list[dict[str, object]]:
    cursor = connection.execute(
        """
        SELECT
            id,
            created_at,
            image_path,
            origin,
            dpi,
            laser_center_x_px,
            laser_center_y_px,
            radiation_center_x_px,
            radiation_center_y_px,
            light_center_x_px,
            light_center_y_px,
            radiation_edge_length_x_px,
            radiation_edge_length_y_px,
            radiation_area_px2,
            light_area_px2
        FROM analyses
        ORDER BY id
        """
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def update_analysis_record(connection: sqlite3.Connection, analysis_id: int, origin: str, dpi: float) -> None:
    connection.execute(
        """
        UPDATE analyses
        SET origin = ?, dpi = ?
        WHERE id = ?
        """,
        (origin, _round1(dpi), analysis_id),
    )
    connection.commit()


def delete_analysis(connection: sqlite3.Connection, analysis_id: int) -> None:
    connection.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    connection.commit()


def export_rows_to_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    if not rows:
        return

    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _drop_outdated_analyses_table(connection: sqlite3.Connection) -> None:
    table = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'analyses'"
    ).fetchone()
    if table is None:
        return
    existing = {row[1] for row in connection.execute("PRAGMA table_info(analyses)")}
    expected = set(ANALYSES_COLUMNS)
    if not expected.issubset(existing):
        connection.execute("DROP TABLE analyses")


def _round1(value: float) -> float:
    return round(float(value), 1)
