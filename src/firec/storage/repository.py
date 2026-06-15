import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from firec.core.analysis import AnalysisResult


def connect_database(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    initialize_database(connection)
    return connection


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_path TEXT NOT NULL,
            radiation_center_x REAL NOT NULL,
            radiation_center_y REAL NOT NULL,
            radiation_width REAL NOT NULL,
            radiation_height REAL NOT NULL,
            radiation_angle REAL NOT NULL DEFAULT 0,
            radiation_edge_lengths TEXT NOT NULL DEFAULT '[]',
            radiation_points TEXT NOT NULL,
            light_center_x REAL NOT NULL,
            light_center_y REAL NOT NULL,
            light_width REAL NOT NULL,
            light_height REAL NOT NULL,
            light_angle REAL NOT NULL DEFAULT 0,
            light_edge_lengths TEXT NOT NULL DEFAULT '[]',
            light_points TEXT NOT NULL,
            width_difference REAL NOT NULL,
            height_difference REAL NOT NULL,
            width_ratio REAL NOT NULL DEFAULT 0,
            height_ratio REAL NOT NULL DEFAULT 0,
            center_dx REAL NOT NULL DEFAULT 0,
            center_dy REAL NOT NULL DEFAULT 0
        )
        """
    )
    _ensure_columns(connection)
    connection.commit()


def save_analysis(connection: sqlite3.Connection, image_path: str | Path, result: AnalysisResult) -> None:
    connection.execute(
        """
        INSERT INTO analyses (
            created_at,
            image_path,
            radiation_center_x,
            radiation_center_y,
            radiation_width,
            radiation_height,
            radiation_angle,
            radiation_edge_lengths,
            radiation_points,
            light_center_x,
            light_center_y,
            light_width,
            light_height,
            light_angle,
            light_edge_lengths,
            light_points,
            width_difference,
            height_difference,
            width_ratio,
            height_ratio,
            center_dx,
            center_dy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            Path(image_path).name,
            result.radiation_field.center.x,
            result.radiation_field.center.y,
            result.radiation_field.width,
            result.radiation_field.height,
            result.radiation_field.angle,
            json.dumps(result.radiation_field.edge_lengths, ensure_ascii=True),
            _points_to_json(result.radiation_field.points),
            result.light_field.center.x,
            result.light_field.center.y,
            result.light_field.width,
            result.light_field.height,
            result.light_field.angle,
            json.dumps(result.light_field.edge_lengths, ensure_ascii=True),
            _points_to_json(result.light_field.points),
            result.width_difference,
            result.height_difference,
            result.width_ratio,
            result.height_ratio,
            result.center_dx,
            result.center_dy,
        ),
    )
    connection.commit()


def fetch_analysis_rows(connection: sqlite3.Connection) -> list[dict[str, object]]:
    cursor = connection.execute(
        """
        SELECT
            created_at,
            image_path,
            radiation_center_x,
            radiation_center_y,
            radiation_width,
            radiation_height,
            radiation_angle,
            radiation_edge_lengths,
            radiation_points,
            light_center_x,
            light_center_y,
            light_width,
            light_height,
            light_angle,
            light_edge_lengths,
            light_points,
            width_difference,
            height_difference,
            width_ratio,
            height_ratio,
            center_dx,
            center_dy
        FROM analyses
        ORDER BY created_at
        """
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def export_rows_to_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    if not rows:
        return

    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _points_to_json(points) -> str:
    return json.dumps([{"x": point.x, "y": point.y} for point in points], ensure_ascii=True)


def _ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {row[1] for row in connection.execute("PRAGMA table_info(analyses)")}
    columns = {
        "radiation_angle": "REAL NOT NULL DEFAULT 0",
        "radiation_edge_lengths": "TEXT NOT NULL DEFAULT '[]'",
        "light_angle": "REAL NOT NULL DEFAULT 0",
        "light_edge_lengths": "TEXT NOT NULL DEFAULT '[]'",
        "width_ratio": "REAL NOT NULL DEFAULT 0",
        "height_ratio": "REAL NOT NULL DEFAULT 0",
        "center_dx": "REAL NOT NULL DEFAULT 0",
        "center_dy": "REAL NOT NULL DEFAULT 0",
    }
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE analyses ADD COLUMN {name} {definition}")
