#!/usr/bin/env python3
"""Interactively edit GPS CSV paths.

Usage:
    python gps_csv_editor.py gps_paths/gp_raceline_gps.csv
    python gps_csv_editor.py gps_paths/gp_raceline_gps.csv --output edited.csv
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button


REPO_ROOT = Path(__file__).resolve().parent
GP_BBOX = [-86.94512413996848, -86.94340953701301, 40.43720425898327, 40.43850926028827]
DEFAULT_GP_MAP_NAMES = ["gp_map.png", "gp_map.jpg", "gp_map.jpeg"]


def normalize_header(name):
    return name.strip().lstrip("#").strip().lower()


def find_column(header, candidates):
    normalized = [normalize_header(col) for col in header]
    for candidate in candidates:
        if candidate in normalized:
            return normalized.index(candidate)
    return None


def find_coordinate_columns(header):
    y_idx = find_column(header, {"lat", "latitude"})
    x_idx = find_column(header, {"lon", "lng", "longitude"})
    if y_idx is not None and x_idx is not None:
        return y_idx, x_idx, "Latitude", "Longitude"

    y_idx = find_column(header, {"y", "y_m"})
    x_idx = find_column(header, {"x", "x_m"})
    if y_idx is not None and x_idx is not None:
        return y_idx, x_idx, "Y", "X"

    raise ValueError(
        "Could not find latitude/longitude or x/y columns in CSV header: "
        f"{', '.join(header)}"
    )


def format_coord(value):
    text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text if text else "0"


def load_path(csv_path):
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"{csv_path} must contain a header and at least one point")

    header = rows[0]
    data_rows = [row for row in rows[1:] if row]
    lat_idx, lon_idx, y_label, x_label = find_coordinate_columns(header)

    for row_number, row in enumerate(data_rows, start=2):
        if len(row) <= max(lat_idx, lon_idx):
            raise ValueError(f"Row {row_number} does not include editable coordinate columns")

    lats = np.array([float(row[lat_idx]) for row in data_rows], dtype=float)
    lons = np.array([float(row[lon_idx]) for row in data_rows], dtype=float)
    return header, data_rows, lat_idx, lon_idx, lats, lons, y_label, x_label


def write_path(csv_path, header, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class GpsCsvEditor:
    def __init__(self, args):
        self.input_path = Path(args.csv_file)
        self.output_path = Path(args.output) if args.output else self.input_path
        self.backup = args.backup and self.output_path == self.input_path
        self.backup_written = False
        self.pick_radius = args.pick_radius
        (
            self.header,
            self.rows,
            self.lat_idx,
            self.lon_idx,
            self.lats,
            self.lons,
            self.y_label,
            self.x_label,
        ) = load_path(self.input_path)
        self.undo_stack = []
        self.selected_idx = None
        self.drag_started = False
        self.buttons = {}

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.16)
        self.setup_plot(args)
        self.setup_buttons()
        self.connect_events()

    def setup_plot(self, args):
        map_path = self.resolve_map_path(args)
        if map_path:
            bbox = self.parse_bbox(args.bbox) if args.bbox else GP_BBOX
            image = plt.imread(map_path)
            self.ax.imshow(image, extent=bbox, aspect="equal", zorder=0)

        (self.line,) = self.ax.plot(
            self.lons,
            self.lats,
            "-",
            color="tab:blue",
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        self.points = self.ax.scatter(
            self.lons,
            self.lats,
            s=args.marker_size,
            color="tab:orange",
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        (self.selected_marker,) = self.ax.plot(
            [],
            [],
            "o",
            color="red",
            markersize=10,
            fillstyle="none",
            markeredgewidth=2,
            zorder=4,
        )
        self.status = self.ax.text(
            0.01,
            0.99,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        self.ax.set_title(f"GPS CSV Editor: {self.input_path}")
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)
        if map_path:
            self.ax.set_xlim(bbox[0], bbox[1])
            self.ax.set_ylim(bbox[2], bbox[3])
        self.update_status(
            "Drag points. Click Save, Reverse, Undo, or use s/r/z/q keys."
        )
        self.update_artists()

    def setup_buttons(self):
        button_specs = [
            ("save", "Save", 0.12, self.save),
            ("reverse", "Reverse", 0.25, self.reverse),
            ("undo", "Undo", 0.41, self.restore_undo),
            ("quit", "Quit", 0.54, self.close),
        ]
        for key, label, left, callback in button_specs:
            button_ax = self.fig.add_axes([left, 0.04, 0.11, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons[key] = button

    def resolve_map_path(self, args):
        if args.map:
            map_path = Path(args.map).expanduser()
            if map_path.exists():
                return map_path
            search_paths = [
                Path.cwd() / map_path,
                REPO_ROOT / map_path,
                REPO_ROOT / "gp_helper_code" / map_path,
            ]
            for candidate in search_paths:
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"Could not find map image: {args.map}")

        if not args.gp_map:
            return None

        search_dirs = [Path.cwd(), REPO_ROOT, REPO_ROOT / "gp_helper_code"]
        for search_dir in search_dirs:
            for name in DEFAULT_GP_MAP_NAMES:
                candidate = search_dir / name
                if candidate.exists():
                    return candidate

        searched = ", ".join(DEFAULT_GP_MAP_NAMES)
        raise FileNotFoundError(
            f"--gp-map requested, but none of these files were found: {searched}"
        )

    def parse_bbox(self, bbox_text):
        if bbox_text.lower() == "gp":
            return GP_BBOX
        values = [float(item.strip()) for item in bbox_text.split(",")]
        if len(values) != 4:
            raise ValueError("--bbox must be 'gp' or min_lon,max_lon,min_lat,max_lat")
        return values

    def path_bbox(self):
        lon_pad = max((self.lons.max() - self.lons.min()) * 0.05, 0.00001)
        lat_pad = max((self.lats.max() - self.lats.min()) * 0.05, 0.00001)
        return [
            self.lons.min() - lon_pad,
            self.lons.max() + lon_pad,
            self.lats.min() - lat_pad,
            self.lats.max() + lat_pad,
        ]

    def connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)
        canvas.mpl_connect("key_press_event", self.on_key)

    def push_undo(self):
        self.undo_stack.append(
            (
                [row[:] for row in self.rows],
                self.lats.copy(),
                self.lons.copy(),
            )
        )

    def restore_undo(self, event=None):
        if not self.undo_stack:
            self.update_status("Nothing to undo.")
            return
        self.rows, self.lats, self.lons = self.undo_stack.pop()
        self.selected_idx = None
        self.update_artists()
        self.update_status("Undid last edit.")

    def nearest_point(self, event):
        if event.x is None or event.y is None:
            return None
        xy_pixels = self.ax.transData.transform(np.column_stack((self.lons, self.lats)))
        distances = np.hypot(xy_pixels[:, 0] - event.x, xy_pixels[:, 1] - event.y)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self.pick_radius:
            return nearest
        return None

    def on_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        nearest = self.nearest_point(event)
        if nearest is None:
            self.selected_idx = None
            self.update_artists()
            return
        self.push_undo()
        self.selected_idx = nearest
        self.drag_started = True
        self.update_status(f"Moving point {nearest}.")
        self.update_artists()

    def on_motion(self, event):
        if self.selected_idx is None or not self.drag_started or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.lons[self.selected_idx] = event.xdata
        self.lats[self.selected_idx] = event.ydata
        self.rows[self.selected_idx][self.lon_idx] = format_coord(event.xdata)
        self.rows[self.selected_idx][self.lat_idx] = format_coord(event.ydata)
        self.update_artists(draw_idle=True)

    def on_release(self, event):
        if event.button == 1:
            self.drag_started = False

    def on_key(self, event):
        if event.key in {"s", "S"}:
            self.save()
        elif event.key in {"r", "R"}:
            self.reverse()
        elif event.key in {"z", "ctrl+z", "cmd+z"}:
            self.restore_undo()
        elif event.key in {"escape"}:
            self.selected_idx = None
            self.update_artists()
        elif event.key in {"q", "Q"}:
            self.close()

    def reverse(self, event=None):
        self.push_undo()
        self.rows.reverse()
        self.lats = self.lats[::-1].copy()
        self.lons = self.lons[::-1].copy()
        self.selected_idx = None
        self.update_artists()
        self.update_status("Reversed path direction. Click Save or press s.")

    def save(self, event=None):
        if self.backup and not self.backup_written and self.input_path.exists():
            backup_path = self.input_path.with_suffix(self.input_path.suffix + ".bak")
            shutil.copy2(self.input_path, backup_path)
            self.backup_written = True
        write_path(self.output_path, self.header, self.rows)
        self.update_status(f"Saved {self.output_path}.")
        print(f"Saved {self.output_path}")

    def close(self, event=None):
        plt.close(self.fig)

    def update_status(self, message):
        target = self.output_path
        backup_text = " backup enabled" if self.backup else ""
        self.status.set_text(f"{message}\nOutput: {target}{backup_text}")
        self.fig.canvas.draw_idle()

    def update_artists(self, draw_idle=False):
        self.line.set_data(self.lons, self.lats)
        self.points.set_offsets(np.column_stack((self.lons, self.lats)))
        if self.selected_idx is None:
            self.selected_marker.set_data([], [])
        else:
            self.selected_marker.set_data(
                [self.lons[self.selected_idx]], [self.lats[self.selected_idx]]
            )
        if draw_idle:
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactively drag GPS path points, reverse direction, and save a CSV."
    )
    parser.add_argument(
        "csv_file", help="CSV file with latitude/longitude or x/y columns"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to overwriting the input file on save.",
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Do not create input.csv.bak when saving over the input file.",
    )
    parser.add_argument(
        "--map",
        help="Optional background image to draw under the path. Defaults to the GP bbox.",
    )
    parser.add_argument(
        "--bbox",
        help="Map extent for --map as 'gp' or min_lon,max_lon,min_lat,max_lat.",
    )
    parser.add_argument(
        "--gp-map",
        action="store_true",
        help="Use gp_map.png/jpg with the Grand Prix bbox from gp_helper_code/viz.py.",
    )
    parser.add_argument(
        "--pick-radius",
        type=float,
        default=12.0,
        help="Point selection radius in screen pixels. Default: 12.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=35.0,
        help="Point marker size. Default: 35.",
    )
    parser.set_defaults(backup=True)
    return parser.parse_args()


def main():
    args = parse_args()
    editor = GpsCsvEditor(args)
    editor.show()


if __name__ == "__main__":
    main()
