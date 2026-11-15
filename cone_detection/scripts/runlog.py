"""One append-only CSV of every training/eval run: config in, bucketed recall out.

Used by train.py, eval_buckets.py and sweep.py so that run 1 is already logged
in the same shape as run 40. Stdlib only.

    from runlog import log_run
    log_run(run="n_640_baseline", model="yolo26n", imgsz=640, epochs=100,
            recall_10_25=0.41, ...)

Unknown keys are appended as new columns; the file is rewritten with the union
of columns so old rows stay aligned.
"""
import csv
import datetime as _dt
import os
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "results" / "runs.csv"

COLUMNS = [
    "timestamp", "run", "notes",
    # --- config
    "model", "weights", "imgsz", "epochs", "batch", "close_mosaic",
    "fliplr", "hsv_h", "hsv_s", "hsv_v", "mosaic", "scale", "device", "dataset",
    # --- overall metrics (context only - NOT the metric that decides anything)
    "map50", "map50_95", "precision", "recall",
    # --- the numbers that matter: recall by bbox height in POST-LETTERBOX px.
    # Buckets floor at 10px - below FSOCO's ~100 sq px labeling threshold,
    # a missing label does not mean a missing cone, so it is not scored.
    "recall_10_16", "recall_16_32", "recall_32_64", "recall_ge64",
    "n_10_16", "n_16_32", "n_32_64", "n_ge64", "sub_floor_gt",
    "conf_thresh", "iou_thresh",
    # --- deployment
    "engine", "latency_ms_e2e", "latency_ms_infer", "fps", "jetson",
]


def log_run(**kw):
    kw.setdefault("timestamp", _dt.datetime.now().isoformat(timespec="seconds"))
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows, cols = [], list(COLUMNS)
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
            for c in (r.fieldnames or []):
                if c not in cols:
                    cols.append(c)
    for k in kw:
        if k not in cols:
            cols.append(k)
    rows.append({k: kw.get(k, "") for k in cols})

    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    os.replace(tmp, CSV_PATH)
    return CSV_PATH


if __name__ == "__main__":
    print(log_run(run="_selftest", notes="delete me", model="yolo26n", imgsz=640))
