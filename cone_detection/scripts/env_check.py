#!/usr/bin/env python3
"""Readiness check - run this before task 4, and any time the env feels off.

    conda activate cone && python scripts/env_check.py

Verifies every dependency the plan actually leans on, plus the two config
footguns that cost us a debugging cycle already (relative dataset `path:` and
the ultralytics runs_dir pointing outside the project).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ok = True


def check(label, fn):
    global ok
    try:
        msg = fn()
        print(f"  \033[32mPASS\033[0m  {label:<34} {msg}")
    except Exception as e:
        ok = False
        print(f"  \033[31mFAIL\033[0m  {label:<34} {type(e).__name__}: {str(e)[:110]}")


def main():
    print(f"\n{'='*72}\n  environment readiness\n{'='*72}")

    def py():
        v = sys.version_info
        assert v[:2] != (3, 14), "ultralytics/torch have no 3.14 wheels - use the `cone` env"
        assert (3, 9) <= v[:2] <= (3, 13), f"untested python {v.major}.{v.minor}"
        return f"{v.major}.{v.minor}.{v.micro}  ({sys.prefix.split('/')[-1]} env)"
    check("python", py)

    def torch_():
        import torch
        dev = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        assert dev != "cpu", "no GPU backend - training will be unusably slow"
        return f"{torch.__version__}   device={dev}"
    check("torch + accelerator", torch_)

    def ul():
        import ultralytics
        return ultralytics.__version__
    check("ultralytics", ul)

    def models():
        from ultralytics import YOLO
        got = []
        for name in ("yolo26n.pt", "yolo11n.pt"):
            p = ROOT / "weights" / name
            assert p.exists(), f"missing {p} (YOLO('{name}') will fetch it)"
            got.append(name.replace(".pt", ""))
        return " + ".join(got) + "  (cached in weights/)"
    check("pretrained weights", models)

    def p2():
        from ultralytics import YOLO
        m = YOLO("yolo26n-p2.yaml")
        n = sum(x.numel() for x in m.model.parameters())
        return f"yolo26n-p2 builds, {n/1e6:.2f}M params  (task 7 sweep)"
    check("p2 variant", p2)

    def ih():
        import imagehash
        from PIL import Image
        Image.new("RGB", (64, 64))
        return "imagehash + pillow  (task 3 dedupe)"
    check("dedupe deps", ih)

    def onnx():
        import onnx, onnxruntime
        return f"onnx {onnx.__version__}  (task 6 export; TRT builds on the Jetson)"
    check("export deps", onnx)

    def settings():
        from ultralytics.utils import SETTINGS
        rd = Path(SETTINGS["runs_dir"])
        assert ROOT in rd.parents or rd == ROOT / "yolo26n/runs", \
            f"runs_dir is {rd}, outside the project - runs will scatter"
        return str(rd)
    check("ultralytics runs_dir", settings)

    def cfg():
        p = ROOT / "configs" / "cone.yaml"
        assert p.exists(), "configs/cone.yaml missing"
        line = next(l for l in p.read_text().splitlines()
                    if l.startswith("path:"))
        val = line.split(":", 1)[1].strip()
        assert Path(val).is_absolute(), \
            f"path: {val!r} is relative - ultralytics resolves it against CWD, not this file"
        return f"path is absolute -> {val}"
    check("configs/cone.yaml", cfg)

    def data():
        # FSOCO arrives as Supervisely at ~/fsoco_raw and is converted into
        # data/interim/fsoco; data/raw/fsoco is intentionally empty.
        f = ROOT / "data/interim/fsoco/images"
        a = ROOT / "data/raw/aks"
        n_f = sum(1 for _ in f.glob("*")) if f.is_dir() else 0
        n_a = sum(1 for _ in a.rglob("*") if _.suffix.lower() in {".jpg", ".jpeg", ".png"})
        assert n_f or n_a, "no source images - convert FSOCO and unzip AKS first"
        return f"fsoco(converted) {n_f:,} imgs, aks {n_a:,} imgs"
    check("source datasets", data)

    def merged():
        m = ROOT / "data/merged"
        assert m.is_dir(), "data/merged missing - run scripts/merge.py"
        parts = {}
        for s in ("train", "val", "test"):
            ni = len(list((m / s / "images").glob("*")))
            nl = len(list((m / s / "labels").glob("*.txt")))
            assert ni == nl, f"{s}: {ni} images vs {nl} labels"
            assert ni, f"{s} split is empty"
            parts[s] = ni
        return "  ".join(f"{k} {v:,}" for k, v in parts.items())
    check("merged dataset", merged)

    print(f"\n{'='*72}")
    if ok:
        print("  ALL GREEN\n")
    else:
        print("  ^ fix the FAIL lines above.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
