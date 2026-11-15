#!/usr/bin/env python3
"""Whole-repo consistency audit. Run after any restructure or script change.

    python scripts/audit.py

Checks the things that silently invalidate results rather than crashing:
identical image lists across the YOLO/COCO mirrors, absolute dataset paths,
un-subsampled eval splits, bbox conversion accuracy, and stale path
references left by a move.
"""
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = []
WARN = []


def check(label, fn):
    try:
        msg = fn()
        print(f"  \033[32mPASS\033[0m  {label:<42} {msg}")
    except AssertionError as e:
        FAIL.append(label)
        print(f"  \033[31mFAIL\033[0m  {label:<42} {e}")
    except Exception as e:
        WARN.append(label)
        print(f"  \033[33mWARN\033[0m  {label:<42} {type(e).__name__}: {str(e)[:60]}")


def sec(t):
    print(f"\n{'='*78}\n  {t}\n{'='*78}")


def main():
    sec("1. every script parses")

    def parse():
        bad = []
        for f in sorted((ROOT / "scripts").glob("*.py")):
            try:
                ast.parse(f.read_text())
            except SyntaxError as e:
                bad.append(f"{f.name}:{e.lineno}")
        assert not bad, f"syntax errors: {bad}"
        n = len(list((ROOT / "scripts").glob("*.py")))
        return f"{n} scripts"
    check("python syntax", parse)

    def shells():
        bad = []
        for f in list((ROOT / "scripts").rglob("*.sh")) + list((ROOT / "scripts").rglob("*.sub")):
            r = subprocess.run(["bash", "-n", str(f)], capture_output=True)
            if r.returncode:
                bad.append(f.name)
        assert not bad, f"shell syntax errors: {bad}"
        return "train.sub, sync_to_rcac.sh, setup_rfdetr_env.sh"
    check("shell syntax", shells)

    sec("2. dataset configs")

    def yamls():
        out = []
        for y in sorted((ROOT / "configs").glob("*.yaml")):
            if y.name.endswith(".example.yaml"):
                continue  # a template with a placeholder path, not a real config
            txt = y.read_text()
            m = re.search(r"^path:\s*(.+)$", txt, re.M)
            assert m, f"{y.name} has no path:"
            p = Path(m.group(1).strip())
            assert p.is_absolute(), f"{y.name} path is RELATIVE - ultralytics resolves vs CWD"
            assert p.exists(), f"{y.name} path does not exist: {p}"
            out.append(y.name)
        return f"{len(out)} absolute + existing"
    check("yaml path: absolute and real", yamls)

    sec("3. subset integrity (the fairness guarantee)")

    def subsets():
        tags = sorted({re.search(r"subset_(\d+)", d.name).group(1)
                       for d in (ROOT / "data").glob("subset_*") if d.is_dir()})
        assert tags, "no subsets built"
        msgs = []
        for t in tags:
            y = ROOT / f"data/subset_{t}"
            c = ROOT / f"data/subset_{t}_coco"
            for ysplit, csplit in (("train", "train"), ("val", "valid"), ("test", "test")):
                yi = {p.name for p in (y / ysplit / "images").iterdir()}
                j = json.loads((c / csplit / "_annotations.coco.json").read_text())
                ci = {im["file_name"] for im in j["images"]}
                assert yi == ci, (f"subset {t}/{ysplit}: YOLO and COCO image lists DIFFER "
                                  f"(yolo-only {len(yi-ci)}, coco-only {len(ci-yi)})")
            msgs.append(t)
        return f"tags {msgs}: YOLO == COCO image lists"
    check("YOLO/COCO mirrors identical", subsets)

    def evalsplits():
        full = {s: len(list((ROOT / f"data/merged/{s}/images").iterdir()))
                for s in ("val", "test")}
        for d in (ROOT / "data").glob("subset_*"):
            if not d.is_dir() or d.name.endswith("_coco"):
                continue
            for s in ("val", "test"):
                n = len(list((d / s / "images").iterdir()))
                assert n == full[s], f"{d.name}/{s} is {n}, expected FULL {full[s]}"
        return f"val={full['val']} test={full['test']} never subsampled"
    check("eval splits never subsampled", evalsplits)

    def bbox():
        """COCO bbox must round-trip from the YOLO label within a pixel."""
        t = sorted((ROOT / "data").glob("subset_*_coco"))
        assert t, "no coco subset"
        j = json.loads((t[0] / "train/_annotations.coco.json").read_text())
        by = {im["id"]: im for im in j["images"]}
        src = ROOT / "data/merged/train/labels"
        worst, n = 0.0, 0
        for a in j["annotations"][:4000]:
            im = by[a["image_id"]]
            lp = src / (Path(im["file_name"]).stem + ".txt")
            if not lp.exists():
                continue
            W, H = im["width"], im["height"]
            want = []
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                _, cx, cy, bw, bh = (float(x) for x in line.split()[:5])
                want.append((cx * W - bw * W / 2, cy * H - bh * H / 2, bw * W, bh * H))
            if not want:
                continue
            x, y, w, h = a["bbox"]
            d = min(max(abs(x-a2[0]), abs(y-a2[1]), abs(w-a2[2]), abs(h-a2[3])) for a2 in want)
            worst = max(worst, d); n += 1
        assert worst < 1.5, f"bbox conversion off by {worst:.2f}px"
        return f"{n} boxes checked, max error {worst:.3f}px"
    check("YOLO -> COCO bbox accuracy", bbox)

    sec("4. eval instrument")

    def selftest():
        r = subprocess.run([sys.executable, str(ROOT / "scripts/eval_buckets.py"), "--self-test"],
                           capture_output=True, text=True)
        assert "ALL SELF-TESTS PASSED" in r.stdout, "self-tests did not pass"
        return f"{r.stdout.count('PASS')} assertions"
    check("eval_buckets self-tests", selftest)

    def shared():
        src = (ROOT / "scripts/eval_buckets.py").read_text()
        assert "from ultralytics" not in src, \
            "eval_buckets imports ultralytics directly - backend leak, rfdetr env will break"
        assert "backends.get(backend)" in src, "eval_buckets does not dispatch to backends"
        return "model-agnostic; dispatches via backends.py"
    check("eval path has no backend leak", shared)

    sec("5. stale references after the restructure")

    def stale():
        pats = [(r'ROOT / "runs"', "old runs/ dir"),
                ("viz_preds", "deleted viz_preds.py"),
                ("1288", "obsolete 1288 resolution advice"),
                ("% 56", "hardcoded 56 resolution step"),
                ("divisible by 56", "stale: 56 is Base-only, not universal")]
        hits = []
        files = list((ROOT / "scripts").rglob("*.py")) + \
                list((ROOT / "scripts").rglob("*.sub")) + [ROOT / "rfdetr/README.md"]
        for f in files:
            if not f.exists() or f.name == "audit.py":   # it holds the patterns itself
                continue
            txt = f.read_text()
            for pat, why in pats:
                if pat in txt:
                    hits.append(f"{f.name}: {why}")
        assert not hits, "; ".join(hits)
        return "no stale paths or obsolete constants"
    check("no stale references", stale)

    sec("6. results ledger")

    def ledger():
        f = ROOT / "results/runs.csv"
        rows = list(csv.reader(f.open()))
        hdr = rows[0]
        bad = [i for i, r in enumerate(rows[1:], 2) if len(r) != len(hdr)]
        assert not bad, f"rows {bad} have wrong column count"
        return f"{len(rows)-1} runs, {len(hdr)} cols, schema consistent"
    check("runs.csv schema", ledger)

    print()
    if FAIL:
        print(f"\033[31m  {len(FAIL)} FAILED\033[0m: {FAIL}")
    if WARN:
        print(f"\033[33m  {len(WARN)} warnings\033[0m: {WARN}")
    if not FAIL and not WARN:
        print("\033[32m  ALL CHECKS PASSED\033[0m")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
