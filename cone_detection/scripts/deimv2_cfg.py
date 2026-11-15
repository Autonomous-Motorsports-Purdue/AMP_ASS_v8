#!/usr/bin/env python3
"""Build a DEIMv2 run config from our COCO mirror, by OVERRIDING an upstream one.

DEIMv2 is a repo, not a pip package: training is `train.py -c <yaml>` inside a
clone. So unlike rf-detr, the model-specific knobs live in a YAML we generate
rather than in kwargs we pass. This module is that generator, shared by
train_deimv2.py (writes it before training) and backends.py (reads it back at
eval time, so the model is rebuilt exactly as it was trained).

The generated file does NOT restate the upstream config. It `__include__`s it
by absolute path and overrides only what has to change:

    dataset paths, num_classes, output_dir, resolution, epoch schedule

Everything else - architecture, optimizer, loss, matcher - stays whatever the
upstream variant config says, so a DEIMv2 release upgrade is a `git pull`.

Two things this gets right that hand-editing gets wrong:

  * `num_classes` is READ from the COCO file, not assumed. Our mirror follows
    Roboflow's convention (id 0 is a dummy supercategory, id 1 is `cone`) and
    DEIMv2 with `remap_mscoco_category: False` uses category_id AS the label -
    so the label is 1 and num_classes must be 2. Setting 1 trains the wrong
    head width and every box is background.
  * The aug/LR schedule is SCALED, not truncated. Upstream tunes `flat_epoch`,
    `no_aug_epoch`, the transform `policy.epoch` list and the collate
    `stop_epoch`/`mixup_epochs`/`copyblend_epochs` against its own `epoches`.
    Asking for 20 epochs while those still say 64/120 means the no-aug tail
    and the matcher change never happen at all.
"""
import copy
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# arch -> upstream config stem. Both families ship an `s`, `m`, `l` and `x`,
# so those MUST be spelled with the backbone; the ultra-lights are unambiguous.
VARIANTS = {
    "hgnetv2_atto": "deimv2_hgnetv2_atto_coco",
    "hgnetv2_femto": "deimv2_hgnetv2_femto_coco",
    "hgnetv2_pico": "deimv2_hgnetv2_pico_coco",
    "hgnetv2_n": "deimv2_hgnetv2_n_coco",
    "hgnetv2_s": "deimv2_hgnetv2_s_coco",
    "hgnetv2_m": "deimv2_hgnetv2_m_coco",
    "hgnetv2_l": "deimv2_hgnetv2_l_coco",
    "hgnetv2_x": "deimv2_hgnetv2_x_coco",
    "dinov3_s": "deimv2_dinov3_s_coco",
    "dinov3_m": "deimv2_dinov3_m_coco",
    "dinov3_l": "deimv2_dinov3_l_coco",
    "dinov3_x": "deimv2_dinov3_x_coco",
}
ALIASES = {"atto": "hgnetv2_atto", "femto": "hgnetv2_femto",
           "pico": "hgnetv2_pico", "n": "hgnetv2_n"}
AMBIGUOUS = {"s", "m", "l", "x"}

# Variants whose backbone weights are NOT auto-downloaded (see deimv2/README.md).
NEEDS_CKPT = {"dinov3_s": "ckpts/vitt_distill.pt",
              "dinov3_m": "ckpts/vittplus_distill.pt"}

# The upstream eval/inference transform normalises only for the ViT backbones
# (tools/inference/torch_inf.py). Getting this wrong at eval time silently
# costs recall, so backends.py reads it from here.
VIT_BACKBONE = {"dinov3_s", "dinov3_m", "dinov3_l", "dinov3_x"}

# Every DEIMv2 backbone is satisfied by a multiple of 32: HGNetv2 downsamples
# 32x, the ViT-Tiny backbones use patch 16. Unlike rf-detr there is no
# per-variant window product, so 1280 is valid for ALL of them - the same
# resolution as the yolo26n baseline and RFDETRNano/Small.
RESOLUTION_STEP = 32


def deimv2_home():
    """The clone. $DEIMV2_HOME wins, so a cluster can put it on scratch."""
    return Path(os.environ.get("DEIMV2_HOME", ROOT / "deimv2/DEIMv2")).resolve()


def normalize_arch(arch):
    a = arch.strip().lower().removeprefix("deimv2_").removesuffix("_coco")
    if a in AMBIGUOUS:
        raise SystemExit(
            f"[deimv2] --arch {arch!r} is ambiguous: DEIMv2 ships both "
            f"hgnetv2_{a} and dinov3_{a}. Spell the backbone, "
            f"e.g. --arch dinov3_{a}.")
    a = ALIASES.get(a, a)
    if a not in VARIANTS:
        raise SystemExit(f"[deimv2] unknown --arch {arch!r}; "
                         f"choose from {sorted(VARIANTS)} "
                         f"(or the aliases {sorted(ALIASES)})")
    return a


def upstream_config(arch):
    p = deimv2_home() / "configs/deimv2" / f"{VARIANTS[normalize_arch(arch)]}.yml"
    if not p.exists():
        raise SystemExit(
            f"[deimv2] upstream config not found: {p}\n"
            f"         clone the repo first: bash scripts/setup_deimv2_env.sh\n"
            f"         (or point $DEIMV2_HOME at an existing clone)")
    return p


# --- the upstream loader, reimplemented so we can READ resolved defaults -----
# Mirrors engine/core/yaml_utils.load_config: includes are merged first, the
# including file last, so the file wins. We only read with this; the run itself
# still goes through DEIMv2's own loader.
def _merge(dct, other):
    for k in other:
        if k in dct and isinstance(dct[k], dict) and isinstance(other[k], dict):
            _merge(dct[k], other[k])
        else:
            dct[k] = other[k]
    return dct


def resolve(path, cfg=None):
    cfg = {} if cfg is None else cfg
    path = Path(path)
    own = yaml.load(path.read_text(), Loader=yaml.Loader) or {}
    for inc in own.get("__include__", []):
        inc = Path(os.path.expanduser(inc))
        if not inc.is_absolute():
            inc = path.parent / inc
        _merge(cfg, resolve(inc, cfg))
    return _merge(cfg, own)


# --- dataset ----------------------------------------------------------------
def coco_num_classes(ds):
    """max(category_id) + 1 over the COCO file. See the module docstring: with
    remap_mscoco_category False, category_id IS the label."""
    ann = Path(ds) / "train/_annotations.coco.json"
    if not ann.exists():
        raise SystemExit(f"[deimv2] no COCO annotations at {ann}\n"
                         f"         run: python scripts/yolo_to_coco.py")
    cats = json.loads(ann.read_text()).get("categories", [])
    if not cats:
        raise SystemExit(f"[deimv2] {ann} has no categories")
    return max(int(c["id"]) for c in cats) + 1


def _split_dir(ds, name):
    # ABSOLUTE, always: training runs with cwd set to the DEIMv2 clone (its
    # configs reference ./ckpts/ relatively and engine.* only imports from the
    # repo root), so a relative dataset path would resolve against the clone.
    d = (Path(ds).resolve()) / name
    ann = d / "_annotations.coco.json"
    if not ann.exists():
        raise SystemExit(f"[deimv2] missing {ann}\n"
                         f"         run: python scripts/yolo_to_coco.py")
    return str(d), str(ann)


# --- schedule ---------------------------------------------------------------
# Every key upstream expresses in epochs. Scaling all of them by the same
# factor keeps a shortened run shaped like the tuned one instead of one whose
# augmentation never turns off.
EPOCH_KEYS = [
    ("flat_epoch",),
    ("no_aug_epoch",),
    ("train_dataloader", "dataset", "transforms", "policy", "epoch"),
    ("train_dataloader", "collate_fn", "stop_epoch"),
    ("train_dataloader", "collate_fn", "mixup_epochs"),
    ("train_dataloader", "collate_fn", "copyblend_epochs"),
    ("DEIMCriterion", "matcher", "matcher_change_epoch"),
]


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _plant(d, keys, val):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


def scale_schedule(base, epochs):
    """Return the epoch-keyed overrides for a run of `epochs` epochs."""
    src = base.get("epoches")
    out = {}
    if not src or not epochs or epochs == src:
        return out, src, []
    f = epochs / src
    notes = []

    def sc(v):
        return max(1, min(epochs, round(v * f)))

    for keys in EPOCH_KEYS:
        v = _dig(base, keys)
        if v is None:
            continue
        nv = [sc(x) for x in v] if isinstance(v, list) else sc(v)
        _plant(out, keys, nv)
        notes.append(f"{'.'.join(keys)}: {v} -> {nv}")
    return out, src, notes


# --- transforms -------------------------------------------------------------
def _retarget_ops(ops, res, photometric=True):
    """Rewrite an upstream ops list for a new input size.

    Read-and-patch, not restate: the list carries Mosaic/IoUCrop/Sanitize in a
    tuned order, and a hand-written replacement drifts every DEIMv2 release.
    """
    out = []
    for op in copy.deepcopy(ops or []):
        t = op.get("type") if isinstance(op, dict) else None
        if t == "Resize":
            op["size"] = [res, res]
        elif t == "Mosaic":
            # upstream: output_size 320 at 640, i.e. half the input
            op["output_size"] = max(32, res // 2)
        elif t == "RandomPhotometricDistort" and not photometric:
            continue
        out.append(op)
    return out


def build(arch, dataset, resolution=640, epochs=0, output_dir=".",
          batch=0, val_batch=0, workers=4, photometric=True, seed=0):
    """The override dict that becomes the run's config.yml."""
    arch = normalize_arch(arch)
    up = upstream_config(arch)
    base = resolve(up)

    if resolution % RESOLUTION_STEP:
        near = max(RESOLUTION_STEP,
                   round(resolution / RESOLUTION_STEP) * RESOLUTION_STEP)
        raise SystemExit(
            f"[deimv2] --resolution must be divisible by {RESOLUTION_STEP}; "
            f"{resolution} is not. Nearest valid: {near}.")

    tr_dir, tr_ann = _split_dir(dataset, "train")
    va_dir, va_ann = _split_dir(dataset, "valid")

    tops = _dig(base, ("train_dataloader", "dataset", "transforms", "ops"))
    vops = _dig(base, ("val_dataloader", "dataset", "transforms", "ops"))
    policy = _dig(base, ("train_dataloader", "dataset", "transforms", "policy"))

    cfg = {
        "__include__": [str(up)],
        "output_dir": str(Path(output_dir).resolve()),   # cwd is the clone
        "seed": seed,
        "num_classes": coco_num_classes(dataset),
        "remap_mscoco_category": False,
        "eval_spatial_size": [resolution, resolution],
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": tr_dir,
                "ann_file": tr_ann,
                "return_masks": False,
                "transforms": {"ops": _retarget_ops(tops, resolution, photometric)},
            },
            "collate_fn": {"base_size": resolution},
            "num_workers": workers,
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": va_dir,
                "ann_file": va_ann,
                "return_masks": False,
                "transforms": {"ops": _retarget_ops(vops, resolution, True)},
            },
            "num_workers": workers,
        },
    }
    if batch:
        cfg["train_dataloader"]["total_batch_size"] = batch
    if val_batch:
        cfg["val_dataloader"]["total_batch_size"] = val_batch
    if not photometric and policy and "ops" in policy:
        cfg["train_dataloader"]["dataset"]["transforms"]["policy"] = {
            "ops": [o for o in policy["ops"] if o != "RandomPhotometricDistort"]}

    sched, src_epochs, notes = scale_schedule(base, epochs)
    if epochs:
        cfg["epoches"] = epochs
    _merge(cfg, sched)
    return cfg, {"arch": arch, "upstream": str(up), "base_epoches": src_epochs,
                 "schedule_notes": notes, "needs_ckpt": NEEDS_CKPT.get(arch)}


HEADER = """\
# GENERATED by scripts/train_deimv2.py - do not hand-edit.
# Regenerate instead; this file is the record of how the run was configured,
# and scripts/backends.py reads it back to rebuild the model for eval.
#
# It __include__s the upstream variant config by ABSOLUTE path (DEIMv2's loader
# supports that) and overrides only dataset, classes, resolution and schedule.
"""


def write(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump(cfg, sort_keys=False, width=120))
    return path
