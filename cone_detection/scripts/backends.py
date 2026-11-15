#!/usr/bin/env python3
"""Prediction backends - the only model-specific code in the eval path.

Every backend takes the item list from eval_buckets.load_split() and fills
`pr_boxes` (N,4 xyxy in SOURCE pixels) and `pr_conf` (N,). Everything
downstream - bucketing, IoU matching, the confidence sweep, runs.csv - is
shared, so every model is measured by the identical instrument.

SOURCE pixels is the contract. Every backend must undo its own letterbox /
resize before returning, or the comparison is meaningless.

Backends live in different conda envs on purpose (rfdetr and deimv2 each pin
their own torch); the import is therefore lazy and inside the function.
"""
import sys
from pathlib import Path

import numpy as np

EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


def resolution_step(arch="RFDETRBase"):
    """RF-DETR input size must be divisible by patch_size * num_windows, and
    that product is PER VARIANT - it is NOT always 56:

        Nano/Small/Medium/Large  16 * 2 = 32   (same as YOLO -> 1280 is valid)
        Base                     14 * 4 = 56   (1280 is NOT valid)

    Read it from the variant's config rather than hardcoding, so a new variant
    cannot silently break the parity check.
    """
    try:
        import rfdetr.config as C
        cls = getattr(C, f"{arch}Config", None)
        if cls is not None:
            d = {k: v.default for k, v in cls.model_fields.items()}
            ps, nw = d.get("patch_size"), d.get("num_windows")
            if isinstance(ps, int) and isinstance(nw, int):
                return ps * nw
    except ImportError:
        pass   # called from the `cone` env, where rfdetr is not installed
    except Exception:
        pass
    # Fallback table, read off rfdetr 1.10.1 configs. Used only when rfdetr is
    # not importable (e.g. a dry-run from the `cone` env); guessing 56 here
    # wrongly rejects 1280 for every variant except Base.
    return {"RFDETRNano": 32, "RFDETRSmall": 32, "RFDETRMedium": 32,
            "RFDETRLarge": 32, "RFDETRBase": 56}.get(arch, 56)


def predict_yolo(items, weights, imgsz, min_conf, device, batch, arch="",
                 model_config=""):
    from ultralytics import YOLO
    m = YOLO(weights)
    # Feed ultralytics one batch at a time, never the whole list with
    # stream=True. On MPS, with mixed source resolutions, the streamed form
    # tripped "NMS time limit exceeded" and returned EMPTY results for 49 of
    # 50 val images (bucketed recall 0.02 vs 0.98 chunked) - silently, since an
    # empty result is a legal answer. Zipping each chunk with its results also
    # makes a short result list fail loudly instead of shifting every index.
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        rs = m.predict([str(it["img"]) for it in chunk], imgsz=imgsz,
                       conf=min_conf, device=device, batch=batch,
                       verbose=False, max_det=1000)
        if len(rs) != len(chunk):
            raise RuntimeError(f"{len(rs)} results for {len(chunk)} images")
        for it, r in zip(chunk, rs):
            b = r.boxes
            if len(b):
                it["pr_boxes"] = b.xyxy.cpu().numpy().astype(np.float32)
                it["pr_conf"] = b.conf.cpu().numpy().astype(np.float32)
            else:
                it["pr_boxes"], it["pr_conf"] = EMPTY
    return items


def predict_rfdetr(items, weights, imgsz, min_conf, device, batch,
                   arch="RFDETRBase", model_config=""):
    """RF-DETR. No NMS - it is a set predictor, so every returned box is a
    distinct query. That is a real difference from YOLO, not a bug: there is
    no duplicate suppression to tune.
    """
    import rfdetr
    from PIL import Image

    step = resolution_step(arch)
    if imgsz % step:
        near = max(step, round(imgsz / step) * step)
        raise SystemExit(
            f"[rfdetr] {arch} needs a resolution divisible by {step}; "
            f"{imgsz} is not. Nearest valid: {near}.")

    cls = getattr(rfdetr, arch, None)
    if cls is None:
        avail = [n for n in dir(rfdetr) if n.startswith("RFDETR")]
        raise SystemExit(f"[rfdetr] unknown arch {arch!r}; available: {avail}")

    kw = {"resolution": imgsz}
    if weights:
        kw["pretrain_weights"] = weights
    m = cls(**kw)
    if hasattr(m, "optimize_for_inference"):
        try:
            m.optimize_for_inference()
        except Exception:
            pass

    for it in items:
        img = Image.open(it["img"]).convert("RGB")
        det = m.predict(img, threshold=min_conf)
        xyxy = np.asarray(getattr(det, "xyxy", np.zeros((0, 4))), np.float32)
        conf = np.asarray(getattr(det, "confidence", np.zeros(0)), np.float32)
        if len(xyxy) == 0:
            it["pr_boxes"], it["pr_conf"] = EMPTY
            continue
        # Contract check: boxes must already be in this image's pixel space.
        if xyxy[:, 2].max() <= 1.01 and xyxy[:, 3].max() <= 1.01:
            xyxy = xyxy * np.array([it["w"], it["h"], it["w"], it["h"]], np.float32)
        it["pr_boxes"], it["pr_conf"] = xyxy.reshape(-1, 4), conf.reshape(-1)
    return items


def _torch_device(device):
    """eval_buckets speaks ultralytics' dialect ("0", "mps", "cpu")."""
    import torch
    d = str(device)
    if d in ("auto", ""):
        return ("cuda:0" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu")
    return f"cuda:{d}" if d.isdigit() else d


def predict_deimv2(items, weights, imgsz, min_conf, device, batch, arch="",
                   model_config=""):
    """DEIMv2. Like RF-DETR it is a set predictor - no NMS, 300 queries.

    Two things worth knowing about this backend:

    * It needs the RUN'S CONFIG, not just an arch name. DEIMv2 builds the model
      from YAML, and `num_classes` (2 here - see deimv2_cfg) decides the head
      width, so the checkpoint only loads against the config it was trained
      with. train_deimv2.py writes that config next to the checkpoint; this
      looks for it there. Guessing it would load a wrong-shaped head.
    * DEIMv2 resizes to a SQUARE - it stretches, it does not letterbox. Its
      postprocessor already rescales boxes by the original (w, h), so what
      comes back is in SOURCE pixels and the contract holds. But the model saw
      a different effective vertical scale than yolo26n did; that is a
      preprocessing difference, documented in deimv2/README.md.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import deimv2_cfg

    cfg_path = Path(model_config) if model_config else \
        Path(weights).resolve().parent / "config.yml"
    if not cfg_path.exists():
        raise SystemExit(
            f"[deimv2] no run config at {cfg_path}.\n"
            f"         train_deimv2.py writes config.yml next to the "
            f"checkpoint; pass --model-config if it lives elsewhere.")

    home = deimv2_cfg.deimv2_home()
    if not (home / "engine").exists():
        raise SystemExit(f"[deimv2] no DEIMv2 clone at {home}; "
                         f"run scripts/setup_deimv2_env.sh or set $DEIMV2_HOME")
    sys.path.insert(0, str(home))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(cfg_path), resume=str(weights))
    res = cfg.yaml_cfg.get("eval_spatial_size") or [imgsz, imgsz]
    if int(res[0]) != int(imgsz):
        raise SystemExit(
            f"[deimv2] --imgsz {imgsz} but {cfg_path.name} was built for "
            f"{res[0]}. Evaluating at a size the run was not trained for is a "
            f"different experiment; pass --imgsz {res[0]}.")
    # The backbone was already trained; do not re-fetch ImageNet weights.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    dev = _torch_device(device)
    model = Deployed().to(dev).eval()

    # Upstream normalises for the ViT backbones only (tools/inference/torch_inf.py).
    arch_n = deimv2_cfg.normalize_arch(arch) if arch else ""
    ops = [T.Resize((imgsz, imgsz)), T.ToTensor()]
    if arch_n in deimv2_cfg.VIT_BACKBONE:
        ops.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]))
    tf = T.Compose(ops)

    bs = max(1, int(batch))
    with torch.no_grad():
        for i in range(0, len(items), bs):
            chunk = items[i:i + bs]
            ims = torch.stack([tf(Image.open(it["img"]).convert("RGB"))
                               for it in chunk]).to(dev)
            sizes = torch.tensor([[it["w"], it["h"]] for it in chunk],
                                 device=dev)
            labels, boxes, scores = model(ims, sizes)
            for j, it in enumerate(chunk):
                sc = scores[j].float().cpu().numpy()
                bx = boxes[j].float().cpu().numpy().reshape(-1, 4)
                # Single-class task: the label is not used. num_classes is 2
                # only because our COCO ids start the real class at 1.
                keep = sc >= min_conf
                if not keep.any():
                    it["pr_boxes"], it["pr_conf"] = EMPTY
                    continue
                it["pr_boxes"] = bx[keep].astype(np.float32)
                it["pr_conf"] = sc[keep].astype(np.float32)
    return items


BACKENDS = {"yolo": predict_yolo, "rfdetr": predict_rfdetr,
            "deimv2": predict_deimv2}


def get(name):
    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; choose from {list(BACKENDS)}")
    return BACKENDS[name]
