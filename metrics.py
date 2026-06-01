"""
metrics.py — Per-class segmentation skill metrics with DDP support.

Computes IoU, Dice/F1, precision, recall for AR, Hurricane, ETC, MCS from
per-class TP/FP/FN counts accumulated across validation batches and then
SUM-reduced across all DDP ranks.

Decision rule for binarisation: argmax over the 5 softmax classes
(0=background, 1=AR, 2=Hurricane, 3=ETC, 4=MCS). Target masks are read
from the 14-channel input batch at the heuristic-mask channels
[1, 5, 10, 13] (see dataset.py `_compute_single_sample`).

Foreground IoU/Dice only — background (class 0) is not scored because it
dominates pixel-wise and would give a trivially ~99% number.
"""

import torch
import torch.distributed as dist


# (class_name, pred_class_idx in argmax, target_channel_idx in 14-ch batch)
CLASS_INFO = [
    ("ar",        1,  1),
    ("hurricane", 2,  5),
    ("etc",       3, 10),
    ("mcs",       4, 13),
]

NUM_FG_CLASSES = len(CLASS_INFO)


class SegmentationMetricAccumulator:
    """
    Per-class TP/FP/FN accumulator with DDP all_reduce — computes BOTH
    binarisation variants in parallel so they can be compared side-by-side:

      * "argmax"  (existing): pixel assigned to the class with highest
                  softmax prob, even if all classes are low. Multi-class
                  mutually-exclusive. Background dominance can hide rare-class
                  predictions.
      * "thresh"  (new):      per-class independent decision — pixel is
                  positive for class c iff seg_probs[c] > prob_threshold.
                  Allows rare classes to register even when they don't beat
                  background in argmax. Default prob_threshold=0.5; lower
                  to ~0.3 for rare classes where the softmax mass concentrates
                  on background and the rare-class prob rarely crosses 0.5.

    All existing keys (e.g. "ar_iou") remain unchanged → backward-compatible
    with plot_losses.py / loss_summary_report.txt. New keys carry the
    "_thresh" suffix (e.g. "ar_iou_thresh").
    """

    def __init__(self, device,
                 target_threshold: float = 0.5,
                 prob_threshold: float   = 0.5):
        self.device           = device
        self.target_threshold = target_threshold
        self.prob_threshold   = prob_threshold
        # tp/fp/fn shape: (NUM_CLASSES, 2) — last dim: 0=argmax, 1=thresh
        self.tp = torch.zeros(NUM_FG_CLASSES, 2, dtype=torch.float64, device=device)
        self.fp = torch.zeros(NUM_FG_CLASSES, 2, dtype=torch.float64, device=device)
        self.fn = torch.zeros(NUM_FG_CLASSES, 2, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, seg_probs: torch.Tensor, batch: torch.Tensor) -> None:
        """
        seg_probs : (B, 5, H, W) softmax over {bg, AR, Hurr, ETC, MCS}
        batch     : (B, 14, H, W) input; targets at channels [1, 5, 10, 13]
        """
        pred_classes = seg_probs.argmax(dim=1)  # (B, H, W) ∈ {0..4} — argmax variant

        for i, (_, cls_idx, ch_idx) in enumerate(CLASS_INFO):
            target_pos = (batch[:, ch_idx] > self.target_threshold)

            # ── Variant 0: argmax (mutually exclusive) ─────────────────────────
            pred_argmax = (pred_classes == cls_idx)
            self.tp[i, 0] += (pred_argmax & target_pos).sum().to(torch.float64)
            self.fp[i, 0] += (pred_argmax & ~target_pos).sum().to(torch.float64)
            self.fn[i, 0] += (~pred_argmax & target_pos).sum().to(torch.float64)

            # ── Variant 1: per-class threshold (multi-label) ───────────────────
            pred_thresh = (seg_probs[:, cls_idx] > self.prob_threshold)
            self.tp[i, 1] += (pred_thresh & target_pos).sum().to(torch.float64)
            self.fp[i, 1] += (pred_thresh & ~target_pos).sum().to(torch.float64)
            self.fn[i, 1] += (~pred_thresh & target_pos).sum().to(torch.float64)

    def reduce(self, world_size: int) -> None:
        """SUM-reduce counts across all DDP ranks (both variants in one call)."""
        if world_size <= 1 or not dist.is_initialized():
            return
        dist.all_reduce(self.tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.fn, op=dist.ReduceOp.SUM)

    def compute(self, eps: float = 1e-8) -> dict:
        """
        Return flat dict containing BOTH binarisation variants:
          * argmax keys : <class>_iou, _dice, _precision, _recall, _tp, _fp, _fn,
                          mean_iou, mean_dice   (unchanged from prior versions)
          * thresh keys : same names + "_thresh" suffix
        """
        out = {}
        for variant_idx, suffix in ((0, ""), (1, "_thresh")):
            ious, dices = [], []
            for i, (name, _, _) in enumerate(CLASS_INFO):
                tp = float(self.tp[i, variant_idx].item())
                fp = float(self.fp[i, variant_idx].item())
                fn = float(self.fn[i, variant_idx].item())
                iou       = tp / (tp + fp + fn + eps)
                dice      = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
                precision = tp / (tp + fp + eps)
                recall    = tp / (tp + fn + eps)
                out[f"{name}_iou{suffix}"]       = iou
                out[f"{name}_dice{suffix}"]      = dice
                out[f"{name}_precision{suffix}"] = precision
                out[f"{name}_recall{suffix}"]    = recall
                out[f"{name}_tp{suffix}"]        = tp
                out[f"{name}_fp{suffix}"]        = fp
                out[f"{name}_fn{suffix}"]        = fn
                ious.append(iou)
                dices.append(dice)
            out[f"mean_iou{suffix}"]  = sum(ious) / len(ious)
            out[f"mean_dice{suffix}"] = sum(dices) / len(dices)
        return out

    def reset(self) -> None:
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()


def format_metrics(metrics: dict) -> str:
    """
    Human-readable summary of compute() output. Shows argmax (arg) and
    threshold (thr) variants side-by-side per class on multiple lines so
    rare-class differences are visible at a glance.
    """
    lines = []
    for name, _, _ in CLASS_INFO:
        n = name.upper().ljust(9)
        lines.append(
            f"{n} arg: IoU={metrics[f'{name}_iou']:.4f} "
            f"P={metrics[f'{name}_precision']:.4f} "
            f"R={metrics[f'{name}_recall']:.4f}  ||  "
            f"thr: IoU={metrics[f'{name}_iou_thresh']:.4f} "
            f"P={metrics[f'{name}_precision_thresh']:.4f} "
            f"R={metrics[f'{name}_recall_thresh']:.4f}"
        )
    lines.append(
        f"mean_IoU  arg={metrics['mean_iou']:.4f}  thr={metrics['mean_iou_thresh']:.4f}"
    )
    return "\n  ".join(lines)
