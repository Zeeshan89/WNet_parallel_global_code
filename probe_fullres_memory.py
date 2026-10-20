#!/usr/bin/env python3
"""probe_fullres_memory.py — peak GPU memory and step time of WNetMultiLabel training steps at
0.25 deg sizes (full globe padded to 728x1440, and crops), fp32 and bf16 autocast."""
import os, time
_mc = f"/tmp/miopen-cache-rank-{os.environ.get('SLURM_PROCID', 0)}"; os.makedirs(_mc, exist_ok=True)
os.environ.update(MIOPEN_USER_DB_PATH=_mc, MIOPEN_CUSTOM_CACHE_DIR=_mc, MIOPEN_FIND_MODE="2", MIOPEN_DISABLE_CACHE="0")
import torch, torch.nn.functional as F
from WNet_model import WNetMultiLabel
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB")
m = WNetMultiLabel(6, 4).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4)
print("params", sum(p.numel() for p in m.parameters()) / 1e6, "M")
for (h, w) in [(192, 384), (368, 736), (368, 1440), (728, 1440)]:
    for bs in (1, 2, 4):
        for amp in (False, True):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                x = torch.randn(bs, 6, h, w, device=dev); y = (torch.rand(bs, 4, h, w, device=dev) > .9).float()
                for it in range(3):
                    t0 = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        p, rec, lg = m(x)
                        loss = F.binary_cross_entropy_with_logits(lg.float(), y) + F.mse_loss(rec.float(), x)
                    opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
                print(f"{h}x{w} bs{bs} {'bf16' if amp else 'fp32'}: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB  step {time.time()-t0:.2f} s", flush=True)
            except torch.OutOfMemoryError:
                print(f"{h}x{w} bs{bs} {'bf16' if amp else 'fp32'}: OOM", flush=True)
            finally:
                x = y = p = rec = lg = loss = None
