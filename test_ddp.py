from mpi4py import MPI
import torch
import torch.distributed as dist
import os

def main():
    # ── MPI setup ─────────────────────────────────────────────────────────────
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # ── With --gpus-per-task=1, each rank only sees ONE GPU → always cuda:0 ──
    # SLURM_LOCALID would be 0-7 but only cuda:0 is visible per task.
    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda:0")

    # ── Initialize DDP process group ──────────────────────────────────────────
    # MASTER_ADDR is set in the batch script (forced IPv4)
    os.environ.setdefault("MASTER_PORT", "3442")

    dist.init_process_group(
        backend="nccl",        # maps to AMD RCCL on ROCm
        init_method="env://",
        world_size=world_size,
        rank=world_rank,
    )

    # ── Confirm each rank has its own GPU ─────────────────────────────────────
    device_name = torch.cuda.get_device_name(local_rank)
    print(f"[Rank {world_rank}/{world_size}] "
          f"cuda:0 on this task | "
          f"Device: {device_name} | "
          f"Node: {MPI.Get_processor_name()}", flush=True)

    # ── All-reduce test (verifies inter-GPU communication via RCCL) ───────────
    tensor = torch.ones(1, device=device) * world_rank
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    expected = sum(range(world_size))   # 0+1+2+...+7 = 28
    if world_rank == 0:
        status = "✅ PASSED" if int(tensor.item()) == expected else "❌ FAILED"
        print(f"\nAll-reduce result: {int(tensor.item())} "
              f"(expected {expected}) → {status}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
