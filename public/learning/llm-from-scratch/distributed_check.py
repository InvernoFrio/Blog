"""Two CPU processes: exact token-weighted gradients, file exchange or Gloo DDP."""
import argparse
from datetime import timedelta
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(3)
        model = torch.nn.Linear(3, 4).double()
        distributed = DDP(model)
        inputs = torch.tensor([[1., 0., 2.], [0., 1., 1.], [2., 2., 0.], [1., 3., 2.]], dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 3])
        local_x = inputs[:1] if rank == 0 else inputs[1:]
        local_y = labels[:1] if rank == 0 else labels[1:]
        count = torch.tensor(local_y.numel(), dtype=torch.float64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        loss_sum = F.cross_entropy(distributed(local_x), local_y, reduction="sum")
        (2 * loss_sum / count).backward()
        reference = torch.nn.Linear(3, 4).double()
        reference.load_state_dict(model.state_dict())
        F.cross_entropy(reference(inputs), labels).backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-10, atol=1e-12)
        print(f"PASS rank {rank}: unequal local token counts, exact global gradient", flush=True)
    finally:
        dist.destroy_process_group()


def file_worker(rank, directory):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 4).double()
    inputs = torch.tensor([[1., 0., 2.], [0., 1., 1.], [2., 2., 0.], [1., 3., 2.]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3])
    local_x = inputs[:1] if rank == 0 else inputs[1:]
    local_y = labels[:1] if rank == 0 else labels[1:]
    F.cross_entropy(model(local_x), local_y, reduction="sum").backward()
    torch.save({"count": local_y.numel(), "gradients": [p.grad.clone() for p in model.parameters()]},
               Path(directory) / f"rank-{rank}.pt")


def file_check(directory):
    mp.spawn(file_worker, args=(directory,), nprocs=2, join=True)
    states = [torch.load(Path(directory) / f"rank-{r}.pt", weights_only=True) for r in range(2)]
    total = sum(state["count"] for state in states)
    torch.manual_seed(3)
    reference = torch.nn.Linear(3, 4).double()
    inputs = torch.tensor([[1., 0., 2.], [0., 1., 1.], [2., 2., 0.], [1., 3., 2.]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3])
    F.cross_entropy(reference(inputs), labels).backward()
    wrong_differs = False
    for index, parameter in enumerate(reference.parameters()):
        global_gradient = sum(state["gradients"][index] for state in states) / total
        torch.testing.assert_close(global_gradient, parameter.grad, rtol=1e-10, atol=1e-12)
        wrong = sum(state["gradients"][index] / state["count"] for state in states) / 2
        wrong_differs |= not torch.allclose(wrong, parameter.grad)
    assert wrong_differs, "Unequal token counts must expose the rank-mean error"
    print("PASS two actual CPU processes: file-exchanged gradients equal the global token mean")
    print("PASS equal weighting of rank means produces an incorrect gradient")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("file", "gloo"), default="file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="ddp-check-") as directory:
        if args.backend == "file":
            file_check(directory)
        else:
            if not dist.is_available() or not dist.is_gloo_available():
                raise RuntimeError("Gloo unavailable; use --backend file for the portable math experiment")
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(worker, args=(rendezvous,), nprocs=2, join=True)
            print("Two-rank CPU/Gloo DDP gradient check passed.")
    print(f"Backend: {args.backend}. No GPU or communication-throughput benchmark performed.")


if __name__ == "__main__":
    main()
