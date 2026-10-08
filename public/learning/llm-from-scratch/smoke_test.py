"""Integration checks: causality, SFT masks, RNG/optimizer resume, real training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from minillm import BOS, EOS, Config, MiniLM, encode, make_windows

ROOT = Path(__file__).resolve().parent


def run(*arguments, succeeds=True):
    result = subprocess.run([sys.executable, str(ROOT / "minillm.py"), *map(str, arguments)], capture_output=True, text=True, encoding="utf-8")
    if succeeds and result.returncode:
        raise AssertionError(result.stderr)
    if not succeeds:
        assert result.returncode != 0 and "Resume contract differs" in result.stderr
    return result


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), "Resume changed a tensor"
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            same(left, right)
    else:
        assert a == b


def main():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = MiniLM(Config(dropout=0.0)).eval()
    x = torch.tensor([[BOS, 65, 66, 67, 68, 69]])
    changed = x.clone()
    changed[:, 3:] = 70
    with torch.no_grad():
        torch.testing.assert_close(model(x)[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    assert model.head.weight is model.token_embedding.weight
    docs = [{"prompt": "Hi", "response": "Hello"}]
    inputs, labels = make_windows(docs, 64, sft=True)
    prefix = [BOS] + encode("User: Hi\nAssistant: ")
    count = len(encode("Hello")) + 1
    assert (labels != -100).sum().item() == count
    assert labels[0, len(prefix) - 1].item() == ord("H")
    assert labels[0, len(prefix) + count - 2].item() == EOS
    assert (labels[0, :len(prefix) - 1] == -100).all()
    print("PASS causality, tied weights, response-only SFT labels")
    with tempfile.TemporaryDirectory(prefix="minillm-test-") as directory:
        root = Path(directory)
        common = ["train", "--data", ROOT / "demo.jsonl", "--batch-size", "3", "--accum-steps", "2", "--eval-every", "3", "--save-every", "3"]
        run(*common, "--out", root / "full", "--steps", "6")
        run(*common, "--out", root / "part", "--steps", "3")
        run(*common, "--out", root / "resumed", "--steps", "6", "--resume", root / "part/last.pt")
        full = torch.load(root / "full/last.pt", weights_only=True)
        resumed = torch.load(root / "resumed/last.pt", weights_only=True)
        for field in ("model", "optimizer", "step", "seen_tokens", "batch_rng", "torch_rng", "python_rng"):
            same(full[field], resumed[field])
        run(*common, "--out", root / "bad", "--steps", "7", "--lr", "0.004", "--resume", root / "full/last.pt", succeeds=False)
        print("PASS exact CPU resume, including dropout and optimizer; changed schedule rejected")
        run("train", "--out", root / "trained", "--steps", "80")
        report = json.loads((root / "trained/report.json").read_text())
        assert report["final_validation"]["loss"] < report["initial_validation"]["loss"] - 0.5
        generated = run("generate", "--checkpoint", root / "trained/last.pt", "--prompt", "Learning ", "--max-tokens", "80").stdout
        assert generated.startswith("Learning ")
        run("train", "--sft", "--data", ROOT / "demo-sft.jsonl", "--init-from", root / "trained/last.pt", "--out", root / "sft", "--steps", "4")
        stage = torch.load(root / "sft/last.pt", weights_only=True)
        assert stage["step"] == 4 and stage["contract"]["sft"]
        reply = run("generate", "--checkpoint", root / "sft/last.pt", "--prompt-file", ROOT / "demo-prompt.txt", "--max-tokens", "10").stdout
        assert reply.startswith("User: Hi\nAssistant: ")
        print("PASS validation loss decreases, generation, new SFT stage")
    print("All integration checks passed. CPU only; no GPU throughput claims.")


if __name__ == "__main__":
    main()
