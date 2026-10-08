"""Package only the public learning lab, without environments or checkpoints."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1] / "public" / "learning" / "llm-from-scratch"
FILES = ["minillm.py", "math_checks.py", "smoke_test.py", "distributed_check.py",
         "rl_bandit.py", "demo.jsonl", "demo-sft.jsonl", "demo-prompt.txt",
         "requirements-cpu.txt", "README.md", "LICENSE"]
if (ROOT / "experiment_results.json").is_file():
    FILES.append("experiment_results.json")
for name in ("validation-curves.svg", "validation-curves.png"):
    if (ROOT / name).is_file():
        FILES.append(name)
manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
(ROOT / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
FILES.append("checksums.json")
with zipfile.ZipFile(ROOT / "llm-from-scratch.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(FILES):
        entry = zipfile.ZipInfo(name, date_time=(2026, 10, 8, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o644 << 16
        archive.writestr(entry, (ROOT / name).read_bytes())
with zipfile.ZipFile(ROOT / "llm-from-scratch.zip") as archive:
    assert archive.testzip() is None
    for name in FILES:
        assert archive.read(name) == (ROOT / name).read_bytes()
print(f"Packaged and verified {len(FILES)} public files with SHA-256 manifest.")
