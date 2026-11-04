"""Small reproducibility and output helpers."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DEFAULT_DATASET = PROJECT / "datasets" / "dataset"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def unique_name(label):
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + label + "_" + uuid.uuid4().hex[:6]


def fresh(path):
    path = Path(path).resolve()
    # Refuse even an empty existing directory: completed runs are never reused.
    path.mkdir(parents=True, exist_ok=False)
    return path


def environment():
    packages = {}
    for name in ("numpy", "scikit-learn", "torch", "xgboost"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    # Team members can run a downloaded source archive without installing Git.
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True)
        commit = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "machine": platform.machine(), "processor": platform.processor(), "git_commit": commit}
