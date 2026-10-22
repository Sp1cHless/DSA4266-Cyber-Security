"""One-command orchestration; representation fitting remains independently callable."""
from __future__ import annotations

import logging
from pathlib import Path

from CIC_new.preprocessing.sequence_index import signature
from .prepare import prepare
from .export import export
from .storage import read_json, write_json, load_tables
from .validate import validate

LOG = logging.getLogger(__name__)


def build(db, out, *, resume=False, prepare_only=False, embeddings=None, seed=4266,
          shard_size=2048, train_pool=50000, val_pool=2048, embedding_steps=3000,
          batch_size=128, validate_every=200, l7_dim=8, src_port_dim=8, dst_port_dim=8,
          device="auto", cpu_threads=2, skip_validation=False):
    db, out = Path(db).resolve(), Path(out).resolve()
    if not db.is_file():
        raise FileNotFoundError(db)
    if min(shard_size, train_pool, val_pool, embedding_steps, batch_size, validate_every,
           l7_dim, src_port_dim, dst_port_dim, cpu_threads) < 1:
        raise ValueError("Sizes, dimensions and training settings must be positive")
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Output exists: {out}. Use --resume for completed stages or choose a fresh --out.")
    out.mkdir(parents=True, exist_ok=True)
    config = dict(source=signature(db), seed=seed, shard_size=shard_size,
                  train_pool=train_pool, val_pool=val_pool, target_ratios=[.6, .2, .2], shift_buckets=0)
    status_path = out / "workflow.json"
    if status_path.exists():
        status = read_json(status_path)
        if status["config"] != config:
            raise ValueError("Source/preparation settings changed; use a fresh --out")
    else:
        if resume and any(out.iterdir()):
            raise ValueError("No workflow.json; this is not a scheme1 workflow output")
        status = dict(complete=False, config=config)
        write_json(status_path, status)
    prepared = out / "prepared"
    prep_manifest = prepared / "manifest.json"
    if prep_manifest.exists() and read_json(prep_manifest).get("complete"):
        LOG.info("Reuse completed real-bucket preparation: %s", prepared)
    else:
        prepare(db, prepared, seed=seed, shard_size=shard_size, train_pool=train_pool, val_pool=val_pool)
    if prepare_only:
        LOG.info("Real sequences/buckets/splits ready: %s. No embedding fitting or synthesis ran.", prepared)
        return {"prepared": str(prepared), "complete": False}
    embedding_folder = Path(embeddings).resolve() if embeddings else out / "embeddings"
    if embeddings or (embedding_folder / "manifest.json").exists() and read_json(embedding_folder / "manifest.json").get("complete"):
        _, fitted = load_tables(prepared, embedding_folder)
        if not embeddings and fitted["dimensions"] != dict(l7=l7_dim, src_port=src_port_dim, dst_port=dst_port_dim):
            raise ValueError("Requested dimensions differ from completed category tables")
        LOG.info("Reuse frozen category tables: %s", embedding_folder)
    else:
        from .embeddings import train
        train(prepared, embedding_folder, steps=embedding_steps, batch_size=batch_size,
              validate_every=validate_every, l7_dim=l7_dim, src_port_dim=src_port_dim,
              dst_port_dim=dst_port_dim, seed=seed, device=device, cpu_threads=cpu_threads)
    dataset = out / "dataset"
    if (dataset / "manifest.json").exists() and read_json(dataset / "manifest.json").get("complete"):
        current = read_json(dataset / "manifest.json")
        _, fitted = load_tables(prepared, embedding_folder)
        if current["embedding_checkpoint_sha256"] != fitted["embedding_sha256"]:
            raise ValueError("Existing dataset uses different embedding weights; choose a fresh output")
        LOG.info("Reuse completed model inputs: %s", dataset)
    else:
        export(prepared, embedding_folder, dataset, seed=seed + 1, shard_size=shard_size)
    if not skip_validation:
        try:
            validate(dataset, prepared)
        except Exception as error:
            write_json(dataset / "validation.json", dict(passed=False, complete=False, error=str(error)))
            status.update(complete=False, validation_run=True, validation_passed=False)
            write_json(status_path, status)
            raise
    if signature(db) != config["source"]:
        raise ValueError("Source fingerprint changed")
    status.update(complete=True, prepared=str(prepared), embeddings=str(embedding_folder),
                  dataset=str(dataset), validation_run=not skip_validation,
                  validation_passed=True if not skip_validation else None)
    write_json(status_path, status)
    LOG.info("READY: %s", dataset)
    return status
