"""Publish a validated internet-motion sample to OSS staging with readback hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from validate_dataset import validate_sample


def _call(command, **kwargs):
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-500:])
    return p.stdout


def _remote_sha256(ossutil, config, uri):
    p = subprocess.Popen([ossutil, "cat", uri, "-c", config, "--quiet"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = hashlib.sha256()
    for block in iter(lambda: p.stdout.read(4 * 1024 * 1024), b""):
        h.update(block)
    p.stdout.close()
    err = p.stderr.read()
    if p.wait():
        raise RuntimeError(err.decode(errors="replace")[-500:])
    return h.hexdigest()


def publish(root, sample_id, destination, ossutil, config):
    validate_sample(root, sample_id)
    root = Path(root)
    meta = json.loads((root / "metadata" / (sample_id + ".json")).read_text())
    if not meta.get("source_uri") or not meta.get("source_license"):
        raise ValueError("Internet samples need source_uri and source_license before publication")
    if meta.get("source_object_sha256"):
        if _remote_sha256(ossutil, config, meta["source_uri"]) != meta["source_object_sha256"]:
            raise RuntimeError("Original OSS source checksum does not match metadata")
    base = destination.rstrip("/") + "/" + sample_id
    uploaded = {}
    for kind in ("new_joints", "new_joint_vecs", "annotations", "metadata"):
        path = root / (meta["arrays"][kind] if kind != "metadata" else f"metadata/{sample_id}.json")
        uri = base + "/" + kind + path.suffix
        local_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            existing_hash = _remote_sha256(ossutil, config, uri)
        except RuntimeError:
            existing_hash = None
        if existing_hash is not None and existing_hash != local_hash:
            raise RuntimeError("Existing OSS sample differs: " + uri)
        if existing_hash is None:
            _call([ossutil, "cp", str(path), uri, "-c", config, "--quiet"])
            if _remote_sha256(ossutil, config, uri) != local_hash:
                raise RuntimeError("OSS readback mismatch: " + uri)
        uploaded[kind] = {"uri": uri, "sha256": local_hash, "bytes": path.stat().st_size}
    catalog = {
        "schema_version": meta["schema_version"], "sample_id": sample_id,
        "review_status": meta["review_status"], "source_uri": meta["source_uri"],
        "source_license": meta["source_license"],
        "processing_video_sha256": meta["processing_video_sha256"],
        "source_object_sha256": meta["source_object_sha256"],
        "parent_source_uri": meta["parent_source_uri"], "source_page": meta["source_page"],
        "frames": meta["frames"], "output_fps": meta["output_fps"], "files": uploaded,
    }
    catalog_uri = destination.rstrip("/") + "/catalog/" + sample_id + ".json"
    content = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode()
    desired = hashlib.sha256(content).hexdigest()
    try:
        existing = _remote_sha256(ossutil, config, catalog_uri)
    except RuntimeError:
        existing = None
    if existing is not None and existing != desired:
        raise RuntimeError("Existing OSS catalog differs")
    if existing is None:
        _call([ossutil, "cp", "-", catalog_uri, "-c", config, "--quiet"], input=content)
        if _remote_sha256(ossutil, config, catalog_uri) != desired:
            raise RuntimeError("OSS catalog readback mismatch")
    return catalog_uri


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("sample_id")
    p.add_argument("destination", help="OSS staging prefix, e.g. oss://bucket/internet-videos/processed/v1/staging")
    p.add_argument("--ossutil", required=True)
    p.add_argument("--config", required=True)
    a = p.parse_args()
    print(publish(a.root, a.sample_id, a.destination, a.ossutil, a.config))
