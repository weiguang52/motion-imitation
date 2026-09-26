"""Process one licensed OSS clip into a validated, staged motion sample."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dataset_export import SCHEMA_VERSION
from publish_dataset import publish
from validate_dataset import validate_sample


def _run(command, **kwargs):
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if p.returncode:
        raise RuntimeError("Command failed: " + " ".join(map(str, command[:3])) + "\n" +
                           p.stderr.decode(errors="replace")[-2000:])
    return p.stdout


def _oss_bytes(ossutil, config, uri):
    return _run([ossutil, "cat", uri, "-c", config, "--quiet"])


def _download_checked(ossutil, config, uri, expected_sha256, path):
    _run([ossutil, "cp", uri, str(path), "-c", config, "--quiet"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError("Source clip SHA256 mismatch")


def _gpu_environment():
    env = os.environ.copy()
    prefix = Path(sys.prefix)
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    libs = [site / "torch" / "lib", prefix / "lib"]
    libs.extend(sorted((site / "nvidia").glob("*/lib")))
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def process(manifest_uri, clip_uri, scratch_parent, destination, ossutil, config,
            humanml_code_dir, offsets_path):
    manifest = json.loads(_oss_bytes(ossutil, config, manifest_uri))
    records = [record for record in manifest["clips"] if record["uri"] == clip_uri]
    if len(records) != 1:
        raise ValueError("Clip URI must appear exactly once in the licensed clip manifest")
    record = records[0]
    if not record.get("sha256") or not record.get("license") or not record.get("source_page"):
        raise ValueError("Clip manifest lacks checksum, license, or source page")
    repo = Path(__file__).resolve().parents[1]
    scratch = Path(scratch_parent).resolve()
    if not scratch.is_dir():
        raise ValueError("Data-disk scratch directory does not exist")
    with tempfile.TemporaryDirectory(prefix="motion-clip-", dir=scratch) as temp:
        folder = Path(temp)
        source = folder / "source-fragmented.mp4"
        regular = folder / "source-regular.mp4"
        output = folder / "dataset"
        _download_checked(ossutil, config, clip_uri, record["sha256"], source)
        _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
              "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart", str(regular)])
        command = [sys.executable, str(repo / "server_side" / "run.py"),
                   "--input", str(regular), "--raw-motion-only",
                   "--raw-motion-target-fps", "20", "--dataset-output-dir", str(output),
                   "--humanml-code-dir", humanml_code_dir,
                   "--humanml-offsets-path", offsets_path,
                   "--source-uri", clip_uri,
                   "--source-license", record["license"],
                   "--source-object-sha256", record["sha256"],
                   "--parent-source-uri", record["source_uri"],
                   "--source-page", record["source_page"]]
        inference = subprocess.run(command, cwd=repo, env=_gpu_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if inference.returncode:
            raise RuntimeError("Motion reconstruction failed:\n" +
                               inference.stdout.decode(errors="replace")[-3000:])
        processing_hash = hashlib.sha256(regular.read_bytes()).hexdigest()
        sample_id = hashlib.sha256((SCHEMA_VERSION + ":" + processing_hash).encode()).hexdigest()[:24]
        meta_path = output / "metadata" / (sample_id + ".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["source_clip_start_second"] = record["start_second"]
        meta["source_clip_end_second"] = record["end_second"]
        meta["source_manifest_uri"] = manifest_uri
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        validate_sample(output, sample_id)
        catalog_uri = publish(output, sample_id, destination, ossutil, config) if destination else None
        return {"sample_id": sample_id, "frames": meta["frames"], "catalog_uri": catalog_uri}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-uri", required=True)
    p.add_argument("--clip-uri", required=True)
    p.add_argument("--scratch-parent", required=True, help="Existing data-disk directory")
    p.add_argument("--destination", help="OSS staging prefix; omit for validation-only run")
    p.add_argument("--ossutil", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--humanml-code-dir", required=True)
    p.add_argument("--humanml-offsets-path", required=True)
    a = p.parse_args()
    print(json.dumps(process(a.manifest_uri, a.clip_uri, a.scratch_parent,
                             a.destination, a.ossutil, a.config,
                             a.humanml_code_dir, a.humanml_offsets_path), ensure_ascii=False))
