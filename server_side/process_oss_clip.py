"""Process licensed OSS clips sequentially with one persistent GVHMR process."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Source clip SHA256 mismatch")


def _gpu_environment():
    env = os.environ.copy()
    prefix = Path(sys.prefix)
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    libs = [site / "torch" / "lib", prefix / "lib"]
    libs.extend(sorted((site / "nvidia").glob("*/lib")))
    env["LD_LIBRARY_PATH"] = ":".join(map(str, libs)) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _wait_for_result(path, worker, log_path):
    while not path.exists():
        if worker.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:]
            raise RuntimeError(f"GVHMR worker exited {worker.returncode}:\n{tail}")
        time.sleep(0.2)
    return json.loads(path.read_text(encoding="utf-8"))


def process(manifest_uri, clip_uris, scratch_parent, destination, ossutil, config,
            humanml_code_dir, offsets_path):
    manifest = json.loads(_oss_bytes(ossutil, config, manifest_uri))
    by_uri = {record["uri"]: record for record in manifest["clips"]}
    if len(by_uri) != len(manifest["clips"]):
        raise ValueError("Clip manifest contains duplicate URIs")
    requested = list(clip_uris) if clip_uris else list(by_uri)
    if not requested or any(uri not in by_uri for uri in requested):
        raise ValueError("Requested clips are missing from the manifest")
    scratch = Path(scratch_parent).resolve()
    if not scratch.is_dir():
        raise ValueError("Data-disk scratch directory does not exist")
    repo = Path(__file__).resolve().parents[1]
    completed = []
    errors = []
    with tempfile.TemporaryDirectory(prefix="motion-batch-", dir=scratch) as temp:
        folder = Path(temp)
        log_path = folder / "worker.log"
        with log_path.open("w", encoding="utf-8") as worker_log:
            worker = subprocess.Popen(
                [sys.executable, str(repo / "server_side" / "dataset_batch_worker.py")],
                cwd=repo, env=_gpu_environment(), stdin=subprocess.PIPE,
                stdout=worker_log, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for index, uri in enumerate(requested):
                    record = by_uri[uri]
                    if not all(record.get(key) for key in ("sha256", "license", "source_page", "source_uri")):
                        errors.append({"uri": uri, "error": "missing checksum, license, page, or parent source"})
                        continue
                    with tempfile.TemporaryDirectory(prefix=f"clip-{index:06d}-", dir=folder) as clip_temp:
                        clip_dir = Path(clip_temp)
                        source = clip_dir / "source-fragmented.mp4"
                        regular = clip_dir / "source-regular.mp4"
                        output = clip_dir / "dataset"
                        result_path = clip_dir / "worker-result.json"
                        try:
                            _download_checked(ossutil, config, uri, record["sha256"], source)
                            _run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                                  "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart", str(regular)])
                            source.unlink()
                            request = {
                                "result_path": str(result_path),
                                "video_path": str(regular),
                                "output_dir": str(output),
                                "humanml_code_dir": humanml_code_dir,
                                "humanml_offsets_path": offsets_path,
                                "source_uri": uri, "source_license": record["license"],
                                "source_object_sha256": record["sha256"],
                                "parent_source_uri": record["source_uri"],
                                "source_page": record["source_page"],
                            }
                            worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                            worker.stdin.flush()
                            response = _wait_for_result(result_path, worker, log_path)
                            if response["status"] != "success":
                                raise RuntimeError(response.get("type", "WorkerError") + ": " + response["message"])
                            digest = hashlib.sha256()
                            with regular.open("rb") as stream:
                                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                                    digest.update(block)
                            processing_hash = digest.hexdigest()
                            sample_id = hashlib.sha256((SCHEMA_VERSION + ":" + processing_hash).encode()).hexdigest()[:24]
                            meta_path = output / "metadata" / (sample_id + ".json")
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            meta["source_clip_start_second"] = record["start_second"]
                            meta["source_clip_end_second"] = record["end_second"]
                            meta["source_manifest_uri"] = manifest_uri
                            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                            validate_sample(output, sample_id)
                            catalog_uri = publish(output, sample_id, destination, ossutil, config) if destination else None
                            item = {"source_uri": uri, "sample_id": sample_id, "frames": meta["frames"],
                                    "catalog_uri": catalog_uri}
                            completed.append(item)
                            print(json.dumps({"completed": item}, ensure_ascii=False), flush=True)
                        except Exception as exc:
                            error = {"uri": uri, "error": str(exc)}
                            errors.append(error)
                            print(json.dumps({"failed": error}, ensure_ascii=False), flush=True)
            finally:
                worker.stdin.close()
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.terminate()
                    worker.wait(timeout=10)
        log = log_path.read_text(errors="replace")
        model_loads = log.count(">>> Loading GVHMR Model")
    return {"requested": len(requested), "completed": completed,
            "errors": errors, "model_loads": model_loads}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-uri", required=True)
    p.add_argument("--clip-uri", action="append", help="Repeat for multiple clips; omit to process all clips")
    p.add_argument("--scratch-parent", required=True, help="Existing data-disk directory")
    p.add_argument("--destination", help="OSS staging prefix; omit for validation-only run")
    p.add_argument("--ossutil", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--humanml-code-dir", required=True)
    p.add_argument("--humanml-offsets-path", required=True)
    a = p.parse_args()
    result = process(a.manifest_uri, a.clip_uri, a.scratch_parent,
                     a.destination, a.ossutil, a.config,
                     a.humanml_code_dir, a.humanml_offsets_path)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["errors"]:
        raise SystemExit(2)
