"""Continuously discover licensed Commons videos and stage motion samples in OSS."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# CUDA wheel libraries must be visible to the dynamic loader at process start.
if os.environ.get("MOTION_GPU_LIBS_READY") != "1":
    _prefix = Path(sys.prefix)
    _site = _prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    _libs = [_site / "torch" / "lib", _prefix / "lib"]
    _libs.extend(sorted((_site / "nvidia").glob("*/lib")))
    os.environ["LD_LIBRARY_PATH"] = ":".join(map(str, _libs)) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["MOTION_GPU_LIBS_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import numpy as np
import requests

OSS = "/root/gpufree-data/tools/ossutil-2.4.0-linux-amd64/ossutil"
CFG = "/root/gpufree-data/config/ossutilconfig"
PYTHON = "/root/gpufree-data/conda-envs/motion-imitation/bin/python"
PROCESSOR = "/root/gpufree-data/apps/motion-imitation/server_side/process_oss_clip.py"
SCRATCH = "/root/gpufree-data/tmp"
CODE = "/root/gpufree-data/apps/humanml3d-code"
OFFSETS = "/root/gpufree-data/models/humanml3d/target_offsets.npy"
YOLO_WEIGHTS = "/root/gpufree-data/apps/motion-imitation/inputs/checkpoints/yolo/yolov8x.pt"
BASE = "oss://lighto1-motion-dataset/internet-videos/wikimedia-commons"
RUN_BASE = BASE + "/continuous-v1"
STATE_URI = "oss://lighto1-motion-dataset/internet-videos/processed/v1/ingest-state.json"
DESTINATION = "oss://lighto1-motion-dataset/internet-videos/processed/v1/staging"
PILOT_SOURCE = BASE + "/pilot-2026-09-26/manifest.json"
PILOT_CLIPS = BASE + "/pilot-2026-09-26/clips/manifest.json"
SEARCH = [
    "walking person", "running person", "jumping person", "dancing adult",
    "cooking person", "sitting on chair exercise", "sitting on floor activity",
    "seated movement", "chair yoga", "person getting up from floor",
    "person standing from chair", "person crawling",
]
API = "https://commons.wikimedia.org/w/api.php"
PROXIES = {"http": "http://127.0.0.1:10808", "https": "http://127.0.0.1:10808"}
HEADERS = {"User-Agent": "LightO1MotionDataset/0.2 (https://github.com/weiguang52/motion-imitation)"}


def say(*parts):
    print(datetime.now(timezone.utc).isoformat(), *parts, flush=True)


def oss_cat(uri):
    p = subprocess.run([OSS, "cat", uri, "-c", CFG, "--quiet"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("OSS read failed: " + uri + " " + p.stderr.decode(errors="replace")[-300:])
    return p.stdout


def oss_put_bytes(uri, data):
    p = subprocess.run([OSS, "cp", "-", uri, "-c", CFG, "--force", "--quiet"],
                       input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        raise RuntimeError("OSS write failed: " + uri + " " + p.stderr.decode(errors="replace")[-300:])
    if oss_cat(uri) != data:
        raise RuntimeError("OSS readback mismatch: " + uri)


def oss_sha256(uri):
    p = subprocess.Popen([OSS, "cat", uri, "-c", CFG, "--quiet"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = hashlib.sha256()
    size = 0
    for block in iter(lambda: p.stdout.read(4 * 1024 * 1024), b""):
        h.update(block)
        size += len(block)
    p.stdout.close()
    err = p.stderr.read()
    if p.wait():
        raise RuntimeError("OSS hash failed: " + err.decode(errors="replace")[-300:])
    return h.hexdigest(), size


def load_state():
    try:
        state = json.loads(oss_cat(STATE_URI))
    except RuntimeError:
        state = {"schema_version": 1, "seen_page_ids": [], "processed_clip_uris": [],
                 "query_offsets": {}, "pending_manifests": [], "clip_failures": {}}
        try:
            pilot = json.loads(oss_cat(PILOT_SOURCE))
            state["seen_page_ids"] = [int(item["uri"].rsplit("/", 1)[-1].split(".")[0])
                                      for item in pilot["files"]]
        except Exception as exc:
            say("PILOT_BOOTSTRAP_WARNING", str(exc))
        state["pending_manifests"].append(PILOT_CLIPS)
        # The pilot walking clip was already validated and published.
        state["processed_clip_uris"].append(BASE + "/pilot-2026-09-26/clips/97994423-0000-0007.mp4")
    state.setdefault("clip_failures", {})
    return state


def save_state(state):
    state["updated_utc"] = datetime.now(timezone.utc).isoformat()
    oss_put_bytes(STATE_URI, (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode())


def source_candidates(state, max_sources=2):
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.headers.update(HEADERS)
    seen = set(state["seen_page_ids"])
    selected = []
    for term in SEARCH:
        offset = state["query_offsets"].get(term, 0)
        params = {"action": "query", "generator": "search",
                  "gsrsearch": "filetype:video " + term, "gsrnamespace": 6,
                  "gsrlimit": 20, "gsroffset": offset, "prop": "imageinfo",
                  "iiprop": "url|size|mime|extmetadata|sha1|timestamp|user",
                  "format": "json", "formatversion": 2}
        response = session.get(API, params=params, timeout=60)
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        state["query_offsets"][term] = offset + 20 if pages else 0
        for page in pages:
            pageid = int(page["pageid"])
            title_lower = page.get("title", "").lower()
            if any(word in title_lower for word in (
                    "peacock", " bird", " dog", " cat", "horse", " animal",
                    "giraffe", "cattle", " fish", "bear", "cartoon", "animation",
                    "stream of water", "waterfall", "river", "ocean", "walking tour")):
                continue
            if pageid in seen:
                continue
            info = page.get("imageinfo", [{}])[0]
            license_id = info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            size = info.get("size", 0)
            mime = info.get("mime", "")
            if not (mime.startswith("video/") and 0 < size <= 40_000_000):
                continue
            if not (license_id.startswith("CC BY") or license_id in ("CC0", "Public domain")):
                continue
            selected.append((page, info, license_id))
            seen.add(pageid)
            if len(selected) >= max_sources:
                return selected
    return selected


def upload_source(page, info, license_id):
    pageid = int(page["pageid"])
    suffix = page["title"].rsplit(".", 1)[-1].lower()
    if not suffix.isalnum() or len(suffix) > 6:
        raise ValueError("Unsupported source extension")
    uri = f"{RUN_BASE}/sources/{pageid}.{suffix}"
    response = requests.get(info["url"], proxies=PROXIES, headers=HEADERS,
                            stream=True, timeout=(30, 120))
    response.raise_for_status()
    up = subprocess.Popen([OSS, "cp", "-", uri, "-c", CFG, "--quiet"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    try:
        for block in response.iter_content(1024 * 1024):
            if block:
                up.stdin.write(block)
                digest.update(block)
                size += len(block)
        up.stdin.close()
        up.stdout.read()
        err = up.stderr.read()
        up_code = up.wait()
        if up_code or size != info["size"]:
            raise RuntimeError("Source upload incomplete: " + err.decode(errors="replace")[-300:])
    finally:
        response.close()
        if up.poll() is None:
            up.kill()
            up.wait()
    if oss_sha256(uri) != (digest.hexdigest(), size):
        raise RuntimeError("Source OSS checksum mismatch")
    meta = info.get("extmetadata", {})
    return {"pageid": pageid, "title": page["title"], "uri": uri,
            "source_page": "https://commons.wikimedia.org/wiki/" + page["title"].replace(" ", "_"),
            "license": license_id, "license_url": meta.get("LicenseUrl", {}).get("value"),
            "sha256": digest.hexdigest(), "bytes": size, "source_sha1_base36": info.get("sha1"),
            "source_url": info["url"], "uploader": info.get("user")}


def scan_person_intervals(source_uri, model, max_seconds=600):
    source = subprocess.Popen([OSS, "cat", source_uri, "-c", CFG, "--quiet"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    decoder = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-vf",
         "fps=1,scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2",
         "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
        stdin=source.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    source.stdout.close()
    positives = []
    frame_bytes = 640 * 640 * 3
    try:
        for second in range(max_seconds):
            data = decoder.stdout.read(frame_bytes)
            if not data:
                break
            if len(data) != frame_bytes:
                raise RuntimeError("Incomplete decoded scan frame")
            image = np.frombuffer(data, dtype=np.uint8).reshape(640, 640, 3)
            result = model.predict(image, classes=[0], conf=0.35, imgsz=640,
                                   device=0, verbose=False)[0]
            if len(result.boxes):
                positives.append(second)
    finally:
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
        decoder.wait()
        if source.poll() is None:
            source.terminate()
        source.wait()
    intervals = []
    start = last = None
    for second in positives:
        if start is None:
            start = last = second
        elif second <= last + 2:
            last = second
        else:
            if last - start + 1 >= 3:
                intervals.append((max(0, start - 1), last + 2))
            start = last = second
    if start is not None and last - start + 1 >= 3:
        intervals.append((max(0, start - 1), last + 2))
    chunks = []
    for start, end in intervals:
        cursor = start
        while cursor < end:
            stop = min(end, cursor + 30)
            if stop - cursor >= 3:
                chunks.append((cursor, stop))
            cursor = stop
    return chunks


def upload_clip(source, start, end):
    uri = f"{RUN_BASE}/clips/{source['pageid']}-{start:05d}-{end:05d}.mp4"
    down = subprocess.Popen([OSS, "cat", source["uri"], "-c", CFG, "--quiet"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    ff = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-ss", str(start),
         "-t", str(end - start), "-map", "0:v:0", "-an", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "25", "-pix_fmt", "yuv420p",
         "-movflags", "frag_keyframe+empty_moov+default_base_moof",
         "-f", "mp4", "pipe:1"],
        stdin=down.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    down.stdout.close()
    up = subprocess.Popen([OSS, "cp", "-", uri, "-c", CFG, "--quiet"],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    try:
        for block in iter(lambda: ff.stdout.read(1024 * 1024), b""):
            up.stdin.write(block)
            digest.update(block)
            size += len(block)
        up.stdin.close()
        ff.stdout.close()
        ff_error = ff.stderr.read()
        ff_code = ff.wait()
        source_code = down.wait()
        up.stdout.read()
        up_error = up.stderr.read()
        up_code = up.wait()
        if ff_code or source_code not in (0, -13, 141) or up_code or size < 1024:
            raise RuntimeError("Clip pipeline failed: " + ff_error.decode(errors="replace")[-200:] +
                               up_error.decode(errors="replace")[-200:])
    finally:
        for proc in (down, ff, up):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if oss_sha256(uri) != (digest.hexdigest(), size):
        raise RuntimeError("Clip OSS checksum mismatch")
    return {"uri": uri, "source_uri": source["uri"], "source_page": source["source_page"],
            "start_second": start, "end_second": end, "sha256": digest.hexdigest(),
            "bytes": size, "license": source["license"], "license_url": source["license_url"],
            "title": source["title"], "review_status": "automatic person detection; human review required"}


def process_manifest(manifest_uri, state):
    manifest = json.loads(oss_cat(manifest_uri))
    pending = [record["uri"] for record in manifest["clips"]
               if record["uri"] not in state["processed_clip_uris"]
               and state["clip_failures"].get(record["uri"], 0) < 3]
    if not pending:
        state["pending_manifests"] = [uri for uri in state["pending_manifests"] if uri != manifest_uri]
        save_state(state)
        return
    command = [PYTHON, PROCESSOR, "--manifest-uri", manifest_uri,
               "--scratch-parent", SCRATCH, "--destination", DESTINATION,
               "--ossutil", OSS, "--config", CFG,
               "--humanml-code-dir", CODE, "--humanml-offsets-path", OFFSETS]
    for uri in pending:
        command.extend(["--clip-uri", uri])
    say("BATCH_START", len(pending), manifest_uri)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in result.stdout.splitlines():
        say("BATCH", line[:1000])
    try:
        summary = json.loads(result.stdout.splitlines()[-1])
    except Exception:
        raise RuntimeError(f"Batch output missing summary, exit={result.returncode}")
    for error in summary.get("errors", []):
        uri = error["uri"]
        state["clip_failures"][uri] = state["clip_failures"].get(uri, 0) + 1
    for item in summary.get("completed", []):
        uri = item["source_uri"]
        if uri not in state["processed_clip_uris"]:
            state["processed_clip_uris"].append(uri)
    if all(record["uri"] in state["processed_clip_uris"] or
           state["clip_failures"].get(record["uri"], 0) >= 3
           for record in manifest["clips"]):
        state["pending_manifests"] = [uri for uri in state["pending_manifests"] if uri != manifest_uri]
    save_state(state)
    if summary.get("errors"):
        say("BATCH_ERRORS", len(summary["errors"]))


def discover_batch(state):
    candidates = source_candidates(state)
    if not candidates:
        save_state(state)
        say("DISCOVERY_EMPTY")
        return
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    clips = []
    sources = []
    for page, info, license_id in candidates:
        try:
            source = upload_source(page, info, license_id)
            sources.append(source)
            say("SOURCE", source["title"], source["bytes"], source["uri"])
            for start, end in scan_person_intervals(source["uri"], model):
                try:
                    clip = upload_clip(source, start, end)
                    clips.append(clip)
                    say("CLIP", clip["uri"], clip["bytes"])
                except Exception as exc:
                    say("CLIP_ERROR", source["uri"], start, end, type(exc).__name__, str(exc)[:300])
            state["seen_page_ids"].append(source["pageid"])
        except Exception as exc:
            say("SOURCE_ERROR", page.get("title"), type(exc).__name__, str(exc)[:300])
    if clips:
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_uri = f"{RUN_BASE}/manifests/{batch_id}.json"
        body = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "source": "Wikimedia Commons API", "sources": sources, "clips": clips}
        oss_put_bytes(manifest_uri, (json.dumps(body, ensure_ascii=False, indent=2) + "\n").encode())
        state["pending_manifests"].append(manifest_uri)
    save_state(state)


def main():
    state = load_state()
    save_state(state)
    say("DAEMON_START", "state", STATE_URI, "sources", len(state["seen_page_ids"]))
    while True:
        try:
            for uri in list(state["pending_manifests"]):
                process_manifest(uri, state)
            discover_batch(state)
            for uri in list(state["pending_manifests"]):
                process_manifest(uri, state)
        except Exception as exc:
            say("CYCLE_ERROR", type(exc).__name__, str(exc)[:1000])
        say("SLEEP", 300, "seconds", "processed_clips", len(state["processed_clip_uris"]))
        time.sleep(300)


if __name__ == "__main__":
    main()
