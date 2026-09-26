"""Keep GVHMR loaded while processing JSONL motion-export requests on stdin."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from run import GVHMRSystem


def main():
    system = GVHMRSystem(load_h1_optimizer=False)
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        result_path = Path(request.pop("result_path"))
        try:
            result = system.process_video_raw_motion(
                video_path_str=request["video_path"],
                output_dir=request["output_dir"],
                raw_motion_target_fps=20,
                dataset_output_dir=request["output_dir"],
                humanml_code_dir=request["humanml_code_dir"],
                humanml_offsets_path=request["humanml_offsets_path"],
                source_uri=request["source_uri"],
                source_license=request["source_license"],
                source_object_sha256=request["source_object_sha256"],
                parent_source_uri=request["parent_source_uri"],
                source_page=request["source_page"],
            )
            response = {"status": "success", "result": result}
        except Exception as exc:
            traceback.print_exc()
            response = {"status": "error", "type": type(exc).__name__, "message": str(exc)}
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(response, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(result_path)


if __name__ == "__main__":
    main()
