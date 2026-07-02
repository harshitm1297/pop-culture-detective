from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.transform.common import write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pipeline(
    *,
    project_root: Path,
    extract_fn,
    extract_args,
    transform_fn,
    load_fn=None,
    load_args=None,
    enable_load: bool = False,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    pipeline_run_id = make_run_id()
    report_dir = paths.reports_root / pipeline_run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "pipeline_manifest.json"

    manifest: dict[str, Any] = {
        "pipeline_run_id": pipeline_run_id,
        "started_at_utc": _utc_now(),
        "status": "running",
        "steps": [],
    }
    write_json(manifest_path, manifest)

    resolved_source_run_id = source_run_id
    try:
        if not resolved_source_run_id:
            manifest["steps"].append(
                {
                    "step": "extract_multisource",
                    "status": "running",
                    "started_at_utc": _utc_now(),
                }
            )
            write_json(manifest_path, manifest)
            resolved_source_run_id = extract_fn(project_root, extract_args)
            manifest["steps"][-1]["status"] = "completed"
            manifest["steps"][-1]["finished_at_utc"] = _utc_now()
            manifest["steps"][-1]["source_run_id"] = resolved_source_run_id
            write_json(manifest_path, manifest)

        manifest["steps"].append(
            {
                "step": "transform_canonical",
                "status": "running",
                "started_at_utc": _utc_now(),
                "source_run_id": resolved_source_run_id,
            }
        )
        write_json(manifest_path, manifest)
        process_run_id = transform_fn(project_root, resolved_source_run_id, None)
        manifest["steps"][-1]["status"] = "completed"
        manifest["steps"][-1]["finished_at_utc"] = _utc_now()
        manifest["steps"][-1]["process_run_id"] = process_run_id

        if enable_load and load_fn is not None:
            manifest["steps"].append(
                {
                    "step": "motherduck_load",
                    "status": "running",
                    "started_at_utc": _utc_now(),
                    "source_run_id": resolved_source_run_id,
                    "process_run_id": process_run_id,
                }
            )
            write_json(manifest_path, manifest)
            load_manifest = load_fn(
                project_root=project_root,
                source_run_id=resolved_source_run_id,
                process_run_id=process_run_id,
                pipeline_run_id=pipeline_run_id,
                enable_motherduck_load=getattr(load_args, "enable_motherduck_load", None),
            )
            manifest["steps"][-1]["status"] = "completed"
            manifest["steps"][-1]["finished_at_utc"] = _utc_now()
            manifest["steps"][-1]["motherduck_manifest_path"] = load_manifest.get("manifest_path")
            manifest["motherduck_manifest_path"] = load_manifest.get("manifest_path")
        manifest["status"] = "completed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["source_run_id"] = resolved_source_run_id
        manifest["process_run_id"] = process_run_id
        write_json(manifest_path, manifest)
        return manifest
    except Exception as exc:  # noqa: BLE001
        if manifest["steps"] and manifest["steps"][-1]["status"] == "running":
            manifest["steps"][-1]["status"] = "failed"
            manifest["steps"][-1]["finished_at_utc"] = _utc_now()
            manifest["steps"][-1]["error"] = str(exc)
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["error"] = str(exc)
        write_json(manifest_path, manifest)
        raise
