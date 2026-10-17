"""Immutable, timestamped experiment artifacts with SHA-256 manifests."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_file(path: Path, full: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {"path": str(path), "bytes": stat.st_size}
    h = hashlib.sha256()
    if full:
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        record["sha256"] = h.hexdigest()
        record["hash_mode"] = "content"
    else:
        h.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}".encode())
        record["sha256"] = h.hexdigest()
        record["hash_mode"] = "metadata"
    return record


def fingerprint(path: str | os.PathLike[str], mode: str = "full") -> dict[str, Any]:
    """Hash one file or a directory tree deterministically."""
    root = Path(path).resolve()
    if not root.exists():
        return {"path": str(root), "exists": False}
    full = mode == "full"
    if root.is_file():
        return {"path": str(root), "exists": True, **_hash_file(root, full)}
    files = []
    tree_hash = hashlib.sha256()
    total = 0
    for child in sorted(
        p for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ):
        item = _hash_file(child, full)
        rel = child.relative_to(root).as_posix()
        files.append({"relative_path": rel, **{k: v for k, v in item.items() if k != "path"}})
        total += item["bytes"]
        tree_hash.update(rel.encode("utf-8") + b"\0")
        tree_hash.update(item["sha256"].encode("ascii") + b"\0")
    return {
        "path": str(root), "exists": True, "kind": "directory",
        "file_count": len(files), "bytes": total,
        "sha256": tree_hash.hexdigest(), "hash_mode": mode, "files": files,
    }


def _git_state() -> dict[str, Any]:
    try:
        root = Path(config.PROJECT_ROOT)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


class RunContext:
    """Owns one immutable run directory and its machine-readable manifest."""

    def __init__(self, stage: str, parameters: dict[str, Any], root: str | None = None,
                 hash_mode: str = "full", run_group: str | None = None):
        self.stage = stage
        self.parameters = parameters
        self.hash_mode = hash_mode
        self.started_at = utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cfg_hash = canonical_hash({"stage": stage, "parameters": parameters})
        self.run_id = f"{stamp}_{stage}_{cfg_hash[:10]}"
        self.root = Path(root or config.RUNS_DIR).resolve()
        self.path = self.root / self.run_id
        self.artifacts_dir = self.path / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=False)
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_group": run_group,
            "stage": stage,
            "status": "running",
            "started_at_utc": self.started_at,
            "parameters": parameters,
            "parameter_sha256": cfg_hash,
            "command": sys.argv,
            "cwd": os.getcwd(),
            "python": sys.version,
            "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
            "git": _git_state(),
            "code": {
                "src": fingerprint(Path(config.PROJECT_ROOT) / "src", "full"),
                "requirements": fingerprint(Path(config.PROJECT_ROOT) / "requirements.txt", "full"),
                "docker_compose": fingerprint(Path(config.PROJECT_ROOT) / "docker" / "docker-compose.yml", "full"),
            },
            "inputs": {},
            "artifacts": {},
        }
        self._finalized = False
        self._write_manifest()
        atexit.register(self._finalize_if_aborted)

    def artifact_path(self, relative: str) -> str:
        path = self.artifacts_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def record_input(self, name: str, path: str, mode: str | None = None) -> None:
        self.manifest["inputs"][name] = fingerprint(path, mode or self.hash_mode)
        self._write_manifest()

    def record_artifact(self, name: str, path: str, mode: str | None = None) -> None:
        self.manifest["artifacts"][name] = fingerprint(path, mode or self.hash_mode)
        self._write_manifest()

    def write_json(self, name: str, value: Any, relative: str | None = None) -> str:
        path = Path(self.artifact_path(relative or f"{name}.json"))
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
        self.record_artifact(name, str(path), "full")
        return str(path)

    def finalize(self, status: str = "complete", error: str | None = None) -> None:
        if self._finalized:
            return
        self.manifest["status"] = status
        self.manifest["completed_at_utc"] = utc_now()
        if error:
            self.manifest["error"] = error
        self._write_manifest()
        manifest_path = self.path / "manifest.json"
        manifest_sha = _hash_file(manifest_path, True)["sha256"]
        (self.path / "manifest.sha256").write_text(f"{manifest_sha}  manifest.json\n")
        pointer_name = f"latest_{self.stage}.json" if status == "complete" else f"last_{self.stage}_attempt.json"
        pointer = self.root / pointer_name
        tmp = pointer.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "run_id": self.run_id, "stage": self.stage,
            "run_dir": str(self.path), "manifest": str(self.path / "manifest.json"),
            "manifest_sha256": manifest_sha, "status": status,
        }, indent=2) + "\n")
        os.replace(tmp, pointer)
        self._finalized = True

    def _finalize_if_aborted(self) -> None:
        if not self._finalized:
            try:
                self.finalize("aborted", "Process exited before normal run finalization")
            except Exception:
                pass

    def _write_manifest(self) -> None:
        path = self.path / "manifest.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(tmp, path)


def latest_artifact(stage: str, name: str, root: str | None = None) -> str | None:
    pointer = Path(root or config.RUNS_DIR) / f"latest_{stage}.json"
    if not pointer.exists():
        return None
    run = json.loads(pointer.read_text())
    manifest = json.loads(Path(run["manifest"]).read_text())
    item = manifest.get("artifacts", {}).get(name)
    return item.get("path") if item and item.get("exists") else None


def publish_copy(source: str, destination: str) -> None:
    src, dst = Path(source), Path(destination)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def publish_tree(source: str, destination: str) -> None:
    """Merge a directory into a legacy output location without deleting peers."""
    src, dst = Path(source), Path(destination)
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.rglob("*"):
        relative = child.relative_to(src)
        target = dst / relative
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect tracked experiment artifacts")
    ap.add_argument("--latest", metavar="STAGE")
    ap.add_argument("--artifact")
    ap.add_argument("--fingerprint")
    ap.add_argument("--hash-mode", choices=["full", "metadata"], default="full")
    args = ap.parse_args()
    if args.fingerprint:
        print(json.dumps(fingerprint(args.fingerprint, args.hash_mode), indent=2))
    elif args.latest:
        if args.artifact:
            value = latest_artifact(args.latest, args.artifact)
            if not value:
                raise SystemExit(f"No artifact {args.artifact!r} for stage {args.latest!r}")
            print(value)
        else:
            pointer = Path(config.RUNS_DIR) / f"latest_{args.latest}.json"
            if not pointer.exists():
                raise SystemExit(f"No completed run for stage {args.latest!r}")
            print(pointer.read_text(), end="")
    else:
        ap.error("use --latest STAGE or --fingerprint PATH")


if __name__ == "__main__":
    main()
