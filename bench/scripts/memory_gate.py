#!/usr/bin/env python3
"""Compare deterministic engine memory across two Cherenkov revisions.

For Lavapipe, export
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
XDG_RUNTIME_DIR=/tmp/runtime-ubuntu RUST_LOG=error before running this
script. It passes the caller's environment through unchanged.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any


SCENES = ("map", "chart", "text-page", "ui-list", "effects")
DEFAULT_ENGINES = ("cherenkov", "cherenkov-cpu")
ROOT_MARKER = ".memory-gate-root.json"
WORKTREE_MARKER = ".memory-gate-worktree.json"
HARNESS_HINT = "pass --harness <ref with #101>"


class GateError(Exception):
    pass


def command(
    args: list[str],
    *,
    cwd: Path | None = None,
    binary_output: bool = False,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=not binary_output,
    )


def git_output(repo: Path, *args: str) -> str:
    result = command(["git", "-C", str(repo), *args])
    if result.returncode:
        raise GateError(
            f"git {' '.join(args)} failed:\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout.strip()


def resolve_commit(repo: Path, ref: str) -> str:
    return git_output(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def ensure_work_root(work_root: Path, repo: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    marker = work_root / ROOT_MARKER
    expected = {"repo": str(repo.resolve())}
    if marker.exists():
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GateError(f"invalid work directory marker {marker}: {error}") from error
        if actual != expected:
            raise GateError(
                f"{work_root} is owned by another repository; use a fresh --work directory"
            )
    else:
        existing = list(work_root.iterdir())
        if existing:
            raise GateError(
                f"{work_root} is not an owned memory-gate directory; "
                "use a fresh --work directory"
            )
        marker.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")


def ensure_worktree(
    repo: Path, work_root: Path, side: str, commit: str
) -> Path:
    path = work_root / f"{side}-{commit[:12]}"
    marker = path / WORKTREE_MARKER
    expected = {"commit": commit, "repo": str(repo.resolve()), "side": side}
    if path.exists():
        if not path.is_dir() or not marker.is_file():
            raise GateError(
                f"refusing to reuse {path}: it is not a marked memory-gate worktree"
            )
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GateError(f"invalid worktree marker {marker}: {error}") from error
        if actual != expected:
            raise GateError(f"worktree marker mismatch at {path}")
        top = Path(git_output(path, "rev-parse", "--show-toplevel")).resolve()
        head = git_output(path, "rev-parse", "HEAD")
        if top != path.resolve() or head != commit:
            raise GateError(f"worktree identity changed at {path}")
        return path

    result = command(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), commit]
    )
    if result.returncode:
        raise GateError(
            f"could not create {side} worktree at {path}:\n"
            f"{result.stdout}{result.stderr}"
        )
    marker.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
    return path


def ensure_clean_except_owned_files(path: Path, *, allow_bench: bool) -> None:
    result = command(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"]
    )
    if result.returncode:
        raise GateError(f"could not inspect owned worktree {path}: {result.stderr}")
    unexpected = []
    for line in result.stdout.splitlines():
        name = line[3:]
        if name == WORKTREE_MARKER:
            continue
        if allow_bench and (name == "bench" or name.startswith("bench/")):
            continue
        unexpected.append(line)
    if unexpected:
        raise GateError(
            f"refusing to build modified {path} worktree:\n" + "\n".join(unexpected)
        )


def overlay_bench(repo: Path, harness: str, base: Path) -> None:
    result = command(
        ["git", "-C", str(repo), "archive", "--format=tar", harness, "bench"],
        binary_output=True,
    )
    if result.returncode:
        raise GateError(f"could not archive bench from harness {harness}:\n{result.stderr}")
    with tempfile.TemporaryDirectory(prefix=".memory-gate-overlay-", dir=base) as temporary:
        temporary_path = Path(temporary)
        archive_root = temporary_path.resolve()
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
                members = archive.getmembers()
                for member in members:
                    destination = (temporary_path / member.name).resolve()
                    if not destination.is_relative_to(archive_root):
                        raise GateError(f"unsafe path in harness archive: {member.name}")
                    if member.issym() or member.islnk():
                        target = (destination.parent / member.linkname).resolve()
                        if not target.is_relative_to(archive_root):
                            raise GateError(
                                f"unsafe link in harness archive: {member.name}"
                            )
                archive.extractall(temporary_path)
        except (OSError, tarfile.TarError) as error:
            raise GateError(f"could not extract harness bench archive: {error}") from error

        replacement = temporary_path / "bench"
        if not replacement.is_dir():
            raise GateError(f"harness {harness} contains no bench directory")
        destination = base / "bench"
        backup = base / ".memory-gate-bench-old"
        if backup.exists():
            raise GateError(f"refusing to replace leftover path {backup}")
        destination.rename(backup)
        try:
            replacement.rename(destination)
        except OSError:
            backup.rename(destination)
            raise
        shutil.rmtree(backup)


def cargo_target_directory(path: Path) -> Path:
    result = command(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"], cwd=path
    )
    if result.returncode:
        raise GateError(f"cargo metadata failed in {path}:\n{result.stdout}{result.stderr}")
    metadata = json.loads(result.stdout)
    return Path(metadata["target_directory"])


def build(path: Path, out: Path, side: str) -> Path:
    log_path = out / f"build-{side}.log"
    result = command(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "cherenkov-bench",
            "--features",
            "cherenkov,cherenkov-cpu",
        ],
        cwd=path,
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise GateError(
            f"release build failed for {side}; see {log_path}\n"
            f"{result.stdout}{result.stderr}"
        )
    executable = cargo_target_directory(path) / "release" / "cherenkov-bench"
    if os.name == "nt":
        executable = executable.with_suffix(".exe")
    if not executable.is_file():
        raise GateError(f"build succeeded but executable is missing: {executable}")
    return executable


def counter_engine_memory(
    report_path: Path, report: dict[str, Any]
) -> dict[str, Any] | None:
    counters = report.get("counters")
    if not isinstance(counters, dict):
        return None
    has_cpu = "memory_cpu_bytes" in counters
    has_gpu = "memory_gpu_bytes" in counters
    if not has_cpu and not has_gpu:
        return None
    if not has_cpu or not has_gpu:
        raise GateError(f"{report_path} has incomplete CPU/GPU engine memory counters")
    cpu_bytes = counters["memory_cpu_bytes"]
    gpu_bytes = counters["memory_gpu_bytes"]
    if cpu_bytes is None and gpu_bytes is None:
        return None
    if (
        not isinstance(cpu_bytes, int)
        or isinstance(cpu_bytes, bool)
        or not isinstance(gpu_bytes, int)
        or isinstance(gpu_bytes, bool)
        or cpu_bytes < 0
        or gpu_bytes < 0
    ):
        raise GateError(f"{report_path} has invalid CPU/GPU engine memory counters")
    return {"measured": {"cpu_bytes": cpu_bytes, "gpu_bytes": gpu_bytes}}


def read_engine_memory(report_path: Path) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"could not read report {report_path}: {error}") from error
    if not isinstance(report, dict):
        raise GateError(f"{report_path} has no memory report; {HARNESS_HINT}")
    memory = report.get("memory")
    if not isinstance(memory, dict):
        counters = counter_engine_memory(report_path, report)
        if counters is not None:
            return counters
        raise GateError(f"{report_path} has no memory report; {HARNESS_HINT}")
    steady = memory.get("steady")
    engine = steady.get("engine") if isinstance(steady, dict) else None
    if not isinstance(engine, dict):
        raise GateError(f"{report_path} has no steady engine memory reading")
    if "unavailable" in engine:
        engine_memory = {"unavailable": engine["unavailable"]}
    else:
        measured = engine.get("measured")
        if not isinstance(measured, dict):
            raise GateError(f"{report_path} has an invalid steady engine memory reading")
        cpu_bytes = measured.get("cpu_bytes")
        gpu_bytes = measured.get("gpu_bytes")
        if (
            not isinstance(cpu_bytes, int)
            or isinstance(cpu_bytes, bool)
            or not isinstance(gpu_bytes, int)
            or isinstance(gpu_bytes, bool)
            or cpu_bytes < 0
            or gpu_bytes < 0
        ):
            raise GateError(f"{report_path} has invalid CPU/GPU engine bytes")
        engine_memory = {"measured": {"cpu_bytes": cpu_bytes, "gpu_bytes": gpu_bytes}}

    counters = counter_engine_memory(report_path, report)
    if counters is not None and counters != engine_memory:
        raise GateError(
            f"{report_path} has inconsistent memory readings: "
            f"memory.steady.engine={engine_memory}, counters={counters}"
        )
    return engine_memory


def measure_side(
    side: str,
    root: Path,
    binary: Path,
    scene_root: Path,
    engines: tuple[str, ...],
    warmup: int,
    frames: int,
    out: Path,
) -> dict[tuple[str, str], dict[str, Any] | str]:
    readings: dict[tuple[str, str], dict[str, Any] | str] = {}
    for scene in SCENES:
        scene_path = scene_root / "scenes" / "perf" / scene
        for engine in engines:
            samples: list[dict[str, Any] | str] = []
            for attempt in (1, 2):
                report_path = out / f"{side}-{scene}-{engine}-{attempt}.json"
                log_path = out / f"{side}-{scene}-{engine}-{attempt}.log"
                result = command(
                    [
                        str(binary),
                        "measure",
                        "--engine",
                        engine,
                        "--scene",
                        str(scene_path),
                        "--warmup",
                        str(warmup),
                        "--frames",
                        str(frames),
                        "--out",
                        str(report_path),
                    ],
                    cwd=root,
                )
                log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
                if result.returncode:
                    samples.append(
                        f"command error: measure exited {result.returncode}; "
                        f"see {log_path}"
                    )
                    continue
                try:
                    samples.append(read_engine_memory(report_path))
                except GateError as error:
                    samples.append(f"report error: {error}")
            key = (scene, engine)
            if len(samples) != 2:
                readings[key] = "nondeterministic: expected two samples"
            elif isinstance(samples[0], dict) != isinstance(samples[1], dict):
                readings[key] = "nondeterministic: same-side memory readings differ"
            elif isinstance(samples[0], dict) and samples[0] != samples[1]:
                readings[key] = "nondeterministic: same-side memory readings differ"
            elif isinstance(samples[0], str) and isinstance(samples[1], str):
                first_kind = samples[0].split(":", maxsplit=1)[0]
                second_kind = samples[1].split(":", maxsplit=1)[0]
                if first_kind != second_kind:
                    readings[key] = "nondeterministic: same-side measure outcomes differ"
                else:
                    readings[key] = samples[0]
            elif isinstance(samples[0], str):
                readings[key] = samples[0]
            elif "unavailable" in samples[0]:
                readings[key] = (
                    f"unavailable: {samples[0]['unavailable']}"
                )
            else:
                readings[key] = samples[0]
    return readings


def format_bytes(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def markdown_table(
    base_ref: str,
    head_ref: str,
    rows: list[tuple[str, str, str, str, str, str, str]],
) -> str:
    lines = [
        "# Engine memory landing gate",
        "",
        f"- Base: `{base_ref}`",
        f"- Head: `{head_ref}`",
        "",
        "| Scene | Engine | Base CPU (B) | Base GPU (B) | Head CPU (B) | Head GPU (B) | Delta |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for scene, engine, base_cpu, base_gpu, head_cpu, head_gpu, delta in rows:
        delta = delta.replace("|", "\\|")
        lines.append(
            f"| {scene} | {engine} | {base_cpu} | {base_gpu} | "
            f"{head_cpu} | {head_gpu} | {delta} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base git ref")
    parser.add_argument("--head", required=True, help="head git ref")
    parser.add_argument(
        "--harness",
        help="ref with #101 bench sources to overlay onto the base worktree",
    )
    parser.add_argument(
        "--engines", default=",".join(DEFAULT_ENGINES), help="comma-separated adapters"
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", required=True, type=Path, help="report output directory")
    parser.add_argument(
        "--work", type=Path, default=Path("/home/ubuntu/memory-gate")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.frames <= 0:
        raise GateError("--warmup must be nonnegative and --frames must be positive")
    engines = tuple(
        engine.strip() for engine in args.engines.split(",") if engine.strip()
    )
    if not engines:
        raise GateError("--engines must name at least one adapter")
    unknown_engines = set(engines) - set(DEFAULT_ENGINES)
    if unknown_engines:
        raise GateError(
            f"this build supports only {', '.join(DEFAULT_ENGINES)}; "
            f"unknown engines: {', '.join(sorted(unknown_engines))}"
        )

    repo = Path(__file__).resolve().parents[2]
    repo = Path(git_output(repo, "rev-parse", "--show-toplevel")).resolve()
    work_root = args.work.expanduser().resolve()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ensure_work_root(work_root, repo)

    base_commit = resolve_commit(repo, args.base)
    head_commit = resolve_commit(repo, args.head)
    harness_commit = resolve_commit(repo, args.harness) if args.harness else None
    base = ensure_worktree(repo, work_root, "base", base_commit)
    head = ensure_worktree(repo, work_root, "head", head_commit)
    ensure_clean_except_owned_files(base, allow_bench=harness_commit is not None)
    ensure_clean_except_owned_files(head, allow_bench=False)
    if harness_commit is not None:
        overlay_bench(repo, harness_commit, base)

    scene_root = head
    if not all(
        (scene_root / "scenes" / "perf" / scene / "scene.json").is_file()
        for scene in SCENES
    ):
        raise GateError(f"head worktree {head} is missing one or more perf scenes")

    base_binary = build(base, out, "base")
    base_readings = measure_side(
        "base", base, base_binary, scene_root, engines, args.warmup, args.frames, out
    )
    head_binary = build(head, out, "head")
    head_readings = measure_side(
        "head", head, head_binary, scene_root, engines, args.warmup, args.frames, out
    )

    rows = []
    failed = False
    for scene in SCENES:
        for engine in engines:
            key = (scene, engine)
            base_value = base_readings[key]
            head_value = head_readings[key]
            if isinstance(base_value, str) or isinstance(head_value, str):
                failed = True
                error = "; ".join(
                    value for value in (base_value, head_value) if isinstance(value, str)
                )
                rows.append(
                    (scene, engine, "—", "—", "—", "—", f"ERROR: {error}")
                )
                continue

            base_bytes = base_value["measured"]
            head_bytes = head_value["measured"]
            cpu_delta = head_bytes["cpu_bytes"] - base_bytes["cpu_bytes"]
            gpu_delta = head_bytes["gpu_bytes"] - base_bytes["gpu_bytes"]
            delta = f"CPU {cpu_delta:+,} B; GPU {gpu_delta:+,} B"
            if cpu_delta or gpu_delta:
                failed = True
            rows.append(
                (
                    scene,
                    engine,
                    format_bytes(base_bytes["cpu_bytes"]),
                    format_bytes(base_bytes["gpu_bytes"]),
                    format_bytes(head_bytes["cpu_bytes"]),
                    format_bytes(head_bytes["gpu_bytes"]),
                    delta,
                )
            )

    table_path = out / "memory-gate.md"
    table = markdown_table(args.base, args.head, rows)
    table_path.write_text(table, encoding="utf-8")
    print(table, end="")
    print(f"table: {table_path}")
    print(f"exit code: {1 if failed else 0}")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as error:
        print(f"memory gate: {error}", file=sys.stderr)
        sys.exit(1)
