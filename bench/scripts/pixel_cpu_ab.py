#!/usr/bin/env python3
"""Interleaved CPU measurements under pixel-adb's exclusive device lock.

Invoke through ~/.local/bin/pixel-adb --run python3 bench/scripts/pixel_cpu_ab.py.
All adb subprocesses inherit the serial and lock owned by that wrapper.
Cooldown waits consume Android battery-change events, without timed polling.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import time
from zoneinfo import ZoneInfo


def adb(*args):
    return subprocess.check_output(["adb", *map(str, args)], text=True)


def thermal():
    battery = adb("shell", "dumpsys", "battery")
    service = adb("shell", "dumpsys", "thermalservice")
    return {
        "battery_tenths_c": int(re.search(r"temperature: (-?\d+)", battery)[1]),
        "thermal_status": int(re.search(r"Thermal Status: (\d+)", service)[1]),
    }


def ready(state):
    return state["battery_tenths_c"] < 450 and state["thermal_status"] < 2


def wait_for_cooldown():
    state = thermal()
    if ready(state):
        return state
    logging.info("Waiting for battery-change notifications: %s", state)
    with subprocess.Popen(
        ["adb", "logcat", "-b", "events", "-v", "brief", "-T", "1", "battery_level:I", "*:S"],
        stdout=subprocess.PIPE,
        text=True,
    ) as events, selectors.DefaultSelector() as watch:
        watch.register(events.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 30 * 60
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not watch.select(remaining):
                    raise TimeoutError("battery/thermal cooldown exceeded 30 minutes")
                event = events.stdout.readline()
                if not event:
                    raise RuntimeError("battery notification stream ended before cooldown")
                if "battery_level" in event:
                    state = thermal()
                    if ready(state):
                        return state
        finally:
            events.terminate()
            events.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-rev", required=True)
    parser.add_argument("--candidate-rev", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--scenes", nargs="+", default=["map", "chart", "text-page", "ui-list", "effects"])
    args = parser.parse_args()
    if "ANDROID_SERIAL" not in os.environ:
        parser.error("invoke through pixel-adb --run to hold the device lock")
    args.out.mkdir(parents=True, exist_ok=False)
    run_key = hashlib.sha256(str(args.out.resolve()).encode()).hexdigest()[:12]
    root = f"/data/local/tmp/cherenkov-cpu-ab-{args.candidate_rev[:12]}-{run_key}"
    manifest = {
        "baseline_rev": args.baseline_rev,
        "candidate_rev": args.candidate_rev,
        "frames": args.frames,
        "warmup": args.warmup,
        "device": adb("shell", "getprop", "ro.product.model").strip(),
        "serial": os.environ["ANDROID_SERIAL"],
        "device_directory": root,
        "started_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runs": [],
    }
    adb("shell", "mkdir", "-p", root)
    for side, binary in [("A", args.baseline), ("B", args.candidate)]:
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        manifest[f"{side}_sha256"] = digest
        remote = f"{root}/{side}"
        adb("push", binary, remote)
        if adb("shell", "sha256sum", remote).split()[0] != digest:
            raise RuntimeError(f"device binary hash mismatch: {side}")
        adb("shell", "chmod", "755", remote)
    adb("push", args.corpus, f"{root}/perf")
    adb("shell", "input", "keyevent", "223")
    for workers, cpus in [(1, "7"), (8, "0-7")]:
        for scene in args.scenes:
            for index, side in enumerate("ABAB", 1):
                name = f"{scene}-t{workers}-{index}-{side}"
                start = wait_for_cooldown()
                power = adb("shell", "dumpsys", "power")
                if "mHalInteractiveModeEnabled=false" not in power:
                    raise RuntimeError("screen became interactive")
                logging.info("Starting %s: %s", name, start)
                remote = f"{root}/{name}.json"
                command = [
                    "env", f"RAYON_NUM_THREADS={workers}", f"{root}/{side}",
                    "measure", "--engine", "cherenkov-cpu", "--scene", f"{root}/perf/{scene}",
                    "--cpu", cpus, "--frames", str(args.frames), "--warmup", str(args.warmup),
                    "--out", remote,
                ]
                with (args.out / f"{name}.log").open("w") as log:
                    subprocess.run(["adb", "shell", shlex.join(command)], stdout=log, stderr=log, check=True)
                end = thermal()
                adb("pull", remote, args.out / f"{name}.json")
                manifest["runs"].append({
                    "name": name, "start": start, "end": end,
                    "completed_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z"),
                })
                (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                logging.info("Completed %s: %s", name, end)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
