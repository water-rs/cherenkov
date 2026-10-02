#!/usr/bin/env python3
"""iPhone 16 Pro driver for `cherenkov-bench external-cost` (#168).

Runs on the Mac mini where the phone is attached, through devicectl.
Takes the device lock `/tmp/device-locks/<udid>.lock` with fcntl.flock
for each run only and releases it between runs — other jobs share the
phone.

iOS carries the same matrix minus `--energy` (the meter is ODPM/powermetrics
only) and `--cpu` (affinity is Linux/Android only).

Per run: one bench-args.json entry, one app launch, one pull of
`Documents/out`. Raw reports land in `--out-dir` (never in git).
"""

import argparse
import fcntl
import json
import os
import pathlib
import subprocess
import time

UDID = "00008140-001845681E98801C"
LOCK = "/tmp/device-locks/00008140-001845681E98801C.lock"
BUNDLE = "dev.cherenkov.bench"
DEFAULT_APP = (
    "~/Coding/water-rs/cherenkov-wt-168/bench/ios/build/Build/Products/"
    "Release-iphoneos/CherenkovBench.app"
)

CELLS = [("1080p", "sdr"), ("1080p", "pq"), ("4k", "sdr"), ("4k", "pq")]
ORDERS = [("e", "c", "e", "c"), ("c", "e", "c", "e")]
FRAMES = 240
WARMUP = 30
RATE = 120


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


class Lock:
    """fcntl.flock over the iPhone's device-lock file."""

    def __enter__(self):
        self.f = open(LOCK, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def devicectl(*args):
    r = run(["xcrun", "devicectl", *args])
    if r.returncode != 0:
        raise RuntimeError(f"devicectl {args}: {r.stderr or r.stdout}")
    return r.stdout


def install(app):
    devicectl("device", "install", "app", "--device", UDID, app)


def push_args(out_dir, args):
    path = out_dir / "bench-args.json"
    path.write_text(json.dumps([args]))
    devicectl(
        "device", "copy", "to", "--device", UDID,
        "--domain-type", "appDataContainer",
        "--domain-identifier", BUNDLE,
        "--source", str(path), "--destination", "Documents/bench-args.json",
    )
    # Read back: a stale args file is the silent wrong-suite failure.
    back = out_dir / "args-readback.json"
    devicectl(
        "device", "copy", "from", "--device", UDID,
        "--domain-type", "appDataContainer",
        "--domain-identifier", BUNDLE,
        "--source", "Documents/bench-args.json",
        "--destination", str(back),
    )
    if back.read_text() != path.read_text():
        raise RuntimeError("bench-args.json readback mismatch")


def launch():
    devicectl(
        "device", "process", "launch",
        "--terminate-existing", "--device", UDID, BUNDLE,
    )


def wait_done(dest, launched_at, want_args):
    """Copy Documents/out when a fresh done.json exists; retry to the
    timeout. Freshness is proven by run-0.log's first tracing timestamp
    at/after the launch instant and by the recorded args matching."""
    deadline = time.time() + 300
    while time.time() < deadline:
        r = run([
            "xcrun", "devicectl", "device", "copy", "from",
            "--device", UDID, "--domain-type", "appDataContainer",
            "--domain-identifier", BUNDLE,
            "--source", "Documents/out", "--destination", str(dest),
        ])
        done = dest / "done.json"
        log = dest / "run-0.log"
        if r.returncode == 0 and done.exists() and log.exists():
            try:
                d = json.loads(done.read_text())
                fresh_args = d["results"][0]["args"] == want_args
                fresh_ts = (
                    log.read_text(errors="replace").splitlines()[0][:27]
                    >= launched_at
                )
                if fresh_args and fresh_ts:
                    return
            except (OSError, KeyError, IndexError, json.JSONDecodeError):
                pass
        time.sleep(4)
    raise RuntimeError(f"timeout waiting for done.json in {dest}")


def one_run(app, path, size, transfer, rep, out_dir):
    name = f"ext-{size}-{transfer}-{path}-{rep}"
    args = [
        "external-cost", "--path", path, "--size", size,
        "--transfer", transfer, "--frames", str(FRAMES),
        "--warmup", str(WARMUP), "--rate", str(RATE),
        "--out", f"Documents/out/{name}.json",
    ]
    with Lock():
        print(f"[{size}/{transfer} rep{rep}] path {path} — lock held", flush=True)
        push_args(out_dir, args)
        launched_at = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime())
        launch()
        dest = out_dir / name
        wait_done(dest, launched_at, args)
    print(f"    done — lock released", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default=os.path.expanduser(DEFAULT_APP))
    parser.add_argument("--out-dir", default="/tmp/external-cost-iphone")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Lock():
        install(args.app)
        print("installed", flush=True)

    for size, transfer in CELLS:
        for order in ORDERS:
            for rep, path in enumerate(order):
                one_run(args.app, path, size, transfer, rep, out_dir)
    print("matrix complete")


if __name__ == "__main__":
    main()