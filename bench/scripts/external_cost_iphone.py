#!/usr/bin/env python3
"""iPhone 16 Pro driver for `cherenkov-bench external-cost` (#168).

Runs on the Mac mini where the phone is attached, through devicectl.
Several jobs share the phone and the same bundle id, so the device lock
(`fcntl.flock`) is held for a run's whole cycle — uninstall, install,
push args, launch (always `--terminate-existing`: the app never exits,
so a plain launch would reuse a stale instance), wait for done, pull —
and released after the pull.

iOS carries the same matrix minus `--energy` (the meter is ODPM/
powermetrics only) and `--cpu` (affinity is Linux/Android only).

Raw reports land in `--out-dir`, never in git.
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


def wait_done(dest, launched_at, want_args):
    """Copy Documents/out when a fresh done.json exists; retry to the
    timeout. Freshness is proven by the recorded args matching and by
    run-0.log's first tracing timestamp at/after the launch instant."""
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


def verify(report_path, size, transfer, path):
    """The pulled report must come from this build's `external-cost`
    run — fields only that subcommand writes, at the right cell."""
    r = json.loads(pathlib.Path(report_path).read_text())
    want_path = {"e": "external", "c": "copy-convert"}[path]
    want_layout = {"sdr": "nv12", "pq": "p010"}[transfer]
    assert r["path"] == want_path, r["path"]
    assert r["layout"] == want_layout, r["layout"]
    assert r["transfer"] == f"bt{'709-sdr' if transfer == 'sdr' else '2020-pq'}"
    assert (r["width"], r["height"]) == (
        {"1080p": (1920, 1080), "4k": (3840, 2160)}[size]
    )
    assert r["samples"], "no per-frame samples"
    return r


def one_run(app, path, size, transfer, rep, out_dir):
    name = f"{size}-{transfer}-{path}-{rep}"
    remote = f"Documents/out/ext-{name}.json"
    args = [
        "external-cost", "--path", path, "--size", size,
        "--transfer", transfer, "--frames", str(FRAMES),
        "--warmup", str(WARMUP), "--rate", str(RATE),
        "--out", remote,
    ]
    dest = out_dir / f"run-{name}"
    with Lock():
        print(f"[{size}/{transfer} rep{rep}] path {path} — lock held", flush=True)
        # Another job may have installed its own dev.cherenkov.bench —
        # uninstall first so install() provably lands this binary.
        devicectl("device", "uninstall", "app", "--device", UDID, BUNDLE)
        devicectl("device", "install", "app", "--device", UDID, app)

        args_file = out_dir / "bench-args.json"
        args_file.write_text(json.dumps([args]))
        devicectl(
            "device", "copy", "to", "--device", UDID,
            "--domain-type", "appDataContainer",
            "--domain-identifier", BUNDLE,
            "--source", str(args_file),
            "--destination", "Documents/bench-args.json",
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
        if back.read_text() != args_file.read_text():
            raise RuntimeError("bench-args.json readback mismatch")

        launched_at = time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime())
        devicectl(
            "device", "process", "launch",
            "--terminate-existing", "--device", UDID, BUNDLE,
        )
        wait_done(dest, launched_at, args)
    report = verify(dest / f"ext-{name}.json", size, transfer, path)
    print(
        f"    done — lock released; gpu p50 {report['composite_seconds'][0]*1e3:.2f} ms",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default=os.path.expanduser(DEFAULT_APP))
    parser.add_argument("--out-dir", default="/tmp/external-cost-iphone")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for size, transfer in CELLS:
        for order in ORDERS:
            for rep, path in enumerate(order):
                r = one_run(args.app, path, size, transfer, rep, out_dir)
                results.append({"cell": f"{size}/{transfer}", "path": path,
                                "order": "abab" if order == ORDERS[0] else "baba",
                                "rep": rep, "report": name_of(r)})
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print("matrix complete")


def name_of(r):
    return r["path"]


if __name__ == "__main__":
    main()
