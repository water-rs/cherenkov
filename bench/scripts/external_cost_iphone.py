#!/usr/bin/env python3
"""iPhone 16 Pro driver for `cherenkov-bench external-cost` (#168).

Runs on the Mac mini where the phone is attached, through devicectl.
Several jobs share the phone and the same bundle id, so the device lock
(`fcntl.flock`) is held for one run's whole cycle — uninstall, install,
push args, launch (always `--terminate-existing`: the app never exits,
so a plain launch would reuse a stale instance), wait for done, pull —
and released after the pull. Time blocked on the lock accumulates; past
15 minutes the driver stops with whatever it has already pulled.

Before the measurement the host writes `Documents/thermal.json` from
`ProcessInfo.thermalState` and does not run the bench unless the state
is nominal or fair. A serious or critical reading ends that launch, the
lock is released, and the driver waits outside the lock until the next
probe. iOS has no host-side thermal query, so the probe is that launch.

iOS carries the same matrix minus `--energy` (the meter is ODPM/
powermetrics only) and `--cpu` (affinity is Linux/Android only).

A pull is accepted only when `done.json`'s args are this launch's
(they carry the run id), the log contains that id, and the report is
the `external-cost` cell this binary was installed to measure.

Raw reports land in `--out-dir`, never in git.

SIGTERM and SIGHUP end the run through one exit path: status.json
records ``matrix stopped: <signal>`` and the log prints that line.
A driver launched under nohup, which ignores SIGHUP, becomes its own
session before the handler is installed, so the launching shell's
hangup is not delivered and an explicit ``kill -HUP`` still is.
"""

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid

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
LOCK_BUDGET_S = 15 * 60
COOL_LIMIT_S = 300
COOL_GAP_S = 15
OK_THERMAL = ("nominal", "fair")

# Seconds spent blocked in flock across the process.
lock_waited = 0.0


class LockBudget(Exception):
    """Cumulative time blocked on the device lock passed 15 minutes."""


class ThermalTimeout(Exception):
    """The phone stayed above fair for the cool-down limit."""


class Stopped(Exception):
    """SIGTERM or SIGHUP. The process records the stop and exits."""

    def __init__(self, signame):
        super().__init__(signame)
        self.signame = signame


def run(cmd, timeout=180):
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def devicectl(*args):
    result = run(["xcrun", "devicectl", *args])
    if result.returncode != 0:
        raise RuntimeError(
            "devicectl {} failed: {}".format(args, result.stderr or result.stdout)
        )
    return result.stdout


def device_copy_from(source, dest):
    """Copy `source` out of the app container.

    The destination must not already exist: a file source lands as that
    file, and a directory source lands as its contents, without a
    Documents/ wrapper. Copying a file onto an existing directory fails.
    """
    if dest.exists():
        if dest.is_dir():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    return run([
        "xcrun", "devicectl", "device", "copy", "from",
        "--device", UDID, "--domain-type", "appDataContainer",
        "--domain-identifier", BUNDLE,
        "--source", source, "--destination", str(dest),
    ])


def fetched(source, dest):
    """The file `device_copy_from` just wrote, wherever devicectl put it."""
    if dest.is_file():
        return dest
    if dest.is_dir():
        return find_named(dest, pathlib.Path(source).name)
    return None


def find_named(root, name):
    direct = root / name
    if direct.is_file():
        return direct
    matches = [path for path in root.rglob(name) if path.is_file()]
    if not matches:
        return None
    return min(matches, key=lambda path: len(path.parts))


def uninstall():
    result = run([
        "xcrun", "devicectl", "device", "uninstall", "app",
        "--device", UDID, BUNDLE,
    ])
    if result.returncode == 0:
        return
    blob = (result.stderr or "") + (result.stdout or "")
    lowered = blob.lower()
    if "not installed" in lowered or "could not find" in lowered or "not found" in lowered:
        return
    raise RuntimeError("devicectl uninstall failed: {}".format(blob))


class Lock:
    """Exclusive flock. The blocked time counts toward the 15-minute budget."""

    def __enter__(self):
        global lock_waited
        self.held = False
        self.f = open(LOCK, "a+")
        remaining = LOCK_BUDGET_S - lock_waited
        if remaining <= 0:
            self.f.close()
            raise LockBudget(
                "lock wait already {:.0f}s".format(lock_waited)
            )
        started = time.monotonic()
        acquired = threading.Event()

        def grab():
            try:
                fcntl.flock(self.f, fcntl.LOCK_EX)
            except OSError:
                return
            acquired.set()

        threading.Thread(target=grab, daemon=True).start()
        ok = acquired.wait(remaining)
        waited = time.monotonic() - started
        lock_waited += waited
        if not ok:
            raise LockBudget(
                "lock wait {:.0f}s exceeds 15 min".format(lock_waited)
            )
        if lock_waited > LOCK_BUDGET_S:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            raise LockBudget(
                "lock wait {:.0f}s exceeds 15 min".format(lock_waited)
            )
        self.held = True
        print(
            "lock acquired after {:.1f}s (cumulative {:.1f}s)".format(
                waited, lock_waited
            ),
            flush=True,
        )
        return self

    def __exit__(self, *exc):
        if self.held:
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            self.held = False


def cdhash(app):
    result = run(["codesign", "-dvvv", app])
    blob = result.stderr + result.stdout
    for line in blob.splitlines():
        if line.startswith("CDHash="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("codesign reported no CDHash for {}".format(app))


def exe_sha256(app):
    path = pathlib.Path(app) / "CherenkovBench"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def wait_file(source, dest, timeout, predicate):
    """Poll `source` until it copies to `dest` and `predicate` accepts it."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        result = device_copy_from(source, dest)
        found = fetched(source, dest)
        if found is not None:
            try:
                if predicate(found):
                    return found.read_bytes()
            except (OSError, json.JSONDecodeError, KeyError, IndexError, AssertionError) as exc:
                last = exc
        else:
            last = (result.stderr or result.stdout or "").strip()
        time.sleep(2)
    raise RuntimeError(
        "timeout waiting for {} ({})".format(source, last)
    )


def verify(report_path, size, transfer, path):
    """The pulled report is this build's `external-cost` cell."""
    report = json.loads(pathlib.Path(report_path).read_text())
    want_path = {"e": "external", "c": "copy-convert"}[path]
    want_layout = {"sdr": "nv12", "pq": "p010"}[transfer]
    assert report["path"] == want_path, report["path"]
    assert report["layout"] == want_layout, report["layout"]
    assert report["transfer"] == "bt{}-{}".format(
        "709" if transfer == "sdr" else "2020",
        "sdr" if transfer == "sdr" else "pq",
    ), report["transfer"]
    assert (report["width"], report["height"]) == {
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }[size]
    assert report["measured_frames"] == FRAMES
    assert len(report["samples"]) == FRAMES, len(report["samples"])
    assert report["total_seconds"], "no per-frame gpu total"
    assert any(sample["gpu_seconds"] is not None for sample in report["samples"]), (
        "no gpu timestamps"
    )
    if path == "c":
        assert any(
            sample["handoff_seconds"] is not None for sample in report["samples"]
        ), "path c recorded no handoff stamps"
    assert str(report["backend"]).lower() == "metal", report["backend"]
    assert "apple" in report["adapter"].lower(), report["adapter"]
    return report


def cycle(app, path, size, transfer, rep, order_name, run_id, out_dir, identity):
    """One locked cycle. Returns ("ok", row) or ("hot", state)."""
    name = "{}-{}-{}-{}".format(size, transfer, path, rep)
    remote = "Documents/out/ext-{}-{}.json".format(name, run_id)
    args = [
        "external-cost", "--path", path, "--size", size,
        "--transfer", transfer, "--frames", str(FRAMES),
        "--warmup", str(WARMUP), "--rate", str(RATE),
        "--out", remote,
    ]
    print(
        "[{}/{} {} rep{}] path {} run {} — installing".format(
            size, transfer, order_name, rep, path, run_id
        ),
        flush=True,
    )
    uninstall()
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
    back_path = out_dir / "args-readback.json"
    back_result = device_copy_from("Documents/bench-args.json", back_path)
    back = fetched("Documents/bench-args.json", back_path)
    if back is None or json.loads(back.read_text()) != [args]:
        raise RuntimeError(
            "bench-args.json readback failed: {}".format(
                (back_result.stderr or back_result.stdout or "").strip()
            )
        )

    stale_path = out_dir / "thermal-stale.json"
    device_copy_from("Documents/thermal.json", stale_path)
    if fetched("Documents/thermal.json", stale_path) is not None:
        raise RuntimeError("stale thermal.json survived uninstall")

    launched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    devicectl(
        "device", "process", "launch",
        "--terminate-existing", "--device", UDID, BUNDLE,
    )

    thermal_bytes = wait_file(
        "Documents/thermal.json",
        out_dir / "thermal.json",
        60,
        lambda path: "state" in json.loads(path.read_text()),
    )
    thermal = json.loads(thermal_bytes.decode())["state"]
    print("    thermal {}".format(thermal), flush=True)
    if thermal not in OK_THERMAL:
        wait_file(
            "Documents/out/done.json",
            out_dir / "done-hot.json",
            45,
            lambda path: json.loads(path.read_text()).get("thermal") == thermal,
        )
        uninstall()
        return ("hot", thermal)

    def done_ok(path):
        done = json.loads(path.read_text())
        row = done["results"][0]
        return row["args"] == args and row["exit_code"] == 0

    run_dir = out_dir / "run-{}-{}".format(name, run_id)
    scratch = out_dir / ".out-pull"
    deadline = time.monotonic() + 300
    pulled = None
    while time.monotonic() < deadline:
        result = device_copy_from("Documents/out", scratch)
        if result.returncode == 0 or scratch.exists():
            done = find_named(scratch, "done.json")
            log = find_named(scratch, "run-0.log")
            report_file = find_named(scratch, "ext-{}-{}.json".format(name, run_id))
            if (
                done is not None
                and log is not None
                and report_file is not None
                and done_ok(done)
                and run_id in log.read_text(errors="replace")
            ):
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                shutil.copytree(scratch, run_dir)
                pulled = run_dir
                break
        time.sleep(2)
    if pulled is None:
        names = []
        if scratch.exists():
            names = sorted(
                item.name for item in scratch.rglob("*") if item.is_file()
            )
        raise RuntimeError(
            "timeout waiting for done.json run {}; pulled {}".format(run_id, names)
        )

    report_path = find_named(pulled, "ext-{}-{}.json".format(name, run_id))
    report = verify(report_path, size, transfer, path)
    accept = {
        "run_id": run_id,
        "launched_at_utc": launched_at,
        "thermal": thermal,
        "cdhash": identity["cdhash"],
        "exe_sha256": identity["exe_sha256"],
        "args": args,
        "adapter": report["adapter"],
        "backend": report["backend"],
    }
    (pulled / "accept.json").write_text(json.dumps(accept, indent=2))
    gpu = report["total_seconds"]
    print(
        "    accepted {} thermal {} gpu p50 {:.2f} ms".format(
            run_id, thermal, gpu[0] * 1e3
        ),
        flush=True,
    )
    return ("ok", {
        "cell": "{}/{}".format(size, transfer),
        "path": path,
        "order": order_name,
        "rep": rep,
        "run_id": run_id,
        "thermal": thermal,
        "dir": run_dir.name,
        "gpu_p50_s": gpu[0],
        "gpu_p99_s": gpu[2],
    })


def one_run(app, path, size, transfer, rep, order_name, out_dir, identity):
    """Thermal-gated run. Retries once on an operational failure."""
    last = None
    for attempt in (1, 2):
        run_id = uuid.uuid4().hex[:12]
        cool_start = None
        try:
            while True:
                with Lock():
                    kind, payload = cycle(
                        app, path, size, transfer, rep, order_name,
                        run_id, out_dir, identity,
                    )
                if kind == "ok":
                    return payload
                if cool_start is None:
                    cool_start = time.monotonic()
                elapsed = time.monotonic() - cool_start
                if elapsed > COOL_LIMIT_S:
                    raise ThermalTimeout(
                        "thermal stayed {} for {:.0f}s".format(payload, elapsed)
                    )
                print(
                    "    thermal {}; cooling {:.0f}s outside the lock".format(
                        payload, elapsed
                    ),
                    flush=True,
                )
                time.sleep(COOL_GAP_S)
        except (LockBudget, ThermalTimeout, Stopped):
            raise
        except Exception as exc:
            last = exc
            print(
                "    attempt {} failed: {}".format(attempt, exc),
                flush=True,
            )
    raise last


def write_status(path, payload):
    payload["lock_wait_s"] = round(lock_waited, 1)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def signal_done(out_dir, state):
    fifo = out_dir / "done.fifo"
    try:
        fd = os.open(str(fifo), os.O_WRONLY | os.O_NONBLOCK)
    except OSError:
        return
    try:
        os.write(fd, (state + "\n").encode())
    finally:
        os.close(fd)


def install_stop_signals():
    """Raise Stopped on SIGTERM and SIGHUP.

    nohup sets SIGHUP to ignore so an ssh hangup does not kill a
    detached driver. Leave that process group first — an explicit
    ``kill -HUP`` is still delivered to the pid — then install the
    handler. A foreground run, whose SIGHUP is not ignored, keeps its
    session and records the hangup.
    """
    if (
        signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        and os.getpid() != os.getsid(0)
    ):
        try:
            os.setsid()
        except OSError:
            pass

    names = {signal.SIGTERM: "SIGTERM", signal.SIGHUP: "SIGHUP"}

    def handle(signum, _frame):
        raise Stopped(names[signum])

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGHUP, handle)


def finish(out_dir, status_path, status, code, line):
    """Write status.json, wake a fifo waiter, print the matrix line, exit."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    write_status(status_path, status)
    signal_done(out_dir, status["state"])
    print(line, flush=True)
    sys.exit(code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", default=os.path.expanduser(DEFAULT_APP))
    parser.add_argument("--out-dir", default="/tmp/external-cost-iphone")
    parser.add_argument("--head", default="unknown")
    parser.add_argument(
        "--lock-waited",
        type=float,
        default=0.0,
        help="seconds already spent blocked on the device lock",
    )
    args = parser.parse_args()
    global lock_waited
    lock_waited = args.lock_waited
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "status.json"
    results = []
    status = {
        "state": "running",
        "reason": "",
        "head": args.head,
        "cdhash": "",
        "exe_sha256": "",
        "results": results,
    }
    write_status(status_path, status)
    install_stop_signals()
    code = 0
    line = "matrix complete"
    try:
        identity = {"cdhash": cdhash(args.app), "exe_sha256": exe_sha256(args.app)}
        status["cdhash"] = identity["cdhash"]
        status["exe_sha256"] = identity["exe_sha256"]
        print(
            "binary {} {} lock_waited {:.1f}s".format(
                identity["cdhash"], identity["exe_sha256"][:16], lock_waited
            ),
            flush=True,
        )
        for size, transfer in CELLS:
            for order in ORDERS:
                order_name = "abab" if order == ORDERS[0] else "baba"
                for rep, path in enumerate(order):
                    row = one_run(
                        args.app, path, size, transfer, rep,
                        order_name, out_dir, identity,
                    )
                    results.append(row)
                    status["results"] = results
                    write_status(status_path, status)
        status["state"] = "complete"
    except Stopped as exc:
        code = 2
        status["state"] = "stopped"
        status["reason"] = "matrix stopped: {}".format(exc.signame)
        line = status["reason"]
    except LockBudget as exc:
        code = 2
        status["state"] = "stopped"
        status["reason"] = str(exc)
        print("stopped: {}".format(exc), flush=True)
        line = "matrix stopped"
    except Exception as exc:
        code = 1
        status["state"] = "failed"
        status["reason"] = str(exc)
        traceback.print_exc()
        line = "matrix failed"
    status["results"] = results
    finish(out_dir, status_path, status, code, line)


if __name__ == "__main__":
    main()