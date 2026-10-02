"""Pixel 9 Pro driver for `cherenkov-bench external-cost` (#168).

Run only under `pixel-adb --run`: its device lock covers the whole
experiment (one adb channel, `su -c` for the root-only ODPM rails).

Matrix: per size×transfer cell the two paths run interleaved —
e,c,e,c then the reversed control c,e,c,e — so drift cancels. The
battery/thermal gate is checked before every measured window, with the
screen off the whole time (the bench draws offscreen).

Raw reports land in `--out-dir` (default /tmp/external-cost-pixel),
never in git.
"""

import argparse
import datetime
import json
import pathlib
import re
import subprocess
import time

BENCH = "/data/local/tmp/cherenkov-bench"
CELLS = [("1080p", "sdr"), ("1080p", "pq"), ("4k", "sdr"), ("4k", "pq")]
# ABAB, then the reversed BABA control.
ORDERS = [("e", "c", "e", "c"), ("c", "e", "c", "e")]
# The X4 prime core: `measure`'s documented pin for the Pixel.
CPU = "7"
# 240 frames at 120 Hz is a 2 s window — the ODPM energy counters tick
# at a coarse cadence and a sub-second window can read zero.
FRAMES = 240
WARMUP = 30
RATE = 120.0


def adb(*args):
    result = subprocess.run(
        ["adb", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"adb {args}: {result.stdout}")
    return result.stdout


def shell(command):
    return adb("shell", command)


def thermal():
    service = shell("dumpsys thermalservice")
    status = int(re.search(r"Thermal Status:\s*(\d+)", service)[1])
    # The cached section may be hours old. The same dumpsys call also asks
    # the HAL for current readings, including the battery in degrees Celsius.
    current = service.split("Current temperatures from HAL:", 1)[1].split(
        "Current cooling devices from HAL:", 1
    )[0]
    battery = re.search(
        r"Temperature\{mValue=([\d.]+), mType=2, mName=battery,", current
    )
    if battery is None:
        raise RuntimeError("thermalservice returned no current battery temperature")
    return {"battery_c": float(battery[1]), "status": status}


def cool_enough(sample):
    return sample["battery_c"] < 45.0 and sample["status"] < 2


def cooled():
    shell("input keyevent KEYCODE_SLEEP")
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        sample = thermal()
        if cool_enough(sample):
            return sample
        print(
            f"screen-off cooling: battery={sample['battery_c']} C, status={sample['status']}",
            flush=True,
        )
        time.sleep(5)
    raise RuntimeError("cooling timeout: battery must be below 45 C and status below 2")


def push(binary):
    subprocess.run(["adb", "push", binary, BENCH], check=True, capture_output=True)
    shell(f"chmod 755 {BENCH}")


def run(path, size, transfer, rep, out_dir):
    remote = f"/data/local/tmp/ext-{size}-{transfer}-{path}-{rep}.json"
    cmd = (
        f"su -c '{BENCH} external-cost --path {path} --size {size} "
        f"--transfer {transfer} --frames {FRAMES} --warmup {WARMUP} "
        f"--rate {RATE} --energy --cpu {CPU} --out {remote}'"
    )
    print(f"[{size}/{transfer} rep{rep}] path {path} ...", flush=True)
    out = shell(cmd)
    if "external-cost" not in out:
        raise RuntimeError(f"bench failed: {out}")
    local = out_dir / f"{size}-{transfer}-{path}-{rep}.json"
    subprocess.run(["adb", "pull", remote, str(local)], check=True, capture_output=True)
    shell(f"rm {remote}")
    return json.loads(local.read_text())


def summarize(report):
    p = lambda key: (report.get(key) or [None, None, None])
    composite = p("composite_seconds")
    total = p("total_seconds")
    energy = report.get("energy") or {}
    memory = report.get("memory") or {}
    steady = (memory.get("steady") or {}).get("process") or {}
    gpu = (((memory.get("steady") or {}).get("engine") or {}).get("value") or {})
    return {
        "gpu_ms_p50": (composite[0] or 0.0) * 1e3,
        "gpu_ms_p99": (composite[2] or 0.0) * 1e3,
        "total_ms_p50": (total[0] or 0.0) * 1e3,
        "total_ms_p99": (total[2] or 0.0) * 1e3,
        "encode_ms_p50": report["encode_seconds"][0] * 1e3,
        "encode_ms_p99": report["encode_seconds"][2] * 1e3,
        "submit_ms_p50": report["submit_seconds"][0] * 1e3,
        "handoff_ms_p50": (report.get("handoff_seconds") or [None])[0],
        "joules_per_frame": energy.get("joules_per_frame"),
        "watts": energy.get("watts"),
        "steady_rss_bytes": steady.get("rss_bytes"),
        "steady_gpu_bytes": gpu.get("gpu_bytes"),
        "missed": (report.get("pacing") or {}).get("missed_deadlines"),
        "thermal": (report.get("conditions") or {}).get("thermal_status"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="aarch64 release binary")
    parser.add_argument("--out-dir", default="/tmp/external-cost-pixel")
    parser.add_argument("--runs", type=int, default=4, help="runs per order (ABAB => 4)")
    args = parser.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    push(args.binary)
    shell("input keyevent KEYCODE_SLEEP")

    results = []
    initial = cooled()
    print(f"initial thermal: {initial}", flush=True)
    for size, transfer in CELLS:
        for order in ORDERS:
            for rep, path in enumerate(order[: args.runs]):
                sample = cooled()
                report = run(path, size, transfer, rep, out_dir)
                row = {
                    "cell": f"{size}/{transfer}",
                    "path": path,
                    "order": "abab" if order == ORDERS[0] else "baba",
                    "rep": rep,
                    "battery_c": sample["battery_c"],
                    "status": sample["status"],
                    **summarize(report),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
    summary = {
        "device": "pixel9pro",
        "binary": args.binary,
        "finished": datetime.datetime.now(datetime.UTC).isoformat(),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()