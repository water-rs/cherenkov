# cherenkov-bench

Engine adapters and the render/measure CLI for the Cherenkov cross-engine
suite. Adapters are feature-gated: `vello-classic`, `vello-hybrid`,
`vello-cpu`, `skia`, `skia-metal`.

## Measuring on big.LITTLE hardware

On hosts whose CPUs have differing `cpuinfo_max_freq` (e.g. Tensor G4:
four 1.95 GHz cores, three 2.6 GHz, one 3.1 GHz) an unpinned `measure`
run lands on any cluster — the same scene can time 10x apart depending
on which core the scheduler picked. Pass `--cpu` to pin the measuring
thread (and every thread the adapter spawns afterwards) to one set:

    cherenkov-bench measure --engine vello-cpu --scene scenes/perf/chart \
        --frames 60 --cpu 7 --out measure.json        # big core only
    cherenkov-bench measure ... --cpu 4-6              # mid cores
    cherenkov-bench measure ... --cpu 0-3              # little cores

`--cpu` works on Linux and Android; elsewhere it is refused with an
error. Each sample records the CPU it started and ended on
(`sched_getcpu`); a sample that migrated mid-phase is flagged and the
report counts them. The report's `placement` block shows the requested
set, `controlled`/`heterogeneous` flags, per-CPU sample counts and each
CPU's `cpuinfo_max_freq`. Without `--cpu` on a heterogeneous host the
run still proceeds but the report marks `controlled: false` and the log
warns once — publish those numbers only with the caveat attached.
