#!/usr/bin/env python3
"""Run the DEM / hydrology pipeline end to end.

Created on Tue Jul 7 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

The stage scripts already read and write the shared ``data/`` tree, so each
one also runs standalone with no arguments; this runner only executes them
in the right order and stops at the first failure.

    stage         script                       reads                writes
    ------------  ---------------------------  -------------------  ------------------
    carve         01_carve_dem.py              data/00_source_dems  data/01_carved
    fill          02_fill_dem.py               data/01_carved       data/02_filled
    route         03_flow_router.py            data/02_filled       data/03_flows
    catchments    04_delineate_catchments.py   data/03_flows        data/04_catchments
    accumulation  05_flow_accumulation.py      data/03_flows        data/05_accumulation
    hand          06_hand.py                   data/02_filled +     data/06_hand
                                               data/03_flows +
                                               data/05_accumulation
    floodplains   07_floodplains.py            (same as hand)       data/07_floodplains

USAGE (inside the ``water`` conda environment)
    python 00_run_pipeline.py                     # carve -> fill -> route
                                               #   -> catchments + accumulation
                                               #   -> hand -> floodplains
    python 00_run_pipeline.py --from route        # resume after an earlier run
    python 00_run_pipeline.py --only fill route   # just these stages
    python 00_run_pipeline.py --skip catchments   # everything else
    python 00_run_pipeline.py --area 0.5          # stream threshold, km2 (route
                                               #   and catchments together;
                                               #   hand/floodplains have their
                                               #   own --upa-min)

Catchment delineation defaults to method 2 (confluence criterion) because it
is the only method that needs no outlet coordinates; run
04_delineate_catchments.py directly for methods 1 and 3.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Pipeline order.
STAGES = ("carve", "fill", "route", "catchments", "accumulation",
          "hand", "floodplains")


def stage_commands(args) -> dict[str, list[str]]:
    py = sys.executable
    return {
        "carve": [py, str(HERE / "01_carve_dem.py")],
        "fill": [py, str(HERE / "02_fill_dem.py")],
        "route": [py, str(HERE / "03_flow_router.py"), "--area", str(args.area)],
        "catchments": [py, str(HERE / "04_delineate_catchments.py"),
                       "--method", "2",
                       "--stream-threshold", str(args.area)],
        "accumulation": [py, str(HERE / "05_flow_accumulation.py")],
        # hand/floodplains read stage 5's D8 accumulation; no flags passed,
        # so they run on their USER SETTINGS defaults (--upa-min 0.2 etc.)
        "hand": [py, str(HERE / "06_hand.py")],
        "floodplains": [py, str(HERE / "07_floodplains.py")],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run the DEM/hydrology pipeline stages in order.",
        epilog=f"stages, in order: {', '.join(STAGES)}")
    ap.add_argument("--only", nargs="+", choices=STAGES, metavar="STAGE",
                    help="run only these stages (in pipeline order)")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, metavar="STAGE",
                    help="start from this stage instead of 'carve'")
    ap.add_argument("--skip", nargs="+", default=[], choices=STAGES,
                    metavar="STAGE", help="leave these stages out")
    ap.add_argument("--area", type=float, default=1.0, metavar="KM2",
                    help="minimum contributing area defining a stream, used "
                         "by both route and catchments (default 1.0); hand "
                         "and floodplains have their own --upa-min instead")
    args = ap.parse_args(argv)

    if args.only:
        selected = [s for s in STAGES if s in args.only]
    else:
        selected = list(STAGES)
        if args.from_stage:
            selected = selected[selected.index(args.from_stage):]
    selected = [s for s in selected if s not in args.skip]
    if not selected:
        ap.error("no stages left to run")

    commands = stage_commands(args)
    print(f"Pipeline: {' -> '.join(selected)}\n")
    t0 = time.perf_counter()
    for stage in selected:
        cmd = commands[stage]
        print(f"=== {stage} " + "=" * max(1, 66 - len(stage)))
        t1 = time.perf_counter()
        # cwd=HERE so the stages' relative paths resolve next to the scripts.
        ret = subprocess.run(cmd, cwd=str(HERE)).returncode
        if ret != 0:
            print(f"\nStage '{stage}' failed (exit {ret}); pipeline stopped. "
                  f"Fix the problem and resume with: "
                  f"python 00_run_pipeline.py --from {stage}")
            return ret
        print(f"=== {stage} done in {time.perf_counter() - t1:.1f} s\n")

    print(f"Pipeline finished in {time.perf_counter() - t0:.1f} s. "
          f"Results are under {HERE / 'data'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
