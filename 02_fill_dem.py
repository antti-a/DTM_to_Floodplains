#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hydrologically condition (fill) DEMs with WhiteboxTools.

Created on Fri Jul 3 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 2 (see README.md):
    reads   data/01_carved/*.tif        (01_carve_dem.py output)
    writes  data/02_filled/filled_<name>.tif
which is the default input of the next stage (03_flow_router.py).

For every GeoTIFF DEM in the input folder, removes the artefacts that
break downstream flow routing:

  1. fill pits and depressions    (WBT ``FillDepressions``, priority-flood;
                                   a single-cell pit is just a one-cell
                                   depression, so one pass removes both --
                                   WBT's separate ``FillSingleCellPits`` tool
                                   corrupts valid cells to nodata and is
                                   deliberately not used)
  2. resolve flats                (``fix_flats=True`` bakes a tiny gradient
                                   into the elevations of filled flats, so the
                                   output DEM itself drains -- not just some
                                   side-channel flow-direction raster)

The inputs are mosaicked *virtually* with ``gdalbuildvrt`` and filled as one
surface, so depressions spanning tile edges fill correctly (filling is a
global operation and cannot be done lazily inside the VRT itself). The filled
mosaic is then cropped back onto each input's exact grid and written as
``data/02_filled/filled_<name>.tif`` -- the same DEMs, but filled.

The outputs are float64 on purpose. Source elevations are quantized to
the centimetre, so flat terrain is full of exactly tied cells; the
``fix_flats`` gradient that makes those ties drain is far smaller than
float32 can represent at these elevations, and a float32 output silently
collapses the flats right back -- downstream flow routing then dies on a
DEM that merely looks conditioned. float64 keeps the gradients, and the
fill step verifies that the written mosaic actually drains.

Notes:
  * Depression removal is fill-only (no breaching), by design.
  * Depressions draining across the mosaic's outer edge fill to the edge
    elevation ("outlets at edge"); nodata regions likewise act as outlets.
  * Cells that are nodata in an input stay nodata in its output.
  * All inputs must share CRS, cell size and grid alignment (validated).

Run inside the ``water`` conda environment:

    conda activate water
    python 02_fill_dem.py
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import from_bounds

logger = logging.getLogger("fill_dem")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
NODATA = -9999.0             # KM2 convention; also stamped on the outputs
KEEP_INTERMEDIATE = False    # keep outputs/_work/ after a successful run

# Used when the input carries stage-1 provenance tags (dem_source_tiles).
SOURCE_DATA_CREDIT = (
    "National Land Survey of Finland 2 m elevation model (KM2), CC BY 4.0, "
    "carved with the SYKE 'Tierumpujen uomakorjaus' culvert correction "
    "(CC BY 4.0); source tiles in the dem_source_tiles tag."
)
# Fallback for pre-convention inputs without tags.
SOURCE_DATA_CREDIT_PRESUMED = (
    "2 m DEM in EPSG:3067 (ETRS89 / TM35FIN), vertical datum N2000; "
    "presumed source: National Land Survey of Finland 2 m elevation model "
    "(KM2), CC BY 4.0."
)

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
INPUT_DIR = DATA_DIR / "01_carved"    # carved DEMs (01_carve_dem.py output)
OUT_DIR = DATA_DIR / "02_filled"      # filled DEMs -> 03_flow_router.py input
WORK_DIR = OUT_DIR / "_work"          # mosaic + WBT intermediates


# --------------------------------------------------------------------------- #
# GDAL VRT helpers (mosaic the input DEMs virtually)
# --------------------------------------------------------------------------- #
def _gdalbuildvrt_exe() -> str:
    """Locate the ``gdalbuildvrt`` executable shipped with the conda env."""
    exe = shutil.which("gdalbuildvrt")
    if exe:
        return exe
    # Windows-only fallback: on conda/Windows the GDAL CLI tools live in
    # <env>/Library/bin. On Linux/macOS shutil.which() above finds them.
    cand = Path(sys.executable).parent / "Library" / "bin" / "gdalbuildvrt.exe"
    if cand.exists():
        return str(cand)
    raise RuntimeError(
        "gdalbuildvrt not found -- run inside the 'water' conda env."
    )


def build_vrt(
    tiles: Sequence[Path], vrt_path: Path, nodata: float = NODATA
) -> Path:
    """Mosaic ``tiles`` virtually into ``vrt_path`` (no physical merge)."""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _gdalbuildvrt_exe(), "-overwrite", str(vrt_path),
        *[str(t) for t in tiles],
    ]
    logger.info("Building VRT from %d tile(s) -> %s", len(tiles), vrt_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    with rasterio.open(vrt_path) as src:
        if src.nodata is None:
            logger.warning(
                "VRT has no nodata; downstream reads assume %s", nodata
            )
        elif src.nodata != nodata:
            logger.warning("VRT nodata %s != expected %s", src.nodata, nodata)
    return vrt_path


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def validate_inputs(dems: Sequence[Path]) -> None:
    """Require one shared CRS, cell size and grid lattice across ``dems``.

    Filling runs on a mosaic of all inputs, so a tile on a shifted grid or in
    another CRS would be silently resampled by the VRT -- fail instead. Also
    warns on overlapping tiles (gdalbuildvrt stacks them last-wins).
    """
    infos = []
    for path in dems:
        with rasterio.open(path) as src:
            infos.append((path.name, src.crs, src.transform, src.bounds))

    name0, crs0, t0, _ = infos[0]
    res = (abs(t0.a), abs(t0.e))
    for name, crs, t, _ in infos[1:]:
        if crs != crs0:
            raise ValueError(f"{name}: CRS {crs} != {crs0} ({name0})")
        if (abs(t.a), abs(t.e)) != res:
            raise ValueError(
                f"{name}: cell size {(abs(t.a), abs(t.e))} != {res} ({name0})"
            )
        dx = (t.c - t0.c) / t.a
        dy = (t.f - t0.f) / t.e
        if abs(dx - round(dx)) > 1e-6 or abs(dy - round(dy)) > 1e-6:
            raise ValueError(
                f"{name}: grid origin misaligned with {name0} by "
                f"({dx % 1:.6f}, {dy % 1:.6f}) cells"
            )

    for i, (name_a, _, _, ba) in enumerate(infos):
        for name_b, _, _, bb in infos[i + 1:]:
            if (ba.left < bb.right and bb.left < ba.right
                    and ba.bottom < bb.top and bb.bottom < ba.top):
                logger.warning(
                    "Tiles %s and %s overlap; VRT keeps the later one "
                    "in the overlap", name_a, name_b,
                )


# --------------------------------------------------------------------------- #
# Mosaic materialisation (WBT cannot read VRTs)
# --------------------------------------------------------------------------- #
def materialize_mosaic(vrt_path: Path, out_tif: Path,
                       nodata: float = NODATA) -> Path:
    """Copy the virtual mosaic into a real GeoTIFF that WBT can open.

    WBT's GeoTIFF reader panics on the floating-point predictor
    (PREDICTOR=3), so this intermediate is written with predictor 1.
    """
    with rasterio.open(vrt_path) as src:
        dem = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            dem[dem == src.nodata] = nodata
        dem[~np.isfinite(dem)] = nodata
        logger.info(
            "Mosaic grid: %d x %d cells @ %g m, bounds %s",
            src.width, src.height, src.transform.a, tuple(src.bounds),
        )
        return write_dem(
            out_tif, dem, src.transform, src.crs, nodata,
            dtype="float64", predictor=1,
        )


# --------------------------------------------------------------------------- #
# The fill (WhiteboxTools)
# --------------------------------------------------------------------------- #
def count_undrainable(dem_tif: Path, nodata: float = NODATA) -> tuple[int, int]:
    """Count interior cells with no strictly lower neighbour.

    Such a cell (a pit or a flat tie) stops every flow-routing algorithm.
    Nodata cells count as outlets (like the fill treats them) and edge
    cells drain off-grid, so a healthy conditioned DEM leaves only a
    scattered handful. Returns (undrainable_cells, valid_cells).
    """
    with rasterio.open(dem_tif) as src:
        z = src.read(1).astype("float64")
        if src.nodata is not None:
            nodata = src.nodata
    valid = np.isfinite(z) & (z != nodata)
    z[~valid] = -np.inf  # nodata is an outlet: neighbours drain into it
    centre = z[1:-1, 1:-1]
    has_lower = np.zeros(centre.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                has_lower |= (
                    z[1 + dr:z.shape[0] - 1 + dr,
                      1 + dc:z.shape[1] - 1 + dc] < centre
                )
    stuck = valid[1:-1, 1:-1] & ~has_lower
    return int(stuck.sum()), int(valid.sum())


def fill(mosaic_tif: Path, work_dir: Path = WORK_DIR) -> Path:
    """Fill pits + depressions and resolve flats; return the filled mosaic.

    ``FillDepressions`` is a priority-flood fill that removes single-cell
    pits along with larger depressions; ``fix_flats=True`` applies an
    automatically chosen tiny increment across flats so the result drains
    everywhere. That increment only survives the file round-trip in
    float64 (see the module docstring), so the written mosaic is checked
    here: this function fails loudly rather than hand a non-draining
    "filled" DEM downstream.

    WBT panics still exit with code 0, so the existence check on the output
    file is the real success test.
    """
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(work_dir))

    filled_tif = work_dir / "mosaic_filled.tif"

    logger.info("WBT FillDepressions (fix_flats=True) ...")
    ret = wbt.fill_depressions(
        str(mosaic_tif), str(filled_tif),
        fix_flats=True, flat_increment=None, max_depth=None,
    )
    if ret != 0 or not filled_tif.exists():
        raise RuntimeError(f"FillDepressions failed (exit {ret})")

    stuck, valid = count_undrainable(filled_tif)
    logger.info("Drainage check: %d of %d valid cells undrainable", stuck, valid)
    if stuck > max(1, valid // 1000):
        raise RuntimeError(
            f"Filled mosaic does not drain: {stuck} of {valid} cells have no "
            f"lower neighbour. The fix_flats gradients were lost - check that "
            f"every step keeps the DEM in float64."
        )
    return filled_tif


# --------------------------------------------------------------------------- #
# Output helper
# --------------------------------------------------------------------------- #
def write_dem(
    path: Path,
    array: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    nodata: float = NODATA,
    dtype: str = "float32",
    predictor: int | None = None,
    tags: dict | None = None,
) -> Path:
    """Write a single-band GeoTIFF (tiled + compressed) in ``dtype``.

    ``predictor`` overrides the dtype-based default; pass 1 for rasters
    that WBT must read (its reader rejects PREDICTOR=3).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if predictor is None:
        # predictor 3 = floating-point, 2 = integer horizontal differencing
        predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="deflate",
        predictor=predictor,
        bigtiff="IF_SAFER",   # large rasters can exceed 4 GB
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)
    logger.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Crop the filled mosaic back onto each input grid
# --------------------------------------------------------------------------- #
def crop_back(
    filled_tif: Path,
    dem_path: Path,
    out_dir: Path = OUT_DIR,
    nodata: float = NODATA,
) -> Path:
    """Write ``out_dir/filled_<name>.tif`` on ``dem_path``'s exact grid.

    Reads the input's window out of the filled mosaic (same lattice by
    construction, so this is a pure crop), re-applies the input's nodata
    mask, and logs how much was filled as a QC summary.
    """
    dem_path = Path(dem_path)
    with rasterio.open(dem_path) as src:
        orig = src.read(1).astype("float64")
        if src.nodata is not None and src.nodata != nodata:
            orig[orig == src.nodata] = nodata
        transform, crs, bounds = src.transform, src.crs, src.bounds
        src_tags = src.tags()

    with rasterio.open(filled_tif) as src:
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        filled = src.read(
            1, window=window, boundless=True, fill_value=nodata
        ).astype("float64")
    if filled.shape != orig.shape:
        raise RuntimeError(
            f"{dem_path.name}: crop shape {filled.shape} != {orig.shape}"
        )

    valid = (orig != nodata) & np.isfinite(orig)
    filled[~valid] = nodata

    diff = filled[valid] - orig[valid]
    if diff.size and diff.min() < -1e-3:
        raise RuntimeError(
            f"{dem_path.name}: fill lowered cells by up to {-diff.min():.3f} m"
        )
    raised = int(np.count_nonzero(diff > 0))
    logger.info(
        "%s: %d of %d valid cells raised (%.2f %%), max fill depth %.3f m",
        dem_path.name, raised, int(valid.sum()),
        100.0 * raised / max(1, valid.sum()), float(diff.max()) if diff.size else 0.0,
    )

    forwarded = {k: src_tags[k] for k in ("dem_source_tiles", "dem_carve")
                 if k in src_tags}
    if "dem_source_tiles" not in forwarded:
        logger.warning(
            "%s: input carries no provenance tags (pre-convention carved "
            "tile); recording presumed source", dem_path.name,
        )
    return write_dem(
        Path(out_dir) / f"filled_{dem_path.stem}.tif",
        filled, transform, crs, nodata, dtype="float64",
        tags=dict(
            title="Hydrologically conditioned (depression-filled) DEM",
            fill_method="WhiteboxTools FillDepressions, fix_flats=True, run "
                        "on the virtual mosaic of all input tiles and "
                        "cropped back to this tile's grid; float64 preserves "
                        "the flat-fix gradients",
            dem_fill="wbt_fill_depressions_fix_flats",
            source_data_credit=(SOURCE_DATA_CREDIT
                                if "dem_source_tiles" in forwarded
                                else SOURCE_DATA_CREDIT_PRESUMED),
            horizontal_crs="EPSG:3067 (ETRS89 / TM35FIN), units metres",
            vertical_datum="N2000, units metres (datum of the source DEM)",
            generated_by="02_fill_dem.py",
            **forwarded,
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def fill_all(
    input_dir: Path = INPUT_DIR,
    out_dir: Path = OUT_DIR,
    work_dir: Path = WORK_DIR,
    keep_intermediate: bool = KEEP_INTERMEDIATE,
) -> list[Path]:
    """Fill every ``*.tif`` in ``input_dir``; return the written paths."""
    dems = sorted(Path(input_dir).glob("*.tif"))
    if not dems:
        raise FileNotFoundError(f"No .tif DEMs found in {input_dir}")
    logger.info("Found %d DEM(s) in %s", len(dems), input_dir)

    validate_inputs(dems)
    work_dir.mkdir(parents=True, exist_ok=True)

    vrt = build_vrt(dems, work_dir / "inputs_mosaic.vrt")
    mosaic_tif = materialize_mosaic(vrt, work_dir / "mosaic.tif")
    filled_tif = fill(mosaic_tif, work_dir)

    written = [crop_back(filled_tif, d, out_dir) for d in dems]

    if not keep_intermediate:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Removed intermediates in %s", work_dir)
    return written


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    written = fill_all()
    print("\nFilled DEM(s):")
    for p in written:
        print(f"  {p}")
