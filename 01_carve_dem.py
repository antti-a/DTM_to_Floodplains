#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carve DEMs with the SYKE culvert / road-crossing correction layer.

Created on Fri Jul 3 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

Pipeline stage 1 (see README.md):
    reads   data/00_source_dems/*.tif
    writes  data/01_carved/carved_<name>.tif
which is the default input of the next stage (02_fill_dem.py).

For every GeoTIFF DEM in the input folder, downloads the matching window
of the SYKE "Tierumpujen uomakorjaus" correction raster (windowed WCS, never
the whole country), aligns it onto the DEM's exact grid, and applies a
cell-wise minimum ("carve"): ``min(DEM, culvert)`` where the culvert layer
has data, DEM elsewhere. The carved DEM lowers elevations at culverts and
pipe crossings so flow routes through road embankments instead of being
falsely dammed.

The *whole* DEM is carved -- no catchment polygon, no buffering, and no
depression filling (that is stage 2's job). The only output per input DEM
is ``data/01_carved/carved_<name>.tif``.

Coordinate / vertical reference (KM2 and the culvert layer both ship this way):
  * Horizontal: EPSG:3067 (ETRS89 / TM35FIN), metres.
  * Vertical:   N2000, metres.

Run inside the ``water`` conda environment:

    conda activate water
    python 01_carve_dem.py
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
import requests
from owslib.wcs import WebCoverageService
from rasterio.warp import Resampling, reproject

logger = logging.getLogger("carve_dem")

# --------------------------------------------------------------------------- #
# Defaults / constants
# --------------------------------------------------------------------------- #
TARGET_CRS = 3067            # EPSG:3067 ETRS89 / TM35FIN
RES = 2.0                    # native cell size (m), shared by KM2 and culvert
NODATA = -9999.0             # KM2 and culvert both use this

# SYKE culvert-correction WCS -- windowed GetCoverage only
# (never the whole country).
WCS_URL = (
    "https://paikkatiedot.ymparisto.fi/geoserver/"
    "syke_korkeusmallinuomakorjaus/wcs"
)
WCS_COVERAGE = "syke_korkeusmallinuomakorjaus__Tierumpujen_uomakorjaus"
WCS_AXIS_LABELS = ("E", "N")   # subset axis labels of the coverage envelope
WCS_TIMEOUT = 120              # per-tile WCS read timeout [s] (fail fast)
WCS_RETRIES = 4                # per-tile download attempts on timeout
CULVERT_TILE_KM = 10.0         # culvert tile size [km] (try 5 if WCS stalls)

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
INPUT_DIR = DATA_DIR / "00_source_dems"   # source DEM GeoTIFFs to carve
OUT_DIR = DATA_DIR / "01_carved"          # carved DEMs -> 02_fill_dem.py input
CULVERT_DIR = DATA_DIR / "culvert_cache"  # cached windowed culvert downloads


# --------------------------------------------------------------------------- #
# GDAL VRT helpers (mosaic the downloaded culvert tiles virtually)
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
# Culvert layer (windowed WCS download + grid alignment)
# --------------------------------------------------------------------------- #
def _tile_edges(lo, hi, step, res=RES):
    """Return tile edges over ``lo``..``hi`` snapped to ``res``."""
    lo = math.floor(lo / res) * res
    hi = math.ceil(hi / res) * res
    n = max(1, math.ceil((hi - lo) / step))
    edges = [lo + i * step for i in range(n)]
    edges.append(hi)
    return edges


def _tile_bounds(bounds, tile_m, res=RES):
    """Split ``bounds`` into a grid of <= ``tile_m`` windows."""
    minx, miny, maxx, maxy = bounds
    xs = _tile_edges(minx, maxx, tile_m, res)
    ys = _tile_edges(miny, maxy, tile_m, res)
    return [
        (x0, y0, x1, y1)
        for x0, x1 in zip(xs[:-1], xs[1:])
        for y0, y1 in zip(ys[:-1], ys[1:])
    ]


def _get_coverage(wcs, coverage, fmt, crs_uri, subsets, timeout,
                  retries=WCS_RETRIES):
    """Fetch one WCS coverage window as bytes, retrying flaky timeouts."""
    for attempt in range(1, retries + 1):
        try:
            resp = wcs.getCoverage(
                identifier=coverage, format=fmt, crs=crs_uri,
                timeout=timeout, subsets=subsets,
            )
            return resp.read()
        except requests.exceptions.RequestException as exc:
            if attempt == retries:
                raise
            wait = 5 * attempt
            logger.warning(
                "    attempt %d/%d failed (%s); retrying in %ds",
                attempt, retries, type(exc).__name__, wait,
            )
            time.sleep(wait)


def download_culvert(
    bounds: tuple[float, float, float, float],
    out_path: Path,
    wcs_url: str = WCS_URL,
    coverage: str = WCS_COVERAGE,
    axis_labels: tuple[str, str] = WCS_AXIS_LABELS,
    fmt: str = "image/tiff",
    target_crs: int = TARGET_CRS,
    force: bool = False,
    tile_km: float = CULVERT_TILE_KM,
    timeout: int = WCS_TIMEOUT,
    nodata: float = NODATA,
) -> Path:
    """Download the culvert layer for ``bounds`` as tiles, mosaicked to a VRT.

    ``bounds`` is (minx, miny, maxx, maxy) in EPSG:3067 metres. The window is
    split into ``tile_km`` sub-windows, each fetched with a windowed WCS
    GetCoverage (tiles cached on disk, so a re-run skips finished tiles and
    resumes after a failure). Tiling keeps every request small enough for the
    server's timeout/limits, so any DEM size works; the whole-country raster
    is never fetched. The VRT is rebuilt from the tiles every run, so a copied
    tile cache stays valid on another machine. The VRT path is returned.
    """
    out_path = Path(out_path)
    tiles_dir = out_path.parent / f"{out_path.stem}_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    ex, ny = axis_labels
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{target_crs}"
    windows = _tile_bounds(bounds, tile_km * 1000.0)
    logger.info("Culvert WCS: %d tile(s) of <=%g km", len(windows), tile_km)

    wcs = None
    tiles = []
    for i, (tminx, tminy, tmaxx, tmaxy) in enumerate(windows):
        tile = tiles_dir / f"culvert_{int(tminx)}_{int(tminy)}.tif"
        tiles.append(tile)
        if tile.exists() and not force:
            continue
        if wcs is None:
            wcs = WebCoverageService(wcs_url, version="2.0.1")
        logger.info(
            "  tile %d/%d E[%s,%s] N[%s,%s]", i + 1, len(windows),
            tminx, tmaxx, tminy, tmaxy,
        )
        subsets = [(ex, tminx, tmaxx), (ny, tminy, tmaxy)]
        data = _get_coverage(wcs, coverage, fmt, crs_uri, subsets, timeout)
        fd, tmp = tempfile.mkstemp(suffix=".tif", dir=str(tiles_dir))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, tile)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    build_vrt(tiles, out_path, nodata)
    return out_path


def align_to_reference(
    src_path: Path,
    ref_transform: rasterio.Affine,
    ref_shape: tuple[int, int],
    ref_crs: rasterio.crs.CRS,
    nodata: float = NODATA,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Resample ``src_path`` onto the exact reference grid.

    KM2 and the culvert layer share a 2 m lattice, so nearest-neighbour is
    an exact copy; doing it explicitly guarantees an identical
    shape/transform even if the WCS server snaps the window differently,
    eliminating any half-cell offset before the cell-wise minimum. Source
    nodata becomes ``nodata``.
    """
    dst = np.full(ref_shape, nodata, dtype="float32")
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=nodata,
            resampling=resampling,
        )
    return dst


# --------------------------------------------------------------------------- #
# Cell-wise minimum (the carve)
# --------------------------------------------------------------------------- #
def carved_minimum(
    dem: np.ndarray, culvert: np.ndarray, nodata: float = NODATA
) -> np.ndarray:
    """``min(DEM, culvert)`` where the culvert layer has data; DEM elsewhere.

    Strict nodata handling: the culvert layer is mostly nodata and only carries
    real (lowered) elevations at crossings. nodata must NOT be read as a low
    number that wins the minimum -- so the minimum is taken only where *both*
    layers are valid. Cells with no culvert value keep the DEM; cells where the
    DEM is nodata stay nodata.

    ``dem`` is modified in place and returned (the arrays cover the whole
    DEM, so avoiding a copy matters at large scale).
    """
    if dem.shape != culvert.shape:
        raise ValueError(
            f"shape mismatch: DEM {dem.shape} vs culvert {culvert.shape}"
        )

    dem_valid = (dem != nodata) & np.isfinite(dem)
    cul_valid = (culvert != nodata) & np.isfinite(culvert)
    both = dem_valid & cul_valid
    lowered = int(np.count_nonzero(both & (culvert < dem)))
    np.minimum(dem, culvert, out=dem, where=both)
    dem[~dem_valid] = nodata

    logger.info(
        "Carved: %d cells carry a culvert value, %d lowered below the DEM",
        int(cul_valid.sum()), lowered,
    )
    return dem


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
) -> Path:
    """Write a single-band GeoTIFF (tiled + compressed) in ``dtype``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        # predictor 3 = floating-point, 2 = integer horizontal differencing
        predictor=3 if np.issubdtype(np.dtype(dtype), np.floating) else 2,
        bigtiff="IF_SAFER",   # large rasters can exceed 4 GB
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    logger.info("Wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# Per-DEM carve
# --------------------------------------------------------------------------- #
def carve_dem(
    dem_path: Path,
    out_dir: Path = OUT_DIR,
    culvert_dir: Path = CULVERT_DIR,
    nodata: float = NODATA,
    force_download: bool = False,
) -> Path:
    """Carve one DEM with the SYKE culvert layer over its full extent.

    Reads the whole DEM, downloads the culvert correction for the DEM's
    bounds (cached in ``culvert_dir``), aligns it onto the DEM grid, applies
    the cell-wise minimum, and writes ``out_dir/carved_<name>.tif``.
    """
    dem_path = Path(dem_path)
    name = dem_path.stem
    logger.info("=== Carving %s ===", dem_path.name)

    with rasterio.open(dem_path) as src:
        if src.crs is None or src.crs.to_epsg() != TARGET_CRS:
            raise ValueError(
                f"{dem_path.name}: expected EPSG:{TARGET_CRS}, got {src.crs}"
            )
        dem = src.read(1)
        if dem.dtype != np.float32:
            dem = dem.astype("float32")
        transform = src.transform
        crs = src.crs
        bounds = tuple(src.bounds)
        src_nodata = src.nodata
    if src_nodata is not None and src_nodata != nodata:
        dem[dem == src_nodata] = nodata
    logger.info(
        "DEM grid: %d x %d cells @ %g m, bounds %s",
        dem.shape[1], dem.shape[0], transform.a, bounds,
    )

    culvert_tif = download_culvert(
        bounds,
        Path(culvert_dir) / f"culvert_{name}.vrt",
        force=force_download,
        nodata=nodata,
    )
    culvert = align_to_reference(
        culvert_tif, transform, dem.shape, crs, nodata
    )

    carved = carved_minimum(dem, culvert, nodata)
    del culvert
    return write_dem(
        Path(out_dir) / f"carved_{name}.tif", carved, transform, crs, nodata
    )


def carve_all(
    input_dir: Path = INPUT_DIR,
    out_dir: Path = OUT_DIR,
    culvert_dir: Path = CULVERT_DIR,
) -> list[Path]:
    """Carve every ``*.tif`` in ``input_dir``; return the written paths."""
    dems = sorted(Path(input_dir).glob("*.tif"))
    if not dems:
        raise FileNotFoundError(f"No .tif DEMs found in {input_dir}")
    logger.info("Found %d DEM(s) in %s", len(dems), input_dir)
    return [carve_dem(d, out_dir, culvert_dir) for d in dems]


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    written = carve_all()
    print("\nCarved DEM(s):")
    for p in written:
        print(f"  {p}")
