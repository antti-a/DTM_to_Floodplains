# DTM_to_Floodplains — DEM / hydrology pipeline

From raw elevation tiles to geomorphic floodplains in six stages. Every
stage reads and writes a shared `data/` tree, so **the output of each stage
is already the default input of the next** — no copying of files between
folders. Each script also works standalone (run it with no arguments, or
point it elsewhere with its own CLI options).

The stage scripts are numbered in pipeline order (`01_` … `06_`), and the
runner that calls them is `00_run_pipeline.py`. Three files carry no
number: they are not stages but companion modules — `mdinf.py` (MDinf
directions, used through `03_flow_router.py` when the optional MDinf
method is requested), `accumulation.py` (the flow-accumulation kernel
shared by stages 3 and 4) and `pipeline_io.py` (the shared I/O layer of
stages 3–6: tile discovery and validation, provenance-tag merging,
in-memory mosaicking, D8/accumulation raster loading, lock-tolerant
tagged-GeoTIFF writing).

```
data/00_source_dems/*.tif                 source KM2 DEM tiles (drop them here)
        |
        v  01_carve_dem.py        carve SYKE culvert corrections (WCS download,
        |                         cached in data/culvert_cache/)
data/01_carved/carved_*.tif
        |
        v  02_fill_dem.py         fill pits/depressions,
        |                         resolve flats (pysheds, float64)
data/02_filled/filled_*.tif
        |
        v  03_flow_router.py      route D8 (default; MFD, Dinf,
        |                         MDinf via --fdir)
data/03_flows/flow_direction_*.tif  (+ networks, summary CSV)
        |
        v  04_flow_accumulation.py
data/04_accumulation/flow_accumulation_*.tif
        |
        |  5. and 6. read flow_accumulation_d8.tif together with
        |  data/02_filled and the D8 raster from data/03_flows
        |
        v  05_hand.py
data/05_hand/hand.tif
        |
        v  06_floodplains.py      (same inputs as stage 5;
                                   hand.tif itself is not read)
data/06_floodplains/floodplains.tif
```

## Quick start

```bash
git clone https://github.com/antti-a/DTM_to_Floodplains.git
cd DTM_to_Floodplains
conda env create -f environment.yml
conda activate water
# drop your KM2 DEM tiles (GeoTIFF, EPSG:3067) into data/00_source_dems/
python 00_run_pipeline.py
```

The runner executes the stages in order and stops at the first failure;
resume with `python 00_run_pipeline.py --from <stage>` after fixing the
problem, or run any subset with `--only` / `--skip`.

The stages can equally be run by hand, in order — each knows its own
defaults:

```bash
python 01_carve_dem.py
python 02_fill_dem.py
python 03_flow_router.py
python 04_flow_accumulation.py
python 05_hand.py
python 06_floodplains.py
```

## Changing parameters

Every stage is a command-line script: `python <script> --help` lists all of
its options. The ones that matter most:

| stage | option | meaning |
|---|---|---|
| 3 | `--area KM2` (also on `00_run_pipeline.py`) | minimum contributing area defining a stream (default 1.0) |
| 3 | `--fdir d8 mfd dinf mdinf` / `--fdir all` | which routing algorithms to run; **default is D8 only** — what stages 4–6 consume |
| 4 | `--units m2/cells` | accumulation in square metres (default) or cell counts |
| 5 + 6 | `--upa-min KM2` | stream-initiation threshold for HAND/floodplains (default 0.2 — denser than the stage-3 network) |
| 6 | `--a`, `--b` | GFPLAIN power law `h = a·A^b` (defaults 0.63 and 0.3); calibrate against observed flood extents |

Stages 1–2 take no options — their few constants (CRS, resolution, nodata,
folder names) sit at the top of each script. Stages 5 and 6 additionally
have a `USER SETTINGS` block near the top of the script whose values act as
the defaults for a no-argument run; any CLI flag overrides them.

## The stages

| # | script | reads | writes |
|---|--------|-------|--------|
| 1 | `01_carve_dem.py` | `data/00_source_dems/` | `data/01_carved/` (+ `data/culvert_cache/`) |
| 2 | `02_fill_dem.py` | `data/01_carved/` | `data/02_filled/` |
| 3 | `03_flow_router.py` | `data/02_filled/` | `data/03_flows/` |
| 4 | `04_flow_accumulation.py` | `data/03_flows/flow_direction_*.tif` | `data/04_accumulation/` |
| 5 | `05_hand.py` | `data/02_filled/` + `data/03_flows/flow_direction_d8.tif` + `data/04_accumulation/flow_accumulation_d8.tif` | `data/05_hand/` |
| 6 | `06_floodplains.py` | same as stage 5 | `data/06_floodplains/` |

Notes per stage:

1. **Carve** — lowers the DEM at culverts / road crossings with the SYKE
   "Tierumpujen uomakorjaus" WCS layer so flow crosses road embankments.
   Downloads are windowed, tiled and cached in `data/culvert_cache/`
   (a re-run skips finished tiles).
2. **Fill** — pysheds `fill_depressions` (priority-flood; Barnes et al.,
   2014) plus `resolve_flats` (Barnes et al., 2014) on the virtual mosaic
   of all tiles (depressions spanning tile edges fill correctly), then
   crops back to each tile's grid. Outputs are **float64** on purpose: the
   flat-resolution gradients are smaller than float32 can hold at these
   elevations, and saving as float32 would silently un-condition the DEM.
   `03_flow_router.py` verifies this and stops if the DEM does not drain.
3. **Route** — mosaics the filled tiles, verifies drainage, and routes flow.
   **D8 alone by default** — the format every later stage consumes; MFD,
   Dinf (pysheds) and MDinf (directions from `mdinf.py`, accumulated by
   `accumulation.py`) run on request via `--fdir` (e.g. `--fdir all`
   to compare all four). Each selected algorithm gets its stream network
   (GeoJSON, WGS84), its row in the comparison table (CSV) and its
   `flow_direction_*.tif` raster.
4. **Accumulation** (required by stages 5–6) — standalone weighted flow
   accumulation for every flow-direction raster it finds, in m² (default)
   or cell counts.
5. **HAND** (optional) — height above nearest drain (Nobre et al., 2016):
   for every cell, the elevation difference to the stream cell it drains to
   along the D8 flow path. pyflwdir only turns the existing
   `flow_direction_d8.tif` into a flow graph, and the stream mask comes
   from thresholding stage 4's `flow_accumulation_d8.tif` at `--upa-min`
   (default 0.2 km²). D8 only by design: the down-to-upstream propagation
   needs exactly one downstream cell per cell, which MFD/Dinf do not
   provide ("nearest drain" is ambiguous under divergent flow). One
   deviation from pyflwdir: cells whose flow path ends in a pit or at the
   grid edge without meeting a stream are nodata instead of getting a
   misleading "height above pit" (matters on a whole-DEM run, where small
   edge catchments never reach the threshold).
6. **Floodplains** — GFPLAIN geomorphic floodplains (Nardi et al., 2019).
   Every stream cell (same `--upa-min` mask as HAND) carries the flood
   level `h = a·A^b` (h in m, A = upstream area in km², from stage 4);
   a cell joins the floodplain of the stream cell it drains to (D8) if it
   rises no more than `h` above it. `--a` (default 0.63) and `--b` (default
   0.3) are meant to be calibrated against observed/modelled flood extents —
   fix the literature exponent b ≈ 0.30 first, then fit `a`.

## Conventions the stages share

* **CRS** EPSG:3067 (ETRS89 / TM35FIN, metres); **vertical** N2000 (metres).
* **DEM nodata** is −9999.0 through stages 1–3; flow-direction and
  accumulation rasters use NaN (or 0 for the D8 code raster). Stage 5's
  `hand.tif` is float32 with nodata −9999 (also marking cells that never
  reach a stream); stage 6's `floodplains.tif` is int8 with 1 = floodplain,
  0 = upland, −1 = nodata.
* **Tiles**: every DEM stage accepts 1..n adjacent GeoTIFF tiles, validates
  that they share one CRS / cell size / grid lattice, mosaics them, and
  processes the mosaic as one surface so nothing stops at a tile seam.
* **Filled DEMs stay float64** end-to-end (see stage 2 above).
* **Flow-direction encodings** (written into each file's band descriptions
  and tags by `03_flow_router.py`):
  * `d8` — 1 int32 band, ESRI codes 64=N 128=NE 1=E 2=SE 4=S 8=SW 16=W 32=NW;
    −1 flat, −2 pit, 0 nodata.
  * `dinf` — 1 float32 band, flow angle [0, 2π) CCW from east; −1 flat,
    −2 pit, NaN nodata.
  * `mfd`, `mdinf` — 8 float32 bands: fraction of flow to N, NE, E, SE, S,
    SW, W, NW; NaN nodata.

### Provenance tags

Every raster a stage writes carries machine-readable GeoTIFF dataset tags
(read them with `gdalinfo` or `rasterio.open(...).tags()`), alongside the
human-readable prose tags. Each key is stamped once at the stage where the
value originates and forwarded **verbatim** by every later stage that
consumes the file:

| key | example | origin | meaning |
|---|---|---|---|
| `dem_source_tiles` | `L4142E.tif, L4142F.tif` | 1 | original source-DEM filenames (comma+space separated, sorted); mosaic stages record the union across tiles |
| `dem_carve` | `syke_culvert_min` | 1 | culvert carving: cell-wise `min(DEM, SYKE Tierumpujen uomakorjaus)` |
| `dem_fill` | `pysheds_fill_depressions_resolve_flats` | 2 | pysheds `fill_depressions` + `resolve_flats`, float64 |
| `flow_routing_algorithm` | `d8` | 3 | routing algorithm key (`d8`/`mfd`/`dinf`/`mdinf`); stages 5–6 read it from their inputs instead of assuming D8 |
| `stream_threshold_km2` | `0.2` | 5, 6 | `--upa-min`: min upstream area defining stream/drain cells — **not** stage 3's `--area`, which only shapes the vector network |
| `gfplain_a`, `gfplain_b` | `0.63`, `0.3` | 6 | GFPLAIN parameters in `h = a * A**b` |

When the merged tiles disagree on `dem_carve`/`dem_fill` the tag is omitted
with a note. Pre-convention intermediates (rasters written before these tags
existed) only trigger warnings: downstream stages then assume D8, skip the
missing keys and fall back to the older "presumed source" credit wording.

### NaN vs 0 in the 8-band rasters (MFD vs MDinf)

The two 8-band rasters (produced only when stage 3 runs with `--fdir`)
encode "no flow to this neighbour" differently:

* **MFD** (pysheds) stores **NaN** in the bands of a valid cell that receive
  no flow — only the receiving bands hold numbers.
* **MDinf** (`mdinf.py`) stores **0.0** in non-receiving bands of valid
  cells; a cell is all-NaN only when it is nodata or has no downslope facet
  at all (a pit/flat — essentially absent from a conditioned DEM). Near
  flats and edges its fractions can sum to slightly less than 1: flow aimed
  at a nodata/flat neighbour is simply lost from the domain, as in the
  WhiteboxTools implementation the module was ported from.

The in-pipeline consumer handles both conventions explicitly, treating a
cell as valid when *any* band is finite and NaN bands as 0
(`04_flow_accumulation._fractions_from_bands`).
**If you write a new tool that reads `flow_direction_mfd.tif`, do the same** —
naively summing bands, or treating any-NaN as nodata, silently discards most
of the valid MFD cells.

## Credits

The pipeline follows **Rolim da Paz (2025)** — the condition-route-accumulate
workflow of stages 1–4 — and the **pyflwdir** library (Eilander et al., 2021;
https://github.com/Deltares/pyflwdir), the original inspiration for this
project. Stages 5–6 turn the pipeline's D8 raster into a pyflwdir flow graph
(one downstream index per cell plus a down-to-upstream cell ordering), and
their Numba kernels are adapted from `pyflwdir.dem` (compiled for whole-DEM
grids): HAND after Nobre et al. (2016) with the reached-drain nodata rule
described under stage 5, and GFPLAIN after Nardi et al. (2019) with the
coefficient `a` made an explicit parameter.

Other essential tools: pysheds (D8/MFD/Dinf routing; stage 2 depression
filling and flat resolution after Barnes, Lehman and Mulla, 2014),
rasterio/GDAL, NumPy and Numba. The MDinf direction mathematics in
`mdinf.py` follow Seibert and McGlynn (2007), ported via WhiteboxTools'
MIT-licensed implementation (John Lindsay).

Source data: KM2 2 m DEM © National Land Survey of Finland (CC BY 4.0);
culvert corrections: SYKE "Tierumpujen uomakorjaus" WCS (CC BY 4.0).

## References

Barnes, R., Lehman, C. and Mulla, D. (2014) 'Priority-flood: an optimal
depression-filling and watershed-labeling algorithm for digital elevation
models', *Computers & Geosciences*, 62, pp. 117–127. Available at:
https://doi.org/10.1016/j.cageo.2013.04.024

Barnes, R., Lehman, C. and Mulla, D. (2014) 'An efficient assignment of
drainage direction over flat surfaces in raster digital elevation models',
*Computers & Geosciences*, 62, pp. 128–135. Available at:
https://doi.org/10.1016/j.cageo.2013.01.009

Eilander, D., van Verseveld, W., Yamazaki, D., Weerts, A., Winsemius, H.C.
and Ward, P.J. (2021) 'A hydrography upscaling method for scale-invariant
parametrization of distributed hydrological models', *Hydrology and Earth
System Sciences*, 25(9), pp. 5287–5313. Available at:
https://doi.org/10.5194/hess-25-5287-2021

Nardi, F., Annis, A., Di Baldassarre, G., Vivoni, E.R. and Grimaldi, S.
(2019) 'GFPLAIN250m, a global high-resolution dataset of Earth's
floodplains', *Scientific Data*, 6, 180309. Available at:
https://doi.org/10.1038/sdata.2018.309

Nobre, A.D., Cuartas, L.A., Momo, M.R., Severo, D.L., Pinheiro, A. and
Nobre, C.A. (2016) 'HAND contour: a new proxy predictor of inundation
extent', *Hydrological Processes*, 30(2), pp. 320–333. Available at:
https://doi.org/10.1002/hyp.10581

Rolim da Paz, A. (2025) *Digital elevation models for environmental studies*.
Cham: Springer. Available at: https://doi.org/10.1007/978-3-032-04523-2

Seibert, J. and McGlynn, B.L. (2007) 'A new triangular multiple flow
direction algorithm for computing upslope areas from gridded digital
elevation models', *Water Resources Research*, 43(4), W04501. Available at:
https://doi.org/10.1029/2006WR005128
