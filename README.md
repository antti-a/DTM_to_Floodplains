# DTM_to_Floodplains — DEM / hydrology pipeline

From raw elevation tiles to geomorphic floodplains in seven stages. Every
stage reads and writes a shared `data/` tree, so **the output of each stage
is already the default input of the next** — no copying of files between
folders. Each script also works standalone (run it with no arguments, or
point it elsewhere with its own CLI options).

The stage scripts are numbered in pipeline order (`01_` … `07_`), and the
runner that calls them is `00_run_pipeline.py`. Only `mdinf.py` carries no
number: it is not a stage but a companion module used only through
`03_flow_router.py` when the optional MDinf method is requested.

```
data/00_source_dems/*.tif                 source KM2 DEM tiles (drop them here)
        |
        v  01_carve_dem.py        carve SYKE culvert corrections (WCS download,
        |                         cached in data/culvert_cache/)
data/01_carved/carved_*.tif
        |
        v  02_fill_dem.py         fill pits/depressions,
        |                         resolve flats (WBT, float64)
data/02_filled/filled_*.tif
        |
        v  03_flow_router.py      route D8 (default; MFD, Dinf,
        |                         MDinf via --fdir)
data/03_flows/flow_direction_*.tif  (+ networks, summary CSV)
        |                   |
        v  4. (optional)    v  5. (needed by stages 6-7)
04_delineate_catchments.py  05_flow_accumulation.py
        |                             |
        v                             v
data/04_catchments/          data/05_accumulation/
                                      |
                                      |  6./7. read flow_accumulation_d8.tif
                                      |  together with data/02_filled and the
                                      |  D8 raster from data/03_flows
                                      |
                                      +--> 6. 06_hand.py
                                      |        -> data/06_hand/hand.tif
                                      |
                                      +--> 7. 07_floodplains.py
                                               -> data/07_floodplains/floodplains.tif
```

Stages 6 and 7 recompute nothing: they read the filled DEM (stage 2), the D8
flow directions (stage 3) and the D8 flow accumulation (stage 5) as-is.

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
python 04_delineate_catchments.py --method 2
python 05_flow_accumulation.py
python 06_hand.py
python 07_floodplains.py
```

## Changing parameters

Every stage is a command-line script: `python <script> --help` lists all of
its options. The ones that matter most:

| stage | option | meaning |
|---|---|---|
| 3 + 4 | `--area KM2` (on `00_run_pipeline.py`) | minimum contributing area defining a stream; passed to both route and catchments so they agree (default 1.0) |
| 3 | `--fdir d8 mfd dinf mdinf` / `--fdir all` | which routing algorithms to run; **default is D8 only** — what stages 4–7 consume |
| 4 | `--method 1/2/3`, `--coords E,N` | delineation method; methods 1 and 3 need outlet coordinates |
| 5 | `--units m2/cells` | accumulation in square metres (default) or cell counts |
| 6 + 7 | `--upa-min KM2` | stream-initiation threshold for HAND/floodplains (default 0.2 — denser than the stage-3 network) |
| 7 | `--a`, `--b` | GFPLAIN power law `h = a·A^b` (defaults 0.63 and 0.3); calibrate against observed flood extents |

Stages 1–2 take no options — their few constants (CRS, resolution, nodata,
folder names) sit at the top of each script. Stages 4, 6 and 7 additionally
have a `USER SETTINGS` block near the top of the script whose values act as
the defaults for a no-argument run; any CLI flag overrides them.

## The stages

| # | script | reads | writes |
|---|--------|-------|--------|
| 1 | `01_carve_dem.py` | `data/00_source_dems/` | `data/01_carved/` (+ `data/culvert_cache/`) |
| 2 | `02_fill_dem.py` | `data/01_carved/` | `data/02_filled/` |
| 3 | `03_flow_router.py` | `data/02_filled/` | `data/03_flows/` |
| 4 | `04_delineate_catchments.py` | `data/03_flows/flow_direction_*.tif` | `data/04_catchments/` |
| 5 | `05_flow_accumulation.py` | `data/03_flows/flow_direction_*.tif` | `data/05_accumulation/` |
| 6 | `06_hand.py` | `data/02_filled/` + `data/03_flows/flow_direction_d8.tif` + `data/05_accumulation/flow_accumulation_d8.tif` | `data/06_hand/` |
| 7 | `07_floodplains.py` | same as stage 6 | `data/07_floodplains/` |

Notes per stage:

1. **Carve** — lowers the DEM at culverts / road crossings with the SYKE
   "Tierumpujen uomakorjaus" WCS layer so flow crosses road embankments.
   Downloads are windowed, tiled and cached in `data/culvert_cache/`
   (a re-run skips finished tiles).
2. **Fill** — WhiteboxTools `FillDepressions` with `fix_flats=True` on the
   virtual mosaic of all tiles (depressions spanning tile edges fill
   correctly), then crops back to each tile's grid. Outputs are **float64**
   on purpose: the flat-fix gradients are smaller than float32 can hold at
   these elevations, and saving as float32 would silently un-condition the
   DEM. `03_flow_router.py` verifies this and stops if the DEM does not
   drain.
3. **Route** — mosaics the filled tiles, verifies drainage, and routes flow.
   **D8 alone by default** — the format every later stage consumes; MFD,
   Dinf (pysheds) and MDinf (WhiteboxTools accumulation; directions
   recomputed by `mdinf.py`) run on request via `--fdir` (e.g. `--fdir all`
   to compare all four). Each selected algorithm gets its stream network
   (GeoJSON, WGS84), its row in the comparison table (CSV) and its
   `flow_direction_*.tif` raster.
4. **Catchments** (optional) — three delineation methods from Rolim da Paz
   (2025) ch. 8. `00_run_pipeline.py` uses method 2 (confluence criterion),
   the only one that needs no outlet coordinates; for method 1/3 run the
   script directly with `--coords`. Runs on **every** flow-direction raster
   it finds in `data/03_flows/` (only D8 in a default pipeline run).
5. **Accumulation** (required by stages 6–7) — standalone weighted flow
   accumulation for every flow-direction raster it finds, in m² (default)
   or cell counts.
6. **HAND** (optional) — height above nearest drain (Nobre et al. 2016):
   for every cell, the elevation difference to the stream cell it drains to
   along the D8 flow path. Recomputes nothing — pyflwdir only turns the
   existing `flow_direction_d8.tif` into a flow graph, and the stream mask
   comes from thresholding stage 5's `flow_accumulation_d8.tif` at
   `--upa-min` (default 0.2 km²). D8 only by design: the down-to-upstream
   propagation needs exactly one downstream cell per cell, which MFD/Dinf
   do not provide ("nearest drain" is ambiguous under divergent flow).
   One deviation from pyflwdir: cells whose flow path ends in a pit or at
   the grid edge without meeting a stream are nodata instead of getting a
   misleading "height above pit" (matters on a whole-DEM run, where small
   edge catchments never reach the threshold).
7. **Floodplains** — GFPLAIN geomorphic floodplains (Nardi et al. 2019).
   Every stream cell (same `--upa-min` mask as HAND) carries the flood
   level `h = a·A^b` (h in m, A = upstream area in km², from stage 5);
   a cell joins the floodplain of the stream cell it drains to (D8) if it
   rises no more than `h` above it. `--a` (default 0.63) and `--b` (default
   0.3) are meant to be calibrated against observed/modelled flood extents —
   fix the literature exponent b ≈ 0.30 first, then fit `a`.

## Conventions the stages share

* **CRS** EPSG:3067 (ETRS89 / TM35FIN, metres); **vertical** N2000 (metres).
* **DEM nodata** is −9999.0 through stages 1–3; flow-direction and
  accumulation rasters use NaN (or 0 for the D8 code raster). Stage 6's
  `hand.tif` is float32 with nodata −9999 (also marking cells that never
  reach a stream); stage 7's `floodplains.tif` is int8 with 1 = floodplain,
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

### NaN vs 0 in the 8-band rasters (MFD vs MDinf)

The two 8-band rasters (produced only when stage 3 runs with `--fdir`)
encode "no flow to this neighbour" differently:

* **MFD** (pysheds) stores **NaN** in the bands of a valid cell that receive
  no flow — only the receiving bands hold numbers.
* **MDinf** (`mdinf.py`) stores **0.0** in non-receiving bands of valid
  cells; a cell is all-NaN only when it is nodata or has no downslope facet
  at all (a pit/flat — essentially absent from a conditioned DEM).

Both in-pipeline consumers handle both conventions explicitly, treating a
cell as valid when *any* band is finite and NaN bands as 0
(`04_delineate_catchments.load_flow_grid`,
`05_flow_accumulation._fractions_from_bands`).
**If you write a new tool that reads `flow_direction_mfd.tif`, do the same** —
naively summing bands, or treating any-NaN as nodata, silently discards most
of the valid MFD cells.

One deliberate difference between the two consumers: the catchment
delineator re-normalises each cell's fractions to sum to 1, the accumulation
script uses them as stored (flow aimed at a nodata/flat neighbour is simply
lost from the domain, as WhiteboxTools does). Near flats and edges MDinf
fractions can sum to slightly less than 1, so the two tools can differ
marginally there.

## Provenance & credits

The pipeline design follows **A. Rolim da Paz (2025), *Digital Elevation
Models for Environmental Studies*, Springer**: the overall
condition-route-accumulate workflow and, in particular, the three catchment
delineation methods of stage 4, which implement its chapter 8 (outlet
coordinates, confluence criterion, Pfafstetter/Ottobasins).

**pyflwdir** (Eilander et al. 2021, *Geoscientific Model Development* 14,
doi:10.5194/gmd-14-5045-2021; https://github.com/Deltares/pyflwdir) powers
stages 6–7: it turns the pipeline's existing D8 raster into a flow graph —
one downstream index per cell plus a down-to-upstream cell ordering — so
HAND and the floodplains propagate along the already-computed directions
instead of re-routing anything. The Numba kernels of both stages are
adapted from `pyflwdir.dem` (compiled for whole-DEM grids): HAND after
Nobre et al. (2016, doi:10.1016/j.jhydrol.2015.10.023) with the
reached-drain nodata rule described under stage 6, and GFPLAIN after Nardi
et al. (2019, doi:10.1038/sdata.2018.309) with the coefficient `a` made an
explicit parameter.

Other essential tools: WhiteboxTools (stage 2 depression filling, MDinf
accumulation), pysheds (D8/MFD/Dinf routing), rasterio/GDAL, NumPy and
Numba. The MDinf direction mathematics in `mdinf.py` follow Seibert &
McGlynn (2007), ported via WhiteboxTools' MIT-licensed implementation.

Source data: KM2 2 m DEM © National Land Survey of Finland (CC BY 4.0);
culvert corrections: SYKE "Tierumpujen uomakorjaus" WCS.
