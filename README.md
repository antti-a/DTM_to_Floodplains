# DTM to Floodplains — Geomorphic floodplains from digital terrain model

The pipeline produces rasters of geomorphic floodplains (GFPLAIN; Nardi et al., 2019) and height
above nearest drain (HAND; Nobre et al., 2016) directly from Finland's
national 2 m elevation model (KM2). This is terrain analysis only, no hydraulic
modelling is done. The pipeline is built for Finnish data provided by the National Land Survey (NLS) and the Environment Institute (SYKE). 



DTM is first carved with SYKE's culvert-correction raster so that flow crosses
road embankments instead of ponding behind them. Carved DTM is then conditioned for hydrological calculations by filling depressions and pits to ensure that every pixel drains out of the modelled area. Flow routing and accumulation are then calculated to be used by HAND and floodplain calculations. The pipeline can be modified
to work in other areas by swapping or skipping the culvert-carving stage which at the moment is specific to data available for Finland.
The floodplain delineation (`h = a·A^b`) is the pipeline's only
parametrized step; choosing `a` and `b` is left to the application.



The six stage scripts are numbered in pipeline order (`01\_` … `06\_`) and
share one `data/` tree: each stage's output is already the next stage's
default input, and `00\_run\_pipeline.py` runs them in order. Each stage is
also a standalone command-line script, so any stage can be re-run alone
with different parameters. The three unnumbered files are companion modules
(`pipeline\_io.py`, `accumulation.py`, `mdinf.py`) imported by the stages.

## Quick start

```bash
git clone https://github.com/antti-a/DTM\_to\_Floodplains.git
cd DTM\_to\_Floodplains
conda env create -f environment.yml
conda activate water
# drop your DTM tiles (GeoTIFF) into data/00\_source\_dems/
python 00\_run\_pipeline.py
```

The result is `data/06\_floodplains/floodplains.tif` (1 = floodplain,
0 = upland) plus every intermediate product.

## Running the pipeline

Full run: All six stages:

```bash
python 00\_run\_pipeline.py
```

Resume after a failure, or run a subset — earlier stages' outputs are
reused:

```bash
python 00\_run\_pipeline.py --from route
python 00\_run\_pipeline.py --only fill route
python 00\_run\_pipeline.py --skip hand
```

Adjust the floodplain parameters: The flood level `h = a·A^b` is the
only parametrized step. How to choose `a` and `b` depends on the application.

For example:

```bash
python 06\_floodplains.py --a 0.5 --b 0.35
```

Denser or sparser stream network for HAND and the floodplains: Lower
or raise the stream-initiation threshold (km² of upstream area):

```bash
python 05\_hand.py --upa-min 0.1
python 06\_floodplains.py --upa-min 0.1
```

Compare d8, mfd, dinf and mdinf flow routing algorithms (not needed for
floodplains, but not widely available elsewhere): Route with all four
algorithms and get each one's stream network (GeoJSON), flow-direction
raster and a comparison table (stream cells, Jaccard overlaps, drainage
density):

```bash
python 03\_flow\_router.py --fdir all
```

### Flag reference

|script|flag|meaning (default)|
|-|-|-|
|`00\_run\_pipeline.py`|`--from`, `--only`, `--skip`|which stages to run|
|`00\_run\_pipeline.py`|`--area KM2`|forwarded to stage 3 (1.0)|
|`03\_flow\_router.py`|`--area KM2`|minimum contributing area defining a stream in the vector network (1.0)|
|`03\_flow\_router.py`|`--fdir d8 mfd dinf mdinf` / `all`|routing algorithms to run (d8)|
|`04\_flow\_accumulation.py`|`--units m2/cells`|accumulation in square metres (m2) or cell counts|
|`05\_hand.py`, `06\_floodplains.py`|`--upa-min KM2`|stream-initiation threshold (0.2)|
|`06\_floodplains.py`|`--a`, `--b`|GFPLAIN power law `h = a·A^b` (0.63, 0.3)|

`python <script> --help` lists everything, including flags that repoint the
input and output locations. Stages 1–2 are configured by the constants at
the top of each script; stages 5–6 also have a `USER SETTINGS` block whose
values are simply the defaults a no-argument run uses.

## The stages

|#|script|reads|writes|
|-|-|-|-|
|1|`01\_carve\_dem.py`|`data/00\_source\_dems/`|`data/01\_carved/` (+ `data/culvert\_cache/`)|
|2|`02\_fill\_dem.py`|`data/01\_carved/`|`data/02\_filled/`|
|3|`03\_flow\_router.py`|`data/02\_filled/`|`data/03\_flows/`|
|4|`04\_flow\_accumulation.py`|`data/03\_flows/flow\_direction\_\*.tif`|`data/04\_accumulation/`|
|5|`05\_hand.py`|`data/02\_filled/` + `data/03\_flows/flow\_direction\_d8.tif` + `data/04\_accumulation/flow\_accumulation\_d8.tif`|`data/05\_hand/`|
|6|`06\_floodplains.py`|same as stage 5|`data/06\_floodplains/`|

1. **Carve** — lowers the DEM at culverts and road crossings with the SYKE
"Tierumpujen uomakorjaus" WCS layer so flow crosses embankments.
Downloads are windowed and cached; a re-run skips finished tiles.
2. **Fill** — pysheds `fill\_depressions` (priority-flood) and
`resolve\_flats` (both Barnes et al., 2014) on the mosaic of all tiles,
cropped back to each tile's grid. Outputs are float64 on purpose:
float32 collapses the flat-resolution gradients and silently
un-conditions the DEM (stage 3 verifies drainage and stops if so).
3. **Route** — mosaics the filled tiles and routes flow: D8 by default
(O'Callaghan and Mark, 1984) — the format every later stage consumes —
with MFD, Dinf and MDinf available via `--fdir` for comparison, each
with its own network, comparison-table row and direction raster.
4. **Accumulate** — weighted flow accumulation (upstream contributing
area) for every flow-direction raster found.
5. **HAND** — height above nearest drain (Nobre et al., 2016): each cell's
elevation above the stream cell it drains to along the D8 flow path,
with streams defined by the `--upa-min` threshold. D8 only by design:
the propagation needs exactly one downstream cell per cell.
6. **Floodplains** — GFPLAIN (Nardi et al., 2019): every stream cell
carries the flood level `h = a·A^b` (h in m, A = upstream area in km²),
and a cell joins the floodplain of the stream cell it drains to if it
rises no more than `h` above it.

## Outputs and metadata

All rasters are GeoTIFFs in EPSG:3067 (ETRS89 / TM35FIN, metres), vertical
datum N2000. Every output documents its own encoding in its GeoTIFF band
descriptions and dataset tags (`gdalinfo <file>` or
`rasterio.open(...).tags()`), so the formats are not duplicated here. The
tags also carry full provenance: Source DEM tiles, algorithms and
parameters, stamped at the originating stage and forwarded downstream.

## Credits

The beginning of the pipeline follows Rolim da Paz (2025): The condition-route-accumulate
workflow of stages 1–4, and then the pyflwdir library (Eilander et al., 2021;
https://github.com/Deltares/pyflwdir) in stages 5–6 creates HAND after Nobre et al. (2016), and GFPLAIN after
Nardi et al. (2019) with the coefficient `a` made an explicit parameter.

Other essential tools for this projects are: pysheds (D8/MFD/Dinf routing; stage 2 depression
filling and flat resolution after Barnes, Lehman and Mulla, 2014),
rasterio/GDAL, NumPy and Numba. The MDinf direction mathematics in
`mdinf.py` follow Seibert and McGlynn (2007), ported via WhiteboxTools'
MIT-licensed implementation (John Lindsay).

Source data: KM2 2 m DEM © National Land Survey of Finland (CC BY 4.0);
culvert corrections: SYKE "Tierumpujen uomakorjaus" WCS (CC BY 4.0).

## References

Barnes, R., Lehman, C. and Mulla, D. (2014) 'Priority-flood: an optimal
depression-filling and watershed-labeling algorithm for digital elevation
models', *Computers \& Geosciences*, 62, pp. 117–127. Available at:
https://doi.org/10.1016/j.cageo.2013.04.024

Barnes, R., Lehman, C. and Mulla, D. (2014) 'An efficient assignment of
drainage direction over flat surfaces in raster digital elevation models',
*Computers \& Geosciences*, 62, pp. 128–135. Available at:
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

O'Callaghan, J.F. and Mark, D.M. (1984) 'The extraction of drainage
networks from digital elevation data', *Computer Vision, Graphics, and
Image Processing*, 28(3), pp. 323–344. Available at:
https://doi.org/10.1016/S0734-189X(84)80011-0

Rolim da Paz, A. (2025) *Digital elevation models for environmental studies*.
Cham: Springer. Available at: https://doi.org/10.1007/978-3-032-04523-2

Seibert, J. and McGlynn, B.L. (2007) 'A new triangular multiple flow
direction algorithm for computing upslope areas from gridded digital
elevation models', *Water Resources Research*, 43(4), W04501. Available at:
https://doi.org/10.1029/2006WR005128

