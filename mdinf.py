"""mdinf.py - MDinf flow directions, companion module to 03_flow_router.py.

Created on Sat Jul 4 2026
@author: Antti Ahokas
Written with Claude Code (Anthropic).

03_flow_router.py computes MDinf flow directions with this module when
the MDinf method is requested (--fdir mdinf, or --fdir all).

PROVENANCE
    This is the pipeline's MDinf implementation: 03_flow_router.py routes
    D8, MFD and Dinf through pysheds, which has no MDinf, so the fractions
    come from here and are accumulated by the shared kernel in
    accumulation.py. The mathematics is the direction-partitioning half of
    WhiteboxTools' MDInfFlowAccumulation (mdinf_flow_accum.rs, MIT license,
    John Lindsay), itself a port of the original Java implementation
    written by the method's authors, Jan Seibert and Marc Vis, for Whitebox
    GAT. An MDinf direction depends only on the 3x3 neighbourhood of a
    pixel, not on the accumulation, so the fractions produced here are the
    very ones a WhiteboxTools accumulation of the same surface would follow
    (with no convergence threshold, so its D8 fallback branch never fires) -
    the two implementations were compared on this pipeline's test tiles
    before WhiteboxTools was retired from the dependencies.

THE ALGORITHM - MDinf, Seibert and McGlynn (2007)
    The terrain around each pixel is modelled as eight planar triangular
    facets between the pixel centre and each pair of adjacent neighbours.
    Each facet has a continuous downslope aspect angle; when that angle
    falls outside the facet's 45-degree span it is clamped to the steeper
    of the facet's two edges. Every DOWNSLOPE facet receives a share of
    the pixel's flow proportional to its slope raised to an exponent p
    (1.1 after Freeman, matching flow_router's MDINF_EXPONENT), and each
    facet's share is split linearly by angle between the two grid
    neighbours bracketing its aspect - so each facet feeds one or two
    pixels, and the pixel as a whole feeds up to eight.
      Founder: Seibert, J. and McGlynn, B.L. (2007) 'A new triangular
      multiple flow direction algorithm for computing upslope areas from
      gridded digital elevation models', Water Resources Research, 43(4),
      W04501.

OUTPUT CONVENTION
    mdinf_flowdir() returns fractions as a (8, rows, cols) float64 array
    in pysheds band order N, NE, E, SE, S, SW, W, NW - the same convention
    03_flow_router.py uses for the MFD direction raster, so the two files
    read identically. Nodata pixels, and valid pixels with no downslope
    facet at all (pits and flats; a conditioned DEM has essentially none),
    are NaN in every band. Exactly like WhiteboxTools, a facet share
    aimed at a neighbour that is not strictly lower is dropped rather
    than rerouted, so next to flats and nodata a pixel's fractions can sum
    to slightly less than one.

    Two deliberate deviations from the Rust, neither affecting a
    conditioned DEM: the facet-aspect buffer is reset for every pixel
    (WhiteboxTools reuses it, letting stale angles from previously
    processed pixels leak into edge cases beside nodata), and undefined
    pixels are NaN rather than silently accumulating nothing.
"""

import math

import numpy as np
from numba import njit, prange

# The band order 03_flow_router.py writes (pysheds convention), expressed as
# indices into WhiteboxTools' facet order N, NW, W, SW, S, SE, E, NE.
_PYSHEDS_BANDS = np.array([0, 7, 6, 5, 4, 3, 2, 1])

_SQRT2 = math.sqrt(2.0)


@njit(parallel=True, cache=True)
def _mdinf_weights(dem, nodata, grid_res, exponent):
    """Per-pixel MDinf flow fractions, WhiteboxTools facet order.

    A line-by-line port of the direction-partitioning half of
    mdinf_flow_accum.rs; variable names follow the Rust.
    """
    rows, cols = dem.shape
    # Neighbour (col, row) offsets in facet order N, NW, W, SW, S, SE, E, NE;
    # facet i lies between neighbours i and i+1.
    xd = np.array([0, -1, -1, -1, 0, 1, 1, 1], dtype=np.int64)
    yd = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.int64)
    dd = np.array([1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2, 1.0, _SQRT2])
    quarter_pi = math.pi / 4.0
    out = np.full((8, rows, cols), np.nan)
    for row in prange(rows):
        p = np.empty(8)
        downslope = np.empty(8, np.bool_)
        r_facet = np.empty(8)
        s_facet = np.empty(8)
        valley = np.empty(8)
        weights = np.empty(8)
        for col in range(cols):
            z = dem[row, col]
            if z == nodata:
                continue
            for i in range(8):
                rr = row + yd[i]
                cc = col + xd[i]
                if 0 <= rr < rows and 0 <= cc < cols:
                    p[i] = dem[rr, cc]
                else:
                    p[i] = nodata
                downslope[i] = False
                r_facet[i] = 0.0
                s_facet[i] = nodata
                valley[i] = 0.0
                weights[i] = 0.0

            # Aspect (r_facet) and slope (s_facet) of the 8 triangular facets.
            for i in range(8):
                ii = (i + 1) % 8
                p1 = p[i]
                p2 = p[ii]
                if p1 < z and p1 != nodata:
                    downslope[i] = True
                if p1 != nodata and p2 != nodata:
                    z1 = p1 - z
                    z2 = p2 - z
                    # Normal to the facet through the pixel and both neighbours.
                    nx = (yd[i] * z2 - yd[ii] * z1) * grid_res
                    ny = (xd[ii] * z1 - xd[i] * z2) * grid_res
                    nz = (xd[i] * yd[ii] - xd[ii] * yd[i]) * grid_res * grid_res
                    if nx == 0.0:
                        hr = 0.0 if ny >= 0.0 else math.pi
                    elif nx >= 0.0:
                        hr = math.pi / 2.0 - math.atan(ny / nx)
                    else:
                        hr = 3.0 * math.pi / 2.0 - math.atan(ny / nx)
                    hs = -math.tan(math.acos(
                        nz / math.sqrt(nx * nx + ny * ny + nz * nz)))
                    # Aspect outside the facet's 45 degrees: clamp to the
                    # steeper of its two edges.
                    if hr < i * quarter_pi or hr > (i + 1) * quarter_pi:
                        if p1 < p2:
                            hr = i * quarter_pi
                            hs = (z - p1) / (dd[i] * grid_res)
                        else:
                            hr = ii * quarter_pi
                            hs = (z - p2) / (dd[ii] * grid_res)
                    r_facet[i] = hr
                    s_facet[i] = hs
                elif p1 != nodata and p1 < z:
                    hr = i * quarter_pi
                    hs = (z - p1) / (dd[ii] * grid_res)  # dd[ii]: as in the Rust
                    r_facet[i] = hr
                    s_facet[i] = hs

            # Share of each downslope facet, slope^exponent weighted.
            valley_sum = 0.0
            valley_max = 0.0
            i_max = 0
            for i in range(8):
                ii = (i + 1) % 8
                if s_facet[i] > 0.0:
                    if i * quarter_pi < r_facet[i] < (i + 1) * quarter_pi:
                        valley[i] = s_facet[i]
                    elif r_facet[i] == r_facet[ii]:
                        valley[i] = s_facet[i]
                    elif (s_facet[ii] == nodata
                            and r_facet[i] == (i + 1) * quarter_pi):
                        valley[i] = s_facet[i]
                    else:
                        ii = (i + 7) % 8
                        if (s_facet[ii] == nodata
                                and r_facet[i] == i * quarter_pi):
                            valley[i] = s_facet[i]
                if exponent != 1.0:
                    valley[i] = valley[i] ** exponent
                valley_sum += valley[i]
                if valley[i] > valley_max:
                    valley_max = valley[i]
                    i_max = i
            if valley_sum <= 0.0:
                continue  # pit or flat: direction undefined, bands stay NaN
            if exponent < 10.0:
                for i in range(8):
                    valley[i] /= valley_sum
            else:
                # Extreme exponents collapse to steepest-facet-takes-all.
                for i in range(8):
                    valley[i] = 1.0 if i == i_max else 0.0
            if r_facet[7] == 0.0:
                r_facet[7] = 2.0 * math.pi

            # Split each facet's share between its two bracketing neighbours,
            # linearly by angle.
            for i in range(8):
                ii = (i + 1) % 8
                if valley[i] > 0.0:
                    weights[i] += (valley[i]
                                   * ((i + 1) * quarter_pi - r_facet[i])
                                   / quarter_pi)
                    weights[ii] += (valley[i]
                                    * (r_facet[i] - i * quarter_pi)
                                    / quarter_pi)
            for i in range(8):
                out[i, row, col] = weights[i] if downslope[i] else 0.0
    return out


def mdinf_flowdir(dem, nodata, dx, dy, exponent=1.1):
    """MDinf flow fractions of a DEM: (8, rows, cols), pysheds band order.

    dem is a 2-D elevation array (a bare numpy array or a pysheds Raster),
    nodata its nodata value, dx/dy the pixel size in map units, exponent
    the facet-slope dispersion exponent p. Returns float64 fractions in
    band order N, NE, E, SE, S, SW, W, NW; NaN bands mark nodata pixels
    and pixels with no downslope facet. See the module docstring for the
    provenance and the exact semantics.
    """
    dem = np.ascontiguousarray(dem, dtype=np.float64)
    if nodata is None or not np.isfinite(nodata):
        nodata = -1.0e38
    dem = np.where(np.isfinite(dem), dem, nodata)
    grid_res = (abs(float(dx)) + abs(float(dy))) / 2.0
    weights = _mdinf_weights(dem, float(nodata), grid_res, float(exponent))
    return weights[_PYSHEDS_BANDS]
