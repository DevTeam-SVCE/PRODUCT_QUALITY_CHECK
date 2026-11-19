# Product Quality Check — Complete Pipeline Flow

---

## 1. Entry Points

There are three ways to trigger the pipeline:

```
run_all.sh              →  inspect_all.py  →  partvision/batch.py  (batch: all photos vs all DXFs)
analyse_part.py         →  partvision/cli.py                        (single photo)
analyse_flange.py       →  partvision/cli.py  (--shape round forced) (single round part)
capture.py              →  partvision/camera_ui.py                  (live camera loop)
calibrate_camera.py     →  standalone (one-time lens calibration, produces camera_intrinsics.json)
```

---

## 2. Prerequisite: One-Time Camera Calibration (optional)

**File:** `calibrate_camera.py`

```
15–25 chessboard photos
        ↓
cv2.findChessboardCornersSB   (find inner corners in each photo)
        ↓
cv2.calibrateCamera           (compute camera matrix K + distortion coefficients)
        ↓
camera_intrinsics.json        (K, dist_coeffs, image_size, rms)
```

This file is passed to `capture.py --intrinsics`. Without it the system assumes 75° diagonal FOV.

---

## 3. Live Capture Path  (`capture.py`)

**Files:** `partvision/camera_ui.py` → `partvision/camera.py` → `partvision/refframe.py`

Each camera frame goes through `CaptureController.process()`:

```
Raw frame (1080p from phone/webcam)
        ↓
[refframe.find_candidates]      Detect A4-like quadrilaterals (edge/brightness proposals at 640px)
        ↓
[refframe.refine_quad]          Sub-pixel edge fitting on each candidate (robust line fit on 24 sample points per edge)
        ↓
─── 10 quality checks (cheapest first, full-res only after geometry passes) ───
  1. Frame found?             A4-like quad seen in 3 consecutive frames
  2. All edges visible?       ≥60% of each edge's samples on its fitted line
  3. Fully inside view?       2% margin to every image border
  4. Is it A4?                Aspect within 3% of 210/297
  5. Camera angle?            Tilt ≤2° (from pose estimation via PnP on 4 corners)
  6. Perspective/keystone?    Opposite sides differ ≤2.5%
  7. Size?                    Frame ≥12% of view and ≥2 px/mm
  8. Lighting?                Brightness 45–50%, edge contrast ≥8, paper evenly lit
  9. Steady?                  ≥6 frames over ≥250ms, corners within 1px, nothing moving
 10. Sharp?                   Edge blur FWHM ≤0.22% of image diagonal
        ↓
All checks pass → CAPTURE that frame
        ↓
[refframe.rectify]              Apply homography: warp the A4 interior to square pixels
[refframe.flatten]              (if illumination_flatten=true) fit smooth surface to paper brightness, equalize
        ↓
Saved files in  captures/<id>/
  <id>_original.png             raw camera frame
  <id>_rectified.png            perspective-corrected A4 interior
  <id>_rectified_uncorrected.png  before illumination flatten (when flatten active)
  <id>_detection.jpg            frame with detected corners overlaid
  <id>_rectified.calib.json     full calibration data (see §4)
        ↓
Hand off  rectified.png + calib.json  →  Inspection pipeline (§5)
```

**calib.json contents:** A4 corner positions (px and mm), homographies (mm_from_image, rectified_from_image),
camera model (matrix, dist), mm/px scale, tilt_deg, sharpness_score, stability, all thresholds.

---

## 4. Reading the Drawing  (`partvision/drawing.py`)

Runs at startup in batch mode (once per DXF), or per-call in single-photo mode.

```
NAME.dxf  (or NAME.json spec)
        ↓
[load_dxf]   Parse with ezdxf
             Drop annotation/dimension/text/hatch/centre-line layers and line types
             Remaining line work → segments + circles
        ↓
[_chain]     Chain segments into closed loops (tolerant of gaps, mid-way junctions)
[build_raster]  Rasterise loops onto a fine grid → find closed shapes reliably
        ↓
[_rect_like_frame]  Detect and discard the drawing border frame
        ↓
[build]      From closed shapes and circles:
               outline   → the closed shape enclosing the most holes (largest view)
               holes     → circles, or loops that fit a circle, inside the outline
               rings     → larger circles concentric with a hole (counterbores, chamfers)
               openings  → slots, rectangles, any non-circular closed shape
               contours  → shapes enclosing other features (recesses, steps)
        ↓
[_finish]    Rank features by prominence (largest first; ≥3% of part size = "prominent")
             Attach per-feature tolerance (from DXF or default_mm)
        ↓
model dict:  outline{pts, bbox, is_round}, features[{id, kind, x, y, d, tol, ...}]

Optional: NAME.part.json  →  height_mm (corrects scale for raised face), size_gate (reject/report)
```

---

## 5. Inspection Pipeline — Step Execution

**Files:** `partvision/cli.py` → `partvision/registry.py` → `partvision/steps/*.py`

`cli.main()` creates a `Context`, loads the `.calib.json` if one exists alongside the image,
then calls `registry.run(ctx, pipeline_steps)`.

The registry executes steps in list order, skipping any whose `when` predicate returns False
or whose dependency was already skipped.

### Pipeline: `inspect`  (drawing-free analysis)
```
Steps: part → outline → bore → round_rim → bolt_circle → rim_notches → rings → holes → patterns → pockets → scale → ratios
```

### Pipeline: `drawing`  (drawing check)
```
Steps: part → drawing_load → drawing_align → drawing_outline → drawing_features → drawing_report
```
In batch mode (`batch.py`), both pipelines run: `drawing` first (to identify and check), then `inspect` on the winner.

---

### Step-by-Step Detail

#### STEP: `part`  (`steps/part.py`)
```
Input:  raw grayscale image

[segment.segment]
  → tries 'edges' first:
       Compute sharpness = fine-blur gradient / coarse-blur gradient
       Flood-fill from all 4 image borders, stopping at sharp edges
       → anything the flood cannot reach = part
       (cast shadows have soft edges → stay background; bright reflective faces → stay part)
  → falls back to 'otsu' if part touches the border:
       Single Otsu threshold, both polarities (dark-on-light and light-on-dark)
  → 'auto': tries edges, then otsu, keeps first whose part doesn't touch the border

Largest non-border-touching connected region = the part contour

Output stored in ctx.part:
  contour, inner (through-holes), polarity, MEC centre/radius, area, centroid, bbox, masks
```

#### STEP: `outline`  (`steps/outline.py`)
```
Input:  ctx.part (contour)

Compute:  area, perimeter, circularity, solidity, min-area bounding rectangle (L×W, angle)

Circle fit (robust, 6 iterations):
  if inlier_fraction ≥ round_min_inliers AND scatter ≤ round_max_scatter_frac × r
    → shape = "round"
  else try ellipse fit (5 RANSAC iterations):
    if well-fit AND axis ratio in [ellipse_min, ellipse_max]
      → shape = "round"  (and warn about tilt)
  else:
    4-corner approxPolyDP + rect_fill ≥ threshold  → "rectangular"
    high solidity + ≤ polygon_max_corners           → "polygon (N corners)"
    else                                             → "irregular"

Convexity defects deeper than cut_in_min_depth → cut-ins C1, C2, ... (named by depth)

Establish part coordinate frame: u along long axis, v perpendicular, origin at corner

Output: shape, circularity, solidity, L, W, angle, cut_ins, round (bool)
  → routes all later steps: "round" → bore/rim/bolt_circle/rings; else → holes/patterns
```

#### STEP: `bore`  (`steps/round_part.py`)  — round parts only
```
HoughCircles to get a starting estimate (multiple param1/param2 trials)
Ray-fit refinement via holefit.py:
  Cast 360 rays from the Hough centre
  Find the wall-level crossing on each ray (edge between material and void)
  Robust circle fit on those wall points (iterative, rejects outliers)
Output: bore centre (x,y), diameter, scatter, ray count, ellipse fit cross-check
```

#### STEP: `round_rim`  (`steps/round_part.py`)  — round parts only
```
Cast 720 rays outward from bore centre
Find the outermost sharp edge on each ray → outer rim circle fit
Output: outer rim centre (x,y), diameter, scatter, ray count
```

#### STEP: `rim_notches`  (`steps/round_part.py`)  — round parts only
```
Per-angle rim profile: angular gaps where the rim is missing or recessed
Merge adjacent gap angles into notch spans
Output: notches N1, N2, ... with start/end angles and max depth
```

#### STEP: `rings`  (`steps/round_part.py`)  — round parts only
```
Scan radii between bore and outer rim for brightness gradient transitions
At each candidate radius: circle fit on 360 rays
Classify by gradient direction (lighter/darker outward) and fit quality
Output: rings R1, R2, ... with diameter, gradient, fit quality, status
```

#### STEP: `bolt_circle`  (`steps/bolt_circle.py`)  — round parts only
```
HoughCircles for all small circles in the part's annular zone
Cluster detected hole centres by radius from bore centre
Dominant cluster = bolt circle
Measure each bolt hole via ray-fit (holefit.py)
Compute angular pitches
Output: bolt hole list, bolt circle radius, mean pitch, uniformity
```

#### STEP: `holes`  (`steps/holes.py`)  — non-round, or round with no bolt circle
```
Three independent candidate sources:

  source 'background':
    Connected background-colour regions enclosed by the part mask
    → through-holes with lit backdrop

  source 'surface':
    Spots darker (or brighter) than local material brightness
    → blind holes, pockets, bosses

  source 'edges':
    Regions closed all-round by sharp edges (from the same edge map as 'edges' segmentation)
    → windows whose backdrop is as dark as the metal; pins and bosses

For each candidate:
  Round candidate  → ray-fit circle via holefit.py (wall level, or strongest edge for a pin)
  Rect candidate   → line fit on each of 4 edges (rect_fit)
  Edge-region only counts if the fit succeeds → rejects scratch texture

Deduplication: overlapping candidates from different sources → one feature

Output: holes list, each with: x, y, d (px), shape, source, polarity (dark=hole / bright=pin)
```

#### STEP: `patterns`  (`steps/patterns.py`)
```
Input: holes list

Group holes by same shape + same size (within size_group_ratio)
For each group (≥ min_holes):
  Try circular:   fit_circle on hole centres; accept if scatter ≤ tolerance AND pitch is uniform
  Try rectangle:  4 holes forming a rectangle (side and diagonal checks)
  Try linear:     greedy line extraction (3+ holes on a line)

Output: patterns list — PC# (circle of holes), RC# (rectangle), R# (row)
        → bolt circle diameter stored in ctx.dims for ratios
```

#### STEP: `pockets`  (`steps/common.py`)
```
User-specified pocket positions via --pockets "x,y,r;..."
Drawn as estimates, not measured
```

#### STEP: `scale`  (`steps/common.py`)
```
Priority:
  1. --known-mm FEATURE=MM  (e.g. bore=25) → mm = val / dims[feat]
  2. --mm-per-px or calib.json mm_per_px   → mm = that value
  3. Neither → pixel units only

Sets ctx.mm for all length outputs
```

#### STEP: `ratios`  (`steps/common.py`)
```
Compute dimensionless ratios from ctx.dims (bore, outer, bolt, length, width):
  length_over_bore, outer_over_bore, width_over_bore, length_over_width, outer_over_bolt
Stored in report as ratios and dims_px
```

---

### Drawing Check Steps  (pipeline: `drawing`)

#### STEPS: `drawing_load` … `drawing_report`  (`steps/drawing_check.py`, work in `drawing_check/`)
This is the heaviest step. It runs the full drawing alignment and feature checking.

```
─── A. Initial alignment (register) ───

Starting guess 1 — Outline overlap search:
  Try every rotation (0–360° in 1° steps), both mirrors (normal + flipped)
  For each: apply similarity transform, compute silhouette IoU with part contour
  Top candidates ranked by IoU

Starting guess 2 — Hole-pair matching (hole_hypotheses):
  Find all round spots (blobs) in the photo
  For each pair of drawing holes, match to each pair of photo blobs
  → estimate rotation + scale from each pair
  Rank by hole agreement

─── B. Refinement (refine) ───

For each candidate alignment:
  Measure edge points along the drawing outline (800 normal-direction probes)
  Measure centres of drawing holes and opening centroids
  Weight groups by their own measured precision:
    hole centres   >> outline edge points  (holes are most reliable)
  Iterative robust fit: re-estimate rotation, scale, translation (Sim class: 4-DoF)

─── C. Perspective (perspective_refine) ───

If ≥6 measured holes:
  Fit homography (RANSAC)
  Keep only if it puts clearly more holes on-target than the similarity transform
  → corrects for angled shots

─── D. Pick best alignment (candidates) ───

Score each alignment by:
  feature agreement (holes + plain openings) + edge fit quality + silhouette overlap
Best candidate wins.
(Proj class handles perspective case; Sim handles affine case)

─── E. Scale check ───

If calib.json present:
  Compare fitted px/mm with calib mm/px
  If part is uniformly larger/smaller than drawing beyond tolerance → warn DEVIATES

─── F. Per-feature checking ───

For each drawing feature:

  HOLE:
    Search near expected position (position tolerance = max(1mm, 0.5×diameter))
    Find nearest round dark (or bright) spot of about the right size
    Must be surrounded by material on most sides (rejects shadows/crescents)
    Ray-fit circle → measure centre + diameter
    If no dark hole found but round edge exists → report as "pin/boss/plug"
    Parallax correction: holes on recess floors searched in parallax direction

  OPENING (slot, window):
    Segment the region at the expected place
    Compare by overlap + length/width ratio
    Result: found/missing, size deviation

  CONTOUR (recess/step edge):
    Probe edge along normals at 800 points around the contour polygon
    Measure median offset (how much the actual edge deviates from drawing)

  OUTLINE:
    800 normal-direction measurements around the full drawing outline
    Stop before reaching any drawing hole (can't snap onto hole rim)
    Measure RMS deviation, max excess, max shortfall

  Per feature result:
    status:          DETECTED | DEVIATES | MISSING | NOT MEASURED
    dimension_check: OK | CAN'T VERIFY | UNCERTAIN | MISSING
    position_dev_mm, diameter_dev_mm, note

  Guard band: when deviation is within the photo's measurement uncertainty of the
  tolerance limit → "CAN'T VERIFY" rather than a pass/fail
```

---

## 6. Batch Matching Logic  (`partvision/batch.py`)

Runs before the drawing steps when multiple DXFs exist:

```
For each photo:
  [_silhouette]       Segment the part at low res, extract contour

  Shape screen:
    cv2.matchShapes (Hu moment distance, I2 method)
    → rank all DXFs by silhouette similarity (rotation/scale/mirror invariant)
    If ≤ check_all_up_to DXFs: check them all (screen only orders)
    Else: take top full_check_candidates

  For each candidate DXF:
    Run the full drawing pipeline (cli.main with --drawing candidate)
    Compute match score:
      hole_coverage (batch.hole_coverage):
        Find holes in the photo that lie inside the part material
        Count how many are explained by drawing holes
      match_score:
        score = w1 × outline_overlap
              + w2 × hole_agreement      (drawing holes that land on photo holes)
              + w3 × hole_coverage       (photo holes explained by the drawing)
              + w4 × detected_fraction   (drawing features found)

    If calib.json present:
      size_gate check: if part is physically impossible given camera geometry → rule out

  Sort candidates by score
  _size_tiebreak: if two candidates score within size_tiebreak_score, prefer the one
                  whose calibrated scale matches

  Winner:
    score ≥ min_match_score  AND  detected_fraction ≥ min_detected_fraction
    → run drawing-free `inspect` pipeline on that photo too
    → copy outputs to final location
    → append to summary.json

  If no winner:
    → run drawing-free `inspect` pipeline only
    → report "no matching drawing"
```

---

## 7. Output Generation  (`partvision/output.py`)

Steps never write files. They deposit into `ctx`:
- `ctx.draw(...)` → drawing primitives (circles, lines, text, polylines)
- `ctx.table(...)` → table rows with optional mm conversion
- `ctx.feature(...)` → CSV row
- `ctx.warn(...)` / `ctx.note(...)` → messages
- `ctx.data[...]` → JSON fields

At the end of `cli.main()`:

```
render(ctx, png)          → Annotated PNG
  or
render_panel(ctx, png)    → Drawing-check PNG: photo+overlay LEFT | drawing preview+table RIGHT

write_csv(ctx, path)      → *_measurements.csv  (one row per feature)
write_json(ctx, path)     → *_report.json        (all step data, calibration, warnings)

batch.main also writes:
  summary.json            → [{photo, drawing, score, score_parts, detected, status_counts, image}]
  pipeline.log            → human-readable per-drawing scores and winner
```

---

## 8. File Output Map

```
captures/<id>/
  <id>_original.png                    Raw camera frame
  <id>_rectified.png                   Perspective-corrected image (input to inspection)
  <id>_rectified_uncorrected.png       Before illumination flattening (when active)
  <id>_detection.jpg                   Frame with A4 corners overlaid
  <id>_rectified.calib.json            Full calibration: mm/px, homography, quality metrics

  inspection/
    <id>_rectified.png                 Annotated result: drawing overlay + colour-coded features
    pipeline.log                       Human-readable match log
    summary.json                       Best drawing match + score + status counts
    data/
      *_analysis_measurements.csv      Drawing-free: bore, rim, notches, rings, etc. (px + mm)
      *_analysis_report.json           All step results for drawing-free analysis
      *_drawing_check_measurements.csv Nominal vs measured for each drawing feature
      *_drawing_check_report.json      Full alignment details + per-feature status

output/                                (batch / run_all.sh)
  photo1.png ... photo7.png            Annotated result images
  data/
    photo*_analysis_measurements.csv
    photo*_analysis_report.json
    photo*_drawing_check_measurements.csv
    photo*_drawing_check_report.json
  summary.json
  .work/                               Intermediate per-drawing check results (deleted after)
```

---

## 9. Status Codes

| Status | Meaning |
|---|---|
| **OK** | Found; position and size within tolerance |
| **DETECTED** | Found at the expected place, but outside tolerance or too close to the limit to confirm |
| **DEVIATES** | Something is there but further off than the detection tolerance |
| **MISSING** | Nothing found at that location |
| **NOT MEASURED** | In the drawing but not measurable from a top view (e.g. counterbore depth) |
| **CAN'T VERIFY** | Deviation is within the photo's measurement uncertainty of the tolerance limit |
| **UNCERTAIN** | Outline/contour edge only partially visible; result may be unreliable |

---

## 10. Complete Code → File Map

| File | Does |
|---|---|
| `analyse_part.py` | CLI entry → `partvision/cli.py` |
| `analyse_flange.py` | Same with `default_shape="round"` |
| `inspect_all.py` | Batch CLI → `partvision/batch.py` |
| `capture.py` | Live camera CLI → `partvision/camera_ui.py` |
| `calibrate_camera.py` | Standalone lens calibration |
| `run_all.sh` | Shell shortcut: `inspect_all.py --photos . --drawings dxf_files --out output` |
| `partvision/cli.py` | Arg parsing, calib loading, pipeline dispatch, output calls |
| `partvision/batch.py` | Folder scanning, shape screening, match scoring, summary |
| `partvision/registry.py` | `@step` decorator, pipeline lists, dependency-ordered execution |
| `partvision/core.py` | `Context` object; geometry helpers (circle fit, ray profiles) |
| `partvision/segment.py` | `edges` and `otsu` segmentation methods |
| `partvision/drawing.py` | DXF + JSON spec parser; feature model; prominence ranking |
| `partvision/holefit.py` | Radial ray circle measurement (shared by round_part + drawing_check) |
| `partvision/refframe.py` | A4 frame detection, sub-pixel edges, pose, sharpness, rectification |
| `partvision/camera.py` | Per-frame capture logic, quality gates, file saving |
| `partvision/camera_ui.py` | Browser MJPEG UI; `capture.py` CLI |
| `partvision/output.py` | PNG renderer, CSV writer, JSON writer |
| `partvision/partmeta.py` | `NAME.part.json` reader; calibrated size check |
| `partvision/config.py` | All numeric thresholds (overridable via `--config`) |
| `partvision/steps/part.py` | Step: segment part from background |
| `partvision/steps/outline.py` | Step: classify shape, measure size, find cut-ins |
| `partvision/steps/round_part.py` | Steps: bore, round_rim, rim_notches, rings |
| `partvision/steps/bolt_circle.py` | Step: bolt holes and pitch |
| `partvision/steps/holes.py` | Step: holes/openings (3 sources + dedup) |
| `partvision/steps/patterns.py` | Step: circular, rectangular, row patterns |
| `partvision/steps/drawing_check.py` | Drawing steps: load, align, outline, features, report |
| `partvision/drawing_check/` | Drawing check modules: transform, edges, checks, align, outline, judge, features, report |
| `partvision/steps/common.py` | Steps: pockets, scale, ratios |
| `dxf_files/` | Engineering drawings (box_front, box_rear, circular_rear, circular_top) |
| `captures/` | 92 capture directories with all outputs |
| `blueprints/` | Reference JPEG images of the drawings |
