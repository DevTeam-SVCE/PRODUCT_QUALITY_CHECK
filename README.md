# Product quality check from photos

Checks machined or sheet-metal parts against their drawings, using ordinary top-down photos.
For every photo the pipeline finds the part, works out which drawing it is, lines the drawing up with
the photo (rotation, scale, mirroring for the other face, perspective for an angled shot), and then
checks each drawing feature where it should be: found or missing, and within tolerance or not.

It also works without a drawing: it then measures what it can see (outline, holes, hole patterns,
notches, rings) in pixels, or in mm when you give one known size.

---

## Quick start

```bash
pip install opencv-python-headless numpy pillow ezdxf    # once

bash run_all.sh            # every photo in this folder vs every drawing in dxf_files/, tolerance from config
bash run_all.sh 0.3        # same, with a +-0.3 mm tolerance
```

Results:

| Path | What |
|---|---|
| `output/<photo>.png` | **One image per photo**: the photo with the drawing overlaid and each feature coloured by status (left), and the drawing's features, a status table, alignment details and warnings (right) |
| `output/data/<photo>_drawing_check_measurements.csv` | One row per drawing feature: nominal vs measured position and size, deviation, tolerance, status |
| `output/data/<photo>_drawing_check_report.json` | Everything, including the alignment transform and match scores |
| `output/data/<photo>_analysis_*` | The same photo measured without the drawing |
| `output/summary.json` | Which drawing each photo was matched to, its score, and the result counts |

### Adding a new part

1. Put its drawing in `dxf_files/` (a DXF, or a JSON spec, see below).
2. Put its photos in the project folder (any of `.jpg .jpeg .png .bmp .tif .tiff .webp`).
3. Run `bash run_all.sh`.

Nothing has to be edited. Each photo is matched to the right drawing automatically. A photo whose part is
not among the drawings is reported as **no matching drawing**, and gets the drawing-free analysis instead.
The live camera (`capture.py`) reads the drawings folder again at every capture, so a new drawing is used from
the next capture on, without a restart.

4. Optional, recommended for camera captures: the height of the face the camera sees (the face with the holes),
   above the surface the part stands on, in `dxf_files/NAME.part.json` next to `NAME.dxf`:

   ```json
   {"height_mm": 30}
   ```

   (or `"height_mm": 30` inside a JSON spec). With it, the A4 frame's absolute scale checks the part's real size
   against the drawing (`calibration.scale_check` in the report: OK / DEVIATES). Without it the size cannot be
   verified exactly (status UNVERIFIED, with the height the photo implies), because a raised face looks larger.
   Every drawing is fitted at its own scale. For a capture, a drawing whose size is physically impossible for
   the part (smaller than drawn, or larger than an 80 mm high face explains) is then ruled out
   (`matching.size_gate = "reject"`, the default; a wrong part of a similar outline otherwise wins on shape
   alone). With `"size_gate": "report"` (in a `--config` file) the size difference is only reported. When two
   drawings fit the geometry about equally well (within `matching.size_tiebreak_score`, e.g. the same shape in
   two sizes), the one whose size fits the calibration wins. Plain photos carry no size and are unaffected.
   Limits: `matching.calib_*`.

### Other commands

```bash
python3 inspect_all.py --help                                    # batch: options below
python3 inspect_all.py --photos new_photos/ --drawings dxf_files/ more_drawings/ --out results/
python3 inspect_all.py --pairs pairs.json                        # force pairings: {"photo9.jpg": "dxf_files/x.dxf"}
python3 inspect_all.py --config my_settings.json                 # change any setting (see "Settings")

python3 analyse_part.py photo.jpg out/ --drawing part.dxf        # one photo against one drawing
python3 analyse_part.py photo.jpg out/                           # one photo, no drawing (one pipeline for every part)
python3 analyse_part.py photo.jpg out/ --known-mm bore=25        # ... with a real size for mm
python3 analyse_part.py --help                                   # every option
python3 analyse_part.py photo.jpg out/ --shape round             # treat the part as round whatever its outline
python3 analyse_part.py --list-steps                             # every step and pipeline
python3 analyse_flange.py photo.jpg out/                         # shortcut for --shape round
python3 capture.py                                               # live camera: automatic capture with an A4 frame
```

---

## Status meanings

| Status | Colour | Meaning |
|---|---|---|
| OK | green | Found, and size and position are within the +-tolerance |
| DETECTED | yellow-green | Found where the drawing puts it, but outside the +-tolerance, or too close to the limit to confirm from this photo. The **dimension** column says which |
| DEVIATES | red | Something is there, but further off than the detection tolerance (wrong place or size) |
| MISSING | magenta | Nothing found at that place |
| NOT MEASURED | grey | Listed from the drawing but not measurable in a top view (e.g. a counterbore ring) |

Two tolerances are involved:

* **+-tolerance** (`tolerance.default_mm`, or `--tol`): the dimensional check.
* **Detection tolerance** (`tolerance.detect_position_mm`, `tolerance.detect_size_mm`): how far a feature may
  be off and still count as "there". Default: position within max(1.0 mm, half its size), size within
  max(1.5 mm, half its size).

The dimensional check uses a **guard band**: when a deviation is within the photo's measurement uncertainty of
the limit, the result is "can't verify" rather than a guess. The uncertainty is computed per photo from the
pixel size and the measured scatter, and is printed in each image.

---

## The pipeline

```
photos/  drawings/
   │         │
   │         ▼
   │   1. read drawing ─────────► features in mm, ranked by prominence
   ▼
2. find the part in the photo
   │
   ▼
3. match photo ↔ drawing  (quick shape screen, then full check of the candidates, best score wins)
   │
   ▼
4. align drawing → photo  (rotation, scale, mirror, perspective)
   │
   ▼
5. check every feature where it should be  (found? within tolerance?)
   │
   ▼
6. outputs: one image per photo + CSV + JSON + summary
```

Every part, round or not, goes through the same steps. The drawing check (steps 2-5) is the same for all
parts. The drawing-free analysis (`--profile inspect`, also written next to each drawing check) is one step
list too. Each step declares when it applies, from what the steps before it measured:

| Step | Applies when |
|---|---|
| `part`, `outline` | always (the outline step classifies the part: round / rectangular / polygon / irregular) |
| `bore`, `round_rim`, `bolt_circle`, `rim_notches`, `rings` | the part is round (`analysis.outline` thresholds, or `--shape round`) |
| `holes`, `patterns` | the part is not round, or it is round but no bolt circle was found |
| `pockets`, `scale`, `ratios` | always |

Skipped steps are listed in the report (`steps_skipped`). In a batch, the matched drawing's outline decides
round or not (`matching.round_drawing`).

**Holes, windows and pins on any part (step `holes`).** Candidates come from pluggable sources
(`analysis.holes.sources`), and every candidate is measured the same way afterwards:

| Source | Finds | Why it is needed |
|---|---|---|
| `background` | regions of background colour inside the part | through-openings with a lit backdrop |
| `surface` | spots darker (or brighter) than the metal around them | holes, blind holes |
| `edges` | regions closed all round by a sharp edge (the edge map that finds the outline) | a window whose backdrop lies in the part's own shadow (as dark as the metal), pins and bosses |

Measuring: a round candidate gets a circle fit on rays from its centre (wall level, else the strongest edge for a
pin), as the bore and bolt holes of a round part do; a rectangular one gets a line fit on each of its four edges
(`analysis.rect_fit`). Edge regions count only when such a fit succeeds, so scratch texture (ragged regions) is
not reported. One feature per place: an unmeasured blob overlapping a measured feature or its counterbore ring is
part of it. The table says how each feature was measured and whether its inside is darker (hole) or brighter
(pin/boss) than the material around it. The batch counts photo holes with this same finder
(`matching.hole_count.method: "features"`).

### 1. Reading the drawing (`partvision/drawing.py`)

* **DXF**: entities on annotation layers or with centre/hidden line types are dropped (dimensions, text,
  hatches, centre lines, title block). Lists are in config `drawing.*`. Units come from `$INSUNITS`.
* The remaining line work is drawn onto a fine raster to find closed shapes. This is robust to lines that
  meet mid-way, tiny gaps, and holes drawn as arcs.
* A rectangular frame around everything is recognised as a drawing border and ignored.
* **Outline**: the closed shape that holds the most holes. Other views on the same sheet are ignored.
* Inside the outline: **holes** (circles, or loops that fit a circle), **rings** (larger circles concentric with
  a hole: counterbores, chamfers), **openings** (slots, rectangles, any closed shape) and **contours** (shapes
  that enclose other features: recesses, steps).
* **Prominence**: features are ranked largest first. Those at least 3% of the part size are "prominent"
  (config `drawing.prominent_fraction`). The ranking is shown in the drawing preview in each output image.

**JSON spec** (for a paper or scanned blueprint): type the key features in mm, y pointing down as on paper:

```json
{"outline": {"type": "rect", "length": 90, "width": 56},
 "tolerance": 0.2, "position_tolerance": 0.3,
 "holes":    [{"id": "H1", "x": 12, "y": 10, "d": 5, "tol": 0.1}],
 "openings": [{"id": "W1", "type": "rect", "x": 40, "y": 28, "length": 25, "width": 25, "angle": 0},
              {"id": "S1", "type": "slot", "x": 20, "y": 40, "length": 12, "width": 4, "angle": 90}],
 "visual":   [{"id": "pin", "x": 80, "y": 10, "d": 6}]}
```

`outline` may also be `{"type": "circle", "d": 110}` or `{"type": "polygon", "points": [[x, y], ...]}`.

### 2. Finding the part (step `part`)

The method is set by `segmentation.method` (or `analyse_part.py --segmentation`); the methods live in
`partvision/segment.py`, and a new one is added there with `@method("name")`:

* **`edges`** (used first by the default `auto`): the background is flooded from the image border without
  crossing a *sharp* edge, and what it cannot reach is the part. A cast shadow's border is soft (its penumbra is
  many px wide), the part's border is sharp, so a shadow stays background even where it is darker than the part,
  and a shiny face as bright as the paper stays part. Sharpness is the fine-blur gradient divided by the
  coarse-blur gradient (`edge_sharpness_ratio`), so it does not depend on contrast. Openings inside the part are
  the background-coloured regions it encloses.
* **`otsu`**: one brightness threshold in both polarities. Fine on an evenly lit plain background; a shadow on
  white paper merges with the part.
* **`auto`**: the methods in `segmentation.auto_order`, first one whose part does not touch the image border.

The largest region that is not cut by the image border is the part; a table edge or shadow touching the border
cannot win just by being bigger. A normalised copy of the image (part radius = `image.normalised_radius_px`) is used by the
detectors, so they behave the same at any resolution.

### 3. Matching photo ↔ drawing (`partvision/batch.py`)

1. **Shape screen**: Hu-moment distance between the part's silhouette and each drawing outline (independent of
   position, rotation, scale and mirroring). With up to `matching.check_all_up_to` drawings, every drawing
   is then fully checked and the screen only orders them. With more, the best `matching.full_check_candidates` are.
2. **Full check** (steps 4–5) against each candidate.
3. **Score** in [0, 1] (weights in `matching.weights`):
   * outline overlap: how well the aligned drawing outline covers the part,
   * hole agreement: share of drawing holes that land on a measured hole (a hole found with a faint or
     doubled wall, e.g. a counterbore, counts `matching.weak_hole_weight`),
   * hole coverage: share of the photo's own holes that the drawing explains. A drawing of the other face of
     the part explains only a subset, and a hole-less drawing explains none. A photo hole must lie inside the
     material (`hole_count.edge_clearance` radii from the outline) and not inside a drawing opening (window
     corners are dark spots too); `matching.coverage_prior` keeps a single stray spot from zeroing the score,
   * detected: share of drawing features found.
4. The best candidate wins if it scores at least `matching.min_match_score` and finds at least
   `matching.min_detected_fraction` of its features; otherwise the photo is "no matching drawing".

### 4. Aligning the drawing to the photo (step `drawing_align`, `partvision/drawing_check/align.py`)

* **Starting guesses** from two independent methods:
  1. *outline search*: every rotation, as drawn and mirrored, scored by silhouette overlap and hole evidence,
  2. *hole-pair matching*: pairs of drawing holes matched to pairs of round spots in the photo. This works
     even when shadows or reflections spoil the silhouette.
* **Refinement** of each guess on measured edge points along the outline, plus measured hole and opening
  centres. Each group is weighted by its own measured precision, so clean hole centres dominate when the
  outline is rusty or unevenly lit.
* **Perspective**: with 6 or more measured holes, a homography is fitted (RANSAC). It is kept only when it puts
  clearly more holes on target, so an angled photo is handled without bending a straight-on one.
* The best candidate by feature agreement, edge fit and overlap is kept. Feature agreement covers the holes and
  the plain openings (window, slot: overlap of the drawn opening with the one found there), so a thick or
  stepped part, whose silhouette is the union of both faces and may fit the mirrored drawing better, is still
  aligned by its holes and windows. With fewer than 6 such features, feature agreement counts for less and
  outline overlap for more. Evidence weights (`matching.*`): a hole with a faint or doubled wall (counterbore)
  counts `weak_hole_weight`; a round edge found by searching at the expected place counts `round_edge_weight`
  (that search is centred on the expected place, so it cannot confirm a position); a hole's credit falls to 1/e
  at `agreement_falloff_radii` hole radii off. A symmetric part that fits both ways is reported as drawn.
* **Stepped parts**: if the outline fits clearly better with its own alignment (outline and holes at
  different heights, seen from a close camera), the outline *shape* is checked with that alignment and a
  warning says so.

### 5. Checking each feature

* **Holes**: the nearest round, enclosed spot of about the right size near the expected place (dark
  holes, or light ones where the background shows through). It must be surrounded by lighter or darker
  material on most sides, so shadows and crescents are rejected. The wall is then traced to measure centre
  and diameter; a blob-area estimate cross-checks it. Large holes and bores are measured by their wall edge.
  If no dark hole is found but a round edge of the right size is, it is reported as a pin, boss or plug.
* **Holes on a recess floor** in an angled photo appear shifted (parallax). They are searched for further away,
  in the direction parallax moves deeper features, and reported as found with their size checked and
  their position marked "n/a".
* **Openings** (slots, windows) are segmented and compared by overlap and length/width. **Contours** that
  enclose other features (recess or step edges) are checked edge by edge along their normals.
* **Outline**: the edge is measured along the normal at 800 points of the drawing outline. The search stops
  before it reaches a drawing hole, so it cannot snap onto a hole rim. A point is out of tolerance only beyond
  `tol + outline_guard_px`.

### 6. Outputs (`partvision/output.py`)

Steps never draw or write files themselves. They put drawing primitives, table rows and data on a shared
context, and the renderer turns that into the image, the CSV and the JSON. A new step therefore gets
outputs for free.

---

## Live camera capture with an A4 reference frame

`capture.py` replaces taking photos by hand. Put the part **on** an A4 sheet (or inside an A4-sized frame,
210 x 297 mm outer size) under the camera. The app finds the sheet, checks that the camera looks straight down,
waits until the image is steady and sharp, captures automatically, corrects perspective, sets mm per pixel from
the sheet, and runs the inspection on the corrected image.

```bash
python3 capture.py                                   # camera 0, UI at http://localhost:8000/
python3 capture.py --source http://PHONE_IP:8080/video --host 0.0.0.0   # phone as camera (IP-camera app)
python3 capture.py --drawing dxf_files/box_front.dxf # check every capture against one drawing (faster)
python3 capture.py --no-inspect                      # capture + calibration only
python3 capture.py --source photo.jpg --headless     # try it on a still image, no UI
python3 capture.py --intrinsics camera_intrinsics.json --part-height-mm 6   # see "Accuracy" below
```

### Phone camera, no app

`--source phone` uses the phone's own camera through its web browser (Chrome or Samsung Internet), with no app
to install. The phone shows the live view with the guidance; the computer does the processing and keeps the
results. Browsers allow the camera only on `localhost` or `https://`, hence two ways to connect:

**USB cable** (phone set to *USB debugging*: Settings > About phone > Software information > tap *Build number*
7 times > Developer options > USB debugging; accept the prompt on the phone when plugging in):

```bash
adb reverse tcp:8000 tcp:8000          # the phone's localhost:8000 now reaches this computer
python3 capture.py --source phone      # then open http://localhost:8000/phone on the phone
```

**Wi-Fi** (phone and computer on the same network):

```bash
python3 capture.py --source phone --https --host 0.0.0.0   # prints https://<computer-ip>:8000/phone
```

Open the printed address on the phone and accept the certificate warning once (the certificate is self-signed).
The browser sends frames at about 5-15 fps depending on the phone; stability then needs at least 3 steady frames
over 250 ms instead of 6.

```
camera ──► detect / track the A4 frame ──► alignment ──► stability ──► capture ──► lens correction
  (newest frame only, 640 px)                                            │       (if calibrated)
                                                                         ▼
       inspection (inspect_all / analyse_part)  ◄──  mm per px  ◄──  rectified image (homography)
```

### What has to be true before it captures

Checks run on every frame, cheapest first; sharpness and lighting run at full resolution only once the
geometric checks pass. All values are in config `camera.*`.

| Check | Rule (default) | Message when it fails |
|---|---|---|
| Frame found | A4-like quadrilateral (aspect within 12%, corners within 20 deg), seen in 3 consecutive frames | Position the reference frame inside the camera view. |
| All four edges | at least 60% of each edge's samples on its fitted line (a finger or part over an edge is tolerated, a covered edge is not) | Keep the edges of the reference frame uncovered. |
| Fully visible | 2% margin to every image border | Ensure the complete reference frame is visible. |
| A4 | aspect within 3% of 210/297 while the view is square-on (a Letter sheet is 9% off) | This rectangle is not A4. |
| Camera angle | tilt <= 2 deg (pose from the 4 corners) **and** interior angles within 90 +- 1 deg | Adjust camera angle. + which side is farther |
| Perspective | opposite sides differ by <= 2.5% (keystone) | Adjust camera angle. |
| Size | frame >= 12% of the view and >= 2 px/mm | Move the camera closer to the reference frame. |
| Lighting | brightness >= 45, <= 50% saturated, edge contrast >= 8, paper evenly lit (darkest / brightest >= 0.4) | Improve the lighting. |
| Steady | >= 6 frames over >= 250 ms, corners within 1 px (processing resolution, about 3 px at 1080p), nothing moving on the sheet | Hold steady. |
| Sharp | edge blur (FWHM across the frame's edges, worst side) <= 0.22% of the image diagonal (4.8 px at 1080p) | Image not sharp. Hold steady or refocus. |

**Uneven light.** A soft shadow over part of the sheet (from you, the phone, a lamp at the side) makes the
paper darker there, and the inspection's brightness threshold then merges that paper with the part. The capture
stage evens out the paper's brightness in the rectified image (a smooth surface fitted to the paper, ignoring the
part and its shadow; brightness only, geometry untouched; `camera.flatten_illumination`). A hard shadow cast by
the part itself stays: light the part from above or from two sides.

When everything passes, that same frame is captured (typically 250-350 ms after the scene becomes still) and
re-checked at full resolution. If the re-check fails, the app says so and goes back to live mode.

**Duplicates and retakes.** After a capture, the same scene is not captured again. Moving the camera or the
frame (more than about 3 px), or changing what is on the sheet (a new part), releases the lock, so a series of
parts can be inspected without touching anything. **Retake** captures the current scene again. **Capture
manually** captures even if a check fails; the failing checks are recorded in the metadata.

### What is saved (`captures/<id>/`)

| File | What |
|---|---|
| `<id>_original.png` | the frame as captured |
| `<id>_rectified.png` | the inside of the frame (2 mm inset, `camera.rectify_inset_mm`), perspective-corrected, square pixels at the frame's own resolution |
| `<id>_rectified.calib.json` | corners (original and undistorted px, and mm), homographies, camera matrix and distortion, rectified size, mm/px horizontally and vertically, timestamp, tilt, alignment and sharpness scores, stability, thresholds |
| `<id>_detection.jpg` | the frame with the detected corners |
| `<id>_rectified_uncorrected.png` | the rectified image before the lighting correction |
| `inspection/` | the normal inspection outputs (image, CSV, JSON, `pipeline.log`) for the rectified image |

### How the calibration reaches the inspection

`analyse_part.py` and `inspect_all.py` read `IMAGE_STEM.calib.json` next to an image automatically
(`--ignore-calib` to skip it). The report then has a `calibration` section, and:

* **Without a drawing**, every length is reported in mm with the frame's scale (CSV `*_mm` columns).
* **With a drawing**, the alignment still fits its own scale (unchanged, and the most robust). The frame's
  independent scale is then compared with it: `calibration.scale_check` in the report, and a warning when the
  part is uniformly larger or smaller than drawn by more than the tolerance. Before, a uniformly wrong size could
  not be detected. `--calib-fixed-scale` forces the frame's scale onto the alignment instead (like `--mm-per-px`);
  note that the fixed-scale alignment does not use the perspective hole-pair starts, so it can fail on round parts
  whose orientation only the holes reveal.

### Lens calibration (optional, recommended for accuracy)

```bash
python3 calibrate_camera.py "calib/*.jpg" --board 9x6 --square-mm 25 --out camera_intrinsics.json
python3 capture.py --intrinsics camera_intrinsics.json
```

15-25 chessboard photos at the same resolution, zoom and focus as for inspection. With it, lens distortion is
removed before measuring and the tilt and camera distance are exact. Without it, the tilt is estimated with an
assumed 75 deg diagonal field of view (`camera.assumed_diagonal_fov_deg`): a focal length 15% off gives a tilt
about 15% off. The interior-angle and keystone checks need no calibration. **Recalibrate after changing
resolution, zoom or focus.** The mm/px calibration itself is recomputed from the frame at every capture.

### Accuracy

* **Same plane.** The homography is exact for points on the plane of the frame. The top face of a part of height
  *h* is closer to the camera and appears larger by Z/(Z-h) (Z = camera distance; 2% for a 10 mm part at 500 mm).
  `--part-height-mm` (config `camera.part_height_mm`) corrects the scale for the measured face using Z from the
  pose. Features at different heights (steps, recess floors) still shift with parallax; a homography cannot
  remove that. Keep the camera square-on and far enough away (zoom in) to keep it small.
* **Resolution.** At 1080p with the sheet filling the view, about 2.7-3 px/mm (0.33-0.37 mm/px). Corners are
  located to about 0.2-0.3 px, which puts the scale within about 0.1%. Feature measurements are limited by
  the pixel size, so +-0.5 mm checks are realistic; tighter tolerances need a higher resolution or a smaller
  frame of view.
* **Lens distortion.** Uncorrected barrel distortion bends the frame's edges and changes the scale across the
  image by up to about 1%. Use `--intrinsics` when that matters.
* **The frame.** The outer edge must contrast with the background (a white sheet on a darker surface works). A
  sheet that curls, or a frame that is not exactly 210 x 297 mm, gives a proportionally wrong scale.

### Tests

```bash
python3 -m unittest discover -s tests -v
PQC_REAL_PHOTO=frame.jpg python3 -m unittest tests.test_camera.TestRealPhoto
```

`tests/synth.py` is a virtual pinhole camera: it renders the sheet (and a part drawn from a DXF) for any tilt,
distance, rotation, blur, motion and lighting, so corners, tilt and scale are known exactly. The tests cover:
aligned, slightly and strongly tilted, partly outside, moving, shaking, part being placed, dark and overexposed,
blurred and motion-blurred, near and far, portrait, landscape and rotated, Letter instead of A4, an occluded
edge, lens distortion with intrinsics, automatic capture, duplicate prevention, retake, re-arming after a
change, capture failure, a disconnected camera, and capture-to-inspection end to end.

---

## Code layout

```
analyse_part.py          single photo CLI           (partvision/cli.py)
analyse_flange.py        same, with --shape round
capture.py               live camera capture with the A4 reference frame (partvision/camera_ui.py)
calibrate_camera.py      lens calibration from chessboard photos (for capture.py --intrinsics)
inspect_all.py           batch with automatic matching (partvision/batch.py)
run_all.sh               inspect_all.py on this folder + dxf_files/
partvision/
  config.py              ALL absolute settings, overridable with --config
  core.py                Context passed to every step, geometry helpers (circle fit, ray profiles, ...)
  registry.py            @step decorator (needs, when), pipelines, dependency check
  drawing.py             DXF / JSON spec reader, feature model, drawing preview
  holefit.py             precise round-hole measurement shared by the steps
  segment.py             part/background separation methods (edges: shadow-robust; otsu)
  output.py              annotated image, results panel, CSV, JSON
  batch.py               folder scanning, photo<->drawing matching, scoring
  partmeta.py            per-part metadata (NAME.part.json) and the calibrated size check
  refframe.py            A4 frame: detection, sub-pixel edges, pose/tilt, sharpness, rectification (no I/O)
  camera.py              camera reader, capture decision, saving, hand-off to the inspection
  camera_ui.py           browser UI (MJPEG preview + status) and the capture.py command line
  drawing_check/         checking a photo against its drawing:
    transform.py         drawing mm -> image px mappings (similarity, homography)
    edges.py             edge offsets along the normals of an expected contour
    checks.py            measuring one feature at its expected place: hole, opening, inner contour
    align.py             aligning the drawing (outline search, hole pairs, refine, perspective, best candidate)
    outline.py           outline edge under the alignment, photo uncertainties
    judge.py             verdicts: tolerance with guard band, detection tolerance, combined status
    features.py          one check per drawing feature kind, registered with @feature_check(kind)
    report.py            table, counts, console summary, warnings, legend
  steps/                 one file per group of detectors, registered automatically:
    part.py              find the part (method from partvision/segment.py)
    outline.py           outline shape, size, part frame, cut-ins
    round_part.py        bore, rim + notches, rings            (when the part is round)
    bolt_circle.py       holes on a bolt circle, pitch         (when the part is round)
    holes.py             holes and openings of any shape       (part not round, or no bolt circle)
    patterns.py          circles, rectangles and rows of holes (after holes)
    drawing_check.py     the drawing pipeline: drawing_load -> drawing_align -> drawing_outline ->
                         drawing_features -> drawing_report (work done in partvision/drawing_check/)
    common.py            by-eye pockets, mm scale, ratios
```

### Adding a detector

Create `partvision/steps/my_check.py`:

```python
from ..config import C
from ..registry import step

@step("my_check", needs=("part", "outline"), when=lambda ctx: ctx.results["outline"]["shape"] == "rectangular")
def my_check(ctx):
    """One-line description shown by --list-steps."""
    k = C["analysis"]["my_check"]          # thresholds: add a section to partvision/config.py
    ...                                    # read ctx.gray, ctx.part, ctx.results["part"], ...
    ctx.table("my results", ["id", "value_px"], rows, length_cols=["value_px"])
    ctx.draw("circle", x=..., y=..., r=..., color=(40, 220, 40), width=2)
    ctx.warn("...")                        # appears in the image and the JSON
    return {"anything": "later steps need"}
```

It is picked up automatically. Run it with `--steps part,outline,my_check`, or add it to the `inspect`
pipeline in `partvision/registry.py`; `when` makes it run only on the parts it applies to.

### Adding a drawing feature check

A new kind of drawing feature (e.g. `slot`) gets one function in `partvision/drawing_check/features.py`:

```python
@feature_check("slot")
def check_slot_feature(ctx, f, S):
    # S.sim maps drawing mm -> px, S.judge.verdict / detected / combine judge it, S.mm is mm per px
    ...
    S.rows.append([...])        # one table row
    S.results.append(dict(id=f["id"], kind="slot", status=st, ...))
```

The `drawing_features` step sends each drawing feature to the check registered for its kind.

---

## Settings (`partvision/config.py`)

Settings live in `partvision/config.py`. Override any subset without editing code:

```json
{"tolerance": {"default_mm": 0.3, "detect_position_mm": 0.8},
 "image":     {"min_contrast": 10},
 "matching":  {"min_match_score": 0.6}}
```

```bash
python3 inspect_all.py --config my_settings.json
```

An unknown key is rejected with an error, so a typo cannot be silently ignored.

| Section | Key settings |
|---|---|
| `tolerance` | `default_mm` (+-tolerance when the drawing has none), `position_mm`, `detect_position_mm`, `detect_size_mm`, `outline_guard_px` |
| `segmentation` | method (`auto` / `edges` / `otsu`), sharp-edge thresholds, gap closing, spur removal and the shadow-pocket refine pass (`fragment_frac`), candidate area limits |
| `image` | contrast thresholds, smallest measurable hole, normalised detection size, low-resolution warning |
| `analysis` | drawing-free steps, one subsection each: `outline` (round / rectangular / polygon classes, which route the part), `bore`, `rim`, `rings`, `bolt_circle`, `hole_fit`, `holes` (sources, fits, edge regions, overlap rules), `rect_fit`, `patterns` |
| `drawing` | entity types, layer and line-type words treated as annotation, raster resolution, prominence fraction, default units, curve flattening, feature shape classes, border-frame and other-view detection, arc-to-circle fitting (`raster`) |
| `drawing_check` | the drawing check, one subsection per stage: `register` (first alignment), `surround`, `hole_search`, `hole_check`, `opening_check` (incl. which openings are alignment datums), `edges`, `blobs`, `hole_pairs`, `refine`, `perspective`, `candidates` (choosing between alignments), `outline_check`, `detection` |
| `matching` | file extensions, how many drawings to check, acceptance thresholds, score weights, hole-count sizes |
| `output` | image and panel width, supersampling |
| `camera` | capture resolution, lens calibration file, frame detection (edge/bright proposals, `detect_*`) and tracking (`track_search_px`, `full_*`), capture thresholds (tilt, keystone, stability, motion, sharpness, lighting), re-arm, illumination flattening, rectification resolution and inset, part height, capture folder |

Thresholds are relative to something measured in the photo where possible (the part radius, a hole radius,
the fitted scale), which keeps them valid at any resolution, framing and part size; the key name says what
(`_frac` = of the part radius, `_radii` = of the hole radius, `_px`, `_deg`). Left in the code: loop counts of
the robust fits, statistical constants (1.4826 for the MAD), numerical epsilons, the geometry of a 4-cornered
sheet, the internals of the A4 sheet's sub-pixel edge and sharpness measurements (arguments of the functions in
`partvision/refframe.py`), and the layout and colours of the annotated image.

---

## Limits and photo advice

* **Flat 2-D checks from a top view.** Depths, thread checks, heights and counterbore depths cannot be measured.
* **Scale comes from fitting the drawing**, so a part that is uniformly too large or too small cannot be detected
  from the drawing alone. Captures from `capture.py` add an independent scale from the A4 frame and check it
  (see "Live camera capture"); otherwise calibrate the camera once at a fixed height and pass `--mm-per-px`.
* **Accuracy is set by the photo.** About 0.1–0.2 mm per pixel is typical for a phone photo of a 100 mm part,
  so +-0.5 mm can be checked, but +-0.1 mm needs a closer, sharper, calibrated setup.
* **Angled photos** are corrected for the face the holes are on. Features at other heights (pocket floors,
  stacked plates) shift with the camera angle and are flagged as such.
* **Scanned or PDF blueprints** are not read automatically. Type their key features into a JSON spec.
* **For best results**: shoot straight down, fill the frame with the part, use a plain matte background
  that contrasts with the part, and use diffuse light without strong shadows or reflections.
# PQC
