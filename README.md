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

### Other commands

```bash
python3 inspect_all.py --help                                    # batch: options below
python3 inspect_all.py --photos new_photos/ --drawings dxf_files/ more_drawings/ --out results/
python3 inspect_all.py --pairs pairs.json                        # force pairings: {"photo9.jpg": "dxf_files/x.dxf"}
python3 inspect_all.py --config my_settings.json                 # change any setting (see "Settings")

python3 analyse_part.py photo.jpg out/ --drawing part.dxf        # one photo against one drawing
python3 analyse_part.py photo.jpg out/                           # one photo, no drawing (auto: round or generic)
python3 analyse_part.py photo.jpg out/ --known-mm bore=25        # ... with a real size for mm
python3 analyse_part.py --help                                   # every option
python3 analyse_part.py --list-steps                             # every pipeline step and profile
python3 analyse_flange.py photo.jpg out/                         # shortcut for the round (flange) profile
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

Otsu threshold in both polarities (dark part on light background, or the reverse). The largest region that
is not cut by the image border is the part; a table edge or shadow touching the border cannot win just by
being bigger. A normalised copy of the image (part radius = `image.normalised_radius_px`) is used by the
detectors, so they behave the same at any resolution.

### 3. Matching photo ↔ drawing (`partvision/batch.py`)

1. **Shape screen**: Hu-moment distance between the part's silhouette and each drawing outline (independent of
   position, rotation, scale and mirroring). With up to `matching.check_all_up_to` drawings, every drawing
   is then fully checked and the screen only orders them. With more, the best `matching.full_check_candidates` are.
2. **Full check** (steps 4–5) against each candidate.
3. **Score** in [0, 1] (weights in `matching.weights`):
   * outline overlap: how well the aligned drawing outline covers the part,
   * hole agreement: share of drawing holes that land on a measured hole,
   * hole coverage: share of the photo's own holes that the drawing explains. A drawing of the other face of
     the part explains only a subset, and a hole-less drawing explains none,
   * detected: share of drawing features found.
4. The best candidate wins if it scores at least `matching.min_match_score` and finds at least
   `matching.min_detected_fraction` of its features; otherwise the photo is "no matching drawing".

### 4. Aligning the drawing to the photo (step `drawing_match`, function `register` and after)

* **Starting guesses** from two independent methods:
  1. *outline search*: every rotation, as drawn and mirrored, scored by silhouette overlap and hole evidence,
  2. *hole-pair matching*: pairs of drawing holes matched to pairs of round spots in the photo. This works
     even when shadows or reflections spoil the silhouette.
* **Refinement** of each guess on measured edge points along the outline, plus measured hole and opening
  centres. Each group is weighted by its own measured precision, so clean hole centres dominate when the
  outline is rusty or unevenly lit.
* **Perspective**: with 6 or more measured holes, a homography is fitted (RANSAC). It is kept only when it puts
  clearly more holes on target, so an angled photo is handled without bending a straight-on one.
* The best candidate by hole agreement, edge fit and overlap is kept. With fewer than 6 holes, hole agreement
  counts for less and outline overlap for more. A symmetric part that fits both ways is reported as drawn.
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

## Code layout

```
analyse_part.py          single photo CLI           (partvision/cli.py)
analyse_flange.py        same, round profile by default
inspect_all.py           batch with automatic matching (partvision/batch.py)
run_all.sh               inspect_all.py on this folder + dxf_files/
partvision/
  config.py              ALL absolute settings, overridable with --config
  core.py                Context passed to every step, geometry helpers (circle fit, ray profiles, ...)
  registry.py            @step decorator, profiles, dependency check
  drawing.py             DXF / JSON spec reader, feature model, drawing preview
  holefit.py             precise round-hole measurement shared by the steps
  output.py              annotated image, results panel, CSV, JSON
  batch.py               folder scanning, photo<->drawing matching, scoring
  steps/                 one file per group of detectors, registered automatically:
    part.py              find the part
    outline.py           outline shape, size, part frame, cut-ins
    round_part.py        bore, rim + notches, rings          (round profile)
    bolt_circle.py       holes on a bolt circle, pitch       (round profile)
    holes.py             holes and openings of any shape     (generic profile)
    patterns.py          circles, rectangles and rows of holes (generic profile)
    drawing_match.py     drawing alignment and per-feature checks (drawing profile)
    common.py            by-eye pockets, mm scale, ratios
```

### Adding a detector

Create `partvision/steps/my_check.py`:

```python
from ..registry import step

@step("my_check", needs=("part",))
def my_check(ctx):
    """One-line description shown by --list-steps."""
    ...                                    # read ctx.gray, ctx.part, ctx.results["part"], ...
    ctx.table("my results", ["id", "value_px"], rows, length_cols=["value_px"])
    ctx.draw("circle", x=..., y=..., r=..., color=(40, 220, 40), width=2)
    ctx.warn("...")                        # appears in the image and the JSON
    return {"anything": "later steps need"}
```

It is picked up automatically. Run it with `--steps part,my_check`, or add it to a profile in
`partvision/registry.py`.

---

## Settings (`partvision/config.py`)

Every absolute value lives in `partvision/config.py`. Override any subset without editing code:

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
| `image` | contrast thresholds, smallest measurable hole, normalised detection size, low-resolution warning |
| `drawing` | entity types, layer and line-type words treated as annotation, raster resolution, prominence fraction, default units |
| `matching` | file extensions, how many drawings to check, acceptance thresholds, score weights, hole-count sizes |
| `output` | image and panel width, supersampling |

Numbers that are not in the config are ratios relative to something measured in the photo, such as the part
radius, a hole radius or the fitted scale. That is what makes the pipeline independent of resolution, framing
and part size, so they are part of the algorithms rather than settings.

---

## Limits and photo advice

* **Flat 2-D checks from a top view.** Depths, thread checks, heights and counterbore depths cannot be measured.
* **Scale comes from fitting the drawing**, so a part that is uniformly too large or too small cannot be detected.
  For absolute size, calibrate the camera once at a fixed height and pass `--mm-per-px`.
* **Accuracy is set by the photo.** About 0.1–0.2 mm per pixel is typical for a phone photo of a 100 mm part,
  so +-0.5 mm can be checked, but +-0.1 mm needs a closer, sharper, calibrated setup.
* **Angled photos** are corrected for the face the holes are on. Features at other heights (pocket floors,
  stacked plates) shift with the camera angle and are flagged as such.
* **Scanned or PDF blueprints** are not read automatically. Type their key features into a JSON spec.
* **For best results**: shoot straight down, fill the frame with the part, use a plain matte background
  that contrasts with the part, and use diffuse light without strong shadows or reflections.
# PQC
