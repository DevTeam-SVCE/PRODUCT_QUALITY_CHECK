"""Checking a photo against its drawing (the "drawing" pipeline; steps in partvision/steps/drawing_check.py).

  common.py     settings access (config drawing_check.*), colours, small geometry helpers
  transform.py  drawing mm -> image px mappings (similarity, homography) and their fit
  edges.py      edge offsets along the normals of an expected contour
  checks.py     measuring one feature at its expected place: hole, opening, inner contour
  align.py      aligning the drawing to the photo (outline search, hole pairs, refine, perspective)
  outline.py    outline edge under the alignment, and the photo's uncertainties
  judge.py      verdicts: tolerance with guard band, detection tolerance, combined status
  features.py   per-feature-kind checks, registered with @feature_check(kind)
  report.py     table, counts, console summary, warnings, legend
"""
