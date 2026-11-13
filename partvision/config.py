"""Every absolute tuning value of the pipeline, in one place.

Override any of them without editing code:
    python3 analyse_part.py photo.jpg out --config my_settings.json
    python3 inspect_all.py --config my_settings.json
where my_settings.json holds only the values to change, e.g.
    {"tolerance": {"default_mm": 0.3}, "image": {"min_contrast": 10}}

Values NOT here are geometric ratios relative to something measured in the photo (the part's radius, a
hole's radius, the fitted scale). Those make the pipeline independent of resolution, framing and part
size, and are part of the algorithms rather than settings.
"""
import copy
import json

DEFAULTS = {
    "tolerance": {
        "default_mm": 0.5,          # +-tolerance for size and position when the drawing carries none
        "position_mm": None,        # separate position tolerance (None = same as default_mm)
        "detect_position_mm": 1.0,  # "found": centre within max(this, 0.5 x nominal size) of the drawing
        "detect_size_mm": 1.5,      # "found": size within max(this, 0.5 x nominal size)
        "outline_guard_px": 1.0,    # outline edge marked out of tolerance only beyond tol + this many px
    },
    "image": {
        "normalised_radius_px": 250.0,  # detectors run on a copy where the part's radius is this many px
        "min_contrast": 15,             # gray levels: below this a hole/opening is "no contrast"
        "min_part_contrast": 25,        # gray levels: part vs background contrast warning
        "spot_contrast": 25,            # gray levels: a hole candidate must differ this much from its surroundings
        "min_hole_radius_px": 3,        # holes smaller than this in the photo cannot be measured
        "big_hole_radius_px": 20,       # from this radius, holes/bores are measured by their wall edge
        "low_resolution_px": 300,       # warn when the part is smaller than this across
    },
    "drawing": {
        "skip_entity_types": ["DIMENSION", "ARC_DIMENSION", "LARGE_RADIAL_DIMENSION", "TEXT", "MTEXT", "ATTRIB",
                              "ATTDEF", "HATCH", "LEADER", "MLEADER", "MULTILEADER", "TOLERANCE", "POINT", "IMAGE",
                              "WIPEOUT", "SOLID", "VIEWPORT", "XLINE", "RAY"],
        "skip_layer_words": ["DIM", "TEXT", "CENTER", "CENTRE", "HIDDEN", "HATCH", "TITLE", "BORDER", "FRAME",
                             "NOTE", "ANNO", "SYMBOL", "DEFPOINTS"],
        "skip_linetype_words": ["CENTER", "CENTRE", "HIDDEN", "DASH", "PHANTOM", "DOT", "DIVIDE", "BORDER"],
        "raster_max_px": 4000,          # line work is rasterised to find closed shapes (longest side, px)
        "raster_max_px_per_mm": 40,
        "prominent_fraction": 0.03,     # a feature >= this fraction of the part size is "prominent"
        "default_units_mm": 1.0,        # mm per unit when the DXF has no unit setting
    },
    "matching": {
        "drawing_extensions": [".dxf", ".json"],
        "photo_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        "check_all_up_to": 6,           # with this many drawings or fewer, every drawing is fully checked
        "full_check_candidates": 3,     # otherwise: drawings fully checked after the quick shape screen
        "min_match_score": 0.65,        # below this, a photo is reported as "no matching drawing"
        "min_detected_fraction": 0.5,   # ... and also when fewer of the drawing's features than this are found
        # photo hole count used to judge a candidate drawing (fractions of the part's area-equivalent radius)
        "hole_count": {"min_radius": 0.012, "max_radius": 0.07, "window": 0.03},
        "weights": {"outline_overlap": 0.4, "hole_agreement": 0.3, "hole_coverage": 0.2, "detected": 0.1},
        # captures with an A4-frame calibration (capture.py): the absolute size vs each drawing (partmeta.py).
        # size_gate "report": match on geometry alone (each drawing is fitted at its own scale), the size
        #   difference is only reported; "reject": also rule out drawings whose size no part height explains
        "size_gate": "report",
        "size_tiebreak_score": 0.03,        # drawings scoring within this of the best: the better size fit wins
        "calib_size_tol_pct": 4.0,          # part height known: measured vs expected size
        "calib_max_smaller_pct": 3.0,       # height unknown: a part never looks smaller than drawn by more
        "calib_max_part_height_mm": 80.0,   # ... nor larger than a face this high above the frame explains
    },
    "output": {
        "image_width_px": 1090,         # width of the annotated photo
        "panel_width_px": 600,          # width of the results panel beside it
        "supersample": 2,
    },
    # Live capture with an A4 reference frame (capture.py, partvision/camera.py, partvision/refframe.py)
    "camera": {
        "reference_mm": [210.0, 297.0],     # outer size of the reference frame (short, long side)
        "width": 1920, "height": 1080,      # requested capture resolution (the camera may give another)
        "fps": 30,
        "intrinsics": None,                 # JSON from calibrate_camera.py; None = assumed pinhole camera
        "assumed_diagonal_fov_deg": 75.0,   # used for the tilt estimate when there is no calibration
        "processing_px": 640,               # longest side of the frame used for detection and tracking
        "preview_px": 960,                  # longest side of the live preview
        "preview_jpeg_quality": 78,
        # detection (lenient, so the user gets guidance instead of "not found")
        "detect_min_area_frac": 0.04,       # quad area / frame area
        "detect_aspect_tol": 0.12,          # relative deviation from 210/297 still recognised as the frame
        "detect_angle_tol_deg": 20.0,
        "confirm_frames": 3,                # consecutive detections before the frame counts as found
        "track_jump_px": 25.0,              # at processing resolution: larger jumps restart tracking
        "min_edge_inliers": 0.6,            # share of each side's edge samples on the fitted line (occlusion)
        # capture gates
        "aspect_tol": 0.03,                 # relative aspect error allowed when aligned (A4 vs. Letter is 9%)
        "max_tilt_deg": 2.0,                # camera axis vs. plane normal (PnP with the intrinsics)
        "max_corner_angle_dev_deg": 1.0,    # interior angles in the image vs. 90 deg (intrinsics-free)
        "max_keystone": 0.025,              # opposite side length ratio - 1 (intrinsics-free)
        "min_margin_frac": 0.02,            # gap between the frame and every image border / shorter side
        "min_area_frac": 0.12,              # below: "move closer"
        "min_px_per_mm": 2.0,               # frame resolution in the full frame; below: "move closer"
        "stable_frames": 6,
        "stable_ms": 250,
        "max_corner_disp_px": 1.0,          # at processing resolution (~3 px at 1080p), over the stability window
        "max_content_motion": 2.5,          # mean gray-level change inside the frame between frames
        "max_motion_frac": 0.003,           # ... and share of pixels inside the frame changed by > 20 levels
        "max_edge_width_frac": 0.0022,      # 10-90% edge rise / image diagonal (about 4.8 px at 1080p)
        "min_edge_width_limit_px": 2.5,
        "min_edge_contrast": 8.0,           # gray levels across the frame edge (full resolution)
        "min_brightness": 45.0,
        "max_clipped_frac": 0.5,            # share of saturated pixels inside the frame
        "min_uniformity": 0.4,              # darkest / brightest paper inside the frame (a shadow across the sheet)
        "flatten_illumination": True,       # even out the paper's brightness in the rectified image (not geometry)
        # after a capture
        "auto_rearm": True,                 # capture again when the scene changes (next part), no Retake needed
        "rearm_corner_px": 3.0,             # frame moved this far (processing px) -> camera moved: re-arm
        "rearm_changed_frac": 0.004,        # share of pixels inside the frame changed vs. the capture -> re-arm
        # rectified output for the measurement pipeline
        "rectify_px_per_mm": None,          # None = the frame's native resolution in the photo
        "rectify_inset_mm": 2.0,            # crop this much inside the frame edge (edge, shadow, printed border)
        "part_height_mm": 0.0,              # height of the measured face above the frame's plane (see README)
        "save_dir": "captures",
    },
}


class _Cfg(dict):
    def get_path(self, path, default=None):
        node = self
        for k in path.split("."):
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


C = _Cfg(copy.deepcopy(DEFAULTS))


def _merge(dst, src, where=""):
    for k, v in src.items():
        if k not in dst:
            raise SystemExit(f"unknown config key '{where}{k}' (see partvision/config.py for valid keys)")
        if isinstance(dst[k], dict) and isinstance(v, dict):
            _merge(dst[k], v, f"{where}{k}.")
        else:
            dst[k] = v


def load(path=None):
    """Reset to defaults, then apply overrides from a JSON file (if given)."""
    C.clear()
    C.update(copy.deepcopy(DEFAULTS))
    if path:
        with open(path) as f:
            _merge(C, json.load(f))
    return C
