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
    },
    "output": {
        "image_width_px": 1090,         # width of the annotated photo
        "panel_width_px": 600,          # width of the results panel beside it
        "supersample": 2,
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
