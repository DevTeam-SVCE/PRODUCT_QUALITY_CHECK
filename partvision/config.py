"""Every absolute tuning value of the pipeline, in one place.

Override any of them without editing code:
    python3 analyse_part.py photo.jpg out --config my_settings.json
    python3 inspect_all.py --config my_settings.json
where my_settings.json holds only the values to change, e.g.
    {"tolerance": {"default_mm": 0.3}, "image": {"min_contrast": 10}}

Thresholds are relative to something measured in the photo where possible (the part's radius, a hole's
radius, the fitted scale), so they hold at any resolution, framing and part size; the key names say what
they are relative to. Left in the code: loop counts of robust fits, statistical constants (e.g. 1.4826 for
the MAD) and the layout/colours of the annotated image.
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
        "low_fill_warn": 0.3,           # warn when the outline fills less than this of its enclosing circle
        "small_in_frame_warn": 0.3,     # warn when the part is smaller than this fraction of the shorter side
        "view_margin_frac": 0.2,        # annotated-image crop margin around the part (x its larger side)
    },
    # Separating the part from the background (partvision/segment.py, step "part")
    "segmentation": {
        "method": "auto",               # "auto", "edges" (shadow-robust) or "otsu" (global brightness threshold)
        "auto_order": ["edges", "otsu"],  # auto: first method whose part does not touch the image border
        # candidate regions (fractions of the image area / of the largest candidate)
        "min_area_frac": 0.002,         # smaller regions are specks
        "max_area_frac": 0.6,           # larger regions are the background itself
        "keep_frac_of_largest": 0.25,   # among regions this big, one inside the frame beats a bigger one at the border
        "min_inner_frac": 2e-4,         # openings smaller than this fraction of the part's area are ignored
        # otsu
        "otsu_blur_min_px": 7,
        "otsu_blur_frac": 1 / 135,      # blur kernel / shorter image side
        # edges: sharp edge map
        "edge_sigma_fine_px": 1.0,      # blur for the fine gradient (px; edges are a few px wide at any resolution)
        "edge_sigma_coarse_px": 4.0,    # blur for the coarse gradient
        "edge_sharpness_ratio": 1.6,    # fine / coarse gradient above this = sharp edge (a soft shadow is ~1.1)
        "edge_k_high": 6.0,             # strong edge: fine gradient > this x image noise (median gradient)
        "edge_k_low": 2.5,              # weak edge, kept when connected to a strong one (hysteresis)
        "edge_noise_floor": 0.25,       # gray levels / px: lower bound for the noise estimate
        "edge_gap_frac": 0.006,         # outline gaps up to about 2x this (x shorter image side) are closed
        "spur_frac": 0.012,             # thin protrusions narrower than about 2x this are removed
        "fragment_frac": 0.1,           # refine pass: edge pieces shorter than this x the part's perimeter are
                                        #   ignored (they close shadow pockets onto the outline); 0 = off
        "refine_max_area_loss": 0.2,    # ... the refined part is kept only if it lost at most this share of area
    },
    # Drawing-free analysis (one pipeline for every part: registry.PIPELINES["inspect"]). Sizes are fractions of
    # something measured in the photo, so they hold at any resolution: "_frac" = of the part's radius, "_radii" =
    # of the hole's (or typical hole's) radius, "_px" = pixels, "_deg" = degrees.
    "analysis": {
        "outline": {  # outline shape class, which decides the round-part steps (override: --shape round|other)
            "circle_fit_floor_frac": 0.006,     # circle-fit outlier floor
            "round_min_inliers": 0.6,           # round: share of outline points on the fitted circle (or ellipse) ...
            "round_max_scatter_frac": 0.01,     # ... scatter about it at most this ...
            "round_min_radius_frac": 0.9,       # ... and its radius at least this x the enclosing circle's
            "ellipse_min_axis_ratio": 0.8,      # an ellipse this round counts as a round part seen at an angle
            "ellipse_max_axis_ratio": 0.995,
            "ellipse_tol_frac": 0.01,           # ellipse-fit inlier band (x mean radius) ...
            "ellipse_tol_min_px": 1.5,          # ... at least this many px
            "corner_approx_frac": 0.01,         # polygon simplification (x perimeter) used to count corners
            "rect_min_fill": 0.9,               # rectangular: 4 corners and fills its min-area rectangle this much
            "polygon_min_solidity": 0.9,        # polygon: convex this much and ...
            "polygon_max_corners": 12,          # ... at most this many corners; otherwise "irregular"
            "square_aspect": 1.05,              # below this length/width the part frame takes the side nearest horizontal
            "cut_in_min_depth_frac": 0.03,      # cut-ins: convexity defects deeper than this x sqrt(area) ...
            "cut_in_min_depth_px": 3.0,         # ... and this many px
            "ellipse_min_points": 20,           # robust ellipse refit stops below this many inliers
        },
        "ray_edge_fit": {  # strongest radial edge on rays + circle fit (bore, rings, pins, bores in the drawing check)
            "radial_step_px": 0.25,
            "profile_blur": 9,                  # samples (odd)
        },
        "bore": {  # central bore of a round part (Hough on the normalised image, then an edge fit)
            "blur_px": 9,
            "hough_dp": 1.2,
            "hough_param1": [100, 60, 40],      # Canny thresholds tried in turn until a bore is found
            "hough_param2": [20, 30, 40, 50],
            "min_dist_frac": 0.1,
            "min_radius_frac": 0.10,
            "max_radius_frac": 0.40,
            "max_offset_frac": 0.2,             # bore centre within this of the part centre
            "edge_search": [0.8, 1.2],          # edge fit between these x the Hough radius
            "rays": 360,
            "min_ray_frac": 0.6,                # warn below this share of rays on the fitted circle ...
            "max_scatter_frac": 0.01,           # ... or above this scatter
        },
        "rim": {  # outer rim of a round part and its notches
            "angle_step_deg": 0.5,
            "radial_step_px": 0.5,
            "background_ring": [1.06, 1.12],    # background level sampled between these x the enclosing radius
            "search_to": 0.55,                  # rim searched inward from the background ring to this
            "surface_band": [0.90, 0.96],       # part surface level sampled here
            "level_annulus": [0.85, 0.97],      # overall part level (contrast check)
            "surface_percentile": 75,           # background-side percentile of the band (holes do not drag it)
            "min_ray_contrast": 0.3,            # a ray with less contrast than this x overall uses the overall level
            "run_frac": 0.01,                   # rim = first run of part pixels this long ...
            "run_min_samples": 3,
            "smooth_frac": 0.02,                # ray profile smoothing
            "smooth_min_px": 5,
            "fit_floor_frac": 0.006,
            "min_ray_frac": 0.7,                # warn below this share of rays on the fitted rim circle
            "notch_depth_frac": 0.015,          # notch: rim inside the fitted circle by more than this x rim radius ...
            "notch_min_span_deg": 2.0,          # ... over at least this angle
            "dropout_centre_hole_widths": 0.6,  # a dip centred on a bolt hole (within this x its angular width) ...
            "dropout_max_hole_widths": 1.5,     # ... and at most this wide is lost contrast, not a notch
            "view": [1.42, 1.33],               # annotated-image crop half width / height (x rim radius)
        },
        "rings": {  # concentric steps from the median radial profile
            "profile_from_frac": 0.12,
            "profile_to_frac": 1.03,            # x rim radius
            "notch_margin_deg": 3,              # rays this close to a notch are left out
            "ray_subsample": 2,
            "smooth_frac": 0.028,
            "smooth_min_px": 5,
            "peak_window_frac": 0.03,
            "peak_window_min": 4,
            "min_gradient": 0.6,                # gray levels / sample
            "exclude_rim_frac": 0.05,           # no rings this close to the rim ...
            "exclude_bolt_hole_radii": 1.6,     # ... to the bolt circle (x hole radius) ...
            "exclude_bore_frac": 0.06,          # ... or to the bore
            "confirm_frac_of_strongest": 0.3,   # edges this strong (x strongest) get a circle fit
            "fit_band_frac": 0.04,
            "fit_floor_frac": 0.004,
            "rays": 360,
            "max_scatter_frac": 0.006,          # confirmed ring: scatter, ray share and radius shift within these
            "min_ray_frac": 0.6,
            "max_shift_frac": 0.03,
        },
        "bolt_circle": {  # holes on a common circle about the bore of a round part
            "blur_px": 5,                       # on the normalised image
            "blur_min_px": 3,
            "hough_dp": 1,
            "hough_param1": [100, 60],
            "coarse_param2": [14, 10, 8],
            "coarse_radius_frac": [0.02, 0.09], # first pass: hole radius range
            "min_dist_radii": 1.5,              # Hough minimum distance between holes
            "radius_range_frac": [0.3, 1.05],   # bolt circle radius range
            "min_bore_radii": 1.3,              # ... and outside the bore by this
            "histogram_bin_frac": 0.02,
            "histogram_bin_min_px": 2.0,
            "band_pct": 3.0,                    # a bolt hole lies within this % of the bolt-circle radius
            "min_holes": 3,
            "fine_radius_radii": [0.6, 1.5],    # second pass: hole radius range (x typical hole radius)
            "min_radius_px_normalised": 3,
            "param2_ref_radius_px": 12,         # Hough votes scale with hole size relative to this
            "fine_param2": [10, 12, 14, 16, 18, 20, 24, 28],
            "fine_param2_base": 14,             # holes found only below this are flagged
            "dedupe_radii": 0.6,                # candidates closer than this are one
            "same_hole_radii": 0.5,             # measured holes closer than this are one
            "core_radii": 0.33,                 # hole core disc
            "surface_ring_radii": [1.6, 2.1],   # surrounding surface ring
            "core_tolerance": 0.35,             # hole-like: core within this x contrast of the typical core ...
            "min_local_contrast": 0.5,          # ... and darker/brighter than its surroundings by this x contrast
            "off_circle_size_ratio": [0.6, 1.6],  # off-circle hole: radius within these x the typical radius
            "uniform_pitch_deg": 1.0,
            "ring_min_dip": 0.25,               # counterbore/groove: dip this deep (x hole contrast) ...
            "ring_max_offset_radii": 0.5,       # ... within this of the wall
            "size_class": [0.9, 1.1],           # small / large hole: diameter below / above these x typical
            "min_reliable_radius_px": 8,        # warn: hole types unreliable below this radius
            "census_level": 0.3,                # independent blob count: threshold between core and surface
            "census_band_radii": 2,
            "census_area_ratio": [0.3, 3.0],
            "census_min_circularity": 0.6,
            "max_tilt_deg": 6,                  # warn: bolt circle this elliptical
            "max_radius_scatter_frac": 0.01,    # warn: hole radii from the centre scatter more (x bolt radius)
            "max_centre_offset_frac": 0.01,     # warn: bolt circle centre this far from the bore
            "big_gap_ratio": 1.5,               # pitch above this x median is a big gap
        },
        "hole_fit": {  # precise wall fit of one round hole (bolt circle, drawing check)
            "rays": 72,
            "radial_step_px": 0.25,
            "min_rays": 8,
            "core_radii": 0.33,
            "surface_ring_radii": [1.6, 2.1],
            "recentre_window_radii": 1.8,
            "wall_search_radii": [0.25, 1.7],
            "fit_floor_radii": 0.05,
            "surface_beyond_wall_radii": [0.33, 0.75],
            "blob_window_radii": 2.5,
            "ring_profile_radii": [0.12, 1.1],  # counterbore search beyond the wall
            "ring_profile_step_px": 0.5,
            "ring_search_frac": 0.75,
            "max_scatter_radii": 0.1,           # good wall: scatter, ray share and blob size within these
            "min_ray_frac": 0.75,
            "blob_size_ratio": [0.8, 1.2],
        },
        "holes": {  # holes and openings of any shape (parts that are not round, round parts without a bolt circle)
            "sources": ["background", "surface", "edges"],  # candidate sources, in order (see steps/holes.py)
            "corner_approx_frac": 0.04,
            "round_min_circularity": 0.8,       # shape classes of one opening
            "round_max_aspect": 1.25,
            "slot_min_aspect": 1.6,
            "slot_min_solidity": 0.85,
            "rect_min_fill": 0.85,
            "round_fit": {"core_radii": 0.5, "surface_ring_radii": [1.3, 1.8], "wall_search_radii": [0.3, 1.6],
                          "fit_floor_px": 0.5, "fit_floor_radii": 0.05, "max_scatter_px": 0.6,
                          "max_scatter_radii": 0.15,  # small holes with a chamfer scatter ~1 px
                          "min_ray_frac": 0.75, "max_radius_error": 0.25,
                          "edge_fallback": True,    # no clean wall level (pin, boss): strongest radial edge
                          "compact_max_aspect": 1.5},  # irregular blobs up to this aspect are tried as circles
            "rect_candidate": {"min_fill": 0.8,   # an irregular region filling its min-area rectangle this much ...
                               "min_solidity": 0.85,  # ... and this convex gets the 4-edge rectangle fit
                               "min_side_frac": 0.05},  # ... when its shorter side is at least this x part radius
            "edges": {  # source "edges": regions closed all round by a sharp edge (segmentation edge map)
                "gap_frac": 0.005,              # edge gaps up to about 2x this (x part radius) are closed
                "outline_clearance_gaps": 2,    # searched this many gaps inside the outline ...
                "outline_band_gaps": 3,         # ... and a region reaching this band is a face, not a feature
                "min_area_frac": 0.001,         # region area (x part area)
                "max_area_frac": 0.5,
                "min_solidity": 0.85,           # texture between scratches is ragged; features are compact
                "min_boundary_strength": 3.0,   # median fine gradient on the border (x image edge noise)
                "require_fit": True,            # kept only when it measures as a clean circle or rectangle
            },
            "overlap": {"unmeasured_min_radius_frac": 0.03,  # an unmeasured blob (no clean fit) counts only from
                                                             # this area-equivalent radius (x part radius) up
                        "unmeasured_max_overlap": 0.0,  # an unmeasured blob overlapping a measured feature by
                                                        # more than this share of itself is part of that feature
                        "measured_ring_radii": 1.0},    # a measured round feature also claims a ring this many
                                                        # radii wide around it (counterbore, chamfer)
            "core_depth": 0.5,                  # interior = deeper than this x inscribed radius
            "opening_min_inner_frac": 0.015,    # through-opening: inscribed radius at least this ...
            "opening_max_background_diff": 0.35,  # ... interior within this x (surface - background) of background
            "highlight_max_inner_frac": 0.04,   # thinner than this and ...
            "highlight_max_solidity": 0.8,      # ... less solid than this: a highlight, not a slot
            "material_erode_frac": 0.02,        # surface holes: searched this far inside the outline
            "background_window_frac": 0.15,     # local surface level window
            "extreme_percentile": 0.5,          # darkest (brightest) surface level
            "threshold_spread": 0.4,
            "min_radius_frac": 0.008,
            "max_area_frac": 0.2,               # x searched area
            "min_peak_contrast": 40,            # gray levels
            "min_peak_spread": 0.6,
            "min_median_over_threshold": 1.25,
            "min_inner_frac": 0.015,
            "core_max_from_extreme": 0.2,
            "min_solidity": 0.75,
            "same_hole_diameters": 0.5,
        },
        "rect_fit": {  # precise rectangle from its four edges (holefit.rect_edge_fit); sizes x its shorter side
            "samples_per_side": 40,
            "profile_blur": 5,                  # samples (odd)
            "end_margin": 0.15,                 # profiles start/end this far from the corners (x side length)
            "band_frac": 0.15,                  # profiles reach this far either side of the rough edge ...
            "band_min_px": 4.0,
            "profile_step_px": 0.25,
            "inlier_px": 1.0,                   # robust line fit: residuals within max(this, 3 x MAD) are kept
            "max_scatter_px": 1.5,              # good fit: edge scatter within max(this, frac x shorter side) ...
            "max_scatter_frac": 0.02,
            "min_point_frac": 0.6,              # ... and this share of the profiles on the lines
        },
        "patterns": {  # circles, rectangles and rows among holes of the same shape and size
            "size_group_ratio": 1.12,
            "min_holes": 3,
            "max_circle_radius_frac": 2,
            "circle_fit_tol_px": 1.0,
            "circle_fit_tol_frac": 0.012,
            "even_pitch_tol_deg": 3,            # 3 or 4 holes count as a circle only when evenly spaced
            "triple_centre_tol_frac": 0.05,     # ... and 3 only when centred on the part
            "uniform_pitch_deg": 1.0,
            "rect_tol_px": 1.5,
            "rect_tol_frac": 0.01,
            "row_tol_px": 1.5,
            "row_tol_diameters": 0.2,
            "uniform_spacing_px": 1.5,
            "uniform_spacing_frac": 0.02,
        },
    },
    # Checking a photo against its drawing (step drawing_match). "_radii" = x the expected hole radius in the
    # photo, "_frac" = x the part's (or drawing outline's) size in px, "_px" = pixels, "_deg" = degrees.
    "drawing_check": {
        "register": {  # first alignment: outline overlap on a small raster, then ICP on the outline
            "margin_frac": 0.15,            # raster margin around the part (x its larger side)
            "raster_px": 220.0,             # the part is searched on a raster this many px across
            "hole_bg_window_frac": 0.15,    # local background for hole evidence (x raster size)
            "hole_contrast_full": 40.0,     # gray levels of local contrast that count as full hole evidence
            "hole_evidence_weight": 0.3,    # coarse score = outline overlap + this x hole evidence
            "coarse_step_deg": 2.0,
            "fine_span_deg": 2.0,           # then +- this around the best angle ...
            "fine_step_deg": 0.25,          # ... in these steps
            "outline_samples": 800,
            "icp_iters": 12,
            "icp_wide_iters": 4,            # first iterations use the wide gate
            "icp_gate_wide_frac": 0.05,     # outline points matched within this x part size ...
            "icp_gate_frac": 0.02,          # ... then this
            "icp_gate_min_px": 2.0,
            "icp_trim_medians": 2.5,        # and within this x the median distance
            "min_points": 20,
            "rms_clip_medians": 3,          # outline rms ignores distances beyond this x the median
            "within_px": 2.0,               # reported share of the outline within this distance
        },
        "surround": {  # is a spot enclosed (a hole) or open on a side (shadow, notch)?
            "core_radii": 0.5, "core_min_px": 1.5,
            "ring_radii": [1.3, 2.0],
            "sectors": 8,
            "margin_gray": 12,              # a sector counts when it differs from the core by this much
        },
        "hole_search": {  # round spots near the expected place of a drawing hole
            "reach_radii": 2.5,             # searched within this many hole radii (= 1.25 diameters)
            "window_margin_radii": 2.5,
            "median_window_radii": 6, "median_window_min_px": 9,
            "peak_percentile": 99.5,
            "peak_frac": 0.45,              # spot = local contrast above this x the strongest
            "size_ratio": [0.5, 1.6],       # spot radius within these x the expected radius
            "max_aspect": 1.5,
            "min_compact": 0.6,             # area / disc of its larger side
            "min_surrounded": 0.75,         # share of ring sectors around it that are surface (see surround)
            "sweep_percentiles": [0.5, 60], # threshold sweep for faint holes: from / to these gray percentiles ...
            "sweep_step_gray": 3.0,         # ... in these steps
            "sweep_size_ratio": [0.45, 1.6],
            "sweep_max_aspect": 1.4,
            "sweep_min_compact": 0.65,
            "same_spot_radii": 0.5,
            "exclude_radii": 0.8,           # spots at another drawing hole's place (within this) are skipped
            "off_axis_deg": 45,             # recess holes: prefer spots shifted toward the image centre
        },
        "hole_check": {
            "bore_edge_search": [0.8, 1.2], # big holes: strongest edge between these x expected radius
            "bore_rays": 360,
            "bore_floor_px": 1.0, "bore_floor_radii": 0.004,
            "bore_min_ray_frac": 0.6,
            "bore_max_scatter_px": 1.5, "bore_max_scatter_radii": 0.03,
            "bore_core_radii": 0.5,
            "bore_surround_radii": [1.15, 1.4],
            "core_radii": 0.5, "core_min_px": 1.5,
            "surround_radii": [1.4, 1.9],
            "edge_search": [0.75, 1.25],    # pin/boss: round edge between these x expected radius ...
            "edge_rays": 120,
            "edge_floor_px": 0.7,
            "edge_min_ray_frac": 0.6,
            "edge_max_scatter_px": 0.7, "edge_max_scatter_radii": 0.06,
            "edge_max_offset_radii": 0.5,   # ... centred within this of the expected place
            "shift_radii": 0.5,             # a spot further than this from the expected place is "shifted"
            "prefer_centred_round_edge": True,  # ... and loses to a clean round edge centred on the expected place
                                                #   whose inside is not darker than around it (a pin/boss in the
                                                #   hole; the shifted dark spot is then its shadow)
            "pin_inside_radii": 0.6, "pin_around_radii": [1.1, 1.4],  # inside / around that edge (x its radius)
            "pin_max_scatter_radii": 0.1,   # a pin's chamfered top scatters more than a plain edge
            "measured_size_ratio": [0.5, 1.6],  # measured hole radius within these x expected
            "max_recentre_radii": 0.6,
            "grid_steps": 7,                # shifted hole: strongest spot on this grid within one diameter
            "recess_reach_radii": 4.5,      # holes on a recess floor (parallax) are searched this far ...
            "recess_max_radius_px": 20,     # ... when smaller than this in the photo
        },
        "opening_check": {
            "window_margin_frac": 0.4, "window_margin_px": 4,
            "min_px": 4,                    # narrower than this in the photo: cannot verify
            "band_frac": 0.12,              # inside / around sampled this far from the edge (x shorter side)
            "min_samples": 5,
            "datum_min_iou": 0.7,           # an opening found this clearly is used as an alignment datum
            "ok_min_iou": 0.8,              # below this overlap the opening deviates
            "detected_min_iou": 0.5,
        },
        "edges": {  # strongest edge along the normals of an expected contour
            "profile_step_px": 0.5,
            "profile_blur_px": 5,
            "normal_probe_px": 1.5,
            "normal_probes": 60,
            "hole_margin_radii": 1.3,       # outline search stops this many hole radii short of a drawing hole
            "hole_margin_min_px": 2.0,
            "strong_frac": 0.3,             # an edge counts when at least this x the median strength ...
            "strong_min": 4.0,              # ... and this many gray levels / px
            "contour_strong_min": 6.0,      # inner contours (recess, step)
            "contour_samples": 400,
            "band_frac": 0.03,              # search band: this x the outline size ...
            "band_min_px": 4.0,             # ... at least this
        },
        "blobs": {  # round spots for hole-pair alignment
            "median_window_radii": 6, "median_window_min_px": 15, "median_window_max_px": 201,
            "size_ratio": [0.6, 1.4],
            "max_aspect": 1.6,
            "min_fill": 0.5,
            "max_count": 40,
        },
        "hole_pairs": {  # alignments from pairs of drawing holes matched to pairs of photo spots
            "max_holes": 8, "extra_holes": 4,
            "distinct_size_mm": 0.3,
            "fixed_scale_tol": 0.03,
            "scale_range": [0.6, 1.5],      # x the area-based scale
            "margin_frac": 0.3,             # search box around the part (x its larger side)
            "small_hole_frac": 0.1,         # fixing holes are smaller than this x outline size (not the bore)
            "ref_size_factor": 1.5,
            "min_spot_px": 2.0,
            "blur_px": 5,
            "edge_ref_percentile": 95,
            "outline_samples": 160,
            "size_tol_radii": 0.5, "size_tol_px": 2,
            "match_radii": 1.0, "match_min_px": 3.0,
            "max_hypotheses": 300,
            "upgrade_match_radii": [1.0, 0.6, 0.45],
            "upgrade_min_holes": 6,
            "ransac_radii": 0.5, "ransac_min_px": 3.0,
            "scale_ratio": [0.7, 1.4],      # an upgraded mapping may change the scale by this much
            "tight_match_radii": 0.45, "tight_match_min_px": 2.0,
            "keep_per_orientation": 3,
            "same_angle_deg": 2, "same_place_px": 5,
        },
        "refine": {  # joint fit on outline edge points + measured hole/opening centres
            "iters": 6,
            "mad_k": 3, "mad_floor_px": 1.0,
            "min_points": 20,
            "datum_outlier_medians": 3, "datum_outlier_min_px": 2.0,
            "sigma_floor_px": 0.5,
            "outline_equiv_points": 40.0,   # all outline samples together count as this many independent points
            "band_shrink": 0.6, "band_final_min_px": 3.0,
            "min_fit_points": 10, "min_datums": 3,
            "edge_score_weight": 0.3,
        },
        "perspective": {
            "rounds": 4,
            "min_holes": 6,
            "ransac_radii": 0.4, "ransac_min_px": 2.0,
            "scale_ratio": [0.7, 1.4],
            "min_gain": 0.02,               # kept only when hole agreement improves by more than this
        },
        "candidates": {  # choosing between alignments
            "few_features": 2, "full_weight_features": 6,  # hole weight ramps from few to full features ...
            "min_hole_weight": 0.25,                       # ... starting at this
            "edge_weight": 0.3,
            "iou_weight": 0.2, "iou_weight_few": 0.6,      # outline overlap counts more with few features
            "perspective_band_frac": 0.02,
            "same_angle_deg": 1.5, "same_place_px": 3,
            "mirror_tie": 0.98,             # a mirrored fit must beat the as-drawn one by more than this ratio
            "ambiguous": [0.9, 0.98],       # runner-up within this ratio: warn
            "scale_mismatch_warn": 0.03,    # holes vs outline scale apart by more: warn
        },
        "outline_check": {
            "p95_percentile": 95,
            "own_iters": 6, "own_band_frac": 0.5, "own_band_min_px": 3.0,
            "silhouette_iters": 15, "silhouette_trim_medians": 2.5, "silhouette_trim_min_px": 2.0,
            "own_better_ratio": 0.85, "own_better_px": 0.5,  # outline checked on its own alignment when this much better
            "edge_unc_px": 1.0,             # uncertainty: this px in quadrature with the edge scatter
            "hole_rms_from_median": 1.2,    # 2-D rms of hole centres ~ this x their median distance
            "min_holes_for_scatter": 3,
            "min_visible": 0.6,             # outline/contour edge visible on less: UNCERTAIN
            "contour_min_visible": 0.5,
            "contour_percentile": 90,
            "detect_min_mm": 1.0,           # outline detected when its median offset is within max(det, this)
            "warn_min_iou": 0.85,
        },
        "detection": {
            "size_frac": 0.5,               # detected: within max(detect tol, this x nominal size)
            "recess_pos_factor": 3, "recess_size_frac": 1.5,  # holes on a recess floor (parallax)
        },
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
        "max_block_depth": 8,           # nested block references followed this deep
        "flatten_mm": 0.01,             # curves (arcs, splines) turned into lines this close to the true curve
        "shapes": {"round_min_circularity": 0.95, "slot_min_aspect": 1.6, "slot_min_solidity": 0.9,
                   "rect_min_fill": 0.95},  # shape class of a drawing feature
        "max_hole_frac": 0.98,          # a circle this big (x part size) is the outline, not a hole
        "concentric_frac": 1e-3,        # circles this close (x part size) are concentric: hole + rings
        "frame_min_inner_frac": 0.02,   # a rectangle around a loop this big (x its area) is a drawing border
        "other_view_frac": 0.1,         # closed shapes outside the outline larger than this are other views
        "raster": {  # line work rasterised to find closed regions
            "pad_px": 10,
            "min_region_px": 25,
            "frame_corner_approx_frac": 0.01, "frame_min_fill": 0.97, "frame_min_empty": 0.5, "max_frames": 3,
            "other_view_frac": 0.05,
            "outline_band_px": 7,           # regions touching this band are the body, not inner features
            "min_face_px": 6,
            "circle_floor_px": 1.5,         # a traced region is a circle when the fit holds this share ...
            "circle_min_inliers": 0.95,
            "circle_max_scatter_px": 1.0, "circle_max_scatter_frac": 0.01,  # ... with this scatter
            "same_circle_mm": 0.1, "same_radius_mm": 0.15,  # traced circle = a CIRCLE entity within these
            "concentric_min_mm": 0.05,
            "ring_min_step_mm": 0.05,       # a concentric circle this much larger is a ring (counterbore)
        },
    },
    "matching": {
        "drawing_extensions": [".dxf", ".json"],
        "photo_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        "check_all_up_to": 6,           # with this many drawings or fewer, every drawing is fully checked
        "full_check_candidates": 3,     # otherwise: drawings fully checked after the quick shape screen
        "min_match_score": 0.65,        # below this, a photo is reported as "no matching drawing"
        "min_detected_fraction": 0.5,   # ... and also when fewer of the drawing's features than this are found
        # photo hole count used to judge a candidate drawing (fractions of the part's area-equivalent radius)
        "hole_count": {"method": "features",  # "features": the pipeline's hole finder (holes < big_hole_frac x
                                              # part length); "blobs": quick spot screen with the sizes below
                       "min_radius": 0.012, "max_radius": 0.07, "window": 0.03,
                       "edge_clearance": 1.5,   # centre at least this many hole radii inside the outline
                       "surrounded": 0.75},     # share of the hole's rim that must be surrounded by material
        "coverage_prior": 1.0,          # pseudo-count in hole coverage: (explained + k) / (holes + k)
        "weak_hole_weight": 0.5,        # hole agreement credit for a hole found with a faint/doubled wall
        "round_edge_weight": 0.25,      # ... for a round edge found by searching at the expected place
        "agreement_falloff_radii": 0.75,  # hole agreement falls to 1/e at this centre error (in hole radii)
        "opening_margin_frac": 0.1,     # photo holes within this x opening size of a drawing opening are not holes
        "opening_rect_fit": True,       # rectangular openings measured by their 4 edges (analysis.rect_fit), not by
                                        #   segmenting the inside by brightness (fallback when the fit fails)
        "big_hole_frac": 0.15,          # hole coverage: drawing holes this large (x part size) are bores, not holes
        "inside_big_hole_diameters": 0.5,  # ... photo holes within this x a bore's diameter are inside it
        "hole_explained_diameters": 0.6,   # a photo hole within this x diameter of a drawing hole is explained ...
        "hole_explained_min_mm": 1.5,      # ... or within this many mm
        # the matched drawing's outline says whether the drawing-free analysis treats the part as round
        "round_drawing": {"min_hull_circularity": 0.95, "min_aspect": 0.95},
        "weights": {"outline_overlap": 0.4, "hole_agreement": 0.3, "hole_coverage": 0.2, "detected": 0.1},
        # captures with an A4-frame calibration (capture.py): the absolute size vs each drawing (partmeta.py).
        # size_gate "reject": rule out drawings whose size no part height explains (a raised face only ever looks
        #   larger, by at most calib_max_part_height_mm); "report": match on geometry alone, only report the size.
        #   Only captures (A4-frame calibration) carry a size; plain photos are unaffected.
        "size_gate": "reject",
        "size_tiebreak_score": 0.03,        # drawings scoring within this of the best: the better size fit wins
        "calib_size_tol_pct": 4.0,          # part height known: measured vs expected size
        "calib_max_smaller_pct": 10.0,      # height unknown: a part never looks smaller than drawn by more (margin for fit error)
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
        "detect_blur_px": 5,
        "detect_canny": [20, 50],           # edge-ring proposals: Canny thresholds (low contrast paper still closes)
        "detect_bright_percentile": 99,     # bright-region proposals: second level halfway to this percentile
        "detect_max_area_frac": 0.97,       # larger regions are the background
        "detect_cut_max_area_frac": 0.8,    # a bright region cut by the border, smaller than this: frame partly out
        "detect_partial_min_solidity": 0.95,
        "detect_partial_approx_frac": 0.015,
        "detect_min_fill": 0.85,            # region area / its quadrilateral's area
        "detect_try_candidates": 3,
        "detect_search_px": [6.0, 3.0],     # edge refinement bands for a new detection (processing px) ...
        "track_search_px": 8.0,             # ... and while tracking
        "partial_hint_frames": 15,          # frames after losing a frame near the border to keep saying "partial"
        "max_read_failures": 10,            # consecutive failed reads before the camera counts as gone
        "full_search_proc_px": 2.0,         # full resolution: corners refined within this (processing px) ...
        "full_search_px": 3.0,              # ... plus this many full-resolution px
        "full_profiles": [48, 64],          # gradient profiles per side in the two full-resolution passes
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
        "max_motion_frac": 0.003,           # ... and share of pixels inside the frame changed by > motion_pixel_level
        "motion_pixel_level": 20.0,
        "max_edge_width_frac": 0.0022,      # 10-90% edge rise / image diagonal (about 4.8 px at 1080p)
        "min_edge_width_limit_px": 2.5,
        "min_edge_contrast": 8.0,           # gray levels across the frame edge (full resolution)
        "min_brightness": 45.0,
        "max_clipped_frac": 0.5,            # share of saturated pixels inside the frame
        "min_uniformity": 0.4,              # darkest / brightest paper inside the frame (a shadow across the sheet)
        "flatten_illumination": True,       # even out the paper's brightness in the rectified image (not geometry)
        "flatten_level": 225.0,             # ... to this gray level
        # after a capture
        "auto_rearm": True,                 # capture again when the scene changes (next part), no Retake needed
        "rearm_corner_px": 3.0,             # frame moved this far (processing px) -> camera moved: re-arm
        "rearm_changed_frac": 0.004,        # share of pixels inside the frame changed vs. the capture -> re-arm
        "rearm_pixel_level": 25.0,          # ... by more than this many gray levels
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
