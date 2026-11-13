"""Command line: analyse_part.py IMAGE OUTDIR [options]   (python3 analyse_part.py --help)"""
import argparse
import os
import re

from . import steps as _steps  # noqa: F401  (registers every step)
from .core import Context
from .output import render, write_csv, write_json
from .registry import PROFILES, STEPS, run

DESCRIPTION = """Measure a flat part photographed top-down on a contrasting background.

Profiles (--profile):
  auto     measure the outline, then use 'round' for round parts and 'generic' for anything else
  round    flanges/hubs: bore, rim + notches, holes on a bolt circle, rings
  generic  any outline: size, cut-ins, holes of any shape (round/slot/rectangular/irregular), hole patterns
  drawing  check the part against its drawing (--drawing FILE.dxf or FILE.json): every drawing feature is
           looked for where it should be and reported OK / DEVIATES / MISSING / CAN'T VERIFY
           (chosen automatically when --drawing is given)
Or give an explicit step list with --steps (see --list-steps).

Outputs in OUTDIR, prefixed with --name (default: the image file name):
  NAME_annotated.png, NAME_measurements.csv, NAME_report.json
Check the annotated image before trusting the numbers; self-check failures are listed under WARNINGS."""


def load_calibration(image):
    """The capture.py calibration sidecar of an image (IMAGE_STEM.calib.json), or None."""
    import json
    p = os.path.splitext(image)[0] + ".calib.json"
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        d = json.load(f)
    if not d.get("mm_per_px"):
        return None
    d["_path"] = p
    return d


def _calibration_scale_check(ctx, calib):
    """Absolute size: the drawing alignment fits its own scale, so a part uniformly too large or too small fits
    perfectly. The reference frame's independent scale, with the part's height (partvision/partmeta.py), shows it."""
    from .config import C
    from .partmeta import size_check
    d = ctx.data["drawing"]
    size = max(d.get("outline_size_mm") or [0.0])
    tol = float(ctx.opt_get("tol") or C["tolerance"]["default_mm"])
    chk = size_check(calib, d["registration"]["mm_per_px"], ctx.opt_get("drawing"), size, tol)
    ctx.data["calibration"]["scale_check"] = chk
    if chk["status"] == "UNVERIFIED":
        ih = chk.get("implied_height_mm")
        ctx.say(f"absolute size: part looks {chk['deviation_pct']:+.1f}% vs the drawing; height unknown"
                + (f" (implies a face ~{ih:.0f} mm above the frame)" if ih is not None else ""))
        ctx.note("Absolute size not verified: the part's height is unknown. Add "
                 f"{os.path.splitext(os.path.basename(ctx.opt_get('drawing')))[0]}.part.json "
                 '{"height_mm": H} next to the drawing'
                 + (f" (this photo implies about {ih:.0f} mm)." if ih is not None else "."))
    else:
        ctx.say(f"absolute size vs reference frame: {chk['deviation_pct']:+.2f}% "
                f"({chk['deviation_over_part_mm']:+.2f} mm over {size:.0f} mm, face {chk['height_mm']:g} mm high) "
                f"-> {chk['status']}")
        if chk["status"] != "OK":
            ctx.warn(f"Absolute size: with the reference frame's scale and a {chk['height_mm']:g} mm high face "
                     f"({chk['height_source']}), the part is {chk['deviation_pct']:+.2f}% "
                     f"({chk['deviation_over_part_mm']:+.2f} mm over {size:.0f} mm) "
                     f"{'larger' if chk['deviation_pct'] > 0 else 'smaller'} than the drawing.")
    if not chk["plausible"]:
        if C["matching"].get("size_gate") == "reject":
            ctx.warn(f"Size mismatch: the part looks {chk['deviation_pct']:+.1f}% vs this drawing, more than any part "
                     "height explains. Probably the wrong drawing.")
        else:
            ctx.note(f"Scale-free match: the part looks {chk['deviation_pct']:+.1f}% vs the drawing's nominal size; "
                     "geometry was compared at the fitted scale (matching.size_gate = \"report\").")


def build_parser(default_profile="auto"):
    ap = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--profile", choices=["auto", *PROFILES], default=default_profile)
    ap.add_argument("--steps", help="comma-separated steps to run, overriding --profile")
    ap.add_argument("--list-steps", action="store_true", help="list available steps and profiles, then exit")
    ap.add_argument("--known-mm", action="append", default=[], metavar="FEATURE=MM",
                    help="real size of a measured feature for scale, e.g. bore=25, bolt=90, outer=110, "
                         "length=120, width=60, hole=8 (the run prints what is available)")
    ap.add_argument("--bore-mm", type=float, default=None, help="shorthand for --known-mm bore=MM")
    ap.add_argument("--pockets", default="", help="BY-EYE pocket estimates 'x,y,r;x,y,r' in source pixels")
    ap.add_argument("--hole-polarity", choices=["dark", "bright"], default="dark",
                    help="generic profile: blind holes look darker (default) or brighter than the surface")
    ap.add_argument("--drawing", help="DXF drawing or JSON spec of the part to check against")
    ap.add_argument("--config", default=None, help="JSON file overriding settings in partvision/config.py")
    ap.add_argument("--tol", type=float, default=None,
                    help="tolerance in mm when the drawing has none (default: config tolerance.default_mm)")
    ap.add_argument("--pos-tol", type=float, default=None, help="default position tolerance in mm (default: --tol)")
    ap.add_argument("--detect-pos-tol", type=float, default=None,
                    help="detection tolerance (mm): a feature counts as found if its centre is within "
                         "max(this, half its nominal size) of the drawing position (default: config)")
    ap.add_argument("--detect-size-tol", type=float, default=None,
                    help="detection tolerance (mm): ... and its size within max(this, half its nominal size) "
                         "(default: config)")
    ap.add_argument("--mm-per-px", type=float, default=None,
                    help="fixed scale from a camera calibration; enables absolute size checks against the drawing. "
                         "With a calibration sidecar (IMAGE_STEM.calib.json, written by capture.py) its scale is used "
                         "for the drawing-free analysis and to cross-check the drawing's fitted scale")
    ap.add_argument("--ignore-calib", action="store_true", help="do not read the IMAGE_STEM.calib.json sidecar")
    ap.add_argument("--calib-fixed-scale", action="store_true",
                    help="also force the sidecar's scale onto the drawing alignment (as --mm-per-px). Off by default: "
                         "the fitted alignment is more robust, and the calibration then checks its absolute scale")
    ap.add_argument("--mirror", choices=["auto", "no", "yes"], default="auto",
                    help="whether the photo shows the mirror image of the drawing view (the other face)")
    ap.add_argument("--name", default=None, help="output file prefix (default: from the image file name)")
    ap.add_argument("--out-image", default=None, help="path of the output image (default: OUTDIR/NAME_annotated.png)")
    ap.add_argument("--no-image", action="store_true", help="write only the CSV and JSON data")
    ap.add_argument("--data-dir", default=None, help="folder for the CSV/JSON data (default: OUTDIR)")
    ap.add_argument("--debug-dir", default=None, help="also write an image with debug overlays here")
    return ap


def main(argv=None, default_profile="auto"):
    args = build_parser(default_profile).parse_args(argv)
    from . import config
    config.load(args.config)
    if args.list_steps:
        print("steps:")
        for n, s in STEPS.items():
            print(f"  {n:<12} {s['doc']}" + (f"  [needs {', '.join(s['needs'])}]" if s["needs"] else ""))
        print("profiles:")
        for n, p in PROFILES.items():
            print(f"  {n:<8} {','.join(p)}")
        return 0
    if not args.image or not args.outdir:
        build_parser().error("IMAGE and OUTDIR are required")
    os.makedirs(args.outdir, exist_ok=True)
    name = args.name or re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.splitext(os.path.basename(args.image))[0]).strip("_")
    calib = None if args.ignore_calib else load_calibration(args.image)
    args.calib_mm_per_px = float(calib["mm_per_px"]) if calib and args.mm_per_px is None else None
    if args.calib_mm_per_px and args.calib_fixed_scale:
        args.mm_per_px = args.calib_mm_per_px
    ctx = Context(args.image, args)
    if calib:
        ctx.data["calibration"] = {k: calib.get(k) for k in ("capture", "reference_frame", "pixel_to_mm", "part_plane",
                                                             "mm_per_px", "quality", "camera", "homography", "corners")}
        ctx.data["calibration"]["file"] = calib["_path"]
        if args.calib_mm_per_px:
            ctx.note(f"Scale {args.calib_mm_per_px:.5f} mm/px from the A4 reference-frame calibration "
                     f"({os.path.basename(calib['_path'])}); valid on the frame's plane"
                     + (f", corrected for a face {calib['part_plane']['height_mm']:g} mm above it."
                        if calib.get("part_plane") else ". A face above that plane measures larger by Z/(Z-h)."))

    if args.steps:
        names = [s.strip() for s in args.steps.split(",") if s.strip()]
        profile = "custom"
    else:
        profile = args.profile
        if args.drawing and profile == "auto":
            profile = "drawing"
        if profile == "auto":
            run(ctx, ["part", "outline"])
            profile = "round" if ctx.results["outline"]["shape"] == "round" else "generic"
        names = PROFILES[profile]
    ctx.data["profile"] = profile
    pre_scale = [n for n in names if n not in ("scale", "ratios")]
    run(ctx, pre_scale)
    if args.profile == "auto" and profile == "round" and not ctx.results.get("bolt_circle", {}).get("ok"):
        ctx.warn("Round part without a bolt circle: also ran the generic hole finder.")
        run(ctx, ["holes", "patterns"])
    run(ctx, [n for n in names if n in ("scale", "ratios")])
    if calib and args.calib_mm_per_px and "drawing" in ctx.data:
        _calibration_scale_check(ctx, calib)

    data_dir = args.data_dir or args.outdir
    os.makedirs(data_dir, exist_ok=True)
    png = None if args.no_image else (args.out_image or os.path.join(args.outdir, f"{name}_annotated.png"))
    csv_path = os.path.join(data_dir, f"{name}_measurements.csv")
    js = os.path.join(data_dir, f"{name}_report.json")
    if png:
        os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
        if getattr(ctx, "drawing_model", None):
            # one image: photo with the drawing overlaid + the drawing's features + per-feature results
            from .drawing import render_preview
            from .output import render_panel
            prev = render_preview(ctx.drawing_model, None, os.path.basename(args.drawing), caption=False)
            render_panel(ctx, png, f"{os.path.basename(args.image)}  vs  {os.path.basename(args.drawing)}", prev)
        else:
            render(ctx, png)
    write_csv(ctx, csv_path)
    write_json(ctx, js)
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        render(ctx, os.path.join(args.debug_dir, f"{name}_debug.png"), debug=True)

    print(f"profile: {profile}  (steps: {', '.join(n for n in ctx.results)})")
    for line in ctx.summary:
        print(line)
    if ctx.dims:
        print("lengths available for --known-mm: " + ", ".join(f"{k}={v:.1f}px" for k, v in sorted(ctx.dims.items())))
    print(f"\nWARNINGS ({len(ctx.warnings)}):" if ctx.warnings else "\nWARNINGS: none")
    for m in ctx.warnings:
        print(" - " + m)
    for m in ctx.notes:
        print("note: " + m)
    print("\nwrote " + "\n      ".join(p for p in (png, csv_path, js) if p))
    return 0
