"""Steps any profile can use: by-eye pockets, scale, ratios."""
from ..core import angle_deg
from ..registry import step

OR = (255, 150, 30)


@step("pockets", needs=("part",))
def pockets(ctx):
    """By-eye pocket estimates from --pockets "x,y,r;..." (reported as estimates, never measured)."""
    spec = ctx.opt_get("pockets", "") or ""
    out = []
    ox, oy = ctx.origin[:2]
    for i, tok in enumerate(filter(None, spec.split(";")), 1):
        x, y, r = map(float, tok.split(","))
        out.append(dict(name=f"P{i}", x=x, y=y, r=r, ang=angle_deg(x, y, ox, oy),
                        R=((x - ox) ** 2 + (y - oy) ** 2) ** 0.5, source="estimated by eye"))
    ctx.data["pockets"] = out
    for p in out:
        ctx.feature("pocket " + p["name"], f'{2 * p["r"]:.1f}', f'{p["ang"]:.1f}',
                    note=f'estimated by eye at ({p["x"]:.0f}, {p["y"]:.0f}), {p["R"]:.0f} px from the {ctx.origin[2]}')
        ctx.draw("circle", z=3, x=p["x"], y=p["y"], r=p["r"], color=OR, width=2.5)
        ctx.draw("text", z=6, x=p["x"], y=p["y"], text=p["name"], color=OR, size=14)
    if out:
        ctx.add_legend("orange", f"{len(out)} pockets (P1-P{len(out)}, estimated by eye)", OR, order=21)
    return dict(pockets=out)


@step("scale")
def scale(ctx):
    """mm per px from --known-mm FEATURE=MM, using any length a step registered in ctx.dims."""
    known = list(ctx.opt_get("known_mm", []) or [])
    if ctx.opt_get("bore_mm"):
        known.append(f"bore={ctx.opt_get('bore_mm')}")
    if not known:
        ctx.add_legend("info", "No scale: pixels and degrees.", (255, 0, 230), order=99)
        return dict(mm=None)
    feat, val = known[0].split("=")
    feat = feat.strip().lower()
    if feat not in ctx.dims:
        raise SystemExit(f"cannot use --known-mm {known[0]}: '{feat}' was not measured on this part. "
                         f"Available: {', '.join(sorted(ctx.dims)) or 'none'}")
    ctx.mm = float(val) / ctx.dims[feat]
    ctx.data["mm_per_px"], ctx.data["scale_from"] = ctx.mm, feat
    ctx.warn(f"Scale {ctx.mm:.5f} mm/px comes from the {feat}; features at other heights of the part are scaled "
             "less accurately because of perspective.")
    ctx.add_legend("info", f"Scale {ctx.mm:.4f} mm/px from {feat}.", (255, 0, 230), order=99)
    return dict(mm=ctx.mm)


@step("ratios")
def ratios(ctx):
    """Dimensionless ratios between the registered lengths (valid without a scale)."""
    d = ctx.dims
    r = {}
    if "outer" in d and "bolt" in d:
        r["outer_over_bolt"] = d["outer"] / d["bolt"]
    if "bore" in d:
        for k in sorted(d):
            if k not in ("bore", "hole"):
                r[f"{k}_over_bore"] = d[k] / d["bore"]
    if "length" in d and "width" in d:
        r["length_over_width"] = d["length"] / d["width"]
    ctx.data["ratios"] = r
    ctx.data["dims_px"] = d
    if r:
        ctx.say("ratios " + str({k: round(v, 4) for k, v in r.items()}))
    return dict(ratios=r)
