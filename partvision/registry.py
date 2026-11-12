"""Step registry and profiles.

Add a detector by creating a module in partvision/steps/ with a function decorated by @step(...).
It is imported automatically. Use it in a profile below, or run any list with --steps.
"""
STEPS = {}

# Ordered step lists. 'auto' runs part + outline, then picks round or generic from the outline shape.
PROFILES = {
    "round": ["part", "outline", "bore", "round_rim", "bolt_circle", "rim_notches", "rings", "pockets", "scale",
              "ratios"],
    "generic": ["part", "outline", "holes", "patterns", "pockets", "scale", "ratios"],
    "drawing": ["part", "drawing_match"],
}


def step(name, needs=()):
    """Register fn(ctx) -> dict as a pipeline step. `needs` lists steps that must have run first."""
    def deco(fn):
        STEPS[name] = dict(fn=fn, needs=tuple(needs), doc=(fn.__doc__ or "").strip().split("\n")[0])
        return fn
    return deco


def run(ctx, names):
    for name in names:
        if name in ctx.results:
            continue
        if name not in STEPS:
            raise SystemExit(f"unknown step '{name}'. Available: {', '.join(sorted(STEPS))}")
        missing = [n for n in STEPS[name]["needs"] if n not in ctx.results]
        if missing:
            raise SystemExit(f"step '{name}' needs {', '.join(missing)} to run before it")
        out = STEPS[name]["fn"](ctx)
        ctx.results[name] = out if out is not None else {}
