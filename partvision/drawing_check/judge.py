"""Verdicts of the drawing check: dimensional tolerance (with a guard band for the photo's uncertainty), detection
tolerance (present near the drawing position at roughly the right size), and how the two combine."""
from .common import _k


class Judge:
    """Holds the uncertainties and detection tolerances of one photo; judges each measured feature."""

    def __init__(self, unc_edge, unc_pos, unc_size, det_pos, det_size, on_recess):
        self.unc_edge, self.unc_pos, self.unc_size = unc_edge, unc_pos, unc_size
        self.det_pos, self.det_size, self.on_recess = det_pos, det_size, on_recess
        self.kd = _k("detection")
        self.dsf = self.kd["size_frac"]

    def verdict(self, dev_pos, dev_size, f, extra_bad=False, u_pos=None, u_size=None):
        """Guard band: OK only if clearly inside tolerance, DEVIATES if clearly outside,
        CAN'T VERIFY when the deviation is within +-uncertainty of the limit."""
        u_pos = self.unc_edge if u_pos is None else u_pos
        u_size = self.unc_edge if u_size is None else u_size
        worst, close, near = False, False, []
        for dev, lim, u in ((dev_pos, f["pos_tol"], u_pos),
                            (None if dev_size is None else abs(dev_size), f["tol"], u_size)):
            if dev is None:
                continue
            if dev > lim + u:
                worst = True
            elif dev > lim - u:
                close = True
                near.append(u)
        if worst or extra_bad:
            return "DEVIATES", ""
        if close:
            return "CAN'T VERIFY", f"within +-{max(near):.2f} mm (photo uncertainty) of the tolerance limit"
        return "OK", ""

    def detected(self, dpos_v, dsize_v, size, fid_=None):
        """Present near where the drawing puts it, at roughly the right size (detection tolerance)."""
        lim = max(self.det_pos, self.dsf * size) if fid_ not in self.on_recess else \
            max(self.kd["recess_pos_factor"] * self.det_pos, self.kd["recess_size_frac"] * size)
        return (dpos_v is None or dpos_v <= lim) and \
               (dsize_v is None or abs(dsize_v) <= max(self.det_size, self.dsf * size))

    @staticmethod
    def combine(st_dim, det):
        """Main status: OK = detected and within the dimensional tolerance; DETECTED = present but outside
        (or not verifiable against) the dimensional tolerance; DEVIATES = outside the detection tolerance."""
        if st_dim == "OK":
            return "OK"
        if det:
            return "DETECTED"
        if st_dim in ("MISSING", "NOT MEASURED") or (st_dim == "UNCERTAIN" and det is None):
            return st_dim
        return "DEVIATES"
