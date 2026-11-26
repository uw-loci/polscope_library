"""Comparison rules for the golden replay in tests/test_goldens.py.

Vendored on purpose and kept small: CI runs without the verify-by-oracle skill whose
oracle.py recorded the goldens, so the test cannot import it. This file is identical in
ppm_library, polscope_library and microscope_imageprocessing (the skill's
analysis-libraries profile lists the three copies); only cases.py is repo-specific.

Every case names one rule in cases.RULES:

  {"kind": "exact"}
      Same shape, same dtype, every element equal; NaN equals NaN. For integer, bool and
      label outputs and for I/O round trips.
  {"kind": "tolerance", "atol": a, "rtol": r, "reason": "..."}
      |new - golden| <= atol + rtol * |new| at every finite element (numpy's rule, as
      oracle.py applies it). NaN and infinite positions must match the golden exactly.
  {"kind": "angle", "period": p, "atol": a, "reason": "..."}
      Compared on the circle: the difference is wrapped into [-p/2, p/2) first, so a
      value just below the period and one just above zero are the same angle. NaN
      positions must match. For orientations (period pi or 180) and hue (period 1).

A rule other than exact needs a reason naming the cause of the spread it allows.
"""

import numpy as np

KINDS = ("exact", "tolerance", "angle")


def _index(flat, shape, labels=None):
    idx = tuple(int(i) for i in np.unravel_index(int(flat), shape))
    text = str(idx[0]) if len(idx) == 1 else str(idx)
    if labels is not None and len(idx) == 1 and idx[0] < len(labels):
        text += " (%s)" % labels[idx[0]]
    return text


def validate_rule(rule):
    kind = rule.get("kind", "exact")
    if kind not in KINDS:
        raise ValueError("unknown comparison kind %r (one of %s)" % (kind, ", ".join(KINDS)))
    if kind != "exact" and not rule.get("reason"):
        raise ValueError("comparison kind %r needs a 'reason'" % kind)
    if kind == "angle" and not rule.get("period"):
        raise ValueError("an angle comparison needs a 'period'")
    return rule


def compare(golden, new, rule, labels=None):
    """Compare ``new`` with ``golden`` under ``rule``.

    Returns ``(ok, message, measured)``: ``message`` locates the first differing element
    when ``ok`` is False; ``measured`` is a one-line summary of the largest difference
    for tolerance and angle rules (None for exact), so a run can log the spread it saw.
    """
    validate_rule(rule)
    golden = np.asarray(golden)
    new = np.asarray(new)
    if golden.shape != new.shape:
        return False, "shape differs: golden %s, new %s" % (golden.shape, new.shape), None
    if golden.dtype != new.dtype and not rule.get("ignore_dtype"):
        return False, "dtype differs: golden %s, new %s" % (golden.dtype, new.dtype), None
    kind = rule.get("kind", "exact")
    if kind == "exact":
        return _compare_exact(golden, new, labels)
    g = golden.astype(np.float64)
    n = new.astype(np.float64)
    nan_g, nan_n = np.isnan(g), np.isnan(n)
    bad = nan_g != nan_n
    if bad.any():
        flat = int(np.flatnonzero(bad)[0])
        return (
            False,
            "NaN mask differs at %s: golden %r, new %r (%d positions)"
            % (_index(flat, g.shape, labels), g.flat[flat], n.flat[flat], int(bad.sum())),
            None,
        )
    finite = np.isfinite(g) & np.isfinite(n)
    special = ~finite & ~nan_g
    if special.any() and not np.array_equal(g[special], n[special]):
        bad = special & (g != n)
        flat = int(np.flatnonzero(bad)[0])
        return (
            False,
            "infinite value differs at %s: golden %r, new %r"
            % (_index(flat, g.shape, labels), g.flat[flat], n.flat[flat]),
            None,
        )
    diff = np.zeros_like(g)
    diff[finite] = n[finite] - g[finite]
    if kind == "angle":
        period = float(rule["period"])
        diff = (diff + period / 2.0) % period - period / 2.0
        limit = np.full_like(g, float(rule.get("atol", 0.0)))
        label = "wrapped angle diff"
    else:
        limit = float(rule.get("atol", 0.0)) + float(rule.get("rtol", 0.0)) * np.abs(n)
        label = "abs diff"
    absdiff = np.abs(diff)
    over = finite & (absdiff > limit)
    if finite.any():
        where = int(np.flatnonzero(finite)[np.argmax(absdiff[finite])])
        measured = "max %s %.3g at %s (golden %r, new %r)" % (
            label,
            float(absdiff.flat[where]),
            _index(where, g.shape, labels),
            golden.flat[where],
            new.flat[where],
        )
        if kind == "tolerance":
            denom = np.abs(n[finite])
            rel = np.where(denom > 0, absdiff[finite] / np.where(denom > 0, denom, 1.0), 0.0)
            measured += ", max rel diff %.3g" % float(rel.max())
    else:
        measured = "no finite elements"
    if over.any():
        flat = int(np.flatnonzero(over)[0])
        return (
            False,
            "%s at %s: golden %r, new %r (%s > allowed %.3g; %d of %d elements over; %s)"
            % (
                label,
                _index(flat, g.shape, labels),
                golden.flat[flat],
                new.flat[flat],
                "%.3g" % float(absdiff.flat[flat]),
                float(limit.flat[flat]),
                int(over.sum()),
                int(g.size),
                measured,
            ),
            measured,
        )
    return True, "", measured


def _compare_exact(golden, new, labels):
    if golden.dtype.kind in "fc":
        equal = (golden == new) | (np.isnan(golden) & np.isnan(new))
    else:
        equal = golden == new
    bad = np.asarray(~equal)
    if not bad.any():
        return True, "", None
    flat = int(np.flatnonzero(bad)[0])
    message = "differs at %s: golden %r, new %r (%d of %d elements differ" % (
        _index(flat, golden.shape, labels),
        golden.flat[flat],
        new.flat[flat],
        int(bad.sum()),
        int(golden.size),
    )
    if golden.dtype.kind in "biuf":
        g = golden.astype(np.float64)
        n = new.astype(np.float64)
        ok = np.isfinite(g) & np.isfinite(n)
        if ok.any():
            message += ", max abs diff %.3g" % float(np.abs(g[ok] - n[ok]).max())
    return False, message + ")", None
