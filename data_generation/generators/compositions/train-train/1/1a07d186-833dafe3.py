"""Colored mirror docks.

Every full-span, one-cell-wide bar is a colored mirror.  A floating shape (an
8-connected group of non-background cells off the bars, possibly of several
colors) attaches to every bar that shares a color with it -- at most one bar per
orientation:

* no shared color: the shape is erased;
* one bar: the shape slides perpendicular to that bar as a rigid unit (keeping
  its parallel coordinates and its side, passing through any other bars) until
  its facing edge touches the bar, and the bar then reflects it, so the bar
  becomes the axis of a bilateral copy;
* one horizontal and one vertical bar (a single-colour cross, or a shape that
  carries the colours of two differently coloured crossing bars): the shape
  slides into the facing corner of their crossing and is reflected across both
  bars, giving original / left-right mirror / top-bottom mirror / 180-degree
  rotation around the crossing.

Bars, crossings and background are copied unchanged; every cell keeps its colour.
"""
from __future__ import annotations

import numpy as np

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_N4 = ((0, 1), (1, 0), (0, -1), (-1, 0))


class _Retry(Exception):
    """Internal rejection signal for constructive sampling."""


# --------------------------------------------------------------------------
# Input-only parsing and the reference transformation
# --------------------------------------------------------------------------

def _mode(values):
    vals, counts = np.unique(np.asarray(values).ravel(), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask, conn=8):
    steps = _N8 if conn == 8 else _N4
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for r, c in zip(*np.nonzero(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(sorted(cells))
    return result


def _bars(g):
    height, width = g.shape
    bg = _mode(g)
    hrows = [r for r in range(height) if not np.any(g[r, :] == bg)]
    vcols = [c for c in range(width) if not np.any(g[:, c] == bg)]
    hset, vset = set(hrows), set(vcols)
    open_cols = [c for c in range(width) if c not in vset] or list(range(width))
    open_rows = [r for r in range(height) if r not in hset] or list(range(height))
    hbars = {r: _mode(g[r, open_cols]) for r in hrows}
    vbars = {c: _mode(g[open_rows, c]) for c in vcols}
    return bg, hbars, vbars


def _parse(grid, conn=8, by_color=False):
    """Background, bars (row->color, col->color) and floating shapes [(r, c, color)]."""
    g = np.asarray(grid).astype(int)
    bg, hbars, vbars = _bars(g)
    mask = g != bg
    if hbars:
        mask[sorted(hbars), :] = False
    if vbars:
        mask[:, sorted(vbars)] = False
    shapes = []
    if by_color:
        for color in sorted({int(v) for v in g[mask]}):
            for cells in _components(mask & (g == color), conn):
                shapes.append([(r, c, color) for r, c in cells])
    else:
        for cells in _components(mask, conn):
            shapes.append([(r, c, int(g[r, c])) for r, c in cells])
    return g, bg, hbars, vbars, shapes


def _pick(options, lo, hi):
    # Generated inputs never offer two parallel options; nearest is for robustness only.
    if not options:
        return None
    return min(options, key=lambda b: (min(abs(b - lo), abs(b - hi)), b))


def _dock_at(shape, hr, vc, move=True):
    rows = [r for r, _, _ in shape]
    cols = [c for _, c, _ in shape]
    dr = dc = sr = sc = 0
    if hr is not None:
        if max(rows) < hr:
            dr, sr = hr - 1 - max(rows), -1
        else:
            dr, sr = hr + 1 - min(rows), 1
    if vc is not None:
        if max(cols) < vc:
            dc, sc = vc - 1 - max(cols), -1
        else:
            dc, sc = vc + 1 - min(cols), 1
    if not move:
        dr = dc = 0
    base = [(r + dr, c + dc, k) for r, c, k in shape]
    return {"hr": hr, "vc": vc, "shift": (dr, dc), "side": (sr, sc), "base": base}


def _dock(shape, hbars, vbars, move=True):
    """Docked position of one shape, or None when no bar shares a colour with it."""
    colors = {k for _, _, k in shape}
    hs = sorted(r for r, k in hbars.items() if k in colors)
    vs = sorted(c for c, k in vbars.items() if k in colors)
    if not hs and not vs:
        return None
    rows = [r for r, _, _ in shape]
    cols = [c for _, c, _ in shape]
    dock = _dock_at(shape, _pick(hs, min(rows), max(rows)), _pick(vs, min(cols), max(cols)), move)
    dock["ambiguous"] = len(hs) > 1 or len(vs) > 1
    return dock


def _foot(dock, mode="rule"):
    """Coloured footprint {(r, c): color} of a docked shape under the rule or a variant."""
    hr, vc, base = dock["hr"], dock["vc"], dock["base"]
    sr, sc = dock["side"]
    out = {}

    def put(cells):
        for r, c, k in cells:
            out[(r, c)] = k

    def mir(cells, across_h, across_v):
        return [((2 * hr - r) if across_h else r, (2 * vc - c) if across_v else c, k) for r, c, k in cells]

    two = hr is not None and vc is not None
    if mode == "no_mirror":
        put(base)
        return out
    rows = [r for r, _, _ in base]
    cols = [c for _, c, _ in base]
    if mode in ("translate", "wrong_side"):
        jr = -sr * (max(rows) - min(rows) + 2) if hr is not None else 0
        jc = -sc * (max(cols) - min(cols) + 2) if vc is not None else 0
        moved = [(r + jr, c + jc, k) for r, c, k in base]
        if mode == "translate":
            put(base)
            put(moved)
            if two:
                put([(r + jr, c, k) for r, c, k in base])
                put([(r, c + jc, k) for r, c, k in base])
        else:
            put(moved)
            if hr is not None:
                put(mir(moved, True, False))
            if vc is not None:
                put(mir(moved, False, True))
            if two:
                put(mir(moved, True, True))
        return out
    put(base)
    if two:
        if mode in ("pinwheel", "rotate"):
            for r, c, k in base:
                y, x = r - hr, c - vc
                for _ in range(3):
                    y, x = x, -y
                    out[(hr + y, vc + x)] = k
        else:
            put(mir(base, True, False))
            put(mir(base, False, True))
            put(mir(base, True, True))
        return out
    if mode in ("point", "rotate"):
        if hr is not None:
            s = min(cols) + max(cols)
            put([(2 * hr - r, s - c, k) for r, c, k in base])
        else:
            s = min(rows) + max(rows)
            put([(s - r, 2 * vc - c, k) for r, c, k in base])
        return out
    put(mir(base, hr is not None, vc is not None))
    return out


def _frame(g, bg, hbars, vbars):
    out = np.full(g.shape, bg, dtype=int)
    for r in hbars:
        out[r, :] = g[r, :]
    for c in vbars:
        out[:, c] = g[:, c]
    return out


def _paint(out, foot, hbars, vbars):
    height, width = out.shape
    for (r, c), k in foot.items():
        if 0 <= r < height and 0 <= c < width and r not in hbars and c not in vbars:
            out[r, c] = k


def solve(input_grid: np.ndarray) -> np.ndarray:
    g, bg, hbars, vbars, shapes = _parse(input_grid)
    out = _frame(g, bg, hbars, vbars)
    for shape in shapes:
        dock = _dock(shape, hbars, vbars)
        if dock is not None:
            _paint(out, _foot(dock), hbars, vbars)
    return out


# --------------------------------------------------------------------------
# Shortcut rules used to prove that the demonstrations are discriminating
# --------------------------------------------------------------------------

_FOOT_MODE = {"no_mirror": "no_mirror", "translate": "translate", "wrong_side": "wrong_side",
              "point": "point", "pinwheel": "pinwheel", "rotate": "rotate"}


def _shortcut(grid, mode):
    """Outputs of plausible simpler or different readings (each changes one ingredient)."""
    if mode == "split":          # every colour part is its own object
        g, bg, hbars, vbars, shapes = _parse(grid, by_color=True)
    elif mode == "four_conn":    # diagonal links do not hold an object together
        g, bg, hbars, vbars, shapes = _parse(grid, conn=4)
    else:
        g, bg, hbars, vbars, shapes = _parse(grid)
    out = _frame(g, bg, hbars, vbars)
    for shape in shapes:
        if mode == "nearest":
            # Colour-blind: dock to the closest bar of either orientation, keep everything.
            best = None
            for r in hbars:
                cand = (min(abs(r - y) for y, _, _ in shape), 0, r)
                best = cand if best is None or cand < best else best
            for c in vbars:
                cand = (min(abs(c - x) for _, x, _ in shape), 1, c)
                best = cand if best is None or cand < best else best
            if best is not None:
                dock = _dock_at(shape, best[2] if best[1] == 0 else None, best[2] if best[1] == 1 else None)
                _paint(out, _foot(dock), hbars, vbars)
            continue
        dock = _dock(shape, hbars, vbars, move=(mode != "no_dock"))
        if dock is None:
            if mode == "keep_distractors":
                _paint(out, {(r, c): k for r, c, k in shape}, hbars, vbars)
            continue
        _paint(out, _foot(dock, _FOOT_MODE.get(mode, "rule")), hbars, vbars)
    return out


_SHORTCUTS = ("no_mirror", "no_dock", "translate", "wrong_side", "nearest", "keep_distractors",
              "point", "pinwheel", "split", "four_conn")


# --------------------------------------------------------------------------
# Structural analysis used by validation (asserts the family invariants)
# --------------------------------------------------------------------------

def _dilate(block, cells):
    height, width = block.shape
    for r, c in cells:
        block[max(0, r - 1):min(height, r + 2), max(0, c - 1):min(width, c + 2)] = True


def _mask_of(cells):
    rows = [p[0] for p in cells]
    cols = [p[1] for p in cells]
    M = np.zeros((max(rows) - min(rows) + 1, max(cols) - min(cols) + 1), dtype=bool)
    for p in cells:
        M[p[0] - min(rows), p[1] - min(cols)] = True
    return M


def _has_hole(M):
    padded = np.pad(~M, 1, constant_values=True)
    outside = _components(padded, 4)
    return len(outside) > 1


def _analyze(inp):
    g, bg, hbars, vbars, shapes = _parse(inp)
    height, width = g.shape
    vals, counts = np.unique(g, return_counts=True)
    assert int(np.sum(counts == counts.max())) == 1, "background must be a strict plurality"
    assert hbars or vbars, "no bar"
    assert len(set(hbars.values())) == len(hbars), "repeated horizontal bar color"
    assert len(set(vbars.values())) == len(vbars), "repeated vertical bar color"
    assert bg not in hbars.values() and bg not in vbars.values(), "bar has the background color"
    hrs, vcs = sorted(hbars), sorted(vbars)
    assert all(b - a >= 2 for a, b in zip(hrs, hrs[1:])), "touching horizontal bars"
    assert all(b - a >= 2 for a, b in zip(vcs, vcs[1:])), "touching vertical bars"
    layers = set()
    for r in hrs:
        for c in range(width):
            v = int(g[r, c])
            if c in vbars and vbars[c] != hbars[r]:
                assert v in (hbars[r], vbars[c]), "bad crossing color"
                layers.add("h" if v == hbars[r] else "v")
            else:
                assert v == hbars[r], "broken horizontal bar"
    for c in vcs:
        for r in range(height):
            if r not in hbars:
                assert int(g[r, c]) == vbars[c], "broken vertical bar"
    assert len(layers) <= 1, "inconsistent crossing layering"
    hset, vset = set(hrs), set(vcs)
    claimed = np.zeros(g.shape, dtype=bool)
    per_crossing = {}
    feats = []
    for shape in shapes:
        colors = sorted({k for _, _, k in shape})
        M = _mask_of(shape)
        touched = set()
        for r, c, _ in shape:
            for dy, dx in _N4:
                y, x = r + dy, c + dx
                if y in hset and 0 <= x < width:
                    touched.add(("h", y))
                if x in vset and 0 <= y < height:
                    touched.add(("v", x))
        f = {"colors": colors, "multicolor": len(colors) > 1, "n": len(shape), "size": M.shape,
             "diag": len(_components(M, 4)) > 1, "hole": _has_hole(M),
             "solid": bool(M.all()), "touched": touched,
             "canon": min((Q.shape, Q.tobytes()) for k in range(4)
                          for Q in (np.rot90(M, k), np.rot90(M, k)[:, ::-1]))}
        dock = _dock(shape, hbars, vbars)
        if dock is None:
            f.update({"kind": "distractor", "move": 0, "zero": False, "crossed": 0, "keys": set(),
                      "informative": False, "rot_inf": False, "key_off_edge": False,
                      "touch_own": False, "touch_other": bool(touched)})
            feats.append(f)
            continue
        assert not dock["ambiguous"], "shape matches two parallel bars"
        foot = _foot(dock)
        assert all(0 <= r < height and 0 <= c < width for r, c in foot), "footprint leaves grid"
        assert all(r not in hbars and c not in vbars for r, c in foot), "footprint hits a bar"
        assert not any(claimed[r, c] for r, c in foot), "footprints collide"
        _dilate(claimed, list(foot))
        hr, vc = dock["hr"], dock["vc"]
        dr, dc = dock["shift"]
        sr, sc = dock["side"]
        rows = [r for r, _, _ in shape]
        cols = [c for _, c, _ in shape]
        own, keys, crossed = set(), set(), 0
        if hr is not None:
            own.add(("h", hr))
            keys.add(hbars[hr])
            lo, hi = (max(rows), hr) if sr < 0 else (hr, min(rows))
            crossed += sum(lo < b < hi for b in hrs)
        if vc is not None:
            own.add(("v", vc))
            keys.add(vbars[vc])
            lo, hi = (max(cols), vc) if sc < 0 else (vc, min(cols))
            crossed += sum(lo < b < hi for b in vcs)

        def on_edge(r, c):
            # Facing edge of the docked shape: the row/column directly beside its bar(s).
            return ((hr is not None and r == (hr - 1 if sr < 0 else hr + 1))
                    or (vc is not None and c == (vc - 1 if sc < 0 else vc + 1)))

        if hr is not None and vc is not None:
            key = (hr, vc)
            per_crossing[key] = per_crossing.get(key, 0) + 1
            kind = "2d" if hbars[hr] == vbars[vc] else "dual"
            move, zero = abs(dr) + abs(dc), dr == 0 or dc == 0
            axis, side = "rc", (sr, sc)
        else:
            kind = "1d"
            axis = "r" if hr is not None else "c"
            move = abs(dr) if axis == "r" else abs(dc)
            zero = move == 0
            side = sr if axis == "r" else sc
        f.update({"kind": kind, "axis": axis, "side": side, "crossed": crossed, "move": move,
                  "zero": zero, "keys": keys, "crossing": (hr, vc) if axis == "rc" else None,
                  "informative": foot != _foot(dock, "translate") and foot != _foot(dock, "wrong_side"),
                  "rot_inf": foot != _foot(dock, "rotate"),
                  "key_off_edge": any(k in keys and not on_edge(r, c) for r, c, k in dock["base"]),
                  "touch_own": bool(touched & own), "touch_other": bool(touched - own)})
        feats.append(f)
    assert all(v == 1 for v in per_crossing.values()), "two unfoldings share a crossing"
    cross_colors = set(hbars.values()) & set(vbars.values())
    return {"bg": bg, "hbars": hbars, "vbars": vbars, "shapes": feats, "layers": layers,
            "diff_cross": any(hbars[r] != vbars[c] for r in hrs for c in vcs), "shape": g.shape,
            "cross_colors": cross_colors,
            "edge_bar": any(b in (0, height - 1) for b in hrs) or any(b in (0, width - 1) for b in vcs),
            "untouched_bar": any(col not in {k for f in feats if f["kind"] != "distractor" for k in f["keys"]}
                                 for col in list(hbars.values()) + list(vbars.values())),
            "untouched_cross": sum(1 for col in cross_colors
                                   if not any(f["kind"] == "2d" and col in f["keys"] for f in feats))}


def _axes(analysis):
    axes = set()
    for f in analysis["shapes"]:
        if f["kind"] == "1d":
            axes.add(f["axis"])
        elif f["kind"] in ("2d", "dual"):
            axes.update("rc")
    return axes


# --------------------------------------------------------------------------
# Shape vocabulary
# --------------------------------------------------------------------------

def _glyph(text):
    rows = text.split("/")
    assert len({len(row) for row in rows}) == 1, text
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _forms(M):
    seen, forms = set(), []
    for k in range(4):
        for Q in (np.rot90(M, k), np.rot90(M, k)[:, ::-1]):
            key = (Q.shape, Q.tobytes())
            if key not in seen:
                seen.add(key)
                forms.append(np.ascontiguousarray(Q))
    return forms


_GLYPH_TEXT = (
    "#./##", "#./#./##", "#../#../###", "###/.#.", "###/.#./.#.", ".##/##./.#.", "##/##/#.",
    ".##/##.", "##./.#./.##", "#.#/###", "#.#/#.#/###", "###/#../###", "###/#../###/#../###",
    "#../##./.##", ".#./###/.#.", "#.#/###/#.#", ".#../####", ".#/.#/##/#.", "####/#...",
    "#.../##../.##./..##", "#.#.#/#####", "##../.##./..##", "###/#.#/#..", "#..#/####",
    "####/#..#/#...", "###/..#/.##", "#.##/####", ".#./.#./###", "#.../####/...#",
    "..#/..#/###/#..", "##/.#/.#/##", "###./..##", "#.#/###/.#.", "##.#/.###", "#..../#####",
    ".##./####", "##/#./#./##", "#.#/###/#..",
)
_DIAG_TEXT = (
    "#./.#", "#../.#./..#", "#.../.#../..#./...#", "#...#/.#.#./..#..", "#.#/.#./#.#", "#.#/.#.",
    "#../.#./..#/.#.", "##./..#/..#", "#../.##", ".#/#./.#", "#.#./.#.#", "##../..##",
    "#..../.#.../..###", "###./...#", ".#./#.#/.#.", "#.#/.#./.#.", "#.../.##./...#", "##./..#/.#.",
)
_GLYPHS = tuple(_forms(_glyph(t)) for t in _GLYPH_TEXT)
_DIAGS = tuple(_forms(_glyph(t)) for t in _DIAG_TEXT)


def _side(rng, cap, level):
    sizes = np.arange(1, cap + 1)
    base = np.array([0.10, 0.30, 0.28, 0.14 + 0.10 * level, 0.07 + 0.09 * level,
                     0.04 + 0.07 * level, 0.03 + 0.05 * level][:cap])
    return int(rng.choice(sizes, p=base / base.sum()))


def _blob(rng, h, w):
    top = h * w if min(h, w) == 1 or rng.random() < 0.25 else h * w - 1
    for _ in range(60):
        size = int(rng.integers(max(h, w), max(max(h, w), top) + 1))
        cells = [(int(rng.integers(h)), int(rng.integers(w)))]
        chosen = set(cells)
        for _ in range(300):
            if len(cells) >= size:
                break
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = _N4[int(rng.integers(4))]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                chosen.add((nr, nc))
                cells.append((nr, nc))
        P = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            P[r, c] = True
        if P.any(axis=1).all() and P.any(axis=0).all():
            return P
    return np.ones((h, w), dtype=bool)


def _sparse8(rng, h, w):
    for _ in range(40):
        target = int(rng.integers(max(h, w), max(h, w) + max(1, (h * w - max(h, w)) // 2) + 1))
        cells = [(int(rng.integers(h)), int(rng.integers(w)))]
        chosen = set(cells)
        for _ in range(300):
            if len(cells) >= target:
                break
            r, c = cells[int(rng.integers(len(cells)))]
            dr, dc = _N8[int(rng.integers(8))]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen:
                chosen.add((nr, nc))
                cells.append((nr, nc))
        P = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            P[r, c] = True
        if P.any(axis=1).all() and P.any(axis=0).all():
            return P
    return None


def _stretch(rng, M, hmax, wmax):
    for _ in range(2):
        if rng.random() < 0.5 and M.shape[0] < hmax:
            i = int(rng.integers(M.shape[0]))
            M = np.insert(M, i, M[i], axis=0)
        elif M.shape[1] < wmax:
            j = int(rng.integers(M.shape[1]))
            M = np.insert(M, j, M[:, j], axis=1)
    return M


def _from_library(rng, library, hmax, wmax, level):
    choices = [forms for forms in library if any(f.shape[0] <= hmax and f.shape[1] <= wmax for f in forms)]
    if not choices:
        return None
    forms = choices[int(rng.integers(len(choices)))]
    fits = [f for f in forms if f.shape[0] <= hmax and f.shape[1] <= wmax]
    M = fits[int(rng.integers(len(fits)))]
    if rng.random() < 0.35 * level:
        M = _stretch(rng, M, hmax, wmax)
    return M


def _mask_sample(rng, pz, hmax, wmax, family=None):
    """One shape mask within hmax x wmax (rows x cols of the frame used by the caller)."""
    level = pz["level"]
    if pz["regime"] == "minimal":
        hmax, wmax = min(hmax, 3), min(wmax, 3)
    if hmax < 1 or wmax < 1:
        return None
    if family is None:
        if pz["regime"] == "minimal":
            names, weights = ["blob", "glyph", "line", "dot"], [0.38, 0.42, 0.14, 0.06]
        else:
            names = ["blob", "glyph", "line", "frame", "dot"]
            weights = [0.26 - 0.10 * level, 0.42, 0.13, 0.06 + 0.07 * level, 0.04]
            if pz["diag"]:
                names += ["diagonal", "sparse"]
                weights += [0.12 + 0.08 * level, 0.04 + 0.04 * level]
        weights = np.array(weights) / np.sum(weights)
        family = names[int(rng.choice(len(names), p=weights))]
    if family == "dot":         # parent A's single-cell marker
        return np.ones((1, 1), dtype=bool)
    if family == "blob":
        return _blob(rng, _side(rng, hmax, level), _side(rng, wmax, level))
    if family == "glyph":
        return _from_library(rng, _GLYPHS, hmax, wmax, level)
    if family == "diagonal":
        return _from_library(rng, _DIAGS, hmax, wmax, level)
    if family == "line":
        n = int(rng.integers(2, 8))
        thick = 2 if rng.random() < 0.25 else 1
        opts = [(thick, n), (n, thick)]
        opts = [o for o in opts if o[0] <= hmax and o[1] <= wmax]
        if not opts:
            n = max(1, min(max(hmax, wmax), n))
            opts = [o for o in ((1, n), (n, 1)) if o[0] <= hmax and o[1] <= wmax]
            if not opts:
                return None
        h, w = opts[int(rng.integers(len(opts)))]
        return np.ones((h, w), dtype=bool)
    if family == "frame":
        if hmax < 3 or wmax < 3:
            return None
        h = int(rng.integers(3, min(hmax, 6) + 1))
        w = int(rng.integers(3, min(wmax, 6) + 1))
        M = np.zeros((h, w), dtype=bool)
        M[0, :] = M[-1, :] = M[:, 0] = M[:, -1] = True
        if rng.random() < 0.5:
            border = [(r, c) for r in range(h) for c in range(w) if M[r, c]
                      and (r in (0, h - 1)) != (c in (0, w - 1))]
            r, c = border[int(rng.integers(len(border)))]
            M[r, c] = False
            if not (M.any(axis=1).all() and M.any(axis=0).all()) or len(_components(M, 4)) != 1:
                M[r, c] = True
        return M
    if family == "sparse":
        if hmax < 2 or wmax < 2:
            return None
        return _sparse8(rng, int(rng.integers(2, min(hmax, 5) + 1)), int(rng.integers(2, min(wmax, 5) + 1)))
    return None


_STYLES = ("tip", "stripe", "rim", "checker", "half", "random")


def _colorize(rng, M, keys, body):
    """Fill mask M (-1 = empty).  Every key colour appears; body colours are extra."""
    P = np.full(M.shape, -1, dtype=int)
    cells = [(int(a), int(b)) for a, b in np.argwhere(M)]
    if len(cells) < len(keys):
        return None
    palette = list(keys) + [b for b in body if b not in keys]
    palette = palette[:len(cells)]
    if len(palette) == 1:
        P[M] = palette[0]
        return P
    h, w = M.shape
    interior = [(r, c) for r, c in cells
                if all(0 <= r + dy < h and 0 <= c + dx < w and M[r + dy, c + dx] for dy, dx in _N4)]
    for _ in range(16):
        style = _STYLES[int(rng.integers(len(_STYLES)))]
        order = [palette[int(i)] for i in rng.permutation(len(palette))]
        Q = np.full(M.shape, -1, dtype=int)
        if style == "tip":
            for r, c in cells:
                Q[r, c] = order[0]
            spots = [cells[int(i)] for i in rng.permutation(len(cells))]
            extremes = sorted(cells, key=lambda p: (p[0] + p[1], p[0]))
            if rng.random() < 0.5:
                spots = [extremes[0], extremes[-1]] + spots
            used = set()
            for color in order[1:]:
                for _ in range(int(rng.integers(1, 3))):
                    spot = next((s for s in spots if s not in used), None)
                    if spot is not None and len(used) < len(cells) - 1:
                        used.add(spot)
                        Q[spot] = color
        elif style == "stripe":
            axis = int(rng.integers(2))
            width = int(rng.integers(1, 3))
            for r, c in cells:
                Q[r, c] = order[((r, c)[axis] // width) % len(order)]
        elif style == "rim":
            if not interior:
                continue
            for r, c in cells:
                Q[r, c] = order[1 if (r, c) in interior else 0]
            if len(order) > 2:
                r, c = interior[int(rng.integers(len(interior)))]
                Q[r, c] = order[2]
        elif style == "checker":
            for r, c in cells:
                Q[r, c] = order[(r + c) % len(order)]
        elif style == "half":
            axis = int(rng.integers(2))
            coords = sorted({(r, c)[axis] for r, c in cells})
            if len(coords) < 2:
                continue
            cut = coords[int(rng.integers(1, len(coords)))]
            for r, c in cells:
                Q[r, c] = order[0] if (r, c)[axis] < cut else order[1 % len(order)]
            if len(order) > 2:
                r, c = cells[int(rng.integers(len(cells)))]
                Q[r, c] = order[2]
        else:
            for r, c in cells:
                Q[r, c] = order[int(rng.integers(len(order)))]
        present = {int(v) for v in Q[M]}
        if all(k in present for k in keys) and len(present) >= 2:
            return Q
    for i, (r, c) in enumerate(cells):
        P[r, c] = palette[i] if i < len(palette) else palette[0]
    return P


def _sym(P, axis):
    return np.array_equal(P, P[::-1, :]) if axis == 0 else np.array_equal(P, P[:, ::-1])


def _canon_feet(P, corner):
    """Rule and variant footprints of pattern P docked at a canonical crossing (0, 0)."""
    sr, sc = corner
    h, w = P.shape
    r0 = -h if sr < 0 else 1
    c0 = -w if sc < 0 else 1
    base = [(r0 + int(a), c0 + int(b), int(P[a, b])) for a, b in np.argwhere(P >= 0)]
    dock = {"hr": 0, "vc": 0, "side": (sr, sc), "base": base}
    return {m: _foot(dock, m) for m in ("rule", "translate", "wrong_side", "rotate")}


def _need_ok(P, need, side=None, corner=None):
    """need: None, 'perp', 'both' (1D, pattern in bar frame) or 'inf', 'pin' (2D)."""
    if need is None:
        return True
    if need == "perp":
        return not _sym(P, 0)
    if need == "both":
        return not _sym(P, 0) and not _sym(P, 1)
    feet = _canon_feet(P, corner)
    ok = feet["rule"] != feet["translate"] and feet["rule"] != feet["wrong_side"]
    if need == "pin":
        ok = ok and feet["rule"] != feet["rotate"]
    return ok


def _key_off(P, keys, side=None, corner=None):
    h, w = P.shape
    for a, b in np.argwhere(P >= 0):
        if int(P[a, b]) not in keys:
            continue
        if corner is None:
            if a != (h - 1 if side < 0 else 0):
                return True
        else:
            if a != (h - 1 if corner[0] < 0 else 0) and b != (w - 1 if corner[1] < 0 else 0):
                return True
    return False


# --------------------------------------------------------------------------
# Constructive placement
# --------------------------------------------------------------------------

class _Canvas:
    def __init__(self, height, width, bg, hbars, vbars, layer):
        self.H, self.W, self.bg = height, width, bg
        self.hbars, self.vbars, self.layer = dict(hbars), dict(vbars), layer
        hs, vs = set(hbars), set(vbars)
        self.row_free = np.array([r not in hs for r in range(height)], dtype=bool)
        self.col_free = np.array([c not in vs for c in range(width)], dtype=bool)
        self.row_far = np.array([all(abs(r - b) > 1 for b in hs) for r in range(height)], dtype=bool)
        self.col_far = np.array([all(abs(c - b) > 1 for b in vs) for c in range(width)], dtype=bool)
        self.in_block = np.zeros((height, width), dtype=bool)
        self.out_block = np.zeros((height, width), dtype=bool)
        self.items = []

    def _commit(self, cells, foot, info):
        _dilate(self.in_block, [(r, c) for r, c, _ in cells])
        if foot:
            _dilate(self.out_block, list(foot))
        self.items.append((cells, foot, info))

    @staticmethod
    def _accept(kind, crossed, move):
        if kind == "cross":
            return crossed > 0
        if kind == "plain":
            return crossed == 0
        if kind == "near":
            return crossed == 0 and move <= 3
        if kind == "far":
            return move >= 5
        return True

    def place_1d(self, rng, axis, bar, Q, side, kind="any", loose=False):
        """Q is given in the bar frame: its rows run perpendicular to the bar."""
        if axis == "r":
            L, M = self.H, self.W
            bars, other = sorted(self.hbars), set(self.vbars)
            perp_ok = self.row_free if loose else self.row_far
            par_ok = self.col_free if loose else self.col_far
            inb, outb = self.in_block, self.out_block
        else:
            L, M = self.W, self.H
            bars, other = sorted(self.vbars), set(self.hbars)
            perp_ok = self.col_free if loose else self.col_far
            par_ok = self.row_free if loose else self.row_far
            inb, outb = self.in_block.T, self.out_block.T
        mask = Q >= 0
        h, w = Q.shape
        d0 = bar - h if side < 0 else bar + 1
        m0 = bar + 1 if side < 0 else bar - h
        if min(d0, m0) < 0 or max(d0, m0) + h > L:
            return False
        if any(b != bar and min(d0, m0) <= b < max(d0, m0) + h for b in bars):
            return False
        Qm = Q[::-1, :]
        mm = Qm >= 0
        qs = [q0 for q0 in range(M - w + 1) if par_ok[q0:q0 + w].all()
              and not outb[d0:d0 + h, q0:q0 + w][mask].any()
              and not outb[m0:m0 + h, q0:q0 + w][mm].any()]
        bset = set(bars)
        prange = [d0] if kind == "zero" else (range(0, bar - h + 1) if side < 0 else range(bar + 1, L - h + 1))
        ps = []
        for p0 in prange:
            if not perp_ok[p0:p0 + h].all():
                continue
            move = abs(p0 - d0)
            if (kind == "zero") != (move == 0):
                continue
            lo, hi = (p0 + h - 1, bar) if side < 0 else (bar, p0)
            crossed = sum(lo < b < hi for b in bars)
            adj = (p0 - 1 in bset and p0 - 1 != bar) or (p0 + h in bset and p0 + h != bar)
            if self._accept(kind, crossed, move):
                ps.append((p0, crossed, move, adj))
        if not qs or not ps:
            return False
        for index in rng.permutation(len(qs) * len(ps))[:700]:
            p0, crossed, move, adj = ps[int(index) // len(qs)]
            q0 = qs[int(index) % len(qs)]
            if kind == "touch" and not (adj or q0 - 1 in other or q0 + w in other):
                continue
            if inb[p0:p0 + h, q0:q0 + w][mask].any():
                continue
            cells = [(p0 + int(a), q0 + int(b), int(Q[a, b])) for a, b in np.argwhere(mask)]
            foot = ([(d0 + int(a), q0 + int(b), int(Q[a, b])) for a, b in np.argwhere(mask)]
                    + [(m0 + int(a), q0 + int(b), int(Qm[a, b])) for a, b in np.argwhere(mm)])
            if axis == "c":
                cells = [(b, a, k) for a, b, k in cells]
                foot = [(b, a, k) for a, b, k in foot]
            self._commit(cells, {(r, c): k for r, c, k in foot}, {"kind": "1d", "axis": axis})
            return True
        return False

    def place_2d(self, rng, hr, vc, P, sr, sc, kind="any", loose=False):
        h, w = P.shape
        if hr - h < 0 or hr + h >= self.H or vc - w < 0 or vc + w >= self.W:
            return False
        if any(b != hr and hr - h <= b <= hr + h for b in self.hbars):
            return False
        if any(b != vc and vc - w <= b <= vc + w for b in self.vbars):
            return False
        r0d = hr - h if sr < 0 else hr + 1
        c0d = vc - w if sc < 0 else vc + 1
        mask = P >= 0
        foot = {}
        for a, b in np.argwhere(mask):
            r, c, k = r0d + int(a), c0d + int(b), int(P[a, b])
            for y, x in ((r, c), (2 * hr - r, c), (r, 2 * vc - c), (2 * hr - r, 2 * vc - c)):
                foot[(y, x)] = k
        if any(self.out_block[r, c] for r, c in foot):
            return False
        row_ok = self.row_free if loose else self.row_far
        col_ok = self.col_free if loose else self.col_far
        rows, cols = [], []
        for r0 in (range(0, hr - h + 1) if sr < 0 else range(hr + 1, self.H - h + 1)):
            if row_ok[r0:r0 + h].all():
                lo, hi = (r0 + h - 1, hr) if sr < 0 else (hr, r0)
                rows.append((r0, sum(lo < b < hi for b in self.hbars), abs(r0 - r0d)))
        for c0 in (range(0, vc - w + 1) if sc < 0 else range(vc + 1, self.W - w + 1)):
            if col_ok[c0:c0 + w].all():
                lo, hi = (c0 + w - 1, vc) if sc < 0 else (vc, c0)
                cols.append((c0, sum(lo < b < hi for b in self.vbars), abs(c0 - c0d)))
        if not rows or not cols:
            return False
        for index in rng.permutation(len(rows) * len(cols))[:700]:
            r0, cr, mr = rows[int(index) // len(cols)]
            c0, cc, mc = cols[int(index) % len(cols)]
            if mr + mc == 0 or (kind == "zero" and mr and mc) or (kind == "cross" and cr + cc == 0):
                continue
            if self.in_block[r0:r0 + h, c0:c0 + w][mask].any():
                continue
            cells = [(r0 + int(a), c0 + int(b), int(P[a, b])) for a, b in np.argwhere(mask)]
            self._commit(cells, foot, {"kind": "2d"})
            return True
        return False

    def place_free(self, rng, P, loose=False):
        h, w = P.shape
        mask = P >= 0
        row_ok = self.row_free if loose else self.row_far
        col_ok = self.col_free if loose else self.col_far
        rows = [r0 for r0 in range(self.H - h + 1) if row_ok[r0:r0 + h].all()]
        cols = [c0 for c0 in range(self.W - w + 1) if col_ok[c0:c0 + w].all()]
        if not rows or not cols:
            return False
        for index in rng.permutation(len(rows) * len(cols))[:500]:
            r0 = rows[int(index) // len(cols)]
            c0 = cols[int(index) % len(cols)]
            if self.in_block[r0:r0 + h, c0:c0 + w][mask].any():
                continue
            cells = [(r0 + int(a), c0 + int(b), int(P[a, b])) for a, b in np.argwhere(mask)]
            self._commit(cells, None, {"kind": "distractor"})
            return True
        return False

    def render(self):
        inp = np.full((self.H, self.W), self.bg, dtype=int)
        order = ("h", "v") if self.layer == "v" else ("v", "h")
        for which in order:
            if which == "h":
                for r, color in self.hbars.items():
                    inp[r, :] = color
            else:
                for c, color in self.vbars.items():
                    inp[:, c] = color
        out = inp.copy()
        for cells, foot, _ in self.items:
            for r, c, k in cells:
                inp[r, c] = k
            for (r, c), k in (foot or {}).items():
                out[r, c] = k
        return inp, out


def _bar_positions(rng, length, count, min_diff, margin):
    lo, hi = margin, length - 1 - margin
    span = hi - lo + 1 - (min_diff - 1) * (count - 1)
    if span < count:
        raise _Retry("bars")
    base = np.sort(rng.choice(span, size=count, replace=False))
    return [int(lo + v + (min_diff - 1) * i) for i, v in enumerate(base)]


_SIZES = {"small": (7, 13), "medium": (11, 20), "large": (17, 30), "full": (26, 30)}


def _grid_size(rng, spec):
    nh, nv = spec["nh"], spec["nv"]
    lo, hi = _SIZES[spec["size"]]
    need_h = max(7, 4 * nh + 3) if nh else 6
    need_w = max(7, 4 * nv + 3) if nv else 6
    if need_h > 30 or need_w > 30:
        raise _Retry("size")
    height = int(rng.integers(max(lo, need_h), max(hi, need_h) + 1))
    width = int(rng.integers(max(lo, need_w), max(hi, need_w) + 1))
    if nv == 0 and rng.random() < 0.3:
        width = int(rng.integers(6, 31))
    if nh == 0 and rng.random() < 0.3:
        height = int(rng.integers(6, 31))
    return height, width


_DENSITY = {"sparse": (0.42, 0.43, 0.15, 0.0), "normal": (0.22, 0.42, 0.25, 0.11),
            "dense": (0.06, 0.36, 0.36, 0.22)}


def _longest_run(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def _try_pair(rng, spec, pz):
    nh, nv, k = spec["nh"], spec["nv"], spec["k"]
    erase = spec["type"] == "E"
    musts = spec["musts"]
    level, minimal = pz["level"], pz["regime"] == "minimal"
    height, width = _grid_size(rng, spec)
    bg = spec["bg"] if spec["bg"] is not None else int(rng.integers(10))
    unfold = k > 0 or spec.get("dual", 0) > 0
    min_diff = int(rng.choice((2, 3, 4, 5, 6, 7), p=(0.08, 0.22, 0.25, 0.2, 0.15, 0.1)))
    if unfold:
        min_diff = max(min_diff, 4)
    margin = int(rng.choice((0, 1, 2), p=(0.12, 0.13, 0.75)))
    hpos = _bar_positions(rng, height, nh, min_diff, margin) if nh else []
    vpos = _bar_positions(rng, width, nv, min_diff, margin) if nv else []
    palette = [int(c) for c in rng.permutation(10) if int(c) != bg]
    hcol = palette[:nh]
    rest = palette[nh:]
    hi_idx = [int(i) for i in rng.permutation(nh)[:k]] if k else []
    vi_idx = [int(i) for i in rng.permutation(nv)[:k]] if k else []
    vcol = [None] * nv
    for a, b in zip(hi_idx, vi_idx):
        vcol[b] = hcol[a]
    for b in range(nv):
        if vcol[b] is None:
            vcol[b] = rest.pop(0)
    pool = list(rest)                      # colours owning no bar in this pair
    hbars = dict(zip(hpos, hcol))
    vbars = dict(zip(vpos, vcol))
    layer = "v" if rng.random() < 0.5 else "h"
    canvas = _Canvas(height, width, bg, hbars, vbars, layer)
    crosses = [(r, c, hbars[r]) for r in hpos for c in vpos if hbars[r] == vbars[c]]
    cross_colors = {col for _, _, col in crosses}
    duals = [(r, c) for r in hpos for c in vpos if hbars[r] != vbars[c]
             and hbars[r] not in cross_colors and vbars[c] not in cross_colors]
    one_d = ([("r", r, col) for r, col in hbars.items() if col not in cross_colors]
             + [("c", c, col) for c, col in vbars.items() if col not in cross_colors])
    multi = pz["multi"] and not erase
    contact = pz["contact"] and not erase
    free = [pool[int(i)] for i in rng.permutation(len(pool))[:int(rng.integers(1, 4))]] if pool else []
    multi_rate = float(rng.uniform(0.3, 0.7)) if multi else 0.0
    perp_cap = 3 if minimal else 2 + int(round(4 * level))
    par_cap = 3 if minimal else 3 + int(round(4 * level))
    cap2d = 2 + int(round(3 * level))

    def bodies():
        return [free[int(i)] for i in rng.permutation(len(free))[:int(rng.integers(1, 3))]] if free else []

    def make(hmax, wmax, keys, need=None, side=None, corner=None, force_multi=None, family=None, keyoff=False):
        for _ in range(30):
            M = _mask_sample(rng, pz, hmax, wmax, family)
            if M is None:
                continue
            if len(keys) == 2:
                use = bodies() if free and rng.random() < 0.5 else []
            else:
                want = force_multi if force_multi is not None else (multi and rng.random() < multi_rate)
                use = bodies() if want else []
                if force_multi and not use:
                    return None
            P = _colorize(rng, M, keys, use)
            if P is None or (force_multi and len({int(v) for v in P[P >= 0]}) < 2):
                continue
            if not _need_ok(P, need, side, corner):
                continue
            if keyoff and not _key_off(P, set(keys), side, corner):
                continue
            return P
        return None

    def caps_1d(axis, bar, side, loose):
        L = height if axis == "r" else width
        bars = sorted(hbars) if axis == "r" else sorted(vbars)
        prev = max([b for b in bars if b < bar], default=-1)
        nxt = min([b for b in bars if b > bar], default=L)
        h = min(bar - prev - 1, nxt - bar - 1, perp_cap)
        h = min(h, bar - (0 if loose else 1)) if side < 0 else min(h, L - bar - 1 - (0 if loose else 1))
        if axis == "r":
            par = canvas.col_free if loose else canvas.col_far
        else:
            par = canvas.row_free if loose else canvas.row_far
        return h, min(par_cap, _longest_run(par))

    def caps_2d(hr, vc, sr, sc, loose):
        hb, vb = sorted(hbars), sorted(vbars)
        ph = max([b for b in hb if b < hr], default=-1)
        nh_ = min([b for b in hb if b > hr], default=height)
        pv = max([b for b in vb if b < vc], default=-1)
        nv_ = min([b for b in vb if b > vc], default=width)
        slack = 0 if loose else 1
        h = min(hr - ph - 1, nh_ - hr - 1, cap2d, (hr - slack) if sr < 0 else (height - hr - 1 - slack))
        w = min(vc - pv - 1, nv_ - vc - 1, cap2d, (vc - slack) if sc < 0 else (width - vc - 1 - slack))
        return h, w

    def try_1d(targets, side=None, kind="any", need="rand", attempts=8, keyoff=False,
               force_multi=None, family=None, loose=None):
        targets = list(targets)
        for _ in range(attempts):
            if not targets:
                return False
            axis, bar, col = targets[int(rng.integers(len(targets)))]
            s = side if side is not None else (-1 if rng.random() < 0.5 else 1)
            lz = bool(loose) if loose is not None else (contact and rng.random() < 0.2)
            lz = lz or kind in ("zero", "touch")
            hmax, wmax = caps_1d(axis, bar, s, lz)
            if hmax < 1 or wmax < 1 or (keyoff and hmax < 2):
                continue
            want = need if need != "rand" else ("perp" if rng.random() < 0.72 else None)
            P = make(hmax, wmax, [col], want, s, None, force_multi, family, keyoff)
            if P is None:
                continue
            if canvas.place_1d(rng, axis, bar, P, s, kind, lz):
                return True
        return False

    def try_2d(hr, vc, keys, need, kind="any", attempts=10, keyoff=False, force_multi=None, loose=None):
        for _ in range(attempts):
            sr = -1 if rng.random() < 0.5 else 1
            sc = -1 if rng.random() < 0.5 else 1
            lz = bool(loose) if loose is not None else (contact and rng.random() < 0.2)
            lz = lz or kind == "zero"
            hmax, wmax = caps_2d(hr, vc, sr, sc, lz)
            if hmax < 1 or wmax < 1:
                continue
            P = make(hmax, wmax, keys, need, None, (sr, sc), force_multi, None, keyoff)
            if P is None:
                continue
            if canvas.place_2d(rng, hr, vc, P, sr, sc, kind, lz):
                return True
        return False

    def matched_count():
        return sum(info["kind"] != "distractor" for _, _, info in canvas.items)

    if not erase:
        test_unfold = "informative" in musts
        # Unfoldings first: they need the most free space.
        for index, (hr, vc, col) in enumerate(crosses):
            if index == 0 and ("2d" in musts or (test_unfold and not one_d)):
                if not (try_2d(hr, vc, [col], "pin", "cross" if rng.random() < 0.4 else "any")
                        or try_2d(hr, vc, [col], "pin")):
                    raise _Retry("2d must")
            elif test_unfold:
                # Test unfoldings always distinguish mirror quadrants from a pinwheel;
                # crosses may stay untouched when the test also docks on plain bars.
                if rng.random() < (0.85 if index == 0 else 0.7):
                    if not try_2d(hr, vc, [col], "pin") and index == 0:
                        raise _Retry("test unfolding")
            elif rng.random() < 0.7:
                roll = rng.random()
                need = "pin" if roll < 0.55 else ("inf" if roll < 0.8 else None)
                try_2d(hr, vc, [col], need, "cross" if rng.random() < 0.35 else "any")
        if spec.get("dual", 0):
            if not duals:
                raise _Retry("no dual crossing")
            order = [duals[int(i)] for i in rng.permutation(len(duals))]
            placed = 0
            for hr, vc in order:
                if placed >= spec["dual"]:
                    break
                keys = [hbars[hr], vbars[vc]]
                if try_2d(hr, vc, keys, "pin", "cross" if rng.random() < 0.3 else "any", attempts=12):
                    placed += 1
            if "dual" in musts and not placed:
                raise _Retry("dual must")
        if "multi" in musts:
            ok = try_1d(one_d, need="perp", keyoff=True, force_multi=True, attempts=14) if one_d else False
            if not ok and crosses and not one_d:
                hr, vc, col = crosses[int(rng.integers(len(crosses)))]
                ok = try_2d(hr, vc, [col], "pin", keyoff=True, force_multi=True)
            if not ok:
                raise _Retry("multi must")
        if "cross1d" in musts and not try_1d(one_d, kind="cross", attempts=14):
            raise _Retry("cross1d")
        if "sides" in musts:
            first = -1 if rng.random() < 0.5 else 1
            for s in (first, -first):
                if not try_1d(one_d, side=s, need="both" if s == first else "perp", attempts=14):
                    raise _Retry("sides")
        for axis_name in ("r", "c"):
            if "axis_" + axis_name in musts:
                if not try_1d([t for t in one_d if t[0] == axis_name], need="perp", attempts=12):
                    raise _Retry("axis")
        if "zero" in musts and not try_1d(one_d, kind="zero", attempts=14):
            raise _Retry("zero must")
        if "touch" in musts and not try_1d(one_d, kind="touch", attempts=14):
            raise _Retry("touch must")
        if "diag" in musts and not try_1d(one_d, need="perp", family="diagonal", attempts=14):
            raise _Retry("diag must")
        if test_unfold and one_d and not try_1d(one_d, need="both", attempts=14):
            raise _Retry("informative")
        counts = _DENSITY[spec["shapes"]]
        for target in one_d:
            amount = int(rng.choice(4, p=counts))
            for _ in range(amount):
                kind = str(rng.choice(("any", "cross", "near", "far", "plain"), p=(0.35, 0.25, 0.15, 0.15, 0.10)))
                if not try_1d([target], kind=kind, attempts=3):
                    try_1d([target], attempts=2)
        want = spec.get("min_matched", 1)
        for _ in range(40):
            if matched_count() >= want:
                break
            if one_d:
                try_1d(one_d, attempts=2)
            elif crosses:
                break
        if matched_count() < want:
            raise _Retry("too few matched shapes")
    lo, hi = spec["distract"]
    n_distract = int(rng.integers(lo, hi + 1))
    if musts & {"distractor", "nearfail", "erase"}:
        n_distract = max(1 if not erase else int(rng.integers(1, 4)), n_distract)
    if n_distract and not pool:
        if musts & {"distractor", "erase"}:
            raise _Retry("no distractor color")
        n_distract = 0
    placed = 0
    templates = [cells for cells, _, info in canvas.items if info["kind"] != "distractor"]
    others = [c for c in pool if c not in free]
    dpool = free if (free and (not others or rng.random() < 0.35)) else (others or pool)
    for _ in range(n_distract):
        if templates and rng.random() < 0.2:
            M = _mask_of(templates[int(rng.integers(len(templates)))])
        else:
            M = _mask_sample(rng, pz, min(5, height - 2), min(6, width - 2))
        if M is None:
            continue
        colors = [dpool[int(i)] for i in rng.permutation(len(dpool))]
        if pz["multi"] and len(colors) >= 2 and rng.random() < 0.45:
            P = _colorize(rng, M, [colors[0]], [colors[1]])
        else:
            P = _colorize(rng, M, [colors[0]], [])
        if P is None:
            continue
        if canvas.place_free(rng, P, loose=pz["contact"] and rng.random() < 0.3):
            placed += 1
    if musts & {"distractor", "nearfail", "erase"} and not placed:
        raise _Retry("distractor")
    inp, out = canvas.render()
    try:
        analysis = _analyze(inp)
    except AssertionError as error:
        raise _Retry(str(error))
    if analysis["hbars"] != hbars or analysis["vbars"] != vbars:
        raise _Retry("bar detection mismatch")
    if len(analysis["shapes"]) != len(canvas.items):
        raise _Retry("shape segmentation mismatch")
    if not np.array_equal(solve(inp), out) or np.array_equal(inp, out):
        raise _Retry("solver mismatch")
    _pair_checks(inp, out, analysis, musts)
    return inp, out, analysis


def _differs(inp, out, mode):
    return not np.array_equal(_shortcut(inp, mode), out)


def _pair_checks(inp, out, analysis, musts):
    feats = analysis["shapes"]
    matched = [f for f in feats if f["kind"] != "distractor"]
    if "erase" in musts:
        if matched or not feats:
            raise _Retry("erase pair")
        return
    if not any(f["move"] > 0 for f in matched):
        raise _Retry("nothing moves")
    checks = {
        "cross1d": lambda: any(f["crossed"] > 0 and f["move"] > 0 for f in matched),
        "sides": lambda: {-1, 1} <= {f["side"] for f in matched if f["kind"] == "1d" and f["informative"]},
        "distractor": lambda: any(f["kind"] == "distractor" for f in feats),
        "2d": lambda: any(f["kind"] == "2d" and f["informative"] and f["rot_inf"] for f in matched),
        "dual": lambda: any(f["kind"] == "dual" and f["informative"] and f["rot_inf"] for f in matched),
        "multi": lambda: any(f["multicolor"] and f["key_off_edge"] and f["move"] > 0 for f in matched),
        "zero": lambda: any(f["zero"] for f in matched),
        "touch": lambda: any(f["touch_other"] and f["move"] > 0 for f in matched),
        "diag": lambda: any(f["diag"] and f["move"] > 0 for f in matched) and _differs(inp, out, "four_conn"),
        "nearfail": lambda: _differs(inp, out, "nearest"),
        "axis_r": lambda: "r" in _axes(analysis),
        "axis_c": lambda: "c" in _axes(analysis),
        "informative": lambda: _test_strength(inp, out, analysis) is None,
    }
    for name in musts:
        if name in checks and not checks[name]():
            raise _Retry(name)


def _test_strength(inp, out, analysis):
    """None when a test pair needs both parents and every mirror detail; else a reason."""
    feats = analysis["shapes"]
    matched = [f for f in feats if f["kind"] != "distractor"]
    if not any(f["move"] > 0 for f in matched):
        return "test pair moves nothing"
    if not any(f["move"] > 0 and f["informative"] for f in matched):
        return "test pair has no informative mirror"
    modes = ["nearest", "translate", "wrong_side"]
    if any(f["kind"] == "1d" for f in matched):
        modes.append("point")
    if any(f["kind"] in ("2d", "dual") for f in matched):
        modes.append("pinwheel")
    for mode in modes:
        if not _differs(inp, out, mode):
            return f"test pair does not exclude the {mode} reading"
    return None


# --------------------------------------------------------------------------
# Puzzle-level plan
# --------------------------------------------------------------------------

def _spec(type_, nh, nv, k, size, musts=(), bg=None, shapes="normal", distract=(0, 3), dual=0):
    return {"type": type_, "nh": nh, "nv": nv, "k": k, "size": size, "musts": set(musts),
            "bg": bg, "shapes": shapes, "distract": distract, "dual": dual}


def _oriented(orientation, n, **kwargs):
    if orientation == "H":
        return _spec("H", n, 0, 0, **kwargs)
    return _spec("V", 0, n, 0, **kwargs)


def _spec_axes(spec):
    if spec["type"] == "E":
        return set()
    if spec["type"] == "H":
        return {"r"}
    if spec["type"] == "V":
        return {"c"}
    axes = set()
    if spec["k"] or spec.get("dual"):
        axes |= {"r", "c"}
    if spec["nh"] > spec["k"] and spec["nv"] > spec["k"]:
        axes |= {"r", "c"}
    return axes


def _regime(seed):
    residue = int(seed) % 14
    return "minimal" if residue == 13 else "maximal" if residue in (6, 12) else "intermediate"


def _puzzle_params(rng, regime, num_train, degrade=0):
    if regime == "minimal":
        pz = {"multi": False, "dual": False, "contact": False, "diag": False, "cross": False,
              "erase": False, "level": 0.0}
        bg_policy = "zero" if rng.random() < 0.3 else "fixed"
    elif regime == "maximal":
        pz = {"multi": True, "dual": True, "contact": True, "diag": True, "cross": True,
              "erase": num_train >= 3, "level": 1.0}
        bg_policy = "vary"
    else:
        # Rates sit marginally above the targets because a failed plan is redrawn.
        multi = bool(rng.random() < 0.46)
        pz = {"multi": multi, "dual": multi and bool(rng.random() < 0.37),
              "contact": bool(rng.random() < 0.41), "diag": bool(rng.random() < 0.25),
              "cross": bool(rng.random() < 0.75), "erase": num_train >= 3 and bool(rng.random() < 0.125),
              "level": float(rng.uniform(0.25, 0.85))}
        bg_policy = str(rng.choice(("vary", "fixed", "zero"), p=(0.6, 0.2, 0.2)))
    if degrade >= 1:
        # Rarely needed fallback: keep the regime but drop the most demanding options.
        pz["dual"] = False
        pz["level"] = min(pz["level"], 0.6)
    if degrade >= 2:
        pz["diag"] = pz["erase"] = False
    pz["regime"] = regime
    pz["bg_policy"] = bg_policy
    pz["fixed_bg"] = 0 if bg_policy == "zero" else int(rng.integers(1, 10))
    pz["main"] = "H" if rng.random() < 0.5 else "V"
    return pz


def _plan(rng, pz, num_train, num_test):
    minimal = pz["regime"] == "minimal"
    maximal = pz["regime"] == "maximal"

    def bg():
        return None if pz["bg_policy"] == "vary" else pz["fixed_bg"]

    main = pz["main"]
    other = "V" if main == "H" else "H"
    slots = [int(i) for i in rng.permutation(num_train)]
    specs = [None] * num_train
    anchor = slots.pop()
    anchor_musts = {"cross1d", "sides", "distractor"}
    if minimal or rng.random() < 0.7:
        specs[anchor] = _oriented(main, int(rng.integers(2, 4 if minimal else 5)),
                                  size=str(rng.choice(("medium", "large"))), musts=anchor_musts, bg=bg())
    else:
        n_main, n_other = int(rng.integers(2, 4)), int(rng.integers(1, 3))
        nh, nv = (n_main, n_other) if main == "H" else (n_other, n_main)
        specs[anchor] = _spec("L", nh, nv, 0, size="large", musts=anchor_musts, bg=bg())
    roles = {}
    if pz["cross"]:
        if rng.random() < 0.15:
            spec = _spec("L", 1, 1, 1, size=str(rng.choice(("small", "medium"))), musts={"2d"}, bg=bg(),
                         distract=(1, 3))
        else:
            nh, nv = int(rng.integers(1, 5)), int(rng.integers(1, 5))
            if nh + nv < 3:
                nh, nv = nh + 1, nv + 1
            spec = _spec("L", nh, nv, int(rng.integers(1, min(3, nh, nv) + 1)),
                         size=str(rng.choice(("large", "full"))), musts={"2d"}, bg=bg())
        roles["cross"] = spec
    if pz["dual"]:
        if rng.random() < 0.25:
            spec = _spec("L", 1, 1, 0, size=str(rng.choice(("small", "medium"))), musts={"dual"}, bg=bg(),
                         dual=1, distract=(1, 3))
        else:
            nh, nv = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            k = int(rng.integers(0, 2)) if pz["cross"] and min(nh, nv) >= 2 else 0
            spec = _spec("L", nh, nv, k, size=str(rng.choice(("large", "full"))), musts={"dual"}, bg=bg(),
                         dual=1)
        roles["dual"] = spec
    if pz["erase"]:
        lay = str(rng.choice(("H", "V", "L")))
        nh = int(rng.integers(1, 4)) if lay != "V" else 0
        nv = int(rng.integers(1, 4)) if lay != "H" else 0
        k = int(rng.integers(0, 2)) if lay == "L" and pz["cross"] else 0
        roles["erase"] = _spec("E", nh, nv, k, size=str(rng.choice(("small", "medium", "large"))),
                               musts={"erase"}, bg=bg(), distract=(1, 4))
    for name in ("cross", "dual", "erase"):
        if name not in roles:
            continue
        if slots:
            specs[slots.pop()] = roles[name]
        elif name == "dual" and "cross" in roles:
            spec = roles["cross"]
            spec["nh"], spec["nv"] = max(2, spec["nh"]), max(2, spec["nv"])
            spec["k"] = min(spec["k"], 1)
            spec["dual"] = 1
            spec["musts"].add("dual")
            spec["size"] = "full"
        elif name == "dual":
            pz["dual"] = False
        else:
            pz["erase"] = False
    for i in slots:
        roll = rng.random()
        size = str(rng.choice(("small", "medium", "large"), p=(0.35, 0.4, 0.25)))
        shapes = str(rng.choice(("sparse", "normal", "dense"), p=(0.3, 0.5, 0.2)))
        if minimal:
            specs[i] = _oriented(main, int(rng.integers(1, 4)), size=size, bg=bg(), shapes=shapes)
        elif roll < 0.38:
            specs[i] = _oriented(main, int(rng.integers(1, 6)), size=size, bg=bg(), shapes=shapes)
        elif roll < 0.66:
            specs[i] = _oriented(other, int(rng.integers(1, 6)), size=size, bg=bg(), shapes=shapes)
        else:
            nh, nv = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            k = int(rng.integers(0, min(3, nh, nv) + 1)) if pz["cross"] else 0
            if k and nh + nv < 3:
                nh += 1
            dual = 1 if pz["dual"] and nh > k and nv > k and rng.random() < 0.4 else 0
            specs[i] = _spec("L", nh, nv, k, size=str(rng.choice(("medium", "large", "full"))),
                             musts={"2d"} if k and rng.random() < 0.5 else set(), bg=bg(), shapes=shapes,
                             dual=dual)
    # Puzzle features are each demonstrated by some docking training pair.
    hosts = [i for i in range(num_train) if specs[i]["type"] != "E"]
    plain = [i for i in hosts if specs[i]["type"] in ("H", "V") or specs[i]["nh"] > specs[i]["k"]
             or specs[i]["nv"] > specs[i]["k"]]
    for name, on in (("multi", pz["multi"]), ("zero", pz["contact"]), ("touch", pz["contact"]),
                     ("diag", pz["diag"])):
        if on:
            group = hosts if name == "multi" or not plain else plain
            specs[group[int(rng.integers(len(group)))]]["musts"].add(name)
    # Tests.
    shift = "combine" if maximal else str(rng.choice(("grow", "combine", "matched", "smaller", "transfer"),
                                                     p=(0.27, 0.27, 0.16, 0.14, 0.16)))
    if minimal:
        shift = str(rng.choice(("grow", "matched", "smaller")))
    tests = []
    docking = [s for s in specs if s["type"] != "E"]
    for j in range(num_test):
        mode = shift if j == 0 else (str(rng.choice(("grow", "matched", "smaller"))) if minimal else
                                     str(rng.choice(("grow", "combine", "matched", "smaller", "transfer"))))
        base = docking[int(rng.integers(len(docking)))]
        if mode == "combine":
            nh, nv = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            k = int(rng.integers(1, min(3, nh, nv) + 1)) if pz["cross"] else 0
            dual = 1 if pz["dual"] and nh > k and nv > k and (maximal or rng.random() < 0.6) else 0
            spec = _spec("L", nh, nv, k, size=str(rng.choice(("large", "full"))), bg=bg(), shapes="normal",
                         dual=dual, distract=(1, 4))
        elif mode == "grow":
            if base["type"] == "L":
                nh, nv = min(4, base["nh"] + 1), min(4, base["nv"] + int(rng.integers(0, 2)))
                k = min(max(base["k"], 1 if pz["cross"] and rng.random() < 0.5 else 0), 3, nh, nv)
                dual = base["dual"] if nh > k and nv > k else 0
                spec = _spec("L", nh, nv, k, size="large", bg=bg(), shapes="dense", distract=(1, 4), dual=dual)
            else:
                most = max([s["nh"] if base["type"] == "H" else s["nv"] for s in specs
                            if s["type"] == base["type"]] or [1])
                spec = _oriented(base["type"], min(6, most + int(rng.integers(1, 3))), size="large",
                                 bg=bg(), shapes="dense", distract=(1, 4))
        elif mode == "matched":
            spec = _spec(base["type"], base["nh"], base["nv"], base["k"], base["size"], bg=bg(),
                         dual=base["dual"], distract=(1, 3))
        elif mode == "transfer":
            nh_par = sum(1 for s in docking if s["type"] == "H")
            nv_par = sum(1 for s in docking if s["type"] == "V")
            orient = "H" if nh_par < nv_par else "V" if nv_par < nh_par else other
            spec = _oriented(orient, int(rng.integers(1, 5)), size=str(rng.choice(("medium", "large"))),
                             bg=bg(), distract=(1, 3))
        else:  # smaller
            if base["type"] == "L" and base["k"]:
                spec = _spec("L", 1, 1, 1, size="small", bg=bg(), shapes="sparse", distract=(1, 2))
            elif base["type"] == "L" and base["dual"]:
                spec = _spec("L", 1, 1, 0, size="small", bg=bg(), shapes="sparse", distract=(1, 2), dual=1)
            elif base["type"] == "L":
                spec = _spec("L", 1, 1, 0, size="small", bg=bg(), shapes="sparse", distract=(1, 2))
            else:
                spec = _oriented(base["type"], 1, size="small", bg=bg(), shapes="sparse", distract=(1, 2))
        spec["mode"] = mode
        spec["musts"] = {"informative", "nearfail"}
        tests.append(spec)
    # Coverage patches: every docking axis and crossing style used by a test is demonstrated.
    train_axes = set().union(*(_spec_axes(s) for s in specs))
    need_axes = set().union(*(_spec_axes(t) for t in tests))
    free = [i for i in range(num_train) if specs[i]["type"] in ("H", "V", "L") and i != anchor
            and not specs[i]["musts"] & {"2d", "dual"}]
    for axis in sorted(need_axes - train_axes):
        orientation = "H" if axis == "r" else "V"
        if free:
            i = free.pop(int(rng.integers(len(free))))
            keep = specs[i]["musts"] & {"multi", "zero", "touch", "diag"}
            specs[i] = _oriented(orientation, int(rng.integers(1, 4)),
                                 size=str(rng.choice(("small", "medium"))), bg=bg(), musts=keep)
        else:
            spec = specs[anchor]
            spec["musts"].add("axis_" + axis)
            if spec["type"] != "L":
                n = spec["nh"] or spec["nv"]
                nh, nv = (n, 1) if spec["type"] == "H" else (1, n)
                specs[anchor] = _spec("L", nh, nv, 0, size="large", bg=spec["bg"], musts=set(spec["musts"]))
    if any(t["type"] == "L" and t["nh"] * t["nv"] > t["k"] for t in tests) and not any(
            s["type"] == "L" and s["nh"] * s["nv"] > s["k"] for s in specs):
        target = next((s for s in specs if s["type"] == "L" and s["musts"] & {"2d", "dual"}), None)
        if target is not None:
            target["nh"] = max(2, target["nh"])
            target["size"] = "full"
        elif free:
            i = free.pop(int(rng.integers(len(free))))
            keep = specs[i]["musts"] & {"multi", "zero", "touch", "diag"}
            specs[i] = _spec("L", int(rng.integers(1, 3)), int(rng.integers(1, 3)), 0, size="large",
                             bg=bg(), musts={"axis_r", "axis_c"} | keep)
        else:
            spec = specs[anchor]
            if spec["type"] != "L":
                n = spec["nh"] or spec["nv"]
                nh, nv = (n, 1) if spec["type"] == "H" else (1, n)
                specs[anchor] = _spec("L", nh, nv, 0, size="large", bg=spec["bg"], musts=set(spec["musts"]))
    for s in specs + tests:
        if s["type"] in ("L", "E") and s["nh"] + s["nv"] > 2:
            s["size"] = "large" if s["size"] in ("small", "medium") else s["size"]
    return specs, tests


def _make_pair(rng, spec, pz):
    for _ in range(40):
        try:
            return _try_pair(rng, spec, pz)
        except _Retry:
            continue
    raise _Retry("pair")


def _check_counts(num_train, num_test):
    # None selects the documented defaults (convenient for sampling tools).
    if num_test is None:
        num_test = 1
    if num_train is None:
        num_train = max(4, num_test + 1) if isinstance(num_test, (int, np.integer)) else 4
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = _regime(seed)
    for attempt in range(90):
        degrade = 0 if attempt < 30 else 1 if attempt < 55 else 2
        current = regime if attempt < 75 else "minimal"
        try:
            pz = _puzzle_params(rng, current, num_train, degrade)
            specs, tests = _plan(rng, pz, num_train, num_test)
            puzzle = {"train": [], "test": []}
            seen = set()
            most = 0
            for split, plan in (("train", specs), ("test", tests)):
                for spec in plan:
                    if split == "test" and spec.get("mode") == "grow" and spec["type"] != "L":
                        # A larger test: more bars and more moving shapes than any demonstration.
                        spec["min_matched"] = min(most + 1, 4 * max(spec["nh"], spec["nv"]))
                    for _ in range(6):
                        inp, out, analysis = _make_pair(rng, spec, pz)
                        key = (inp.shape, inp.tobytes())
                        if key not in seen:
                            break
                    else:
                        raise _Retry("duplicate input")
                    seen.add(key)
                    if split == "train":
                        most = max(most, sum(f["kind"] != "distractor" for f in analysis["shapes"]))
                    puzzle[split].append({"input": inp.astype(np.int64), "output": out.astype(np.int64)})
            validate_generated_puzzle(puzzle)
            return puzzle
        except (_Retry, AssertionError):
            continue
    raise RuntimeError("unable to construct a puzzle for this seed")


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts"
    seen = set()
    analyses = {"train": [], "test": []}
    erase_pairs = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "grid size"
                assert np.issubdtype(grid.dtype, np.integer), "integer grid"
                assert np.all((grid >= 0) & (grid <= 9)), "grid colors"
            assert inp.shape == out.shape, "shape preserved"
            key = (inp.shape, inp.astype(np.int64).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            analysis = _analyze(inp)
            assert np.array_equal(solve(inp), out), "output disagrees with rule"
            assert not np.array_equal(inp, out), "identity pair"
            feats = analysis["shapes"]
            matched = [f for f in feats if f["kind"] != "distractor"]
            if not matched:
                # The single all-erase demonstration (training only).
                assert split == "train" and feats, "test pair must dock a shape"
                erase_pairs += 1
            else:
                assert any(f["move"] > 0 for f in matched), "no shape moves"
                for mode in ("no_mirror", "no_dock"):
                    assert _differs(inp, out, mode), f"{mode} shortcut not excluded"
            if split == "test":
                reason = _test_strength(inp, out, analysis)
                assert reason is None, reason
            analysis["out"] = out
            analysis["inp"] = inp
            analyses[split].append(analysis)
    assert erase_pairs <= 1, "more than one all-erase demonstration"
    tr, te = analyses["train"], analyses["test"]
    train_feats = [f for a in tr for f in a["shapes"]]
    test_feats = [f for a in te for f in a["shapes"]]
    train_matched = [f for f in train_feats if f["kind"] != "distractor"]
    test_matched = [f for f in test_feats if f["kind"] != "distractor"]
    assert any(f["crossed"] > 0 and f["move"] > 0 for f in train_matched), \
        "training never shows a shape passing another bar"
    assert any(f["kind"] == "distractor" for f in train_feats), "training never erases a shape"
    sides = {f["side"] for f in train_matched if f["kind"] == "1d" and f["informative"]}
    assert {-1, 1} <= sides, "training lacks informative docking from both sides"
    all_matched = train_matched + test_matched
    applicable = {"pinwheel": any(f["kind"] in ("2d", "dual") for f in all_matched),
                  "split": any(f["multicolor"] for f in all_matched),
                  "four_conn": any(f["diag"] for f in all_matched)}
    for mode in _SHORTCUTS:
        if not applicable.get(mode, True):
            continue
        assert any(_differs(a["inp"], a["out"], mode) for a in tr), \
            f"training does not exclude the {mode} shortcut"
    # Every mechanism a test needs is demonstrated in training.
    train_axes = set().union(*(_axes(a) for a in tr))
    for a in te:
        assert _axes(a) <= train_axes, "test docking axis not demonstrated"
    if any(f["kind"] in ("2d", "dual") for f in test_matched):
        assert any(f["kind"] in ("2d", "dual") and f["informative"] and f["rot_inf"] for f in train_matched), \
            "test unfolds around a crossing that training never demonstrates"
    if any(f["kind"] == "dual" for f in test_matched):
        assert any(f["kind"] == "dual" and f["informative"] and f["rot_inf"] for f in train_matched), \
            "test uses a two-key unfolding that training never demonstrates"
    if any(a["diff_cross"] for a in te):
        assert any(a["diff_cross"] for a in tr), "test bar crossing style not demonstrated"
    if any(f["multicolor"] for f in test_feats):
        assert any(f["multicolor"] and f["key_off_edge"] and f["move"] > 0 for f in train_matched), \
            "test uses multicolor shapes that training never moves"
    if any(f["zero"] for f in test_matched):
        assert any(f["zero"] for f in train_matched), "test zero-travel docking not demonstrated"
    if any(f["touch_other"] for f in test_feats):
        assert any(f["touch_other"] and (f["move"] > 0 or f["kind"] == "distractor") for f in train_feats), \
            "test wrong-bar contact not demonstrated"
    if any(f["diag"] for f in test_feats):
        assert any(f["diag"] and f["move"] > 0 for f in train_matched), \
            "test diagonal-only links not demonstrated"


def _same(a, b):
    return all(len(a[s]) == len(b[s]) for s in ("train", "test")) and \
        all(np.array_equal(x[k], y[k]) for s in ("train", "test")
            for x, y in zip(a[s], b[s]) for k in ("input", "output"))


def self_test() -> None:
    for forms in _GLYPHS:
        assert len(_components(forms[0], 4)) == 1, "glyph must be 4-connected"
    for forms in _DIAGS:
        assert len(_components(forms[0], 8)) == 1 < len(_components(forms[0], 4)), "diagonal glyph"
    stats = {}

    def bump(name, amount=1):
        stats[name] = stats.get(name, 0) + amount

    shapes_seen, bgs, cross_patterns = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        bump("regime_" + _regime(seed))
        tr = [_analyze(p["input"]) for p in puzzle["train"]]
        te = [_analyze(p["input"]) for p in puzzle["test"]]
        all_a = tr + te
        if any(a["cross_colors"] for a in all_a):
            bump("puzzle_with_cross")
        if any(a["hbars"] and a["vbars"] for a in te):
            bump("test_lattice")
        if any(a["hbars"] and not a["vbars"] for a in te):
            bump("test_horizontal_only")
        if any(a["vbars"] and not a["hbars"] for a in te):
            bump("test_vertical_only")
        n_bars = lambda a: len(a["hbars"]) + len(a["vbars"])
        if max(n_bars(a) for a in te) > max(n_bars(a) for a in tr):
            bump("test_more_bars")
        if min(n_bars(a) for a in te) < max(n_bars(a) for a in tr):
            bump("test_fewer_bars")
        if len({a["bg"] for a in all_a}) > 1:
            bump("bg_varies_in_puzzle")
        if any(not any(f["kind"] != "distractor" for f in a["shapes"]) for a in tr):
            bump("all_erase_pair")
        for p, a in zip(puzzle["train"] + puzzle["test"], all_a):
            shapes_seen.add(p["input"].shape)
            bgs.add(a["bg"])
            feats = a["shapes"]
            matched = [f for f in feats if f["kind"] != "distractor"]
            if a["diff_cross"]:
                bump("diff_cross")
                bump("layer_" + "".join(sorted(a["layers"])))
            if a["edge_bar"]:
                bump("edge_bar")
            if a["untouched_cross"]:
                bump("untouched_cross")
            if len(a["cross_colors"]) >= 3:
                bump("three_crosses")
            if max(len(a["hbars"]), len(a["vbars"])) >= 5 and not (a["hbars"] and a["vbars"]):
                bump("five_plus_parallel")
            if len(a["hbars"]) >= 4 and len(a["vbars"]) >= 4:
                bump("lattice_4x4")
            if any(f["crossed"] >= 2 for f in matched):
                bump("deep_crossing")
            if not any(f["kind"] == "distractor" for f in feats):
                bump("no_distractor")
            if any(f["multicolor"] for f in matched):
                bump("multicolor_matched")
            if any(f["multicolor"] for f in feats if f["kind"] == "distractor"):
                bump("multicolor_distractor")
            if any(f["kind"] == "dual" for f in matched):
                bump("dual_key")
            if any(f["kind"] == "2d" and f["multicolor"] for f in matched):
                bump("multicolor_cross")
            if any(f["zero"] for f in matched):
                bump("zero_travel")
            if any(f["touch_other"] for f in matched):
                bump("wrong_bar_contact")
            if any(f["diag"] for f in matched):
                bump("diag_link")
            if any(f["hole"] for f in feats):
                bump("shape_with_hole")
            if any(max(f["size"]) >= 5 for f in matched):
                bump("large_shape")
            if any(f["kind"] == "1d" and f["n"] == 1 for f in matched):
                bump("single_cell_shape")
            if any(f["kind"] in ("2d", "dual") and f["crossed"] > 0 for f in matched):
                bump("unfolding_passes_bar")
            if any(f["kind"] == "2d" and not f["rot_inf"] for f in matched):
                bump("pinwheel_symmetric_cross_shape")
            colors = [tuple(f["colors"]) for f in matched if f["kind"] == "1d"]
            if len(colors) != len(set(colors)):
                bump("bar_with_several_shapes")
            for f in matched:
                if f["kind"] in ("2d", "dual"):
                    cross_patterns.add(f["canon"])
    for name, least in (("regime_minimal", 2), ("regime_maximal", 4), ("puzzle_with_cross", 10),
                        ("test_lattice", 8), ("test_horizontal_only", 3), ("test_vertical_only", 3),
                        ("test_more_bars", 3), ("test_fewer_bars", 3), ("bg_varies_in_puzzle", 10),
                        ("all_erase_pair", 3), ("diff_cross", 10), ("layer_h", 3), ("layer_v", 3),
                        ("edge_bar", 3), ("untouched_cross", 2), ("three_crosses", 1),
                        ("five_plus_parallel", 2), ("deep_crossing", 5), ("no_distractor", 3),
                        ("multicolor_matched", 10), ("multicolor_distractor", 3), ("dual_key", 4),
                        ("multicolor_cross", 2), ("zero_travel", 5), ("wrong_bar_contact", 5),
                        ("diag_link", 5), ("shape_with_hole", 3), ("large_shape", 5),
                        ("single_cell_shape", 3), ("unfolding_passes_bar", 3), ("bar_with_several_shapes", 5),
                        ("lattice_4x4", 1)):
        assert stats.get(name, 0) >= least, f"self_test coverage gap: {name} ({stats.get(name, 0)})"
    assert len(bgs) >= 8, "background colors too uniform"
    assert len(shapes_seen) >= 60, "grid shapes too uniform"
    assert len(cross_patterns) >= 15, "unfolded shapes too uniform"
    assert min(min(s) for s in shapes_seen) <= 10 and max(max(s) for s in shapes_seen) == 30
    assert _same(generate_puzzle(11, 6, 3), generate_puzzle(11, 6, 3)), "nondeterministic"
    assert not _same(generate_puzzle(11, 6, 3), generate_puzzle(12, 6, 3)), "seed ignored"
    for counts in ((2, 1), (10, 4), (5, 4), (10, 1)):
        puzzle = generate_puzzle(4242, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        validate_generated_puzzle(puzzle)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(5, *counts)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {counts}")
