"""Notched-divider territory drops.

Each input is two congruent panels split by a one-cell divider (parent B layout).
The divider is a solid line of a unique separator colour carrying a few coloured
singleton "notches" (parent A markers).  Every notch claims the panel rows that
are nearest to it along the divider, with an exact midpoint tie going to the
upper notch (parent A's one-dimensional Voronoi ownership).  Inside each claimed
territory the right panel's candidate cells are dropped onto the aligned left
panel as one indivisible transaction (parent B's global accept/reject, now scoped
to a territory): if any candidate cell of the territory lands on a base cell the
whole territory's drop is cancelled; otherwise every candidate cell of that
territory lands, painted in the territory's notch colour.  The output is the
left panel after all territories are resolved.

A puzzle may instead stack the panels (horizontal divider); then everything is
transposed: notches claim columns and a tie goes to the left notch.
"""

import numpy as np


# ----------------------------------------------------------------------------
# count / argument handling


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(num_train, num_test):
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


# ----------------------------------------------------------------------------
# parsing and the rule (shared by solve, the validator and the shortcut checks)


def _parse_canonical(grid):
    """Parse a grid whose divider is the middle column; None if it is not one."""
    h, w = grid.shape
    if h < 2 or w < 3 or w % 2 == 0:
        return None
    m = w // 2
    divider = grid[:, m]
    values, counts = np.unique(divider, return_counts=True)
    top = int(np.argmax(counts))
    separator = int(values[top])
    if 2 * int(counts[top]) <= h:
        return None
    left, right = grid[:, :m], grid[:, m + 1:]
    if np.any(left == separator) or np.any(right == separator):
        return None
    left_colors = set(int(v) for v in np.unique(left))
    right_colors = set(int(v) for v in np.unique(right))
    shared = left_colors & right_colors
    if not shared:
        return None
    both = np.concatenate((left.ravel(), right.ravel()))
    background = max(sorted(shared), key=lambda c: int(np.count_nonzero(both == c)))
    positions = [int(r) for r in range(h) if int(divider[r]) != separator]
    if not positions:
        return None
    colors = [int(divider[r]) for r in positions]
    others = sorted(right_colors - {background})
    return {
        "left": left.copy(),
        "right": right.copy(),
        "base": left != background,
        "cand": right != background,
        "background": int(background),
        "separator": separator,
        "positions": positions,
        "colors": colors,
        "base_colors": sorted(left_colors - {background}),
        "cand_colors": others,
        "cand_color": others[0] if len(others) == 1 else None,
    }


def _parse_all(grid):
    """Return every orientation reading of the grid as (transposed, info)."""
    grid = np.asarray(grid, dtype=int)
    readings = []
    for transposed in (False, True):
        info = _parse_canonical(grid.T if transposed else grid)
        if info is not None:
            readings.append((transposed, info))
    return readings


def _owners(length, positions, mode="nearest_upper"):
    """Index of the owning notch for every line along the divider axis."""
    pos = np.asarray(positions, dtype=int)
    idx = np.arange(length)
    if mode == "nearest_upper":
        return np.abs(idx[:, None] - pos[None, :]).argmin(axis=1)
    if mode == "nearest_lower":
        dist = np.abs(idx[:, None] - pos[None, :])
        return len(pos) - 1 - dist[:, ::-1].argmin(axis=1)
    if mode == "down":  # a notch owns itself and everything down to the next notch
        return np.clip(np.searchsorted(pos, idx, side="right") - 1, 0, None)
    if mode == "up":  # a notch owns everything from the previous notch down to itself
        return np.clip(np.searchsorted(pos, idx, side="left"), None, len(pos) - 1)
    raise ValueError(mode)


def _components(mask):
    labels = np.full(mask.shape, -1, dtype=int)
    count = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c] and labels[r, c] < 0:
                labels[r, c] = count
                stack = [(r, c)]
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] < 0:
                            labels[ny, nx] = count
                            stack.append((ny, nx))
                count += 1
    return labels, count


def _render(info, owners="nearest_upper", scope="territory", recolor=True, accept="disjoint"):
    """The true rule (defaults) and the shortcut variants used by the validator."""
    base, cand = info["base"], info["cand"]
    h = base.shape[0]
    own = _owners(h, info["positions"], owners)
    collide = base & cand
    paint = np.zeros_like(cand)
    if scope == "territory":
        for k in range(len(info["positions"])):
            rows = own == k
            ok = not collide[rows].any()
            if accept == "exact":
                ok = ok and np.array_equal(cand[rows], ~base[rows])
            if ok:
                paint[rows] = cand[rows]
    elif scope == "global":
        if not collide.any():
            paint = cand.copy()
    elif scope == "row":
        for r in range(h):
            if not collide[r].any():
                paint[r] = cand[r]
    elif scope == "cell":
        paint = cand & ~base
    elif scope == "piece":
        labels, count = _components(cand)
        for k in range(count):
            piece = labels == k
            if not (piece & base).any():
                paint |= piece
    elif scope == "none":
        paint = cand.copy()
    else:
        raise ValueError(scope)
    out = info["left"].copy()
    if recolor:
        colors = np.asarray(info["colors"], dtype=int)[own]
        fill = np.broadcast_to(colors[:, None], base.shape)
        out[paint] = fill[paint]
    else:
        out[paint] = info["right"][paint]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    readings = _parse_all(grid)
    if not readings:
        raise ValueError("no notched divider found")
    transposed, info = readings[0]
    out = _render(info)
    return out.T.copy() if transposed else out


# Plausible alternative rules that the training pairs must refute.
SHORTCUTS = {
    "tie_goes_down": {"owners": "nearest_lower"},
    "notch_owns_until_next_notch": {"owners": "down"},
    "notch_owns_since_previous_notch": {"owners": "up"},
    "keep_candidate_colour": {"recolor": False},
    "one_global_transaction": {"scope": "global"},
    "parent_B_only": {"scope": "global", "recolor": False},
    "clip_colliding_cells": {"scope": "cell"},
    "cancel_colliding_rows": {"scope": "row"},
    "cancel_colliding_pieces": {"scope": "piece"},
    "exact_fill_required": {"accept": "exact"},
    "paint_everything": {"scope": "none"},
}
IGNORE_A = ("one_global_transaction", "parent_B_only", "keep_candidate_colour")
IGNORE_B = ("paint_everything", "clip_colliding_cells")


def _refuted(info):
    truth = _render(info)
    return {name for name, kw in SHORTCUTS.items() if not np.array_equal(_render(info, **kw), truth)}


# ----------------------------------------------------------------------------
# sampling helpers (canonical orientation: divider is a column, territories are rows)


def _roles(rng):
    background = 0 if rng.random() < 0.5 else int(rng.integers(1, 10))
    rest = [c for c in range(10) if c != background]
    separator, base_color, cand_color = (int(x) for x in rng.choice(rest, size=3, replace=False))
    return background, separator, base_color, cand_color


def _shape(rng, profile):
    ranges = {
        "demo": ((9, 21), (4, 10)),
        "train": ((5, 23), (3, 12)),
        "simple": ((5, 13), (3, 9)),
        "familiar": ((7, 19), (3, 11)),
        "larger": ((14, 31), (5, 15)),
        "tie": ((9, 25), (3, 13)),
    }
    (h0, h1), (w0, w1) = ranges[profile]
    return int(rng.integers(h0, h1)), int(rng.integers(w0, w1))


def _notch_count(rng, profile, h):
    cap = (h - 1) // 2
    low, high = {
        "demo": (3, 5),
        "train": (1, 6),
        "simple": (1, 2),
        "familiar": (2, 4),
        "larger": (4, 7),
        "tie": (2, 5),
    }[profile]
    high = min(high, cap)
    low = min(low, high)
    if profile == "train" and high >= 2 and rng.random() < 0.8:
        low = 2
    return int(rng.integers(low, high + 1))


def _positions(rng, h, k):
    style = int(rng.integers(5))
    if style == 1 and k >= 2:  # equal spacing
        step = int(rng.integers(1, (h - 1) // (k - 1) + 1))
        start = int(rng.integers(0, h - step * (k - 1)))
        return [start + step * i for i in range(k)]
    if style == 2:  # a notch on a boundary
        edge = 0 if rng.random() < 0.5 else h - 1
        rest = rng.choice([r for r in range(h) if r != edge], size=k - 1, replace=False)
        return sorted([edge] + [int(x) for x in rest])
    if style == 3 and k >= 2:  # two adjacent notches
        start = int(rng.integers(0, h - 1))
        pool = [r for r in range(h) if r not in (start, start + 1)]
        rest = rng.choice(pool, size=k - 2, replace=False)
        return sorted([start, start + 1] + [int(x) for x in rest])
    return sorted(int(x) for x in rng.choice(h, size=k, replace=False))


def _tie_layout(rng, h, k):
    """Positions with an adjacent notch pair at an even gap >= 4 (a midpoint tie line)."""
    gaps = [g for g in (4, 6, 8) if g <= h - 1 and h - (g + 1) >= k - 2]
    if not gaps:
        return None
    g = int(rng.choice(gaps))
    a = int(rng.integers(0, h - g))
    outside = [r for r in range(h) if r < a or r > a + g]
    if len(outside) < k - 2:
        return None
    rest = [int(x) for x in rng.choice(outside, size=k - 2, replace=False)] if k > 2 else []
    positions = sorted([a, a + g] + rest)
    upper = positions.index(a)
    return positions, upper, a + g // 2


def _base_mask(rng, h, w):
    rows, cols = np.indices((h, w))
    for _ in range(40):
        style = int(rng.choice(6, p=[0.24, 0.2, 0.18, 0.1, 0.14, 0.14]))
        if style == 0:  # rectangles / blobs
            mask = np.zeros((h, w), dtype=bool)
            for _ in range(int(rng.integers(2, max(3, h * w // 9) + 1))):
                bh, bw = int(rng.integers(1, min(3, h) + 1)), int(rng.integers(1, min(3, w) + 1))
                r, c = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
                mask[r:r + bh, c:c + bw] = True
        elif style == 1:  # skyline growing from one side
            run = np.empty(h, dtype=int)
            run[0] = int(rng.integers(0, w))
            for r in range(1, h):
                run[r] = int(np.clip(run[r - 1] + int(rng.integers(-2, 3)), 0, w - 1))
            mask = cols < run[:, None]
            if rng.random() < 0.5:
                mask = mask[:, ::-1]
        elif style == 2:  # solid wall with cavities
            mask = np.ones((h, w), dtype=bool)
            for _ in range(int(rng.integers(2, max(3, h * w // 7) + 1))):
                bh, bw = int(rng.integers(1, min(3, h) + 1)), int(rng.integers(1, min(3, w) + 1))
                r, c = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
                mask[r:r + bh, c:c + bw] = False
        elif style == 3:  # sparse scatter
            mask = rng.random((h, w)) < float(rng.uniform(0.2, 0.42))
        elif style == 4:  # diagonal stripes
            period = int(rng.integers(3, 6))
            width = int(rng.integers(1, period))
            sign = 1 if rng.random() < 0.5 else -1
            mask = ((rows + sign * cols + int(rng.integers(period))) % period) < width
        else:  # walls crossing the territories, with gaps
            mask = np.zeros((h, w), dtype=bool)
            walls = rng.choice(w, size=int(rng.integers(1, max(2, w // 2) + 1)), replace=False)
            for c in walls:
                mask[:, int(c)] = True
                for _ in range(int(rng.integers(1, max(2, h // 4) + 1))):
                    g0 = int(rng.integers(0, h))
                    mask[g0:g0 + int(rng.integers(1, 3)), int(c)] = False
            for _ in range(int(rng.integers(0, 3))):
                mask[int(rng.integers(h)), int(rng.integers(w))] = True
        density = float(mask.mean())
        if 0.15 <= density <= 0.7 and mask.any() and (~mask).any():
            return mask
    mask = np.zeros((h, w), dtype=bool)
    mask[rows % 2 == cols % 2] = True
    return mask


def _grow(rng, allowed, start, size):
    """A 4-connected piece of up to `size` cells inside `allowed`, containing `start`."""
    h, w = allowed.shape
    cells = [start]
    taken = {start}
    while len(cells) < size:
        options = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                          if 0 <= r + dr < h and 0 <= c + dc < w and allowed[r + dr, c + dc]
                          and (r + dr, c + dc) not in taken})
        if not options:
            break
        nxt = options[int(rng.integers(len(options)))]
        cells.append(nxt)
        taken.add(nxt)
    piece = np.zeros_like(allowed)
    for r, c in cells:
        piece[r, c] = True
    return piece


def _pick(rng, mask):
    cells = np.argwhere(mask)
    r, c = cells[int(rng.integers(len(cells)))]
    return int(r), int(c)


def _pieces(rng, allowed, count, low=1, high=4):
    out = np.zeros_like(allowed)
    for _ in range(count):
        free = allowed & ~out
        if not free.any():
            break
        out |= _grow(rng, free, _pick(rng, free), int(rng.integers(low, high + 1)))
    return out


def _territory_drop(rng, base, rows, outcome):
    """Candidate cells for one territory (`rows` is a boolean row mask)."""
    band = np.zeros_like(base)
    band[rows] = True
    free = band & ~base
    solid = band & base
    if outcome == "full" and free.any():
        return free.copy()
    if outcome == "partial" and free.any():
        drop = _pieces(rng, free, int(rng.integers(1, 4)))
        if np.array_equal(drop, free) and drop.sum() >= 2:
            drop[_pick(rng, drop)] = False
        return drop
    if outcome == "reject" and solid.any():
        if free.sum() >= 3 and rng.random() < 0.25:  # near-miss of an exact fill
            drop = free.copy()
            for _ in range(1 if rng.random() < 0.7 else 2):
                if drop.sum() > 1 and (solid & ~drop).any():
                    drop[_pick(rng, drop & free)] = False
                    drop[_pick(rng, solid & ~drop)] = True
            return drop
        drop = _pieces(rng, free, int(rng.integers(0, 3))) if free.any() else np.zeros_like(base)
        hit = _grow(rng, band & ~drop, _pick(rng, solid & ~drop), int(rng.integers(1, 4)))
        return drop | hit
    return np.zeros_like(base)


def _random_drops(rng, base, own, skip=()):
    cand = np.zeros_like(base)
    for k in range(int(own.max()) + 1):
        if k in skip:
            continue
        outcome = str(rng.choice(["reject", "partial", "full", "empty"], p=[0.38, 0.34, 0.18, 0.10]))
        cand |= _territory_drop(rng, base, own == k, outcome)
    return cand


def _notch_colors(rng, k, roles):
    pool = [c for c in range(10) if c not in roles]
    if rng.random() < 0.75 and k <= len(pool):
        return [int(x) for x in rng.choice(pool, size=k, replace=False)]
    return [int(x) for x in rng.choice(pool, size=k, replace=True)]


def _tie_pair(rng, h, w, k, roles):
    """A pair whose midpoint tie line decides the answer (two variants)."""
    layout = _tie_layout(rng, h, k)
    if layout is None:
        return None
    positions, ia, t = layout
    ib = ia + 1
    own = _owners(h, positions)
    rows = np.arange(h)
    base = _base_mask(rng, h, w)
    up = own == ia
    low = own == ib
    low_top = low & (rows > t) & (rows < positions[ib])
    cand = np.zeros_like(base)
    colors = _notch_colors(rng, k, roles)
    pool = [c for c in range(10) if c not in roles]
    if colors[ia] == colors[ib]:
        colors[ib] = int(rng.choice([c for c in pool if c != colors[ia]]))
    if rng.random() < 0.6:  # variant 1: a collision on the tie line cancels the UPPER territory
        c0 = int(rng.integers(w))
        base[t, c0] = True
        near = np.zeros_like(base)
        near[max(0, t - 1):t + 2, max(0, c0 - 1):c0 + 2] = True
        fit_up = ~base & up[:, None] & (rows != t)[:, None] & ~near
        if not fit_up.any():
            cells = np.argwhere(up[:, None] & (rows != t)[:, None] & ~near)
            if not len(cells):
                return None
            r, c = cells[int(rng.integers(len(cells)))]
            base[r, c] = False
            fit_up = ~base & up[:, None] & (rows != t)[:, None] & ~near
        hit = np.zeros_like(base)
        hit[t, c0] = True
        if rng.random() < 0.4:
            side = [c for c in (c0 - 1, c0 + 1) if 0 <= c < w]
            hit[t, side[int(rng.integers(len(side)))]] = True
        cand |= hit
        cand |= _pieces(rng, fit_up, int(rng.integers(1, 3)))
        accepted = [ib]
        skip = {ia, ib}
    else:  # variant 2: a candidate cell on the tie line is painted in the UPPER colour
        tie_free = ~base[t]
        if not tie_free.any():
            base[t, int(rng.integers(w))] = False
        c1 = _pick(rng, (~base[t])[None, :])[1]
        free_up = ~base & up[:, None]
        piece = _grow(rng, free_up, (t, c1), int(rng.integers(1, 4)))
        cand |= piece
        others = [j for j in range(k) if j not in (ia, ib)
                  and int((own == j).sum()) >= 2]
        if not others:
            return None
        ic = int(rng.choice(others))
        band_c = own == ic
        solid = base & band_c[:, None]
        if not solid.any():
            r = int(rng.choice(np.flatnonzero(band_c)))
            base[r, int(rng.integers(w))] = True
            solid = base & band_c[:, None]
        hr, hc = _pick(rng, solid)
        near = np.zeros_like(base)
        near[max(0, hr - 1):hr + 2, max(0, hc - 1):hc + 2] = True
        fit_c = ~base & band_c[:, None] & (rows != hr)[:, None] & ~near
        if not fit_c.any():
            return None
        cand[hr, hc] = True
        cand |= _pieces(rng, fit_c, int(rng.integers(1, 3)))
        accepted = [ia, ib]
        skip = {ia, ib, ic}
    free_low = ~base & low[:, None]
    start_zone = free_low & low_top[:, None]
    if not start_zone.any():
        cells = np.argwhere(low_top[:, None] & np.ones((1, w), dtype=bool))
        if not len(cells):
            return None
        r, c = cells[int(rng.integers(len(cells)))]
        if cand[r, c]:
            return None
        base[r, c] = False
        free_low = ~base & low[:, None]
        start_zone = free_low & low_top[:, None]
    cand |= _grow(rng, free_low & ~cand, _pick(rng, start_zone & ~cand), int(rng.integers(1, 4)))
    cand |= _pieces(rng, free_low & ~cand, int(rng.integers(0, 2)))
    lower_free = free_low
    if np.array_equal(cand & low[:, None], lower_free):
        return None
    cand |= _random_drops(rng, base, own, skip=skip)
    for j in accepted:
        if (cand & base & (own == j)[:, None]).any():
            return None
    return base, cand, positions, colors


def _plain_pair(rng, h, w, k, roles):
    positions = _positions(rng, h, k)
    own = _owners(h, positions)
    base = _base_mask(rng, h, w)
    cand = _random_drops(rng, base, own)
    return base, cand, positions, _notch_colors(rng, k, roles)


def _assemble(base, cand, positions, colors, roles):
    background, separator, base_color, cand_color = roles
    h, w = base.shape
    left = np.full((h, w), background, dtype=int)
    left[base] = base_color
    right = np.full((h, w), background, dtype=int)
    right[cand] = cand_color
    divider = np.full((h, 1), separator, dtype=int)
    for p, c in zip(positions, colors):
        divider[p, 0] = c
    return np.concatenate((left, divider, right), axis=1)


def _make_pair(rng, profile, roles, fixed_shape, transposed, rich=False):
    for _ in range(60):
        h, w = fixed_shape if fixed_shape is not None else _shape(rng, profile)
        k = _notch_count(rng, profile, h)
        if profile in ("demo", "tie"):
            if profile == "demo":
                k = max(k, 3)
            if k > (h - 1) // 2:
                continue
            built = _tie_pair(rng, h, w, k, roles)
        else:
            built = _plain_pair(rng, h, w, k, roles)
        if built is None:
            continue
        base, cand, positions, colors = built
        if not cand.any() or not base.any() or (~base).all() or base.all():
            continue
        grid = _assemble(base, cand, positions, colors, roles)
        info = _parse_canonical(grid)
        if info is None or len(_parse_all(grid)) != 1:
            continue
        if _pair_problems({"input": grid, "output": _render(info)})[0]:
            continue
        refuted = _refuted(info)
        if profile == "demo" and set(refuted) != set(SHORTCUTS):
            continue
        if profile == "train" and not refuted:
            continue
        if profile in ("simple", "familiar", "larger", "tie") and len(refuted) < 2:
            continue
        if rich and not (set(IGNORE_A) & refuted and set(IGNORE_B) & refuted):
            continue
        out = _render(info)
        if transposed:
            grid, out = grid.T.copy(), out.T.copy()
        return {"input": grid, "output": out}
    return None


def _attempt(rng, num_train, num_test):
    transposed = bool(rng.random() < 0.5)
    fixed_roles = _roles(rng) if rng.random() < 0.4 else None
    fixed_shape = (int(rng.integers(9, 21)), int(rng.integers(4, 11))) if rng.random() < 0.25 else None
    demo_index = int(rng.integers(num_train))
    profiles = [("train", "demo" if i == demo_index else "train", False) for i in range(num_train)]
    rich_index = int(rng.integers(num_test))
    for i in range(num_test):
        if i == rich_index:
            profiles.append(("test", str(rng.choice(["familiar", "larger", "tie"])), True))
        else:
            profiles.append(("test", str(rng.choice(["simple", "familiar", "larger", "tie"])), False))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, profile, rich in profiles:
        for _ in range(20):
            roles = fixed_roles if fixed_roles is not None else _roles(rng)
            pair = _make_pair(rng, profile, roles, fixed_shape, transposed, rich)
            if pair is None:
                continue
            key = (pair["input"].shape, pair["input"].tobytes())
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append(pair)
            break
        else:
            return None
    if _problems(puzzle):
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(50):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError("could not build a valid puzzle")


# ----------------------------------------------------------------------------
# validation


def _pair_problems(pair):
    problems = []
    inp, out = pair["input"], pair["output"]
    readings = _parse_all(inp)
    if len(readings) != 1:
        return [f"divider orientation readings: {len(readings)}"], None, None
    transposed, info = readings[0]
    h, w = info["base"].shape
    if not (5 <= h <= 30 and 3 <= w <= 14):
        problems.append("panel size out of family range")
    k = len(info["positions"])
    if not 1 <= k <= min(7, (h - 1) // 2):
        problems.append("notch count out of range")
    if len(info["base_colors"]) != 1 or len(info["cand_colors"]) != 1:
        problems.append("each panel must hold exactly one foreground colour")
    else:
        roles = {info["background"], info["separator"], info["base_colors"][0], info["cand_colors"][0]}
        if len(roles) != 4:
            problems.append("role colours must be distinct")
        if any(c in roles for c in info["colors"]):
            problems.append("notch colour repeats a panel or divider role colour")
    if not info["base"].any() or info["base"].all() or not info["cand"].any():
        problems.append("degenerate panel")
    both = np.concatenate((info["left"].ravel(), info["right"].ravel()))
    tally = np.bincount(both, minlength=10)
    if int(tally[info["background"]]) * 1 <= int(np.max(np.delete(tally, info["background"]))):
        problems.append("background must be the plurality colour of the two panels")
    expected = _render(info)
    if transposed:
        expected = expected.T
    if not isinstance(out, np.ndarray) or out.shape != expected.shape or not np.array_equal(out, expected):
        problems.append("output does not follow the territory rule")
    return problems, transposed, info


def _problems(puzzle):
    problems = []
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        return ["puzzle must have exactly train/test"]
    train, test = puzzle["train"], puzzle["test"]
    if not isinstance(train, list) or not isinstance(test, list):
        return ["splits must be lists"]
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        problems.append("illegal split counts")
    seen = set()
    orientations = set()
    refuted_train = set()
    refuted_tests = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                return problems + ["pair must have exactly input/output"]
            try:
                pair = {role: np.asarray(grid) for role, grid in pair.items()}
            except (TypeError, ValueError):
                return problems + ["grids must be rectangular"]
            for grid in pair.values():
                if grid.ndim != 2 or grid.size == 0:
                    return problems + ["grids must be 2D arrays"]
                if not np.issubdtype(grid.dtype, np.integer):
                    problems.append("grid dtype must be integer")
                if not all(1 <= n <= 30 for n in grid.shape) or grid.min() < 0 or grid.max() > 9:
                    problems.append("grid outside ARC limits")
            key = (pair["input"].shape, pair["input"].tobytes())
            if key in seen:
                problems.append("duplicate input")
            seen.add(key)
            pair_issues, transposed, info = _pair_problems(pair)
            problems += pair_issues
            if info is None:
                continue
            orientations.add(transposed)
            refuted = _refuted(info)
            if split == "train":
                refuted_train |= refuted
            else:
                refuted_tests.append(refuted)
                if len(refuted) < 2:
                    problems.append("test pair is solved by too many shortcuts")
    if len(orientations) > 1:
        problems.append("divider orientation must be constant within a puzzle")
    if not any(set(IGNORE_A) & r and set(IGNORE_B) & r for r in refuted_tests):
        problems.append("no test pair needs both the territories (A) and the transactions (B)")
    missing = set(SHORTCUTS) - refuted_train
    if missing:
        problems.append(f"training does not refute: {sorted(missing)}")
    return problems


def validate_generated_puzzle(puzzle: dict) -> None:
    problems = _problems(puzzle)
    if problems:
        raise AssertionError("; ".join(problems))


# ----------------------------------------------------------------------------
# self test


def _features(pair):
    transposed, info = _parse_all(pair["input"])[0]
    pos = info["positions"]
    h = info["base"].shape[0]
    gaps = np.diff(pos)
    own = _owners(h, pos)
    collide = info["base"] & info["cand"]
    rejected = [k for k in range(len(pos)) if collide[own == k].any()]
    accepted = [k for k in range(len(pos)) if k not in rejected and info["cand"][own == k].any()]
    return {
        "transposed": transposed,
        "shape": info["base"].shape,
        "k": len(pos),
        "background": info["background"],
        "adjacent": bool(len(gaps) and gaps.min() == 1),
        "boundary": pos[0] == 0 or pos[-1] == h - 1,
        "repeat": len(set(info["colors"])) < len(info["colors"]),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "density": float(info["base"].mean()),
        "refuted": _refuted(info),
    }


def self_test() -> None:
    puzzles = []
    for seed in range(42):
        n_train = 2 + seed % 9
        n_test = 1 + (seed // 9) % min(4, n_train - 1)
        puzzle = generate_puzzle(seed, n_train, n_test)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (n_train, n_test)
        puzzles.append(puzzle)
    keys = {tuple(p[s][i][r].tobytes() for s in ("train", "test") for i in range(len(p[s])) for r in ("input", "output"))
            for p in puzzles}
    assert len(keys) == 42, "seeds must give distinct puzzles"
    feats = [(s, _features(pair)) for p in puzzles for s in ("train", "test") for pair in p[s]]
    tests = [f for s, f in feats if s == "test"]
    alls = [f for _, f in feats]
    assert {f["transposed"] for f in alls} == {False, True}, "both divider orientations expected"
    assert len({f["shape"] for f in alls}) >= 60, "panel shapes lack diversity"
    assert {1, 2, 3, 4, 5, 6} <= {f["k"] for f in alls}, "notch counts lack diversity"
    assert len({f["background"] for f in alls}) >= 8, "backgrounds lack diversity"
    assert any(f["adjacent"] for f in alls) and any(f["boundary"] for f in alls)
    assert any(f["repeat"] for f in alls)
    assert any(f["k"] == 1 for f in tests) and any(f["k"] >= 5 for f in tests)
    assert any(f["accepted"] == 0 for f in tests) and any(f["rejected"] == 0 for f in tests)
    assert any("tie_goes_down" in f["refuted"] for f in tests), "tests should sometimes hinge on the tie"
    assert any(set(IGNORE_A) & f["refuted"] and set(IGNORE_B) & f["refuted"] for f in tests)
    assert min(f["density"] for f in alls) < 0.25 and max(f["density"] for f in alls) > 0.6
    same = generate_puzzle(123, 5, 2)
    again = generate_puzzle(123, 5, 2)
    for s in ("train", "test"):
        for a, b in zip(same[s], again[s]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
    for n, m in ((2, 1), (10, 4), (5, 4), (10, 1)):
        p = generate_puzzle(7 + n * 10 + m, n, m)
        assert (len(p["train"]), len(p["test"])) == (n, m)
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(5, n, m)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {n}/{m}")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
