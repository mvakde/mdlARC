"""Arch threads: color-matched projections joined on complete blank rails.

Only standard Python and NumPy are used. Red (2) is the fixed thread color.
The input itself identifies its blank color, arches, directions, and rails.
"""
from collections import defaultdict
import json

import numpy as np


THREAD = 2
# Normalized component cells, aperture, and opening direction.
TEMPLATES = {
    "D": (((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)), (1, 1), (1, 0)),
    "U": (((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)), (0, 1), (-1, 0)),
    "R": (((0, 0), (0, 1), (1, 0), (2, 0), (2, 1)), (1, 1), (0, 1)),
    "L": (((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)), (1, 0), (0, -1)),
}


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _blank(a):
    candidates = [int(c) for c in np.unique(a)
                  if np.any(np.all(a == c, axis=0))
                  or np.any(np.all(a == c, axis=1))]
    if len(candidates) != 1 or candidates[0] == THREAD:
        raise ValueError("Blank must be the unique color of complete monochrome lines")
    return candidates[0]


def _components(a, blank):
    h, w = a.shape
    seen = np.zeros(a.shape, dtype=bool)
    found = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or a[r, c] == blank:
                continue
            color = int(a[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < h and 0 <= xx < w and not seen[yy, xx]
                            and a[yy, xx] == color):
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            found.append((color, cells))
    return found


def _sources(a, blank):
    sources = []
    for color, cells in _components(a, blank):
        if len(cells) != 5:
            continue
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        normalized = {(r - top, c - left) for r, c in cells}
        for shape, hole, direction in TEMPLATES.values():
            if normalized == set(shape):
                r, c = top + hole[0], left + hole[1]
                if a[r, c] == blank:
                    sources.append((r, c, direction[0], direction[1], color))
                break
    return sorted(sources)


def _read(a):
    blank = _blank(a)
    rows = tuple(int(x) for x in np.flatnonzero(np.all(a == blank, axis=1)))
    cols = tuple(int(x) for x in np.flatnonzero(np.all(a == blank, axis=0)))
    sources = _sources(a, blank)
    groups = defaultdict(list)
    hits, misses = [], []
    for source in sources:
        r, c, dr, dc, color = source
        if dr:
            ahead = [y for y in rows if (y - r) * dr > 0]
            line = min(ahead, key=lambda y: abs(y - r)) if ahead else None
            axis, position = "H", c
        else:
            ahead = [x for x in cols if (x - c) * dc > 0]
            line = min(ahead, key=lambda x: abs(x - c)) if ahead else None
            axis, position = "V", r
        if line is None:
            misses.append(source)
        else:
            groups[(axis, line, color)].append(position)
            hits.append((source, axis, line, position))
    return blank, rows, cols, sources, dict(groups), hits, misses


def _segments(groups):
    return [(axis, line, color, min(positions), max(positions))
            for (axis, line, color), positions in sorted(groups.items())]


def _paint(a, groups):
    out = a.copy()
    for axis, line, _, lo, hi in _segments(groups):
        if axis == "H":
            out[line, lo:hi + 1] = THREAD
        else:
            out[lo:hi + 1, line] = THREAD
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input only, including unseen palettes and geometric arrangements."""
    a = _grid(input_grid)
    return _paint(a, _read(a)[4])


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _bands(rng, count):
    sizes = [int(x) for x in rng.integers(4, 8, size=count)]
    while sum(sizes) + count - 1 > 30:
        candidates = [i for i, n in enumerate(sizes) if n > 4]
        sizes[int(rng.choice(candidates))] -= 1
    bands, rails, cursor = [], [], 0
    for i, n in enumerate(sizes):
        bands.append((cursor, cursor + n))
        cursor += n
        if i != count - 1:
            rails.append(cursor)
            cursor += 1
    return bands, rails, cursor


def _construct(rng, profile):
    # Chambers guide construction only. There are no drawn panels or borders.
    nr = 3 if profile == "horizontal" else int(rng.integers(3, 5))
    nc = int(rng.integers(3, 5))
    rb, rows, h = _bands(rng, nr)
    cb, cols, w = _bands(rng, nc)
    blank = int(rng.choice([c for c in range(10) if c != THREAD]))
    palette = [int(c) for c in rng.permutation([c for c in range(10)
                                               if c not in (blank, THREAD)])]
    ca, cc, extra = palette[:3]
    textures = palette[3:3 + int(rng.integers(1, 3))]
    a = np.full((h, w), blank, dtype=np.int8)
    protected = np.zeros((h, w), dtype=bool)
    protected[rows, :] = True
    protected[:, cols] = True
    used, planted = set(), []

    def place(chamber, direction, color, far=False):
        if chamber in used:
            raise ValueError("Occupied construction chamber")
        iy, ix = chamber
        y0, y1 = rb[iy]
        x0, x1 = cb[ix]
        shape, hole, (dr, dc) = TEMPLATES[direction]
        sh = max(y for y, _ in shape) + 1
        sw = max(x for _, x in shape) + 1
        y = int(rng.integers(y0, y1 - sh + 1))
        x = int(rng.integers(x0, x1 - sw + 1))
        if far:
            if dr:
                y = y0 if dr > 0 else y1 - sh
            else:
                x = x0 if dc > 0 else x1 - sw
        for dy, dx in shape:
            a[y + dy, x + dx] = color
        protected[y:y + sh, x:x + sw] = True
        planted.append((y + hole[0], x + hole[1], dr, dc, color))
        used.add(chamber)

    def horizontal(rail, color, columns=None):
        choices = [((iy, ix), "D" if iy == rail else "U")
                   for iy in (rail, rail + 1) for ix in range(nc)
                   if (iy, ix) not in used]
        if columns is None:
            left = [v for v in choices if v[0][1] <= vcol]
            right = [v for v in choices if v[0][1] > vcol]
            if not left or not right:
                raise ValueError("Cannot straddle vertical rail")
            picked = [left[int(rng.integers(len(left)))],
                      right[int(rng.integers(len(right)))]]
        else:
            picked = []
            for ix in columns:
                options = [v for v in choices if v[0][1] == ix]
                if not options:
                    raise ValueError("No source slot")
                picked.append(options[int(rng.integers(len(options)))])
        for chamber, direction in picked:
            place(chamber, direction, color, far=True)

    vcol = int(rng.integers(nc - 1))
    if profile == "horizontal":
        rail = int(rng.integers(nr - 1))
        # Different sides of one rail explicitly teach both vertical openings.
        place((rail, 0), "D", ca, far=True)
        place((rail + 1, 1), "U", ca, far=True)
        chamber = (rail + int(rng.integers(2)), nc - 1)
        place(chamber, "D" if chamber[0] == rail else "U", cc, far=True)
        other = 1 - rail
        options = [(iy, ix) for iy in (other, other + 1) for ix in range(nc)
                   if (iy, ix) not in used]
        chamber = options[int(rng.integers(len(options)))]
        place(chamber, "D" if chamber[0] == other else "U", ca)
        options = [(iy, ix) for iy in (0, nr - 1) for ix in range(nc)
                   if (iy, ix) not in used]
        chamber = options[int(rng.integers(len(options)))]
        place(chamber, "U" if chamber[0] == 0 else "D", cc)
    else:
        # Three/four arrivals teach the full hull rather than pairwise matching.
        group_size = 3 if profile == "crossing" else 2
        if profile == "crossing" and nr == 4 and rng.random() < .5:
            group_size = 4
        for iy in range(group_size):
            ix = vcol + iy % 2
            place((iy, ix), "R" if ix == vcol else "L", cc, far=True)
        if profile == "crossing":
            horizontal(int(rng.integers(2)), ca)
            # A same-color singleton on another parallel rail teaches grouping
            # by destination, even before the two-span test combination.
            other = next(j for j in range(nc - 1) if j != vcol)
            options = [(iy, ix) for iy in range(nr) for ix in (other, other + 1)
                       if (iy, ix) not in used]
            chamber = options[int(rng.integers(len(options)))]
            place(chamber, "R" if chamber[1] == other else "L", cc)
        else:
            # Test recombines two same-color horizontal bridges with a vertical
            # bridge. The same color must be grouped separately on each rail.
            for rail in (0, nr - 2):
                horizontal(rail, ca)
            # For nr==3 the rails are still distinct (0 and 1).
            extras = int(rng.integers(0, min(4, nr * nc - len(used))))
            for _ in range(extras):
                free = [(iy, ix) for iy in range(nr) for ix in range(nc)
                        if (iy, ix) not in used]
                chamber = free[int(rng.integers(len(free)))]
                place(chamber, str(rng.choice(list(TEMPLATES))),
                      int(rng.choice([ca, cc, extra])))

    # Closed six-cell rectangles are same-color arch lookalikes, not sources.
    free = [(iy, ix) for iy in range(nr) for ix in range(nc) if (iy, ix) not in used]
    if not free:
        raise ValueError("No decoy chamber")
    iy, ix = free[int(rng.integers(len(free)))]
    y0, y1 = rb[iy]
    x0, x1 = cb[ix]
    sh, sw = (2, 3) if rng.random() < .5 else (3, 2)
    y = int(rng.integers(y0, y1 - sh + 1))
    x = int(rng.integers(x0, x1 - sw + 1))
    a[y:y + sh, x:x + sw] = ca
    protected[y:y + sh, x:x + sw] = True

    # Make a whole near-blank line on a real projection path. Its lone pebble
    # is deliberately off the centerline, so local blankness is insufficient.
    near_choices = []
    # Use the intended rails here: the still-untextured canvas has many
    # temporary blank lines which the interruption repair will remove.
    for r, c, dr, dc, _ in planted:
        ahead = [v for v in (rows if dr else cols)
                 if (v - (r if dr else c)) * (dr or dc) > 0]
        if not ahead:
            continue
        line = min(ahead, key=lambda v: abs(v - (r if dr else c)))
        axis, position = ("H", c) if dr else ("V", r)
        for z in range((r if dr else c) + (dr or dc), line, dr or dc):
            view = a[z, :] if dr else a[:, z]
            if np.all(view == blank):
                near_choices.append((axis, z, position))
    if not near_choices:
        raise ValueError("No room for a meaningful near-blank line")
    axis, z, projection = near_choices[int(rng.integers(len(near_choices)))]
    if axis == "H":
        choices = [x for x in range(w) if x not in cols and x != projection
                   and not protected[z, x]]
        if not choices:
            raise ValueError("Cannot interrupt near-blank line")
        a[z, int(rng.choice(choices))] = textures[0]
        protected[z, :] = True
    else:
        choices = [y for y in range(h) if y not in rows and y != projection
                   and not protected[y, z]]
        if not choices:
            raise ValueError("Cannot interrupt near-blank line")
        a[int(rng.choice(choices)), z] = textures[0]
        protected[:, z] = True

    density = float(rng.uniform(.12, .40))
    style = int(rng.integers(3))
    yy, xx = np.indices(a.shape)
    if style == 0:
        probability = np.full(a.shape, density)
    elif style == 1:
        period = int(rng.integers(2, 5))
        probability = np.where((yy // period + xx // period) % 2 == 0,
                               density * 1.6, density * .35)
    else:
        coarse = rng.random(((h + 2) // 3, (w + 2) // 3))
        probability = np.where(coarse[yy // 3, xx // 3] < .5,
                               density * 1.7, density * .4)
    fill = (rng.random(a.shape) < probability) & ~protected
    a[fill] = rng.choice(textures, size=int(fill.sum()))
    # Every nonrail row/column must be interrupted. Do not repair using the
    # protected near-blank line, source holes, or intended full blank rails.
    for axis_number, intended in ((0, rows), (1, cols)):
        length = a.shape[axis_number]
        for i in range(length):
            view = a[i, :] if axis_number == 0 else a[:, i]
            if i in intended or np.any(view != blank):
                continue
            mask = ~protected[i, :] if axis_number == 0 else ~protected[:, i]
            options = np.flatnonzero(mask)
            if not len(options):
                raise ValueError("Cannot interrupt accidental blank line")
            j = int(rng.choice(options))
            if axis_number == 0:
                a[i, j] = int(rng.choice(textures))
            else:
                a[j, i] = int(rng.choice(textures))

    parsed = _read(a)
    if (parsed[0] != blank or parsed[1] != tuple(rows) or parsed[2] != tuple(cols)
            or parsed[3] != sorted(planted)):
        raise ValueError("Texture created unintended structure")
    return {"input": a, "output": _paint(a, parsed[4])}


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    blank, rows, cols, sources, groups, hits, misses = _read(a)
    assert a.shape == out.shape
    assert THREAD not in a
    assert sources and hits
    segments = _segments(groups)
    mask = np.zeros(a.shape, dtype=bool)
    # Cellwise geometric reconstruction rather than invoking solve.
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            mask[r, c] = any((axis == "H" and r == line and lo <= c <= hi)
                            or (axis == "V" and c == line and lo <= r <= hi)
                            for axis, line, _, lo, hi in segments)
    assert np.array_equal(out != a, mask)
    assert np.all(a[mask] == blank) and np.all(out[mask] == THREAD)
    assert np.array_equal(out[~mask], a[~mask])
    assert np.array_equal(out[a != blank], a[a != blank])
    rail_mask = np.zeros(a.shape, dtype=bool)
    rail_mask[list(rows), :] = True
    rail_mask[:, list(cols)] = True
    assert np.all(~mask | rail_mask)
    assert np.any(rail_mask & ~mask)  # B alone, painting all rails, is wrong.
    spans = [s for s in segments if s[4] > s[3]]
    assert spans
    dots = a.copy()
    for _, axis, line, position in hits:
        dots[(line, position) if axis == "H" else (position, line)] = THREAD
    no_color = defaultdict(list)
    for (axis, line, _), positions in groups.items():
        no_color[(axis, line, 0)].extend(positions)
    boundary = defaultdict(list)
    for r, c, dr, dc, color in sources:
        axis = "H" if dr else "V"
        line = (a.shape[0] - 1 if dr > 0 else 0) if dr else (
            a.shape[1] - 1 if dc > 0 else 0)
        boundary[(axis, line, color)].append(c if dr else r)
    assert not np.array_equal(_paint(a, boundary), out)  # A without B.
    near = False
    for source, axis, line, position in hits:
        r, c, dr, dc, _ = source
        for z in range((r if dr else c) + (dr or dc), line, dr or dc):
            view = a[z, :] if dr else a[:, z]
            if np.count_nonzero(view != blank) == 1 and view[position] == blank:
                near = True
    closed = False
    for _, cells in _components(a, blank):
        if len(cells) == 6:
            rh = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
            cw = max(c for _, c in cells) - min(c for _, c in cells) + 1
            closed |= (rh, cw) in ((2, 3), (3, 2))
    crossing = any(hs[0] == "H" and vs[0] == "V"
                   and hs[3] <= vs[1] <= hs[4] and vs[3] <= hs[1] <= vs[4]
                   for hs in segments for vs in segments)
    repeated = defaultdict(set)
    for axis, line, color, lo, hi in spans:
        repeated[(axis, color)].add(line)
    return {"blank": blank, "rows": rows, "cols": cols, "sources": sources,
            "groups": groups, "spans": spans, "near": near, "closed": closed,
            "miss": bool(misses), "crossing": crossing,
            "dots_wrong": not np.array_equal(dots, out),
            "color_wrong": not np.array_equal(_paint(a, no_color), out),
            "parallel_spans": any(len(lines) >= 2 for lines in repeated.values())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            rec = _features(pair)
            records[split].append(rec)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert rec["near"] and rec["closed"] and rec["dots_wrong"]
            assert np.array_equal(solve(pair["input"]), pair["output"])
    train = records["train"]
    directions = {(s[2], s[3]) for r in train for s in r["sources"]}
    assert directions == {(1, 0), (-1, 0), (0, 1), (0, -1)}
    assert any(r["color_wrong"] for r in train)
    assert any(r["miss"] for r in train)
    assert any(r["crossing"] for r in train)
    assert any(any(len(p) >= 3 for p in r["groups"].values()) for r in train)
    assert any(any(len(p) == 1 for p in r["groups"].values()) for r in train)
    for r in records["test"]:
        assert r["parallel_spans"] and r["crossing"]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    profiles = ["horizontal", "crossing"]
    profiles += [str(rng.choice(["horizontal", "crossing"])) for _ in range(nt - 2)]
    rng.shuffle(profiles)
    puzzle, used = {"train": [], "test": []}, set()
    for split, kinds in (("train", profiles), ("test", ["parallel"] * ns)):
        for profile in kinds:
            for _ in range(180):
                try:
                    pair = _construct(rng, profile)
                    f = _features(pair)
                    if not (f["near"] and f["closed"] and f["dots_wrong"]):
                        continue
                    if profile == "horizontal" and not (f["color_wrong"] and f["miss"]):
                        continue
                    if profile != "horizontal" and not f["crossing"]:
                        continue
                    if profile == "parallel" and not f["parallel_spans"]:
                        continue
                    key = _key(pair["input"])
                    if key in used:
                        continue
                except (ValueError, AssertionError):
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("Bounded construction retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _boundary_fixture():
    """Hand-specified expected pixels for cases outside the sampled layouts."""
    a = np.full((14, 17), 8, dtype=np.int8)
    a[[3, 4, 10, 13], :] = 7
    a[:, [7, 14, 16]] = 7
    for top, left, name, color in ((0, 1, "D", 3), (0, 9, "D", 3),
                                   (5, 4, "U", 3), (6, 2, "R", 4),
                                   (6, 10, "L", 4), (6, 8, "R", 6),
                                   (6, 0, "L", 6), (11, 4, "D", 3)):
        shape, hole, _ = TEMPLATES[name]
        for dy, dx in shape:
            a[top + dy, left + dx] = color
        a[top + hole[0], left + hole[1]] = 7
    expected = a.copy()
    expected[3, 2:11] = THREAD
    expected[4, 5] = THREAD
    expected[7, 7] = THREAD  # two identical arrivals collapse to one dot
    expected[7, 14] = THREAD
    expected[13, 5] = THREAD  # the outer edge is a legitimate full blank rail
    assert np.array_equal(solve(a), expected)
    return a, expected


def self_test():
    """Bounded 42-seed structural/semantic checks; returns compact measurements."""
    shapes, layouts, structures, backgrounds = set(), set(), set(), set()
    directions, source_counts, spans, signatures = set(), set(), set(), set()
    features = defaultdict(int)
    equivariance_checks = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _signature(p) == _signature(generate_puzzle(seed))
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                a = pair["input"]
                f = _features(pair)
                shapes.add(a.shape)
                backgrounds.add(f["blank"])
                layouts.add((a.shape, f["rows"], f["cols"], tuple(f["sources"])))
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles))
                                               for c in a.ravel())))
                source_counts.add(len(f["sources"]))
                directions.update((s[2], s[3]) for s in f["sources"])
                spans.update((axis, hi - lo + 1) for axis, _, _, lo, hi in f["spans"])
                for name in ("near", "closed", "miss", "crossing", "color_wrong",
                             "parallel_spans"):
                    features[name] += int(f[name])
        pair = p["test"][0]
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(pair["input"], k)),
                                  np.rot90(pair["output"], k))
            equivariance_checks += 1
        assert np.array_equal(solve(np.fliplr(pair["input"])),
                              np.fliplr(pair["output"]))
        # Permute every non-red role, including blank; no color is a source ID.
        colors = [c for c in range(10) if c != THREAD]
        permutation = np.arange(10, dtype=np.int8)
        for i, color in enumerate(colors):
            permutation[color] = colors[(i + seed % 8 + 1) % 9]
        assert np.array_equal(solve(permutation[pair["input"]]),
                              permutation[pair["output"]])
        equivariance_checks += 2
    assert len(signatures) == 42 and len(shapes) >= 45
    assert len(layouts) >= 190 and len(structures) >= 190
    assert len(backgrounds) >= 8 and len(source_counts) >= 4
    assert len(spans) >= 20 and len(directions) == 4
    assert min(features.values()) >= 30
    fixture, expected = _boundary_fixture()
    for k in range(4):
        assert np.array_equal(solve(np.rot90(fixture, k)), np.rot90(expected, k))
    # Every legal explicit count combination, plus type and range failures.
    legal = 0
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ns, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns
            legal += 1
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "layouts": len(layouts),
            "normalized_structures": len(structures), "blank_colors": sorted(backgrounds),
            "source_counts": sorted(source_counts), "oriented_span_lengths": len(spans),
            "features": dict(features), "legal_count_combinations": legal,
            "equivariance_checks": equivariance_checks, "boundary_fixture_rotations": 4}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
