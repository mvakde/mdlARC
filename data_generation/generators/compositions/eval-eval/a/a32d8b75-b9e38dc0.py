"""Patterned photograms: a negative master, registered paper, and lane shadows.

Only standard-library modules and NumPy are used. No evidence files, seeds, or
previous pairs are consulted by the input-only decoder or solver.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= v <= 30 for v in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _fit_terminal(ys, sign):
    """Fit the longest terminal thin stair with two visible outward steps."""
    xs = np.arange(len(ys)) - len(ys) + 1
    candidates = []
    for period in (1, 2):
        for phase in range(period):
            predicted = int(ys[-1]) + sign * ((xs + phase) // period)
            length = 0
            for same in (predicted == ys)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1 and abs(int(ys[-length]) - int(ys[-1])) >= 2:
                candidates.append((length, -period, phase))
    assert candidates, "two readable terminal steps are required"
    length, neg_period, phase = max(candidates)
    return -neg_period, phase, int(ys[-1]), length


def _registered_blocks(shape, origin, n):
    """Complete squares strictly inside the border, with the master's phase."""
    oy, ox = origin
    for y in range(1, shape[0] - n):
        if (y - oy) % n:
            continue
        for x in range(1, shape[1] - n):
            if (x - ox) % n == 0:
                yield y, x


def _decode(input_grid):
    inp = _grid(input_grid)
    background = int(inp[0, 0])
    assert all(int(inp[y, x]) == background for y, x in
               ((0, 0), (0, -1), (-1, 0), (-1, -1)))
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        for wall in map(int, np.unique(a)):
            if wall == background:
                continue
            positions = np.argwhere(a == wall)
            rear, mouth = int(positions[:, 1].min()), int(positions[:, 1].max())
            if mouth - rear < 6 or rear < 1 or mouth >= a.shape[1] - 3:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 4 or np.any(np.diff(cap) != 1):
                continue
            rails = [np.flatnonzero(a[:, x] == wall) for x in range(rear + 1, mouth + 1)]
            if any(len(rows) != 2 or rows[1] - rows[0] < 3 for rows in rails):
                continue
            top, bottom = np.array(rails).T
            if cap[0] != top[0] or cap[-1] != bottom[0]:
                continue
            if np.any(np.abs(np.diff(top)) > 1) or np.any(np.abs(np.diff(bottom)) > 1):
                continue
            try:
                fits = (_fit_terminal(top, -1), _fit_terminal(bottom, 1))
            except AssertionError:
                continue
            chamber = np.zeros(a.shape, bool)
            for j, x in enumerate(range(rear + 1, mouth + 1)):
                chamber[top[j] + 1:bottom[j], x] = True
            patch = chamber & (a != background) & (a != wall)
            coords = np.argwhere(patch)
            if not len(coords):
                continue
            lo, hi = coords.min(0), coords.max(0) + 1
            n = int(hi[0] - lo[0])
            if not (2 <= n <= 4 and hi[1] - lo[1] == n and len(coords) == n * n):
                continue
            tile = a[lo[0]:hi[0], lo[1]:hi[1]].copy()
            colors = tuple(map(int, np.unique(tile)))
            if len(colors) != 2 or any(np.count_nonzero(a == c) != np.count_nonzero(tile == c) for c in colors):
                continue
            if not np.all(patch[lo[0]:hi[0], lo[1]:hi[1]]):
                continue
            origin = tuple(map(int, lo))
            blocks = [(y, x) for y, x in _registered_blocks(a.shape, origin, n) if x > mouth]
            paper_candidates = []
            for color in map(int, np.unique(a)):
                if color in (background, wall, *colors):
                    continue
                intact = sum(np.all(a[y:y+n, x:x+n] == color) for y, x in blocks)
                if intact >= 2:
                    paper_candidates.append(color)
            if len(paper_candidates) != 1:
                continue
            paper = paper_candidates[0]
            paper_mask = a == paper
            if np.any(paper_mask[:, :mouth + 1]):
                continue
            extras = ~np.isin(a, (background, wall, *colors, paper))
            if np.any(extras[:, :mouth + 1]):
                continue
            material_blocks = []
            recovered_material = np.zeros(a.shape, bool)
            for y, x in blocks:
                s = np.s_[y:y+n, x:x+n]
                if paper_mask[s].any():
                    if not np.all(paper_mask[s] | extras[s]):
                        break
                    recovered_material[s] = True
                    material_blocks.append((y, x))
            else:
                if np.any(paper_mask & ~recovered_material):
                    continue
                # Only terminal flanks are extrapolated. The actual reservoir
                # walls, including any neck, define the visible chamber.
                footprint = chamber.copy()
                rows = np.arange(a.shape[0])
                for x in range(mouth + 1, a.shape[1]):
                    limits = []
                    for fit, sign in zip(fits, (-1, 1)):
                        period, phase, last, _ = fit
                        limits.append(last + sign * ((x - mouth + phase) // period))
                    footprint[:, x] = (rows > limits[0]) & (rows < limits[1])
                matches.append(dict(a=a, rotation=rotation, background=background,
                                    wall=wall, paper=paper, tile=tile, origin=origin,
                                    n=n, rear=rear, mouth=mouth, top=top, bottom=bottom,
                                    fits=fits, footprint=footprint, extras=extras,
                                    blocks=material_blocks, material=recovered_material))
    assert len(matches) == 1, "input must identify exactly one master lamp and paper role"
    return matches[0]


def _predict_decoded(d, behavior="normal"):
    """Lane scan, separate from construction's union-of-shadows arithmetic."""
    a, n, tile = d["a"], d["n"], d["tile"]
    oy, ox = d["origin"]
    colors = sorted(map(int, np.unique(tile)))
    out = a.copy()
    for y in range(a.shape[0]):
        blocked = False
        for x in range(d["mouth"] + 1, a.shape[1]):
            inside = bool(d["footprint"][y, x])
            if inside and d["extras"][y, x] and behavior != "no_shadows":
                blocked = True
            if a[y, x] != d["paper"]:
                continue
            if behavior == "no_beam" or (inside and not blocked):
                source = tile[(y - oy) % n, (x - ox) % n]
                if behavior == "rotate_tile":
                    source = np.rot90(tile)[(y - oy) % n, (x - ox) % n]
                out[y, x] = (source if behavior == "no_exchange" else
                             colors[0] if behavior == "solid_ink" else
                             colors[0] + colors[1] - source)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Develop the registered paper using only visible input evidence."""
    d = _decode(input_grid)
    return np.rot90(_predict_decoded(d), d["rotation"]).copy()


def _analysis(input_grid):
    d = _decode(input_grid)
    a = d["a"]
    effective = d["extras"] & d["footprint"]
    shadow = np.zeros(a.shape, bool)
    for y, x in np.argwhere(effective):
        shadow[y, x + 1:] = True
    paper = a == d["paper"]
    exposed = paper & d["footprint"] & ~shadow
    shaded = paper & d["footprint"] & shadow
    partial = complete = partial_shadow = 0
    n = d["n"]
    for y, x in d["blocks"]:
        s = np.s_[y:y+n, x:x+n]
        count = int(exposed[s].sum())
        complete += count == n * n
        partial += count > 0 and np.any(paper[s] & ~exposed[s])
        partial_shadow += count > 0 and shaded[s].any()
    by, bx = np.nonzero(d["material"])
    box = np.zeros(a.shape, bool)
    box[by.min():by.max()+1, bx.min():bx.max()+1] = True
    d.update(exposed=exposed, shadow=shadow, shaded=shaded,
             features=dict(n=n, rotation=d["rotation"], shape=tuple(a.shape),
                           projection=a.shape[1] - d["mouth"] - 1,
                           palette=(d["background"], d["wall"], d["paper"], *sorted(map(int, np.unique(d["tile"])))),
                           blocks=len(d["blocks"]), complete=complete, partial=partial,
                           partial_shadow=partial_shadow, exposed=int(exposed.sum()),
                           shaded=int(shaded.sum()), outside=int((paper & ~d["footprint"]).sum()),
                           effective=int(effective.sum()), inert=int((d["extras"] & ~d["footprint"]).sum()),
                           blocker_colors=len(np.unique(a[effective])),
                           repeated_lane=bool(np.any(effective.sum(axis=1) > 1)),
                           grouped=bool(np.any(effective[1:] & effective[:-1]) or np.any(effective[:, 1:] & effective[:, :-1])),
                           holes=int((box & ~d["material"] & d["footprint"] & (a == d["background"])).sum()),
                           runs=tuple(f[0] for f in d["fits"]),
                           outline=(tuple(d["top"] - d["top"][0]), tuple(d["bottom"] - d["top"][0])),
                           stencil=tuple((int((y-d["origin"][0])//n), int((x-d["origin"][1])//n)) for y, x in d["blocks"])))
    return d


def _sample_tile(rng, n, colors):
    for _ in range(32):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        if bits.any() and not bits.all() and all(not np.array_equal(bits, np.rot90(bits, k)) for k in (1, 2, 3)):
            return np.where(bits, colors[0], colors[1]).astype(int)
    bits = np.zeros((n, n), bool)
    bits[0, -1] = True
    return np.where(bits, colors[0], colors[1]).astype(int)


def _construct(rng, n, rotation, blocked, projection, teaching=False):
    """Construct geometry and a forward label; return no latent solver state."""
    palette = list(map(int, rng.permutation(10)))
    bg, wall, paper, ca, cb, *obstacle_colors = palette
    periods = [int(rng.integers(1, 3)), int(rng.integers(1, 3))]
    phases = [int(rng.integers(p)) for p in periods]
    flare = 2 * max(periods) + 1
    neck = int(rng.integers(1, 4))
    # The reservoir has a complete master and may narrow before its flare.
    upper = [0] * n
    lower = [n + 3] * n
    for j in range(neck):
        upper.append(min(j + 1, 1))
        lower.append(n + 3 - min(j + 1, int(rng.integers(0, 2))))
    for j in range(1, flare + 1):
        upper.append(upper[n + neck - 1] - (j + phases[0]) // periods[0])
        lower.append(lower[n + neck - 1] + (j + phases[1]) // periods[1])
    upper, lower = np.array(upper), np.array(lower)
    rear = int(rng.integers(1, 4))
    mouth = rear + len(upper)
    projection = min(projection, 29 - mouth)
    assert projection >= 7
    w = mouth + 1 + projection
    span = int(lower.max() - upper.min() + 1)
    h = int(rng.integers(max(18, span + 7), 31))
    offset = int(rng.integers(2 - int(upper.min()), h - int(lower.max()) - 1))
    upper += offset
    lower += offset
    a = np.full((h, w), bg, int)
    a[upper[0]:lower[0] + 1, rear] = wall
    footprint = np.zeros(a.shape, bool)
    for j, x in enumerate(range(rear + 1, mouth + 1)):
        a[upper[j], x] = a[lower[j], x] = wall
        footprint[upper[j] + 1:lower[j], x] = True
    for x in range(mouth + 1, w):
        t = flare + x - mouth
        top = upper[n+neck-1] - (t + phases[0]) // periods[0]
        bottom = lower[n+neck-1] + (t + phases[1]) // periods[1]
        footprint[max(0, top+1):min(h, bottom), x] = True
    origin = (offset + 2, rear + 1)
    tile = _sample_tile(rng, n, (ca, cb))
    oy, ox = origin
    a[oy:oy+n, ox:ox+n] = tile
    blocks = [(y, x) for y, x in _registered_blocks(a.shape, origin, n) if x > mouth]
    assert len(blocks) >= 5
    full = [(y, x) for y, x in blocks if footprint[y:y+n, x:x+n].all()]
    partial = [(y, x) for y, x in blocks if 0 < footprint[y:y+n, x:x+n].sum() < n*n]
    outside = [(y, x) for y, x in blocks if not footprint[y:y+n, x:x+n].any()]
    assert full and partial
    style = int(rng.integers(5))
    rows = sorted({y for y, _ in blocks})
    cols = sorted({x for _, x in blocks})
    chosen = set()
    density = rng.uniform(.35, .8)
    for y, x in blocks:
        iy, ix = rows.index(y), cols.index(x)
        active = (rng.random() < density if style == 0 else
                  iy in (0, len(rows)-1) or ix in (0, len(cols)-1) if style == 1 else
                  (iy + ix) % 2 == 0 if style == 2 else
                  ix == iy % len(cols) or rng.random() < .2 if style == 3 else
                  iy % 2 == 0 or ix == len(cols)//2)
        if active:
            chosen.add((y, x))
    chosen.add(full[int(rng.integers(len(full)))])
    chosen.add(partial[int(rng.integers(len(partial)))])
    if outside:
        chosen.add(outside[int(rng.integers(len(outside)))])
    if blocked:
        # Place two colors in separate physical lanes and retain registered
        # paper downstream. At least one other row of each target stays lit.
        pairs = [(y, x) for y, x in full if (y, x+n) in blocks]
        assert pairs
        sy, sx = pairs[int(rng.integers(len(pairs)))]
        chosen.update(((sy, sx), (sy, sx+n)))
    for y, x in chosen:
        a[y:y+n, x:x+n] = paper
    if blocked:
        lanes = [sy, sy + n - 1] if n > 2 else [sy]
        for i, y in enumerate(lanes):
            a[y, sx] = obstacle_colors[i]
        if n == 2:
            options = [(y, x) for y, x in full if y != sy and (y, x+n) in blocks]
            assert options
            y2, x2 = options[int(rng.integers(len(options)))]
            a[y2:y2+n, x2:x2+n] = paper
            a[y2:y2+n, x2+n:x2+2*n] = paper
            a[y2, x2] = obstacle_colors[1]
        if not teaching:
            for i in range(int(rng.integers(1, 5))):
                # Groups can be on paper or in empty space. No full foreign
                # square is allowed to impersonate a second material role.
                dy_dx = (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1)),
                         ((0, 0), (1, 0), (1, 1)))[int(rng.integers(4))]
                y = int(rng.integers(1, h-2))
                x = int(rng.integers(mouth+1, w-2))
                points = [(y+dy, x+dx) for dy, dx in dy_dx]
                if all(a[yy, xx] in (bg, paper) for yy, xx in points):
                    for yy, xx in points:
                        a[yy, xx] = obstacle_colors[(i+2) % len(obstacle_colors)]
        # An exterior object demonstrates that opacity is footprint-local.
        options = np.argwhere(~footprint & (a == bg) & (np.indices(a.shape)[1] > mouth))
        options = [(int(y), int(x)) for y, x in options if 0 < y < h-1 and x < w-1]
        if options:
            y, x = options[int(rng.integers(len(options)))]
            a[y, x] = obstacle_colors[0]
    extras = ~np.isin(a, (bg, wall, paper, ca, cb))
    shadow = np.zeros(a.shape, bool)
    for y, x in np.argwhere(extras & footprint):
        shadow[y, x+1:] = True
    rr, cc = np.indices(a.shape)
    negative = ca + cb - tile[(rr-oy) % n, (cc-ox) % n]
    active = (a == paper) & footprint & ~shadow
    out = a.copy()
    out[active] = negative[active]
    return {"input": np.rot90(a, rotation).copy(), "output": np.rot90(out, rotation).copy()}


def _integer(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 2, 10, "num_train")
    nq = _integer(num_test, 1, 4, "num_test")
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    preferred = int(rng.integers(4))
    orientation_mode = int(rng.integers(3))
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = split == "train" and i < 2
            n = i + 2 if witness else int(rng.integers(2, 5))
            rotation = (preferred if orientation_mode == 0 else
                        (preferred + (i if split == "train" else i+2)) % 4 if orientation_mode == 1 else
                        int(rng.integers(4)))
            blocked = (i == 1 if witness else bool(rng.random() < .8))
            # Tests extend farther than all demonstrations, without changing
            # the visible cadence, registration or optical behavior.
            projection = int(rng.integers(max(8, 2*n+2 if blocked else 8), 12)
                             if split == "train" else rng.integers(13, 17))
            for attempt in range(96):
                try:
                    pair = _construct(rng, n, rotation, blocked, projection, teaching=witness)
                    d = _analysis(pair["input"])
                    f = d["features"]
                    assert f["exposed"] > 0 and f["complete"] >= 1 and f["partial"] >= 1
                    assert f["outside"] > 0 and f["holes"] > 0
                    if blocked:
                        assert f["shaded"] > 0 and f["partial_shadow"] > 0 and f["blocker_colors"] >= 2
                    if not blocked:
                        assert f["effective"] == 0
                    assert np.array_equal(np.rot90(_predict_decoded(d), rotation), pair["output"])
                    key = _key(pair["input"])
                    if key in seen:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except (AssertionError, ValueError):
                    continue
            else:
                raise RuntimeError(f"bounded sampling exhausted for {split}/{i}, tile {n}")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    features, seen = {"train": [], "test": []}, set()
    alternatives = {k: False for k in ("no_exchange", "solid_ink", "no_beam", "no_shadows", "rotate_tile")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _analysis(inp)
            predicted = _predict_decoded(d)
            canonical_out = np.rot90(out, -d["rotation"])
            assert np.array_equal(predicted, canonical_out), "incorrect developed image"
            paper = d["a"] == d["paper"]
            changed = d["a"] != canonical_out
            assert np.array_equal(changed, d["exposed"])
            assert np.array_equal(canonical_out[~paper], d["a"][~paper]), "non-paper material changed"
            assert set(map(int, np.unique(canonical_out[changed]))) <= set(map(int, np.unique(d["tile"])))
            f = d["features"]
            assert f["complete"] > 0 and f["partial"] > 0 and f["outside"] > 0 and f["holes"] > 0
            features[split].append(f)
            if split == "train":
                for behavior in alternatives:
                    alternatives[behavior] |= not np.array_equal(_predict_decoded(d, behavior), canonical_out)
    assert all(alternatives.values()), "training does not distinguish a specified shortcut"
    assert any(f["effective"] == 0 for f in features["train"])
    assert any(f["blocker_colors"] >= 2 and f["partial_shadow"] > 0 for f in features["train"])
    assert {2, 3} <= {f["n"] for f in features["train"]}
    assert min(f["projection"] for f in features["test"]) > max(f["projection"] for f in features["train"])


def self_test():
    """Bounded: stream 42 seeds; retain hashes and feature sets, never grids."""
    signatures, shapes, outlines, stencils, palettes = set(), set(), set(), set(), set()
    directions = {s: set() for s in ("train", "test")}
    tile_sizes, cadences = set(), set()
    counters = dict(pairs=0, partial_shadow=0, grouped=0, inert=0, repeated_lane=0)
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                d = _analysis(pair["input"])
                f = d["features"]
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                digest.update(pair["input"].astype(np.uint8).tobytes())
                shapes.add(pair["input"].shape)
                outlines.add(f["outline"])
                stencils.add((f["n"], f["stencil"]))
                palettes.add(f["palette"])
                directions[split].add(f["rotation"])
                tile_sizes.add(f["n"])
                cadences.update(f["runs"])
                counters["pairs"] += 1
                for name in ("partial_shadow", "grouped", "inert", "repeated_lane"):
                    counters[name] += bool(f[name])
        signatures.add(digest.hexdigest())
        if seed in (0, 13, 41):
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    assert len(signatures) == 42 and len(shapes) >= 35
    assert len(outlines) >= 40 and len(stencils) >= 100 and len(palettes) >= 100
    assert tile_sizes == {2, 3, 4} and cadences == {1, 2}
    assert all(v == {0, 1, 2, 3} for v in directions.values())
    assert counters["partial_shadow"] >= 42 and counters["inert"] >= 42
    assert counters["grouped"] > 0 and counters["repeated_lane"] > 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1) + 1):
            validate_generated_puzzle(generate_puzzle(883573567, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    damaged = generate_puzzle(7, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), shapes=len(shapes),
                outlines=len(outlines), stencils=len(stencils), palettes=len(palettes), **counters)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
