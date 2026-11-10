"""Reactive scaffold: a multiscale color field dissolves falling rigid rods.

Only NumPy and the standard library are used. All solving conventions are
visible in the input; no puzzle seed or training examples enter solve().
"""
from __future__ import annotations

import numpy as np


# Rod tuple: (top row, left column, height, width, color), chamber coordinates.
def _cells(rod):
    r, c, h, w, _ = rod
    return frozenset((y, x) for y in range(r, r + h)
                     for x in range(c, c + w))


def _down(rod, distance=1):
    return (rod[0] + distance, *rod[1:])


def _moving(rods, height):
    """Greatest set able to move together; one retained support is enough."""
    owners = {cell: i for i, p in enumerate(rods) for cell in _cells(p)}
    blockers = [{owners[cell] for cell in _cells(_down(p)) if cell in owners}
                for p in rods]
    moving = {i for i, p in enumerate(rods) if p[0] + p[2] < height}
    while True:
        blocked = {i for i in moving if blockers[i] - moving}
        if not blocked:
            return moving
        moving -= blocked


def _settle(rods, height):
    rods = list(rods)
    for _ in range(height * (len(rods) + 1)):
        moving = _moving(rods, height)
        if not moving:
            return rods
        rods = [_down(p) if i in moving else p for i, p in enumerate(rods)]
    raise AssertionError("gravity did not converge")


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not np.issubdtype(array.dtype, np.integer)
            or not all(1 <= k <= 30 for k in array.shape)
            or np.any((array < 0) | (array > 9))):
        raise ValueError("expected an integer ARC grid")
    return array.astype(np.int64, copy=False)


def _parse_rods(scene, background):
    """Same-color four-connected components are unambiguous straight rods."""
    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(scene != background))}
    rods = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        color = int(scene[start])
        todo, component = [start], {start}
        while todo:
            r, c = todo.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and int(scene[q]) == color:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        rr, cc = zip(*component)
        r, c = min(rr), min(cc)
        h, w = max(rr) - r + 1, max(cc) - c + 1
        if len(component) < 2 or min(h, w) != 1 or h * w != len(component):
            raise ValueError("each same-color component must be a straight rod")
        rods.append((r, c, h, w, color))
    if not rods:
        raise ValueError("empty scaffold")
    return sorted(rods)


def _decode(input_grid):
    grid = _grid(input_grid)
    # The top right is explicitly unused padding, hence a background swatch.
    bg = int(grid[0, -1])
    blanks = np.flatnonzero(grid[0] == bg)
    if not len(blanks):
        raise ValueError("missing sample separator")
    n = int(blanks[0])
    if not 2 <= n <= 6 or grid.shape[1] < 2 * n + 2 or grid.shape[0] < n + 3:
        raise ValueError("invalid sample header")
    carrier = grid[:n, :n]
    stencil = grid[:n, n + 1:2 * n + 1]
    if (np.any(carrier == bg) or np.any(grid[:n, n] != bg)
            or np.any(grid[:n, 2 * n + 1:] != bg)
            or np.any(grid[n] != bg)):
        raise ValueError("invalid sample padding")
    scene = grid[n + 1:]
    height, width = scene.shape
    # Integral magnification makes replication visually exact on both axes.
    if height < n or height % n or width % n:
        raise ValueError("chamber dimensions must be multiples of sample size")
    rods = _parse_rods(scene, bg)
    if _settle(rods, height) != rods:
        raise ValueError("input scaffold must already be gravity stable")
    return grid, bg, n, carrier, stencil, scene, rods


def _field(carrier, stencil, shape, background, mode="correct"):
    height, width = shape
    n = len(carrier)
    rr, cc = np.arange(height)[:, None], np.arange(width)[None, :]
    sr, sc = rr * n // height, cc * n // width
    base = carrier[sr, sc]
    tile = stencil[rr % n, cc % n]
    if mode == "carrier_tile_rows":
        base = carrier[rr % n, sc]
    elif mode == "carrier_tile_columns":
        base = carrier[sr, cc % n]
    elif mode == "stencil_scaled":
        tile = stencil[sr, sc]
    elif mode == "phase_rows":
        tile = stencil[(rr + 1) % n, cc % n]
    elif mode == "phase_columns":
        tile = stencil[rr % n, (cc + 1) % n]
    elif mode == "no_stencil":
        return base.copy()
    elif mode == "no_carrier":
        return tile.copy()
    elif mode == "opaque_background":
        return tile.copy()
    elif mode == "fuse_then_scale":
        fused = np.where(stencil == background, carrier, stencil)
        return fused[sr, sc]
    elif mode != "correct":
        raise ValueError("unknown field mode")
    return np.where(tile == background, base, tile).astype(np.int64)


def _hit(rod, field):
    r, c, h, w, color = rod
    return bool(np.any(field[r:r + h, c:c + w] == color))


def _react(rods, field, mode="correct"):
    """React before every simultaneous one-cell gravity step, including rest."""
    height = field.shape[0]
    states = list(enumerate(rods))
    trace = {"deleted": [], "steps": 0, "moved": set(), "delayed": set()}
    if mode == "no_reaction":
        final = _settle(rods, height)
        return list(enumerate(final)), trace
    if mode == "global_color":
        colors = set(int(x) for x in np.unique(field))
        ids = [i for i, p in states if p[4] not in colors]
        final = _settle([rods[i] for i in ids], height)
        return list(zip(ids, final)), trace
    if mode == "one_pass":
        states = [(i, p) for i, p in states if not _hit(p, field)]
        return list(zip([i for i, _ in states],
                        _settle([p for _, p in states], height))), trace
    if mode == "no_gravity":
        return [(i, p) for i, p in states if not _hit(p, field)], trace
    for tick in range(height * (len(rods) + 1) + len(rods) + 1):
        removed = [(i, p) for i, p in states if _hit(p, field)]
        if removed:
            trace["deleted"].append((tick, tuple(i for i, _ in removed)))
            trace["delayed"].update(i for i, p in removed if p[0] > rods[i][0])
            dead = {i for i, _ in removed}
            states = [(i, p) for i, p in states if i not in dead]
        current = [p for _, p in states]
        moving = _moving(current, height)
        if not moving:
            return states, trace
        if mode == "land_then_react":
            settled = _settle(current, height)
            states = [(i, p) for (i, _), p in zip(states, settled)]
        else:
            states = [(i, _down(p) if j in moving else p)
                      for j, (i, p) in enumerate(states)]
        trace["moved"].update(states[j][0] for j in moving)
        trace["steps"] += 1
    raise AssertionError("reaction did not terminate")


def _render(rods, shape, background):
    grid = np.full(shape, background, dtype=np.int64)
    occupied = set()
    for p in rods:
        cells = _cells(p)
        assert not occupied & cells
        occupied.update(cells)
        for q in cells:
            grid[q] = p[4]
    return grid


def _paint(states, field):
    result = field.copy()
    for _, p in states:
        for q in _cells(p):
            result[q] = p[4]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution, with no learned palette or generation metadata."""
    grid, bg, n, carrier, stencil, scene, rods = _decode(input_grid)
    field = _field(carrier, stencil, scene.shape, bg)
    final, _ = _react(rods, field)
    output = grid.copy()
    output[n + 1:] = _paint(final, field)
    return output


_FIELD_ALTERNATIVES = ("carrier_tile_rows", "carrier_tile_columns",
                       "stencil_scaled", "no_stencil", "no_carrier",
                       "fuse_then_scale", "phase_rows", "phase_columns")
_PHYSICS_ALTERNATIVES = ("no_reaction", "no_gravity", "global_color",
                         "one_pass", "land_then_react")
# Each field operation must affect actual final rod identity or position,
# not merely the newly painted backdrop.
_REQUIRED = frozenset(_FIELD_ALTERNATIVES + _PHYSICS_ALTERNATIVES
                      + ("rigidity", "whole_rod", "surviving_hazard_color"))


def _state_key(states):
    return tuple(sorted(states))


def _pixel_alternative(rods, field):
    """Deliberate shortcut: keep whole-rod reactions but use granular gravity."""
    cells = [(i, (r, c, 1, 1, p[4])) for i, p in enumerate(rods)
             for r, c in _cells(p)]
    dead = {i for i, p in enumerate(rods) if _hit(p, field)}
    cells = [(i, p) for i, p in cells if i not in dead]
    for _ in range(field.shape[0] * (len(cells) + 1) + 1):
        dead = {i for i, p in cells if _hit(p, field)}
        cells = [(i, p) for i, p in cells if i not in dead]
        moving = _moving([p for _, p in cells], field.shape[0])
        if not moving:
            return _paint(cells, field)
        cells = [(i, _down(p) if j in moving else p)
                 for j, (i, p) in enumerate(cells)]
    raise AssertionError("pixel shortcut did not terminate")


def _erosion_alternative(rods, field):
    """Erase matching cells only; each original rod's remainder stays rigid.

    This changes the deletion rule without also changing gravity to pixels.
    A perforated/disconnected remainder retains its original piece identity.
    """
    states = [(p[4], _cells(p)) for p in rods]
    for _ in range(field.shape[0] * (sum(len(s) for _, s in states) + 1) + 1):
        states = [(color, frozenset(q for q in cells if field[q] != color))
                  for color, cells in states]
        states = [(color, cells) for color, cells in states if cells]
        owners = {q: i for i, (_, cells) in enumerate(states) for q in cells}
        targets = [frozenset((r + 1, c) for r, c in cells) for _, cells in states]
        moving = {i for i, cells in enumerate(targets)
                  if all(r < len(field) for r, _ in cells)}
        while True:
            blocked = {i for i in moving for q in targets[i]
                       if q in owners and owners[q] not in moving}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            out = field.copy()
            for color, cells in states:
                for q in cells:
                    out[q] = color
            return out
        states = [(color, targets[i] if i in moving else cells)
                  for i, (color, cells) in enumerate(states)]
    raise AssertionError("erosion shortcut did not terminate")


def _evidence(input_grid, full=False):
    _, bg, n, carrier, stencil, scene, rods = _decode(input_grid)
    field = _field(carrier, stencil, scene.shape, bg)
    final, trace = _react(rods, field)
    key = _state_key(final)
    correct = _paint(final, field)
    cover = set()
    for name in _FIELD_ALTERNATIVES:
        alternate_field = _field(carrier, stencil, scene.shape, bg, name)
        alternate, _ = _react(rods, alternate_field)
        if (key != _state_key(alternate)
                and not np.array_equal(correct, _paint(alternate, alternate_field))):
            cover.add(name)
    for name in _PHYSICS_ALTERNATIVES:
        alternate, _ = _react(rods, field, name)
        if key != _state_key(alternate) and not np.array_equal(correct, _paint(alternate, field)):
            cover.add(name)
    if not np.array_equal(correct, _pixel_alternative(rods, field)):
        cover.add("rigidity")
    if not np.array_equal(correct, _erosion_alternative(rods, field)):
        cover.add("whole_rod")
    hazard_colors = set(int(x) for x in np.unique(field))
    if any(p[4] in hazard_colors for _, p in final):
        cover.add("surviving_hazard_color")
    if not full:
        return cover
    survivors = {i: p for i, p in final}
    distances = [p[0] - rods[i][0] for i, p in final]
    edges = [(i, j) for i, p in enumerate(rods) for j, q in enumerate(rods)
             if i != j and _cells(_down(p)) & _cells(q)]
    return cover, dict(n=n, shape=scene.shape, background=bg, rods=len(rods),
                      deleted=sum(len(ids) for _, ids in trace["deleted"]),
                      delayed=len(trace["delayed"]),
                      survivors=len(final), steps=trace["steps"],
                      distances=tuple(sorted(set(distances))),
                      redundant=any(sum(i == k for i, j in edges) > 1
                                    for k in range(len(rods))),
                      field_inks=len(hazard_colors),
                      carrier_inks=len(np.unique(carrier)),
                      stencil_inks=len(np.unique(stencil[stencil != bg])),
                      occupancy=float(np.mean(scene != bg)),
                      reactive_survivor=any(p[4] in hazard_colors for p in survivors.values()),
                      horizontal=sum(p[2] == 1 for p in rods),
                      vertical=sum(p[3] == 1 for p in rods),
                      geometry=tuple(p[:4] for p in rods))


def _geometry(rng, height, width, count):
    """Drop randomly positioned beams and posts to make an actual support DAG."""
    rods = []
    for _ in range(count * 7):
        if len(rods) >= count:
            break
        horizontal = bool(rng.random() < .6)
        length = int(rng.integers(2, min(width if horizontal else height - 1,
                                        10 if horizontal else 7) + 1))
        h, w = (1, length) if horizontal else (length, 1)
        c = int(rng.integers(width - w + 1))
        p = (0, c, h, w, 1)
        if any(_cells(p) & _cells(q) for q in rods):
            continue
        for d in range(height):
            target = _down(p, d + 1)
            if target[0] + h > height or any(_cells(target) & _cells(q) for q in rods):
                rods.append(_down(p, d))
                break
    return rods


def _color_rods(rng, rods, palette):
    result = []
    for p in rods:
        neighbors = _cells(p) | {(r + dr, c + dc) for r, c in _cells(p)
                                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        forbidden = {q[4] for q in result if neighbors & _cells(q)}
        choices = [c for c in palette if c not in forbidden]
        if not choices:
            return None
        # Reactive and inert colors are mixed, independent of orientation/role.
        result.append((*p[:4], int(rng.choice(choices))))
    return result


def _patterns(rng, n, hazard, background):
    rr, cc = np.indices((n, n))
    kind = int(rng.integers(5))
    if kind == 0:
        labels = rng.integers(len(hazard), size=(n, n))
    elif kind == 1:
        labels = (rr + int(rng.integers(1, 4)) * cc) % len(hazard)
    elif kind == 2:
        labels = rng.integers(len(hazard), size=(n, 1)) + np.zeros((n, n), int)
    elif kind == 3:
        labels = rng.integers(len(hazard), size=(1, n)) + np.zeros((n, n), int)
    else:
        cuts = [int(rng.integers(1, n)), int(rng.integers(1, n))]
        labels = rng.integers(len(hazard), size=(2, 2))[(rr >= cuts[0]).astype(int),
                                                        (cc >= cuts[1]).astype(int)]
    carrier = np.asarray(hazard)[labels]
    mask_kind = int(rng.integers(5))
    if mask_kind == 0:
        mask = rng.random((n, n)) < float(rng.uniform(.12, .55))
    elif mask_kind == 1:
        mask = (rr + cc + int(rng.integers(2))) % 2 == 0
    elif mask_kind == 2:
        mask = rr == int(rng.integers(n))
    elif mask_kind == 3:
        mask = cc == int(rng.integers(n))
    else:
        mask = (rr == 0) | (cc == 0) | (rr == n - 1) | (cc == n - 1)
        mask &= rng.random((n, n)) < .7
    stencil = np.full((n, n), background, int)
    stencil[mask] = rng.choice(hazard, size=int(mask.sum()))
    if rng.random() < .08:
        stencil[:] = background
    if rng.random() < .5:
        carrier = np.rot90(carrier, int(rng.integers(4))).copy()
    return carrier, stencil


def _sample(rng, background, palette, test=False, sizes=None):
    n = int(rng.choice([2, 3, 4, 5, 6] if sizes is None else sizes))
    max_h = min(5, (29 - n) // n)
    kh = int(rng.integers(2, max_h + 1))
    kw = int(rng.integers(3, min(6, 30 // n) + 1))
    height, width = n * kh, n * kw
    # Test recombination adds more rods, not new physics or conventions.
    count = int(rng.integers(7 if test else 5, 16 if test else 12))
    geometry = _geometry(rng, height, width, count)
    rods = _color_rods(rng, geometry, palette)
    if rods is None or len(rods) < 3:
        return None
    hazard = palette[:int(rng.integers(2, min(4, len(palette) - 1) + 1))]
    carrier, stencil = _patterns(rng, n, hazard, background)
    scene = _render(rods, (height, width), background)
    try:
        parsed = _parse_rods(scene, background)
    except ValueError:
        return None
    if set(parsed) != set(rods):
        return None
    grid = np.full((n + 1 + height, width), background, int)
    grid[:n, :n], grid[:n, n + 1:2 * n + 1] = carrier, stencil
    grid[n + 1:] = scene
    field = _field(carrier, stencil, scene.shape, background)
    final, trace = _react(parsed, field)
    if not final or not trace["deleted"] or not trace["moved"]:
        return None
    return {"input": grid, "output": solve(grid)}


def _key(grid):
    array = np.asarray(grid, dtype=np.uint8)
    return array.shape, array.tobytes()


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    background = int(rng.integers(10))
    palette = [int(x) for x in rng.permutation([c for c in range(10) if c != background])
               [:int(rng.integers(5, 8))]]
    candidates, covers = [], []
    train = None
    # A bounded search chooses two genuinely informative examples. Geometry,
    # field patterns, and palettes are random; no fixed scaffold template.
    for _ in range(900):
        pair = _sample(rng, background, palette)
        if pair is None:
            continue
        cover = _evidence(pair["input"])
        for j in range(len(candidates)):
            other = candidates[j]
            if (_REQUIRED <= cover | covers[j]
                    and other["input"].shape != pair["input"].shape
                    and _decode(other["input"])[2] != _decode(pair["input"])[2]):
                train = [other, pair]
                break
        if train is not None:
            break
        # Retain the strongest, complementary candidates; bound memory/work.
        candidates.append(pair)
        covers.append(cover)
        if len(candidates) > 60:
            weakest = min(range(len(candidates)), key=lambda i: len(covers[i]))
            candidates.pop(weakest)
            covers.pop(weakest)
    if train is None:
        raise RuntimeError("bounded candidate search lacked teaching coverage")
    puzzle = {"train": train, "test": []}
    seen = {_key(p["input"]) for p in train}
    for split, total in (("train", num_train), ("test", num_test)):
        if split == "test":
            train_sizes = {_decode(p["input"])[2] for p in puzzle["train"]}
            train_max_rods = max(len(_decode(p["input"])[-1]) for p in puzzle["train"])
            novel_sizes = sorted(set(range(2, 7)) - train_sizes)
            # Visible new sample size or familiar size with denser geometry.
            test_sizes = novel_sizes if novel_sizes and rng.random() < .5 else sorted(train_sizes)
            require_larger = set(test_sizes) <= train_sizes
        while len(puzzle[split]) < total:
            for _ in range(500):
                pair = _sample(rng, background, palette, test=split == "test",
                               sizes=test_sizes if split == "test" else None)
                if (split == "test" and require_larger and pair is not None
                        and len(_decode(pair["input"])[-1]) <= train_max_rods):
                    continue
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("bounded search lacked unique examples")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, coverage, backgrounds, sizes = set(), set(), set(), set()
    counts = {"train": [], "test": []}
    test_sizes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            _, bg, n, carrier, stencil, scene, rods = _decode(inp)
            backgrounds.add(bg)
            counts[split].append(len(rods))
            assert np.array_equal(inp[:n + 1], out[:n + 1])
            field = _field(carrier, stencil, scene.shape, bg)
            final, trace = _react(rods, field)
            assert final and trace["deleted"] and trace["moved"]
            assert _moving([p for _, p in final], len(scene)) == set()
            assert all(not _hit(p, field) for _, p in final)
            occupied = set()
            for i, p in final:
                assert p[1:] == rods[i][1:] and p[0] >= rods[i][0]
                assert not occupied & _cells(p)
                occupied |= _cells(p)
            assert np.array_equal(out[n + 1:], _paint(final, field))
            assert np.array_equal(solve(inp), out)
            if split == "train":
                coverage |= _evidence(inp)
                sizes.add(n)
            else:
                test_sizes.add(n)
    assert len(backgrounds) == 1
    assert len(sizes) >= 2
    assert test_sizes - sizes or min(counts["test"]) > max(counts["train"]), "missing train/test shift"
    assert _REQUIRED <= coverage, "missing teaching evidence: " + str(_REQUIRED - coverage)


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded deterministic 42-seed audit, including family-specific measures."""
    _mechanics_test()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(generate_puzzle(928149253)) == _fingerprint(generate_puzzle(928149253))
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            _, features = _evidence(pair["input"], full=True)
            records.append(features)
    assert {2, 3, 4, 5, 6} <= {f["n"] for f in records}
    assert len({f["background"] for f in records}) >= 8
    assert len({f["shape"] for f in records}) >= 25
    assert len({f["geometry"] for f in records}) >= 150
    assert len({f["rods"] for f in records}) >= 7
    assert sum(f["delayed"] > 0 for f in records) >= 42
    assert sum(len(set(f["distances"]) - {0}) > 1 for f in records) >= 10
    assert sum(f["redundant"] for f in records) >= 10
    assert any(f["reactive_survivor"] for f in records)
    assert sum(bool(f["horizontal"] and f["vertical"]) for f in records) >= 150
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(4427, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][-1, -1] = (int(damaged["test"][0]["output"][-1, -1]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": len(records),
            "chamber_shapes": len({f["shape"] for f in records}),
            "rod_counts": sorted({f["rods"] for f in records}),
            "delayed_pairs": sum(f["delayed"] > 0 for f in records),
            "unequal_falls": sum(len(set(f["distances"]) - {0}) > 1 for f in records),
            "redundant_support_pairs": sum(f["redundant"] for f in records)}


def _mechanics_test():
    # Explicit multiscale projection: carrier cells become 2-by-3 blocks,
    # whereas the stencil keeps its 2-by-2 native period and transparency.
    carrier = np.array([[1, 2], [3, 4]])
    stencil = np.array([[0, 5], [6, 0]])
    expected = np.array([[1, 5, 1, 5, 2, 5], [6, 1, 6, 2, 6, 2],
                         [3, 5, 3, 5, 4, 5], [6, 3, 6, 4, 6, 4]])
    assert np.array_equal(_field(carrier, stencil, (4, 6), 0), expected)
    # One matching cell dissolves an entire post. Redundant support retains
    # the bridge, while simultaneous dissolution of both posts releases it.
    rods = [(3, 1, 3, 1, 1), (3, 5, 3, 1, 2), (2, 0, 1, 7, 3)]
    field = np.full((6, 7), 4)
    field[5, 1] = 1
    states, _ = _react(rods, field)
    assert states == [(1, rods[1]), (2, rods[2])]
    field[5, 5] = 2
    states, _ = _react(rods, field)
    assert states == [(2, (5, 0, 1, 7, 3))]
    # A beam encounters a matching cell at an intermediate height, dissolves,
    # and releases its cap. Jumping directly to the floor would miss the hit.
    rods = [(2, 2, 4, 1, 1), (1, 1, 1, 3, 2), (0, 0, 1, 5, 3)]
    field = np.full((6, 7), 4)
    field[5, 2], field[3, 2] = 1, 2
    states, trace = _react(rods, field)
    assert states == [(2, (5, 0, 1, 5, 3))]
    assert trace["delayed"] == {1}
    assert _state_key(states) != _state_key(_react(rods, field, "land_then_react")[0])
    assert not np.array_equal(_paint(states, field), _erosion_alternative(rods, field))
    # A no-match member is also solved: it still reveals the projected field.
    # Background is 9 and the carrier contains literal black payload.
    grid = np.full((7, 6), 9)
    grid[:2, :2] = [[0, 1], [2, 3]]
    grid[:2, 3:5] = [[9, 4], [5, 9]]
    rods = [(2, 3, 2, 1, 7), (1, 0, 1, 6, 8)]
    grid[3:] = _render(rods, (4, 6), 9)
    out = solve(grid)
    assert np.array_equal(out[:3], grid[:3])
    assert np.array_equal(out[4], np.full(6, 8))
    assert out[-1, 3] == 7 and out[3, 0] == 0
