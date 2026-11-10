"""A framed core is both a stamp and its own selectively reflected stencil.

Only NumPy and the standard library are used.  The solver reads no generator
state; all roles and dimensions are recovered from the supplied grid.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("a grid must be an integer 1..30 by 1..30 array in 0..9")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _extract(value):
    grid = _grid(value)
    height, width = grid.shape
    if min(grid.shape) < 5:
        raise ValueError("carrier must be at least 5 by 5")
    background = int(grid[0, 0])
    if any(int(grid[r, c]) != background
           for r, c in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("carrier corners disagree")
    border = np.concatenate((grid[0, 1:-1], grid[-1, 1:-1],
                             grid[1:-1, 0], grid[1:-1, -1]))
    frame = int(border[0])
    if frame == background or np.any(border != frame):
        raise ValueError("carrier perimeter must have one distinct frame color")
    interior = grid[1:-1, 1:-1]
    points = np.argwhere(interior != background)
    if len(points) == 0:
        raise ValueError("empty motif")
    lower, upper = points.min(axis=0), points.max(axis=0)
    span = tuple(int(x) for x in upper - lower + 1)
    if any(n not in (1, 3, 4) for n in span):
        raise ValueError("occupied spans must be 1, 3, or 4")
    start = tuple(int(lower[a]) - int(span[a] == 1) for a in (0, 1))
    shape = tuple(3 if n == 1 else n for n in span)
    if any(start[a] < 0 or start[a] + shape[a] > interior.shape[a]
           for a in (0, 1)):
        raise ValueError("centered halo does not fit")
    row, col = start
    h, w = shape
    core = interior[row:row + h, col:col + w].copy()
    outside = interior.copy()
    outside[row:row + h, col:col + w] = background
    if np.any(outside != background):
        raise ValueError("content outside the recovered core")
    if not np.any(core == frame):
        raise ValueError("core needs at least one frame-colored address")
    return {"background": background, "frame": frame, "core": core,
            "span": span, "start": start, "shape": grid.shape,
            "surplus": (height - h - 2, width - w - 2)}


def _tile(core, background, frame):
    h, w = core.shape
    tile = np.full((h + 2, w + 2), frame, dtype=np.int8)
    tile[1:-1, 1:-1] = core
    tile[0, 0] = tile[0, -1] = tile[-1, 0] = tile[-1, -1] = background
    return tile


def _render(core, background, frame, *, reflect_addresses=True,
            reflect_payload=True, all_foreground=False):
    """Direct block placement, also used for explicit concept ablations."""
    h, w = core.shape
    tile = _tile(core, background, frame)
    th, tw = tile.shape
    result = np.full((h * th, w * tw), background, dtype=np.int8)
    addresses = core != background if all_foreground else core == frame
    for r, c in np.argwhere(addresses):
        r, c = int(r), int(c)
        destination = w - 1 - c if r % 2 and reflect_addresses else c
        payload = tile[:, ::-1] if r % 2 and reflect_payload else tile
        result[r * th:(r + 1) * th, destination * tw:(destination + 1) * tw] = payload
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the frame roles and core, then unfold the self-stencil."""
    info = _extract(input_grid)
    return _render(info["core"], info["background"], info["frame"])


def _carrier(core, background, frame, surplus, offset):
    h, w = core.shape
    er, ec = surplus
    r, c = offset
    grid = np.full((h + 2 + er, w + 2 + ec), background, dtype=np.int8)
    grid[0, 1:-1] = grid[-1, 1:-1] = frame
    grid[1:-1, 0] = grid[1:-1, -1] = frame
    grid[r + 1:r + h + 1, c + 1:c + w + 1] = core
    return grid


def _full_span(core, background):
    occupied = core != background
    return bool(occupied[0].any() and occupied[-1].any()
                and occupied[:, 0].any() and occupied[:, -1].any())


def _clear_anchor(core, background, frame):
    """Observable evidence, without using a sampling seed or latent labels."""
    mask = core == frame
    if not (_full_span(core, background) and background in core
            and len(np.unique(core)) >= 3 and not mask[0].any()
            and mask[1].any() and mask[2].any()
            and not np.array_equal(mask[1], mask[1, ::-1])):
        return False
    reflected = core[:, ::-1]
    return all(not np.array_equal(reflected, alternate) for alternate in
               (core, core[::-1, :], core[::-1, ::-1],
                np.roll(core, 1, axis=1), np.roll(core, -1, axis=1)))


def _palette(rng, rich=False, exclude_roles=None):
    while True:
        size = int(rng.integers(3 if rich else 2, 7))
        colors = [int(x) for x in rng.choice(10, size=size, replace=False)]
        if tuple(colors[:2]) != exclude_roles:
            return colors


def _anchor(rng, shape, colors):
    background, frame = colors[:2]
    h, w = shape
    for _ in range(1000):
        core = rng.choice(colors, size=shape).astype(np.int8)
        # An accent-bearing top row creates a deliberately empty macro-band.
        core[0] = rng.choice([background] + colors[2:], size=w)
        if _clear_anchor(core, background, frame):
            return core
    raise RuntimeError("could not sample an informative anchor")


def _random_core(rng, limits, colors, kind):
    background, frame = colors[:2]
    # One-cell occupied axes are always centered in a length-three core.
    if kind == "singleton":
        core = np.full((3, 3), background, dtype=np.int8)
        core[1, 1] = frame
        return core
    h = int(rng.choice([3, 4] if limits[0] == 4 else [3]))
    w = int(rng.choice([3, 4] if limits[1] == 4 else [3]))
    if kind == "horizontal":
        h = 3
    elif kind == "vertical":
        w = 3
    for _ in range(500):
        core = np.full((h, w), background, dtype=np.int8)
        if kind == "horizontal":
            core[1, :] = rng.choice(colors[1:], size=w)
        elif kind == "vertical":
            core[:, 1] = rng.choice(colors[1:], size=h)
        elif kind == "dense":
            core[:] = rng.choice(colors[1:], size=(h, w))
        elif kind == "frame_only":
            core[rng.random((h, w)) < float(rng.uniform(.3, .9))] = frame
        elif kind == "stripe":
            core[:] = rng.choice(colors, size=(h, w))
            if rng.random() < .5:
                core[int(rng.integers(h)), :] = frame
            else:
                core[:, int(rng.integers(w))] = frame
        else:
            density = float(rng.uniform(.25, .8))
            mask = rng.random((h, w)) < density
            core[mask] = rng.choice(colors[1:], size=int(mask.sum()))
        if not np.any(core == frame):
            continue
        if kind in ("horizontal", "vertical") or _full_span(core, background):
            return core
    raise RuntimeError("could not sample core")


def _padding(rng, core_shape, regime):
    if regime in ("anchor", "halo"):
        surplus = tuple(int(rng.integers(1, 10)) for _ in (0, 1))
    else:
        mode = int(rng.choice(3, p=(.28, .42, .30))) if regime == "train" else int(rng.choice(2, p=(.32, .68)))
        surplus = [0, 0]
        if mode == 2:
            surplus = [int(rng.integers(1, 10)) for _ in (0, 1)]
        elif mode == 1:
            axis = int(rng.integers(2))
            cap = 30 - core_shape[axis] - 2 if regime == "test" else 12
            surplus[axis] = int(rng.integers(1, cap + 1))
        surplus = tuple(surplus)
    # Near edge, middle, and far edge are all possible, independently per axis.
    offset = []
    for extra in surplus:
        position = int(rng.integers(3))
        offset.append(0 if position == 0 else extra if position == 2
                      else int(rng.integers(0, extra + 1)))
    return surplus, tuple(offset)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    limits = (int(rng.integers(3, 5)), int(rng.integers(3, 5)))
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            regime = ("anchor" if index == 0 else "halo" if index == 1 else "train") if split == "train" else "test"
            for _ in range(100):
                colors = _palette(rng, rich=regime == "anchor",
                                  exclude_roles=first_roles if regime == "halo" else None)
                background, frame = colors[:2]
                if regime == "anchor":
                    core = _anchor(rng, limits, colors)
                else:
                    kinds = ("random", "dense", "frame_only", "stripe", "horizontal", "vertical", "singleton")
                    weights = (.30, .12, .14, .14, .11, .11, .08) if split == "train" else (.20, .15, .16, .15, .14, .14, .06)
                    kind = "singleton" if regime == "halo" else str(rng.choice(kinds, p=weights))
                    core = _random_core(rng, limits, colors, kind)
                surplus, offset = _padding(rng, core.shape, regime)
                source = _carrier(core, background, frame, surplus, offset)
                if _key(source) not in seen:
                    break
            else:
                raise RuntimeError("could not sample a distinct input")
            seen.add(_key(source))
            if regime == "anchor":
                first_roles = (background, frame)
            # Ground truth is built from the constructed core, not by solving
            # the generated carrier.  Validation recovers it again from input.
            puzzle[split].append({"input": source,
                                  "output": _render(core, background, frame)})
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pixels(output, info):
    """Pixel-level correspondence check; no call to _render or _tile."""
    core, b, f = info["core"], info["background"], info["frame"]
    h, w = core.shape
    th, tw = h + 2, w + 2
    _require(output.shape == (h * th, w * tw), "wrong output dimensions")
    # Undo band reflection in the pixel coordinates and read the source map.
    for row in range(output.shape[0]):
        br, lr = divmod(row, th)
        for col in range(output.shape[1]):
            original_col = output.shape[1] - 1 - col if br % 2 else col
            bc, lc = divmod(original_col, tw)
            if core[br, bc] != f:
                expected = b
            elif lr in (0, th - 1) or lc in (0, tw - 1):
                expected = b if lr in (0, th - 1) and lc in (0, tw - 1) else f
            else:
                expected = int(core[lr - 1, lc - 1])
            _require(int(output[row, col]) == expected, "wrong stencil, frame, or payload pixel")


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            source, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(source) not in seen, "duplicate input")
            seen.add(_key(source))
            item = _extract(source)
            _check_pixels(output, item)
            _require(np.array_equal(solve(source), output), "solve disagreement")
            _require(set(np.unique(output)) == set(np.unique(source)), "a picture color was lost or invented")
            _require(not np.array_equal(source, output), "unexpected identity pair")
            info[split].append(item)
    anchor, halo = info["train"][:2]
    core, b, f = anchor["core"], anchor["background"], anchor["frame"]
    _require(_clear_anchor(core, b, f), "first demonstration must expose addressing and reflection")
    _require(all(n > 0 for n in anchor["surplus"]), "anchor needs two-axis compaction")
    _require(halo["span"] == (1, 1) and all(n > 0 for n in halo["surplus"]), "second demonstration must teach the halo")
    _require((halo["background"], halo["frame"]) != (b, f), "training must vary local roles")
    expected = np.asarray(puzzle["train"][0]["output"])
    for addresses, payload in ((False, False), (True, False), (False, True)):
        _require(not np.array_equal(expected, _render(core, b, f,
                 reflect_addresses=addresses, reflect_payload=payload)), "A ablation is invisible")
    _require(not np.array_equal(expected, _render(core, b, f, all_foreground=True)), "frame-role ablation is invisible")
    # A tight singleton without its retained halo would produce a 3x3 output.
    _require(np.asarray(puzzle["train"][1]["output"]).shape == (15, 15), "halo ablation is invisible")
    _require(tuple(np.asarray(puzzle["train"][0]["input"]).shape) != (core.shape[0] + 2, core.shape[1] + 2), "padding ablation is invisible")
    limits = core.shape
    for split in ("train", "test"):
        for item in info[split]:
            _require(all(item["core"].shape[a] <= limits[a] for a in (0, 1)), "an axis size was not demonstrated")
            if split == "test":
                _require(0 in item["surplus"], "test must recombine normalized and padded axes")


def _fingerprint(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed diversity, semantic, metamorphic and API checks."""
    signatures, shapes, core_shapes, spans = set(), set(), set(), set()
    counts, masks, palettes, roles = set(), set(), set(), set()
    kinds_by_split = {s: set() for s in ("train", "test")}
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed, 6, 2)
        validate_generated_puzzle(puzzle)
        signature = _fingerprint(puzzle)
        signatures.add(signature)
        if seed < 3:
            _require(signature == _fingerprint(generate_puzzle(seed, 6, 2)), "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_count += 1
                source, output = pair["input"], pair["output"]
                item = _extract(source)
                core, b, f = item["core"], item["background"], item["frame"]
                shapes.add(source.shape)
                core_shapes.add(core.shape)
                spans.add(item["span"])
                counts.add(int(np.count_nonzero(core == f)))
                masks.add((core.shape, (core == f).tobytes()))
                palettes.add(len(np.unique(source)))
                roles.add((b, f))
                span = item["span"]
                kind = "singleton" if span == (1, 1) else "horizontal" if span[0] == 1 else "vertical" if span[1] == 1 else "full"
                kinds_by_split[split].add(kind)
                before = source.copy()
                _require(np.array_equal(solve(source), output), "wrong solved pair")
                _require(np.array_equal(before, source), "solve mutated its input")
        # Extra carrier padding and a new offset must have no effect.
        item = _extract(puzzle["train"][0]["input"])
        padded = _carrier(item["core"], item["background"], item["frame"], (4, 7), (3, 1))
        _require(np.array_equal(solve(padded), puzzle["train"][0]["output"]), "padding/translation changes the answer")
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(permutation[padded]), permutation[solve(padded)]), "palette permutation changes roles")
    _require(len(signatures) == 42, "puzzles repeat across seeds")
    _require(core_shapes == {(3, 3), (3, 4), (4, 3), (4, 4)}, "normalized shapes lack range")
    _require(len(shapes) >= 35 and len(masks) >= 65, "geometry lacks range")
    _require({(1, 1), (1, 3), (1, 4), (3, 1), (4, 1), (3, 3), (3, 4), (4, 3), (4, 4)} <= spans, "occupied spans lack range")
    _require(len(counts) >= 8 and 1 in counts, "stamp counts lack range")
    _require({2, 3, 4, 5, 6} <= palettes and len(roles) >= 45, "palettes or roles lack range")
    _require(all(kinds_by_split[s] == {"singleton", "horizontal", "vertical", "full"}
                 for s in kinds_by_split), "motif kinds lack split coverage")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts were ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(5, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # The solver is not limited to shapes in a particular generated curriculum.
    for shape in ((3, 3), (3, 4), (4, 3), (4, 4)):
        core = np.full(shape, 8, dtype=np.int8)
        core[1, 0] = 4
        source = _carrier(core, 0, 4, (0, 5), (0, 5))
        _check_pixels(solve(source), _extract(source))
    bad = generate_puzzle(9)
    bad["train"][0]["output"] = bad["train"][0]["output"].copy()
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("a changed output pixel was accepted")
    return {"seeds": 42, "pairs": pair_count, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "normalized_shapes": sorted(core_shapes),
            "occupied_spans": sorted(spans), "stencil_geometries": len(masks),
            "stamp_counts": sorted(counts), "palette_sizes": sorted(palettes),
            "local_role_assignments": len(roles),
            "fingerprint": hashlib.sha256(repr(sorted(shapes)).encode()).hexdigest()[:16]}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
