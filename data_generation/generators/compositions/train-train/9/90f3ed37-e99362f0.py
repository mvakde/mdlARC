"""Collision relay: exemplar phases drive a changing transparent-layer queue.

Only the standard library and NumPy are used. No parent modules or data are
needed at runtime. The parser and solver use only the presented input.
"""

from itertools import product

import numpy as np


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape),
             "grids must be two dimensional with sides 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), name + " must be an integer")
    _require(low <= int(value) <= high, name + " outside allowed range")
    return int(value)


def _period(slices):
    """The reference contains at least two complete repetitions."""
    for p in range(1, len(slices) // 2 + 1):
        if all(slices[c] == slices[c % p] for c in range(len(slices))):
            return p
    raise ValueError("reference does not display a repeated cycle")


def _options(tile, height, cutoff):
    """All phases/translations, grouped by their actual visible prefix."""
    p = len(tile)
    band = 1 + max(max(s) for s in tile)
    groups = {}
    for phase, offset in product(range(p), range(height - band + 1)):
        prefix = tuple(tuple(r + offset for r in tile[(c + phase) % p])
                       for c in range(cutoff))
        full = tuple(tuple(r + offset for r in tile[(c + phase) % p])
                     for c in range(p))
        groups.setdefault(prefix, []).append((phase, offset, full))
    return groups


def _masks(tile, placements, height, width):
    masks = np.zeros((len(placements), height, width), dtype=bool)
    for i, (phase, offset) in enumerate(placements):
        for c in range(width):
            masks[i, [r + offset for r in tile[(c + phase) % len(tile)]], c] = True
    return masks


def _render(masks, colors, background, dynamic=True):
    """Each column uses the old queue; collision winners rest afterward."""
    n, h, w = masks.shape
    order = list(range(n))
    out = np.full((h, w), background, dtype=np.int8)
    events = []
    for c in range(w):
        overlap = np.sum(masks[:, :, c], axis=0) >= 2
        winners = np.full(h, -1, dtype=int)
        for i in reversed(order):
            winners[masks[i, :, c]] = i
            out[masks[i, :, c], c] = colors[i]
        rest = {int(i) for i in winners[overlap]}
        events.append(tuple(i for i in order if i in rest))
        if dynamic and rest:
            order = [i for i in order if i not in rest] + [i for i in order if i in rest]
    return out, events


def _connected_overlap(masks):
    """Every ribbon participates in the same interaction network."""
    reached = {0}
    while True:
        more = {j for i in reached for j in range(len(masks))
                if np.any(masks[i] & masks[j])}
        if more <= reached:
            return len(reached) == len(masks)
        reached |= more


def _fixed_order_possible(masks, colors, output):
    """Topologically sort the winner>loser comparisons from actual overlaps."""
    edges = {(i, j) for i in range(len(colors)) for j in range(len(colors)) if i != j
             and np.any(masks[i] & masks[j] & (output == colors[i]))}
    remaining = set(range(len(colors)))
    while remaining:
        fronts = {j for j in remaining if not any((i, j) in edges for i in remaining)}
        if not fronts:
            return False
        remaining -= fronts
    return True


def _parse_canonical(a):
    """Normalize the orientation before calling. No output information is used."""
    h, w = a.shape
    _require(h >= 5 and w >= 3, "layout too small")
    bg, divider = int(a[-1, -1]), int(a[-2, 0])
    _require(bg != divider and np.all(a[-2] == divider), "missing queue divider")
    beads = list(map(int, a[-1, :]))
    n = next((j for j, v in enumerate(beads) if v == bg), w)
    colors = tuple(beads[:n])
    _require(2 <= n <= 4 and len(set(colors)) == n
             and divider not in colors and all(v == bg for v in beads[n:]),
             "queue must contain 2..4 distinct beads followed by background")
    bars = [r for r in range(h) if np.all(a[r] == divider)]
    _require(len(bars) == 2 and bars[-1] == h - 2, "need two divider bars")
    cut = bars[0]
    _require(1 <= cut < h - 3, "empty reference or canvas")
    reference, scene = a[:cut], a[cut + 1:-2]
    inks = set(map(int, np.unique(reference))) - {bg}
    _require(len(inks) == 1, "reference must have one ink")
    ink = next(iter(inks))
    _require(ink not in {divider, *colors}, "reference ink has a separate role")
    _require(set(map(int, np.unique(scene))) <= {bg, *colors}, "unexpected canvas ink")
    slices = tuple(tuple(map(int, np.flatnonzero(reference[:, c] == ink)))
                   for c in range(w))
    _require(all(slices), "every reference slice must be nonempty")
    p = _period(slices)
    tile = slices[:p]
    _require(min(min(s) for s in tile) == 0 and
             max(max(s) for s in tile) == reference.shape[0] - 1,
             "reference must use its full vertical bounding box")
    occupied = np.argwhere(scene != bg)
    _require(len(occupied) > 0, "canvas has no prefixes")
    k = 1 + int(occupied[:, 1].max())
    _require(1 <= k < p and k < w, "prefix must be shorter than the cycle")
    groups = _options(tile, scene.shape[0], k)
    placements = []
    for color in colors:
        prefix = tuple(tuple(map(int, np.flatnonzero(scene[:, c] == color)))
                       for c in range(k))
        candidates = groups.get(prefix, [])
        # Duplicate parameterizations are harmless only when the entire masks agree.
        _require(candidates and len({v[2] for v in candidates}) == 1,
                 "prefix does not uniquely register to the reference")
        placements.append(candidates[0][:2])
    masks = _masks(tile, placements, scene.shape[0], w)
    _require(np.all(np.sum(masks[:, :, :k], axis=0) <= 1),
             "input prefixes must not occlude one another")
    for i, color in enumerate(colors):
        _require(np.array_equal(scene[:, :k] == color, masks[i, :, :k]),
                 "a prefix is incomplete")
    return dict(background=bg, divider=divider, ink=ink, colors=colors,
                tile=tile, period=p, cutoff=k, placements=placements,
                masks=masks, scene=scene, reference=reference, bar=cut)


def _parse(input_grid):
    a = _grid(input_grid)
    matches = []
    for turns in range(4):
        try:
            info = _parse_canonical(np.rot90(a, turns))
            matches.append((turns, info))
        except ValueError:
            pass
    _require(len(matches) == 1, "input must have exactly one readable orientation")
    turns, info = matches[0]
    info["turns"] = turns
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    output, _ = _render(info["masks"], info["colors"], info["background"])
    return np.rot90(output, -info["turns"]).copy()


def _motif(rng, p):
    """Five geometric styles; the solver does not classify or assume a style."""
    kind = int(rng.integers(5))
    if kind == 0:  # unequal plateaus and hops
        centers = rng.integers(0, 4, size=p)
    elif kind == 1:  # rounded oscillation, with a local asymmetric hold
        centers = np.rint((1 - np.cos(2 * np.pi * np.arange(p) / p)) * 1.5).astype(int)
        centers[int(rng.integers(p))] = int(rng.integers(4))
    elif kind == 2:  # folded staircase, sometimes with a plateau
        centers = np.array([min(j, p - j) for j in range(p)])
        if rng.integers(2):
            centers[int(rng.integers(p))] = int(rng.integers(4))
    else:
        centers = rng.integers(0, 3, size=p)
    slices = []
    for center in centers:
        center = int(center)
        if kind == 3:  # variable thickness
            sl = tuple(range(center, center + int(rng.integers(1, 3))))
        elif kind == 4 and rng.random() < .55:  # split/rejoin slices
            sl = (center, center + 2)
        else:
            sl = (center,)
        slices.append(sl)
    low = min(min(s) for s in slices)
    tile = tuple(tuple(r - low for r in s) for s in slices)
    return tile


def _assemble(tile, placements, height, width, cutoff, roles):
    bg, divider, ink, *colors = roles
    band = 1 + max(max(s) for s in tile)
    grid = np.full((band + height + 3, width), bg, dtype=np.int8)
    for c in range(width):
        grid[list(tile[c % len(tile)]), c] = ink
    grid[band] = divider
    grid[-2] = divider
    grid[-1, :len(colors)] = colors
    masks = _masks(tile, placements, height, width)
    scene = grid[band + 1:-2]
    for i, color in enumerate(colors):
        scene[:, :cutoff][masks[i, :, :cutoff]] = color
    output, events = _render(masks, colors, bg)
    return {"input": grid, "output": output}, masks, events


def _make_pair(rng, n, is_test, roles, rotation, max_batch=4):
    for _ in range(1200):
        p = int(rng.integers(5, 10))
        tile = _motif(rng, p)
        if _period(tile * 2) != p:
            continue
        band = 1 + max(max(s) for s in tile)
        height = int(rng.integers(max(band + 3, 3 * n), min(19, 27 - band) + 1))
        cutoff = int(rng.integers(2, min(4, p - 1) + 1))
        width = (3 * p + int(rng.integers(0, min(3, 30 - 3 * p) + 1))
                 if is_test else 2 * p + int(rng.integers(0, 3)))
        groups = _options(tile, height, cutoff)
        options = [(prefix, entries[0][:2]) for prefix, entries in groups.items()
                   if len({entry[2] for entry in entries}) == 1]
        rng.shuffle(options)
        chosen = []
        for prefix, placement in options:
            if all(not any(set(prefix[c]) & set(old[c]) for c in range(cutoff))
                   for old, _ in chosen):
                chosen.append((prefix, placement))
                if len(chosen) == n:
                    break
        if len(chosen) != n:
            continue
        placements = [placement for _, placement in chosen]
        observed_phases = {(c + phase) % p for phase, _ in placements for c in range(cutoff)}
        if len(observed_phases) == p:
            continue  # at least one template slice is absent from ALL prefixes
        pair, masks, events = _assemble(tile, placements, height, width, cutoff, roles)
        if not _connected_overlap(masks):
            continue
        unseen = set(range(p)) - observed_phases
        if not any(any(pair["output"][r, c] == roles[3 + i]
                       for r in np.flatnonzero(masks[i, :, c]))
                   for i, (phase, _) in enumerate(placements)
                   for c in range(cutoff, width) if (c + phase) % p in unseen):
            continue
        static, _ = _render(masks, roles[3:], roles[0], dynamic=False)
        tip = np.full_like(pair["output"], roles[0])
        tip[:, :cutoff] = pair["input"][band + 1:-2, :cutoff]
        tip[:, cutoff:] = tip[:, cutoff - 1:cutoff]
        if (np.array_equal(static, pair["output"]) or
                _fixed_order_possible(masks, roles[3:], pair["output"]) or
                np.array_equal(tip, pair["output"]) or
                max(map(len, events)) > max_batch):
            continue
        pair = {name: np.rot90(grid, rotation).copy() for name, grid in pair.items()}
        return pair
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    a = _count(num_train, "num_train", 2, 10)
    b = _count(num_test, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    palette = tuple(map(int, rng.permutation(10)[:7]))
    stable = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    seen = set()
    train_batch = 1
    train_palettes = set()
    for index in range(a + b):
        is_test = index >= a
        # Four interacting ribbons with only one winner in EVERY column are
        # needlessly rare. When training only showed single-winner updates,
        # sample tests with at most three ribbons and retain the same event
        # check. This changes scene sampling, never the solving rule.
        max_layers = min(4, train_batch + 2) if is_test else 4
        n = int(rng.integers(2, max_layers + 1))
        roles = palette[:3 + n] if stable else tuple(map(int, rng.permutation(10)[:3 + n]))
        if is_test:
            # A visibly specified fresh palette defeats literal-color priority.
            while True:
                roles = tuple(map(int, rng.permutation(10)[:3 + n]))
                if roles not in train_palettes:
                    break
        for _ in range(20):
            pair = _make_pair(rng, n, is_test, roles, rotation,
                              max_batch=train_batch if is_test else 4)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("could not make a distinct input")
        seen.add(key)
        puzzle["test" if is_test else "train"].append(pair)
        if not is_test:
            train_palettes.add(roles)
            info = _parse(pair["input"])
            _, events = _render(info["masks"], info["colors"], info["background"])
            train_batch = max(train_batch, max(map(len, events)))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(inp)
    normal = np.rot90(out, info["turns"])
    masks, colors, bg = info["masks"], info["colors"], info["background"]
    expected, events = _render(masks, colors, bg)
    _require(np.array_equal(normal, expected), "output contradicts collision relay")
    _require(np.array_equal(normal[:, :info["cutoff"]], info["scene"][:, :info["cutoff"]]),
             "visible prefixes changed")
    _require(np.array_equal(normal == bg, ~np.any(masks, axis=0)),
             "transparency/union invariant failed")
    static, _ = _render(masks, colors, bg, dynamic=False)
    _require(not np.array_equal(normal, static), "fixed-priority shortcut succeeds")
    _require(not _fixed_order_possible(masks, colors, normal),
             "some static priority order explains this output")
    _require(_connected_overlap(masks), "ribbons split into independent puzzles")
    k, p = info["cutoff"], info["period"]
    observed = {(c + phase) % p for phase, _ in info["placements"] for c in range(k)}
    _require(len(observed) < p, "prefixes reveal the entire reference cycle")
    unseen = set(range(p)) - observed
    _require(any(any(normal[r, c] == colors[i] for r in np.flatnonzero(masks[i, :, c]))
                 for i, (phase, _) in enumerate(info["placements"])
                 for c in range(k, normal.shape[1]) if (c + phase) % p in unseen),
             "all unseen reference slices are hidden in the output")
    tip = info["scene"].copy()
    tip[:, k:] = tip[:, k - 1:k]
    _require(not np.array_equal(normal, tip), "tip-copy shortcut succeeds")
    _require(np.any(normal[:, k:] != bg), "there is no visible completion")
    info["events"] = events
    info["output"] = normal
    info["unseen_phases"] = unseen
    return info


def _reference_counterfactual(input_grid):
    """Find another readable reference with IDENTICAL scene and bead row.

    Used only as a semantic self-test, never by solve or by generation. Return
    the modified input if it changes the solved output. None means no witness
    was found by this bounded search, not a proof of independence.
    """
    info = _parse(input_grid)
    p, k = info["period"], info["cutoff"]
    observed = {(c + phase) % p for phase, _ in info["placements"] for c in range(k)}
    canonical = np.rot90(_grid(input_grid), info["turns"])
    original = solve(input_grid)
    band = info["reference"].shape[0]
    for phase in sorted(set(range(p)) - observed):
        for bits in range(1, 1 << band):
            replacement = tuple(r for r in range(band) if bits & (1 << r))
            if replacement == info["tile"][phase]:
                continue
            modified = canonical.copy()
            for c in range(phase, modified.shape[1], p):
                modified[:band, c] = info["background"]
                modified[list(replacement), c] = info["ink"]
            rotated = np.rot90(modified, -info["turns"]).copy()
            try:
                alternative = solve(rotated)
            except ValueError:
                continue
            if not np.array_equal(original, alternative):
                return rotated
    return None


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _count(a, "num_train", 2, 10)
    _count(b, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    seen, infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos.append(info)
    _require(len({i["turns"] for i in infos}) == 1, "orientation changes inside puzzle")
    train_batch = max(len(e) for i in infos[:a] for e in i["events"])
    _require(all(len(e) <= train_batch for i in infos[a:] for e in i["events"]),
             "test needs an undemonstrated simultaneous update")
    _require(all(i["scene"].shape[1] < 3 * i["period"] for i in infos[:a]) and
             all(i["scene"].shape[1] >= 3 * i["period"] for i in infos[a:]),
             "missing two-cycle to three-cycle train/test shift")
    train_roles = {tuple(i[key] for key in ("background", "divider", "ink")) + i["colors"]
                   for i in infos[:a]}
    _require(all(tuple(i[key] for key in ("background", "divider", "ink")) + i["colors"]
                 not in train_roles for i in infos[a:]), "test palette was not reassigned")


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [_analyze(pair) for puzzle in puzzles for split in ("train", "test")
             for pair in puzzle[split]]
    _require(len({_key(pair["input"]) for p in puzzles for pair in p["train"]}) == 168,
             "42 seeds lack distinct training structures")
    _require({i["turns"] for i in infos} == set(range(4)), "rotation diversity missing")
    _require({len(i["colors"]) for i in infos} == {2, 3, 4}, "layer counts do not vary")
    _require(len({i["scene"].shape for i in infos}) >= 35, "canvas geometry does not vary")
    _require(len({i["tile"] for i in infos}) >= 80, "motif structures do not vary")
    _require(any(any(len(s) > 1 for s in i["tile"]) for i in infos), "no thick/branched slices")
    _require(any(max(map(len, i["events"])) > 1 for i in infos), "no simultaneous winners")
    for puzzle in puzzles:
        _require(_reference_counterfactual(puzzle["train"][0]["input"]) is not None,
                 "no causal exemplar witness in the first training example")
    first, again = generate_puzzle(983922870), generate_puzzle(983922870)
    _require(all(_key(x[name]) == _key(y[name]) for s in first
                 for x, y in zip(first[s], again[s]) for name in ("input", "output")),
             "seed is not reproducible")
    # Palette-equivariance and corruption checks exercise the input-only solver.
    permutation = np.array([7, 4, 9, 1, 0, 6, 3, 8, 2, 5], dtype=np.int8)
    pair = puzzles[0]["test"][0]
    _require(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
             "solver attaches meaning to literal colors")
    for turns in range(4):
        _require(np.array_equal(solve(np.rot90(pair["input"], turns)),
                                np.rot90(pair["output"], turns)),
                 "solver does not preserve whole-arrangement rotation")
    bad = {s: [dict(pair) for pair in puzzles[0][s]] for s in ("train", "test")}
    bad["test"][0]["output"] = bad["test"][0]["output"].copy()
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7000 + 10 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "wrong counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
