"""Color-keyed enclosure relays on upward-right diagonal beams.

The canvas is visible; all non-background cells in its bottom row are sources.
Complete contrasting 3x3 rings are color transducers, not obstacles at their edges.
Only their centers are events.  All recognition uses the original input.
"""

from __future__ import annotations

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _background(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if 2 * int(counts[bg]) <= a.size:
        raise ValueError("input background must occupy a strict majority")
    return bg


def _relays(a, bg, orthogonal=False):
    """Return center -> (border color, payload), without component assumptions."""
    h, w = a.shape
    if min(h, w) < 3:
        return {}
    center = a[1:-1, 1:-1]
    border = a[:-2, 1:-1]
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if not orthogonal:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    mask = (center != bg) & (border != bg) & (center != border)
    for dr, dc in offsets:
        mask &= a[1 + dr:h - 1 + dr, 1 + dc:w - 1 + dc] == border
    rows, cols = np.nonzero(mask)
    return {(int(r + 1), int(c + 1)): (int(border[r, c]), int(center[r, c]))
            for r, c in zip(rows, cols)}


def _trace(a, bg, relays, unconditional=False):
    out = a.copy()
    h, w = a.shape
    events, depths, broken_hits = [], [], []
    crosses = _relays(a, bg, orthogonal=True)
    for start in np.flatnonzero(a[-1] != bg):
        color = int(a[-1, start])
        depth = 0
        for t in range(1, min(h, w - int(start))):
            r, c = h - 1 - t, int(start) + t
            if (r, c) in crosses and (r, c) not in relays:
                broken_hits.append((r, c))
            if (r, c) in relays:
                border, payload = relays[r, c]
                matched = color == border
                events.append((r, c, color, matched))
                if not matched and not unconditional:
                    break
                color = payload
                depth += 1
            if int(a[r, c]) == bg:
                out[r, c] = color
        depths.append(depth)
    return out, events, depths, broken_hits


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the visible grid; no puzzle palette or seed is needed."""
    a = _grid(input_grid)
    bg = _background(a)
    return _trace(a, bg, _relays(a, bg))[0]


def _audit(a):
    """A row-wise check of the output recurrence and causal relay events.

    Unlike solve's per-source traversal, this moves all live states one row
    at a time and recognizes rings by an explicit border list.
    """
    bg = _background(a)
    h, w = a.shape
    relays = {}
    crosses = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            payload = int(a[r, c])
            border = [int(a[r + dr, c + dc])
                      for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
            ring = border[0]
            if payload != bg and ring != bg and payload != ring and len(set(border)) == 1:
                relays[r, c] = (ring, payload)
            orth = [int(a[r - 1, c]), int(a[r + 1, c]),
                    int(a[r, c - 1]), int(a[r, c + 1])]
            if payload != bg and orth[0] != bg and payload != orth[0] and len(set(orth)) == 1:
                crosses.add((r, c))
    sources = tuple(int(c) for c in np.flatnonzero(a[-1] != bg))
    # Column -> (current color, source column, successful transitions).
    states = {c: (int(a[-1, c]), c, 0) for c in sources}
    depths = {c: 0 for c in sources}
    reached, stopped, broken = set(), set(), set()
    matches, broken_states = [], []
    expected = a.copy()
    for r in range(h - 2, -1, -1):
        shifted = {c + 1: state for c, state in states.items() if c + 1 < w}
        states = {}
        for c, (color, start, depth) in shifted.items():
            if (r, c) in crosses and (r, c) not in relays:
                broken.add((r, c))
                broken_states.append((r, c, color))
            if (r, c) in relays:
                reached.add((r, c))
                ring, payload = relays[r, c]
                if ring != color:
                    stopped.add((r, c))
                    continue
                color, depth = payload, depth + 1
                depths[start] = depth
                matches.append((start, r, c, payload))
            if a[r, c] == bg:
                expected[r, c] = color
            states[c] = (color, start, depth)
    source_sums = {h - 1 + c for c in sources}
    def witness(r, c, color):
        return (r >= 2 and c + 2 < w and a[r - 2, c + 2] == bg
                and expected[r - 2, c + 2] == color)

    visible_depth = max((sum(start == s and witness(r, c, color)
                             for start, r, c, color in matches) for s in sources), default=0)
    visible_off_ray = sum(r + c not in source_sums and witness(r, c, bg) for r, c in relays)
    return expected, {
        "background": bg, "sources": sources, "depth": max(depths.values(), default=0),
        "source_count": len(sources), "relay_count": len(relays),
        "stops": len(stopped), "broken_hits": len(broken),
        "visible_depth": int(visible_depth),
        "visible_stops": sum(witness(r, c, bg) for r, c in stopped),
        "visible_broken": sum(int(a[r - 1, c]) == color and witness(r, c, color)
                              for r, c, color in broken_states),
        "visible_off_ray": int(visible_off_ray),
        "dormant": len(set(relays) - reached),
        "off_ray": sum(r + c not in source_sums for r, c in relays),
        "repeated_sources": len({int(a[-1, c]) for c in sources}) < len(sources),
        "positions": tuple(sorted(relays)),
    }


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(a):
    return a.shape, a.tobytes()


def _choose_other(rng, pool, excluded):
    return int(rng.choice([c for c in pool if c not in excluded]))


def _plan(rng, split, index):
    if split == "train" and index == 0:
        return 1, ["broken", "turn"]
    if split == "train" and index == 1:
        return 2, ["turn", "turn", "stop"]
    if split == "train" and index == 2:
        return int(rng.integers(1, 5)), ["turn"]
    if split == "train" and index == 3:
        return int(rng.integers(2, 5)), ["turn", "broken", "turn"]
    depth = 3 if split == "test" and index == 0 else int(rng.integers(1, 3 + (split == "test")))
    events = ["turn"] * depth
    if rng.random() < .65:
        events.insert(int(rng.integers(len(events) + 1)), "broken")
    if rng.random() < .45:
        events.append("stop")
    if events[-1] == "stop" and len(events) < 5 and rng.random() < .35:
        events.append("shadow")
    return int(rng.integers(1, 5)), events


def _make_pair(rng, bg, pool, source_count, kinds, repeat_sources=False):
    # Long chains require room, but canvas margins and aspect ratio are free.
    times = [int(rng.integers(3, 5))]
    for _ in kinds[1:]:
        times.append(times[-1] + int(rng.integers(4, 6)))
    last = times[-1]
    h = int(rng.integers(max(17, last + 4), 31))
    w = int(rng.integers(max(18, last + 4), 31))
    a = np.full((h, w), bg, dtype=np.int8)
    main = int(rng.integers(0, w - last - 3))
    # Prefer visibly separated beams; all rays have distinct diagonal invariants.
    starts = [main]
    for _ in range(source_count - 1):
        choices = [c for c in range(w - 4) if all(abs(c - s) >= 3 for s in starts)]
        if not choices:
            choices = [c for c in range(w - 3) if c not in starts]
        starts.append(int(rng.choice(choices)))
    first_color = int(rng.choice(pool))
    source_colors = [first_color] + [first_color if repeat_sources or rng.random() < .35
                                     else int(rng.choice(pool)) for _ in starts[1:]]
    for c, color in zip(starts, source_colors):
        a[-1, c] = color
    placed, intended = [], set()
    locked = np.zeros((h, w), dtype=bool)
    locked[-1] = True
    # Preserve visible witnesses on both sides of every main-lane event.
    # Without these, an unrelated object could hide a transition's entire gap.
    clear = np.zeros((h, w), dtype=bool)
    for t in times:
        for witness in (t - 2, t + 2):
            clear[h - 1 - witness, main + witness] = True
    locked |= clear

    def fits(r, c):
        return (1 <= r <= h - 3 and 1 <= c <= w - 2
                and not np.any(clear[r - 1:r + 2, c - 1:c + 2])
                and all(abs(r - rr) >= 4 or abs(c - cc) >= 4 for rr, cc in placed))

    def put(r, c, border, payload, broken=False):
        a[r - 1:r + 2, c - 1:c + 2] = border
        a[r, c] = payload
        if broken:
            # Missing NW or SE: four orthogonal neighbors still form a false key.
            dr, dc = ((-1, -1), (1, 1))[int(rng.integers(2))]
            a[r + dr, c + dc] = bg
        else:
            intended.add((r, c))
        placed.append((r, c))
        locked[max(0, r - 2):min(h, r + 3), max(0, c - 2):min(w, c + 3)] = True

    color = first_color
    for t, kind in zip(times, kinds):
        r, c = h - 1 - t, main + t
        if kind in ("turn", "broken"):
            payload = _choose_other(rng, pool, {color})
            put(r, c, color, payload, broken=kind == "broken")
            if kind == "turn":
                color = payload
        elif kind == "stop":
            border = _choose_other(rng, pool, {color})
            put(r, c, border, _choose_other(rng, pool, {color, border}))
        else:  # An intact relay beyond a stop receives no ray.
            put(r, c, color, _choose_other(rng, pool, {color}))

    # Secondary beams can have a local transition or a mismatching stop.
    for start, incoming in zip(starts[1:], source_colors[1:]):
        if rng.random() < .70:
            locations = [(h - 1 - t, start + t)
                         for t in range(3, min(h - 3, w - start - 3))
                         if fits(h - 1 - t, start + t)]
            if locations:
                r, c = locations[int(rng.integers(len(locations)))]
                border = incoming if rng.random() < .70 else _choose_other(rng, pool, {incoming})
                put(r, c, border, _choose_other(rng, pool, {border}))

    # Complete but geometrically unreachable relays refute spontaneous emission.
    sums = {h - 1 + s for s in starts}
    for _ in range(int(rng.integers(1, 3))):
        locations = [(r, c) for r in range(2, h - 3) for c in range(1, w - 3)
                     if r + c not in sums and fits(r, c) and a[r - 2, c + 2] == bg]
        if locations:
            r, c = locations[int(rng.integers(len(locations)))]
            border = int(rng.choice(pool))
            put(r, c, border, _choose_other(rng, pool, {border}))
            clear[r - 2, c + 2] = True
            locked[r - 2, c + 2] = True

    # Clutter remains transparent and never introduces an unplanned full ring.
    free = [(r, c) for r in range(h - 1) for c in range(w) if not locked[r, c]]
    rng.shuffle(free)
    for r, c in free[:int(rng.integers(0, 15))]:
        a[r, c] = int(rng.choice(pool))
    if set(_relays(a, bg)) != intended:
        return None
    expected, info = _audit(a)
    if (info["visible_off_ray"] == 0 or info["depth"] != kinds.count("turn")
            or info["visible_depth"] != info["depth"]
            or np.count_nonzero((a == bg) & (expected != bg)) < 3):
        return None
    return {"input": a, "output": expected}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    bg = 0 if rng.random() < .3 else int(rng.integers(0, 10))
    pool = [int(c) for c in rng.choice([c for c in range(10) if c != bg],
                                      size=int(rng.integers(4, 7)), replace=False)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        if split == "test":
            pool = sorted({int(c) for pair in puzzle["train"]
                           for c in np.unique(pair["input"]) if int(c) != bg})
        for index in range(count):
            k, kinds = _plan(rng, split, index)
            for _ in range(100):
                pair = _make_pair(rng, bg, pool, k, kinds,
                                  repeat_sources=split == "train" and index == 1)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct relay field")
    # Demonstration order carries no semantics.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, backgrounds = set(), set()
    infos = {"train": [], "test": []}
    palettes = {"train": set(), "test": set()}
    orth_diff = unkeyed_diff = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "canvas shape must be preserved")
            bg = _background(a)
            backgrounds.add(bg)
            palettes[split].update(int(c) for c in np.unique(a) if c != bg)
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            expected, info = _audit(a)
            infos[split].append(info)
            _require(np.array_equal(out, expected), "diagonal relay recurrence disagrees")
            _require(np.array_equal(solve(a), expected), "solver disagrees with row audit")
            _require(np.array_equal(out[a != bg], a[a != bg]), "original foreground was overwritten")
            _require(np.count_nonzero((a == bg) & (out != bg)) >= 3, "no informative beam completion")
            _require(1 <= info["source_count"] <= 4 and info["depth"] >= 1, "missing sources or transitions")
            _require(info["visible_off_ray"] >= 1, "no visibly dormant off-ray relay")
            _require(info["visible_depth"] == info["depth"], "chain transitions lack visible witnesses")
            straight = _trace(a, bg, {})[0]
            _require(not np.array_equal(out, straight), "ignoring A gives the same answer")
            if split == "train":
                orth_diff |= not np.array_equal(out, _trace(a, bg, _relays(a, bg, True))[0])
                unkeyed_diff |= not np.array_equal(out, _trace(a, bg, _relays(a, bg), True)[0])
    _require(len(backgrounds) == 1, "puzzle background changed")
    _require(palettes["test"] <= palettes["train"], "test introduces an undemonstrated color")
    train = infos["train"]
    _require({1, 2} <= {i["source_count"] for i in train}, "source-count contrast absent")
    _require({1, 2} <= {i["depth"] for i in train}, "one- and two-transition demonstrations absent")
    _require(any(i["repeated_sources"] for i in train), "repeated source colors absent")
    _require(any(i["visible_broken"] for i in train) and orth_diff, "eight-neighbor condition not demonstrated")
    _require(any(i["visible_stops"] for i in train) and unkeyed_diff, "color-keyed stopping not demonstrated")
    _require(max(i["depth"] for i in train) == 2, "training chain contract changed")
    _require(max(i["depth"] for i in infos["test"]) == 3, "test lacks the longer-chain shift")


def _puzzle_key(puzzle):
    return tuple((_key(_grid(p["input"])), _key(_grid(p["output"])))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded checks over 42 seeds, every legal count, and invalid requests."""
    # Hand-built five-transition paths exceed the sampler's three-transition
    # test shift, include foreground zero and transparent clutter, and exercise
    # both stopping and successful propagation right up to a corner boundary.
    for passes_last in (False, True):
        probe = np.full((30, 30), 6, dtype=np.int8)
        probe[29, 0] = 0
        for t, border, payload in ((3, 0, 1), (8, 1, 2), (13, 2, 3),
                                   (18, 3, 4), (23, 4, 5),
                                   (27, 5 if passes_last else 7, 0 if passes_last else 8)):
            r, c = 29 - t, t
            probe[r - 1:r + 2, c - 1:c + 2] = border
            probe[r, c] = payload
        probe[19, 10] = 9  # Transparent foreground on the active diagonal.
        probe[3:6, 4:7] = 7
        probe[4, 5] = 8  # Dormant off-ray relay.
        expected = probe.copy()
        for t in range(1, 30 if passes_last else 27):
            color = sum(t >= threshold for threshold in (3, 8, 13, 18, 23))
            if t >= 27:
                color = 0
            if probe[29 - t, t] == 6:
                expected[29 - t, t] = color
        _require(np.array_equal(solve(probe), expected), "long-chain/boundary probe failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seeds collapsed")
    _require(_puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0)), "nondeterministic")
    shapes, layouts, backgrounds, sources, rings = set(), set(), set(), set(), set()
    has_zero_foreground = has_top_exit = has_right_exit = has_shadow = False
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                _, info = _audit(a)
                h, w = a.shape
                shapes.add(a.shape)
                layouts.add((a.shape, info["sources"], info["positions"]))
                backgrounds.add(info["background"])
                sources.add(info["source_count"])
                rings.add(info["relay_count"])
                has_zero_foreground |= info["background"] != 0 and np.any(a == 0)
                has_shadow |= info["dormant"] > info["off_ray"]
                out = pair["output"]
                has_top_exit |= bool(np.any((a[0] == info["background"]) & (out[0] != a[0])))
                has_right_exit |= bool(np.any((a[:, -1] == info["background"]) & (out[:, -1] != a[:, -1])))
    _require(len(shapes) >= 45 and len(layouts) >= 190, "structural diversity collapsed")
    _require(len(backgrounds) >= 8 and has_zero_foreground, "palette roles collapsed")
    _require(sources == {1, 2, 3, 4} and len(rings) >= 5, "object-count diversity collapsed")
    _require(has_top_exit and has_right_exit and has_shadow, "boundary or reachability coverage absent")
    for nt in range(2, 11):
        for nx in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1200 + 10 * nt + nx, nt, nx)
            _require(len(p["train"]) == nt and len(p["test"]) == nx, "wrong explicit counts")
    for nt, nx in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nx)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and all legal explicit counts")
