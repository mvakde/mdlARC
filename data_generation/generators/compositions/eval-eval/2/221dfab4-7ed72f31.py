"""Stroboscopic symmetry echoes: a global clock compared at local symmetric endpoints.

No parent modules, generation metadata, training labels, or palette constants are
used by solve. All coordinates used below are in an input-derived top-edge view.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np


SIDES = ("top", "right", "bottom", "left")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "grid colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    while todo:
        root = min(todo)
        todo.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(sorted(cells))
    return result


def _bbox(cells):
    a = np.asarray(cells)
    return int(a[:, 0].min()), int(a[:, 0].max()), int(a[:, 1].min()), int(a[:, 1].max())


def _kind(cells):
    r0, r1, c0, c1 = _bbox(cells)
    if len(cells) == 1:
        return "P"
    if r0 == r1 and len(cells) == c1 - c0 + 1:
        return "H"
    if c0 == c1 and len(cells) == r1 - r0 + 1:
        return "V"
    return None


def _bar(g, color):
    cells = np.argwhere(g == color)
    if not len(cells):
        return None
    rr, cc = cells.T
    found = []
    for side, fixed, boundary, lateral in ((0, rr, 0, cc),
            (1, cc, g.shape[1] - 1, rr), (2, rr, g.shape[0] - 1, cc),
            (3, cc, 0, rr)):
        if np.all(fixed == boundary):
            lo = int(lateral.min())
            if np.array_equal(np.sort(lateral), np.arange(lo, lo + len(cells))):
                found.append(side)
    return found[0] if len(found) == 1 else None


def _destination(marker, kind, source, g, background):
    r0, r1, c0, c1 = _bbox(source)
    a, b = marker[0]
    if min(max(abs(r - x), abs(c - y)) for r, c in source for x, y in marker) != 1:
        return None
    if kind == "H":
        if not (r1 < a or r0 > a):
            return None
        dest = [(2 * a - r, c) for r, c in source]
    elif kind == "V":
        if not (c1 < b or c0 > b):
            return None
        dest = [(r, 2 * b - c) for r, c in source]
    else:
        if not ((r1 <= a or r0 >= a) and (c1 <= b or c0 >= b)):
            return None
        dest = [(2 * a - r, 2 * b - c) for r, c in source]
    h, w = g.shape
    if any(not (0 <= r < h and 0 <= c < w) or int(g[r, c]) != background for r, c in dest):
        return None
    return dest


def _read_input(value):
    original = _grid(value)
    colors, counts = np.unique(original, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(counts.max() > original.size / 2, "background must occupy more than half the input")
    bars = [(int(c), _bar(original, int(c))) for c in colors if c != background]
    bars = [(c, side) for c, side in bars if side is not None]
    _require(len(bars) == 1, "exactly one color must form an unambiguous outer-edge bar")
    clock, side = bars[0]
    g = np.rot90(original, side).copy()
    cols = np.flatnonzero(g[0] == clock)
    start, length = int(cols.min()), len(cols)
    candidates = []
    for color in colors:
        if color in (background, clock):
            continue
        parts = _components(g == color)
        if parts and all(_kind(p) is not None for p in parts):
            candidates.append(int(color))
    _require(len(candidates) == 1, "one common color must consist only of line/point markers")
    marker_color = candidates[0]
    markers = _components(g == marker_color)
    sources = _components((g != background) & (g != clock) & (g != marker_color))
    _require(markers and len(markers) == len(sources), "one payload per marker")
    for source in sources:
        _require(len({int(g[p]) for p in source}) == 1, "payloads must be monochrome")
        _require(_kind(source) is None, "payloads must be non-straight to distinguish the marker role")
    groups, used, occupied = [], set(), set()
    h, w = g.shape
    for marker in markers:
        _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in marker), "local markers must be internal")
        _require(all(not start <= c < start + length for r, c in marker), "clock strip cannot erase a marker")
        kind = _kind(marker)
        options = []
        for index, source in enumerate(sources):
            dest = _destination(marker, kind, source, g, background)
            if dest is not None:
                options.append((index, dest))
        _require(len(options) == 1, "missing or ambiguous marker ownership")
        index, dest = options[0]
        _require(index not in used, "a payload cannot belong to two markers")
        _require(not occupied.intersection(dest), "prospective copies cannot collide")
        used.add(index)
        occupied.update(dest)
        groups.append(dict(kind=kind, marker=marker, source=sources[index], dest=dest,
                           color=int(g[sources[index][0]])))
    _require(len(used) == len(sources), "unowned payload")
    return dict(grid=g, background=background, clock=clock, marker_color=marker_color,
                side=side, start=start, length=length, groups=groups)


def _lit(p):
    return p[0] % 6 == 4


def _inside(info, p):
    return info["start"] <= p[1] < info["start"] + info["length"]


def _translate(group):
    r0, _, c0, _ = _bbox(group["source"])
    a0, _, b0, _ = _bbox(group["dest"])
    return [(r + a0 - r0, c + b0 - c0) for r, c in group["source"]]


def _paint(info, mode="xor", translated=False):
    """Painting implementation; alternatives are used only in diagnostic checks."""
    g = info["grid"]
    out = g.copy()
    special = info["marker_color"]
    for group in info["groups"]:
        for p in group["source"]:
            if _lit(p):
                out[p] = special
        dest = _translate(group) if translated else group["dest"]
        for p, q in zip(group["source"], dest):
            a, b = _lit(p), _lit(q)
            accept = {"xor": a != b, "all": True, "source": a,
                      "destination": b, "or": a or b, "none": False}[mode]
            if accept:
                out[q] = special if b else g[p]
    start, end = info["start"], info["start"] + info["length"]
    out[:, start:end] = info["background"]
    for r in range(g.shape[0]):
        if r % 6 in (0, 2):
            out[r, start:end] = info["clock"]
        elif r % 6 == 4:
            out[r, start:end] = special
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the clock, marker role, ownership, phases and output from this input."""
    info = _read_input(input_grid)
    return np.rot90(_paint(info), -info["side"]).copy()


def _pixel_expected(info):
    """Direct per-cell formula, separately expressed from the painting procedure."""
    g = info["grid"]
    sources, copies = {}, {}
    for group in info["groups"]:
        for p, q in zip(group["source"], group["dest"]):
            sources[p] = group["color"]
            copies[q] = (p, group["color"])
    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            p = (r, c)
            if _inside(info, p):
                out[p] = (info["background"] if r % 2 else
                          info["marker_color"] if r % 6 == 4 else info["clock"])
            elif p in sources:
                out[p] = info["marker_color"] if r % 6 == 4 else sources[p]
            elif p in copies:
                source, color = copies[p]
                # On a special destination, the source must be unlit. On any
                # other destination, the source must be lit. Otherwise stay empty.
                if r % 6 == 4 and source[0] % 6 != 4:
                    out[p] = info["marker_color"]
                elif r % 6 != 4 and source[0] % 6 == 4:
                    out[p] = color
    return out


def _orientation_witness(info, group):
    translated = _translate(group)
    h, w = info["grid"].shape
    if any(not (0 <= r < h and 0 <= c < w) or info["grid"][r, c] != info["background"]
           for r, c in translated):
        return False
    def observed(dest):
        return {q: info["marker_color"] if _lit(q) else group["color"]
                for p, q in zip(group["source"], dest)
                if _lit(p) != _lit(q) and not _inside(info, q)}
    actual = observed(group["dest"])
    return bool(actual) and actual != observed(translated)


def _evidence(info):
    states = Counter()
    evidence = {name: False for name in ("clearing", "ordinary_overwrite", "special_overwrite",
                                         "source_highlight", "source_survives", "memory")}
    witnesses, kinds = set(), set()
    for group in info["groups"]:
        kinds.add(group["kind"])
        if _orientation_witness(info, group):
            witnesses.add(group["kind"])
        for p, q in zip(group["source"], group["dest"]):
            if not _inside(info, q):
                states[f"{int(_lit(p))}{int(_lit(q))}"] += 1
            if _inside(info, p):
                evidence["clearing"] |= p[0] % 2 == 1
                evidence["ordinary_overwrite"] |= p[0] > 0 and p[0] % 6 in (0, 2)
                evidence["special_overwrite"] |= _lit(p)
                evidence["memory"] |= _lit(p) != _lit(q) and not _inside(info, q)
            else:
                evidence["source_highlight"] |= _lit(p)
                evidence["source_survives"] |= not _lit(p)
    evidence.update(states=states, witnesses=witnesses, kinds=kinds, cycle=info["grid"].shape[0] >= 13)
    return evidence


def _training_covered(infos, need_kinds):
    e = [_evidence(x) for x in infos]
    return (all(any(x[name] for x in e) for name in
                ("clearing", "ordinary_overwrite", "special_overwrite", "source_highlight", "source_survives", "cycle"))
            and set().union(*(set(x["states"]) for x in e)) == {"00", "01", "10", "11"}
            and set(need_kinds) <= set().union(*(x["witnesses"] for x in e)))


def _silhouette(rng, h, w, dense=False):
    """Random connected shapes, not a bank of example masks."""
    family = int(rng.integers(6)) if not dense else int(rng.choice([0, 4, 5]))
    mask = np.zeros((h, w), bool)
    if family == 0:
        mask[:] = True
    elif family == 1:
        mask[0, 0] = True
        target = int(rng.integers(max(4, h * w // 3), h * w + 1))
        for _ in range(target - 1):
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        a, b = int(r + dr), int(c + dc)
                        if 0 <= a < h and 0 <= b < w and not mask[a, b]:
                            frontier.add((a, b))
            choices = sorted(frontier)
            mask[choices[int(rng.integers(len(choices)))]] = True
    elif family == 2:
        # A diagonal spine with independently sampled short side branches.
        r = c = 0
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            r += int(r < h - 1)
            c += int(c < w - 1)
            mask[r, c] = True
            if rng.random() < .5 and c > 0:
                mask[r, c - 1] = True
    elif family == 3:
        c = 0
        for r in range(h):
            mask[r, c] = True
            step = int(rng.choice([-1, 1]))
            if not 0 <= c + step < w:
                step = -step
            c += step
    elif family == 4:
        mask[:] = True
        mask[:int(rng.integers(1, h)), -int(rng.integers(1, w)):] = False
    else:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
        if h > 3 and w > 3 and rng.random() < .5:
            mask[1, int(rng.integers(1, w - 1))] = True
    if rng.random() < .5:
        mask = mask[:, ::-1].copy()
    if rng.random() < .5:
        mask = mask[::-1].copy()
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    # Every payload is non-straight; thin diagonals remain allowed.
    if _kind(list(map(tuple, np.argwhere(mask).tolist()))) is not None:
        mask = np.ones((max(2, h), max(2, w)), bool)
        mask[0, -1] = False
    return mask.copy()


def _prototype(rng, kind, mode, large, dense):
    h = int(rng.integers(4, 7) if large else rng.integers(2, 6))
    w = int(rng.integers(3, 7))
    mask = _silhouette(rng, h, w, dense)
    if kind == "P":
        # Connect a variable shape to the center's nearby corner. Closed
        # quadrants permit row/column-aligned source cells, as in parent B.
        cells = set(map(tuple, np.argwhere(mask).tolist()))
        target = min(cells)
        r = c = 0
        cells.add((r, c))
        while (r, c) != target:
            r += int(np.sign(target[0] - r))
            c += int(np.sign(target[1] - c))
            cells.add((r, c))
        dr, dc = ((1, 1), (0, 1), (1, 0), (0, 0))[int(rng.integers(4))]
        sr, sc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
        source = sorted((sr * (r + dr), sc * (c + dc)) for r, c in cells if (r + dr, c + dc) != (0, 0))
        marker = [(0, 0)]
        dest = [(-r, -c) for r, c in source]
        # Removing the corner of a thin branch can split it; retry that shape.
        a0, a1, b0, b1 = _bbox(source)
        local = np.zeros((a1 - a0 + 1, b1 - b0 + 1), bool)
        for r, c in source:
            local[r - a0, c - b0] = True
        if len(_components(local)) != 1 or _kind(source) is not None:
            return None
    else:
        contact = int(rng.choice(np.flatnonzero(mask[0])))
        lo = int(rng.integers(-1, contact + 1))
        hi = int(rng.integers(max(lo + 1, contact), mask.shape[1] + 2))
        marker = [(0, c) for c in range(lo, hi + 1)]
        sign = int(rng.choice([-1, 1]))
        source = [(sign * (int(r) + 1), int(c)) for r, c in np.argwhere(mask)]
        dest = [(-r, c) for r, c in source]
        if kind == "V":
            marker, source, dest = [[(c, r) for r, c in x] for x in (marker, source, dest)]
    footprint = set(marker + source + dest)
    r0, _, c0, _ = _bbox(list(footprint))
    return dict(kind=kind, mode=mode,
                marker=[(r - r0, c - c0) for r, c in marker],
                source=[(r - r0, c - c0) for r, c in source],
                dest=[(r - r0, c - c0) for r, c in dest])


def _pack(rng, prototypes, depth, span, start, length):
    blocked = np.zeros((depth, span), bool)
    # A one-cell halo separates the edge seed from unrelated shapes.
    blocked[:2, max(0, start - 1):min(span, start + length + 1)] = True
    placed = []
    for group in sorted(prototypes, key=lambda x: -len(x["source"])):
        footprint = np.array(sorted(set(group["marker"] + group["source"] + group["dest"])))
        gh, gw = footprint.max(axis=0) + 1
        if gh > depth or gw > span:
            return None
        options = [(r, c) for r in range(depth - int(gh) + 1) for c in range(span - int(gw) + 1)]
        rng.shuffle(options)
        for r, c in options:
            markers = [(a + r, b + c) for a, b in group["marker"]]
            if any(a in (0, depth - 1) or b in (0, span - 1) or start <= b < start + length for a, b in markers):
                continue
            axis = markers[0][0]
            if group["kind"] != "V" and group["mode"] == "open" and axis % 3 == 1:
                continue
            if group["kind"] != "V" and group["mode"] == "locked" and axis % 3 != 1:
                continue
            rr, cc = footprint[:, 0] + r, footprint[:, 1] + c
            if blocked[rr, cc].any():
                continue
            moved = {key: ([(a + r, b + c) for a, b in value] if key in ("marker", "source", "dest") else value)
                     for key, value in group.items()}
            placed.append(moved)
            for a, b in zip(rr, cc):
                blocked[max(0, a - 1):min(depth, a + 2), max(0, b - 1):min(span, b + 2)] = True
            break
        else:
            return None
    return placed


def _input_key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _make_pair(rng, specs, side, roles, payload_pool, seen, prior_infos=(), required=(),
               required_states=(), teaching=False, require_memory=False, require_echo=False):
    background, clock, marker = roles
    for attempt in range(240):
        large = teaching and attempt < 100
        if len(specs) >= 3 or teaching:
            depth, span = int(rng.integers(23, 31)), int(rng.integers(23, 31))
        else:
            depth, span = int(rng.integers(12, 31)), int(rng.integers(12, 31))
        if attempt >= 100:
            depth = span = 30
        length = int(rng.integers(1, 5))
        if rng.random() < .12 and not teaching:
            length = int(rng.integers(4, min(9, span // 3) + 1))
        start = int(rng.integers(1 if length == 1 else 0, span - length + (0 if length == 1 else 1)))
        prototypes = [_prototype(rng, kind, mode, large, dense=attempt >= 60 or (teaching and rng.random() < .45))
                      for kind, mode in specs]
        if any(x is None for x in prototypes):
            continue
        placed = _pack(rng, prototypes, depth, span, start, length)
        if placed is None:
            continue
        g = np.full((depth, span), background, dtype=np.int8)
        g[0, start:start + length] = clock
        for group in placed:
            color = int(rng.choice(payload_pool))
            for p in group["marker"]:
                g[p] = marker
            for p in group["source"]:
                g[p] = color
        inp = np.rot90(g, -side).copy()
        if _input_key(inp) in seen:
            continue
        try:
            info = _read_input(inp)
        except ValueError:
            continue
        evidence = _evidence(info)
        if required and not set(required) <= evidence["witnesses"]:
            continue
        if not set(required_states) <= set(evidence["states"]):
            continue
        if require_memory and not evidence["memory"]:
            continue
        if require_echo and not (evidence["states"]["01"] or evidence["states"]["10"]):
            continue
        if prior_infos and not _training_covered(list(prior_infos) + [info], required):
            continue
        return dict(input=inp, output=np.rot90(_paint(info), -side).copy()), info
    raise RuntimeError("bounded construction could not find a readable teaching scene")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    order = [int(x) for x in rng.permutation(10)]
    roles = order[:3]
    pool = order[3:3 + int(rng.integers(1, 5))]
    kinds = (("H",), ("P",), ("H", "P"))[int(rng.choice(3, p=[.20, .20, .60]))]
    parallel = "H" in kinds and bool(rng.integers(2))
    base = int(rng.integers(4))
    opposed = bool(rng.integers(2))
    train_sides = [base if not opposed or i % 2 == 0 else (base + 2) % 4 for i in range(nt)]
    test_sides = [(base + int(rng.choice([1, 3]))) % 4 for _ in range(ns)]
    puzzle, seen, infos = {"train": [], "test": []}, set(), []
    for i in range(nt):
        if i == 0:
            specs = [(kind, "open") for kind in kinds]
            required, prior, states = kinds, (), ("00", "01", "10")
        elif i == 1:
            specs = [(kind, "locked") for kind in kinds]
            if parallel:
                specs.append(("V", "any"))
            required, prior, states = (), infos, ("11",)
        else:
            count = int(rng.integers(1, 6))
            choices = kinds + (("V",) if parallel else ())
            specs = [(str(rng.choice(choices)), str(rng.choice(["open", "locked", "any"]))) for _ in range(count)]
            required, prior, states = (), (), ()
        pair, info = _make_pair(rng, specs, train_sides[i], roles, pool, seen,
                                prior_infos=prior, required=required, required_states=states,
                                teaching=i < 2, require_memory=i == 0 and "P" in kinds)
        puzzle["train"].append(pair)
        seen.add(_input_key(pair["input"]))
        infos.append(info)
    for i in range(ns):
        choices = kinds + (("V",) if parallel else ())
        count = int(rng.integers(1, 6))
        specs = [(str(rng.choice(choices)), str(rng.choice(["open", "locked", "any"]))) for _ in range(count)]
        # At least one genuine symmetry echo remains visible in every query.
        specs[0] = (str(rng.choice(kinds)), "open")
        test_pool = pool if rng.random() < .65 else list(dict.fromkeys(pool + [order[-1]]))
        pair, info = _make_pair(rng, specs, test_sides[i], roles, test_pool, seen, require_echo=True)
        puzzle["test"].append(pair)
        seen.add(_input_key(pair["input"]))
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check arrays, observable semantics, counterexamples, and teaching coverage."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, training, tests, conventions = set(), [], [], set()
    alternatives = {name: False for name in ("all", "source", "destination", "or", "none", "translated")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _input_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            expected = _pixel_expected(info)
            canonical_out = np.rot90(out, info["side"])
            assert np.array_equal(canonical_out, expected), "incorrect output pixels"
            assert np.array_equal(solve(inp), out), "solver disagreement"
            assert not np.array_equal(inp, out)
            conventions.add((info["background"], info["clock"], info["marker_color"]))
            for group in info["groups"]:
                for p in group["marker"]:
                    assert canonical_out[p] == info["marker_color"], "marker erased"
                for p, q in zip(group["source"], group["dest"]):
                    if not _inside(info, q):
                        wanted = (info["marker_color"] if _lit(q) else group["color"]) if _lit(p) != _lit(q) else info["background"]
                        assert canonical_out[q] == wanted, "echo truth table violated"
            if split == "train":
                training.append(info)
                for name in alternatives:
                    alt = _paint(info, translated=True) if name == "translated" else _paint(info, name)
                    alternatives[name] |= not np.array_equal(alt, canonical_out)
            else:
                tests.append(info)
    assert len(conventions) == 1, "one palette convention per puzzle"
    needed = {g["kind"] for info in tests for g in info["groups"] if g["kind"] != "V"}
    trained = {g["kind"] for info in training for g in info["groups"]}
    assert _training_covered(training, needed), "demonstrations lack a required interaction"
    if "P" in trained:
        assert any(_evidence(x)["memory"] for x in training), "no erased-source echo demonstration"
    assert all(_evidence(x)["states"]["01"] or _evidence(x)["states"]["10"] for x in tests), "query has no visible echo"
    assert all(alternatives.values()), "demonstrations conceal a simpler alternative"
    assert {g["kind"] for info in tests for g in info["groups"]} <= trained


def _normalized(grid):
    mapping = {}
    return grid.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in grid.flat)


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed audit plus legal-count, mutation and covariance checks."""
    shapes, structures, sides, signatures, silhouettes = set(), set(), set(), set(), set()
    roles, counts, lengths, geometries = [set() for _ in range(3)], set(), set(), Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_input_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                shapes.add(pair["input"].shape)
                structures.add(_normalized(pair["input"]))
                sides.add((split, info["side"]))
                lengths.add(info["length"])
                counts.add(len(info["groups"]))
                for colors, name in zip(roles, ("background", "clock", "marker_color")):
                    colors.add(info[name])
                for group in info["groups"]:
                    r0, r1, c0, c1 = _bbox(group["source"])
                    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), bool)
                    for r, c in group["source"]:
                        local[r - r0, c - c0] = True
                    silhouettes.add((local.shape, local.tobytes()))
                    geometries["diagonal"] += len(_components(local, False)) > 1
                    geometries["hole"] += len(_components(~np.pad(local, 1), False)) > 1
                    geometries["rectangle"] += bool(local.all())
                    geometries[group["kind"]] += 1
                    if group["kind"] == "P":
                        a, b = group["marker"][0]
                        geometries["aligned_center"] += any(r == a or c == b for r, c in group["source"])
                    else:
                        coord = 1 if group["kind"] == "H" else 0
                        geometry = [p[coord] for p in group["source"]]
                        axis = [p[coord] for p in group["marker"]]
                        geometries["overhang"] += min(geometry) < min(axis) or max(geometry) > max(axis)
                # Quarter-turn covariance includes rectangular grids and palette roles.
                if split == "test":
                    rotated = np.rot90(pair["input"])
                    assert np.array_equal(solve(rotated), np.rot90(pair["output"]))
        if seed < 3:
            assert _equal(puzzle, generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(structures) >= 150 and len(silhouettes) >= 80
    assert sides == {(s, k) for s in ("train", "test") for k in range(4)}
    assert len(counts) >= 4 and len(lengths) >= 4
    assert all(len(c) >= 8 for c in roles)
    assert all(geometries[name] > 0 for name in
               ("H", "V", "P", "diagonal", "hole", "rectangle", "aligned_center", "overhang"))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(3100 + nt * 10 + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A direct small example makes source memory and strip priority concrete.
    small = np.zeros((14, 15), dtype=np.int8)
    small[0, 11] = 1
    small[6, 9] = 2
    for p in ((2, 11), (3, 11), (4, 11), (5, 10)):
        small[p] = 3
    result = solve(small)
    assert (result[2, 11], result[3, 11], result[4, 11]) == (1, 0, 2)
    assert (result[10, 7], result[8, 7], result[9, 7], result[7, 8]) == (2, 3, 0, 0)
    assert result[6, 9] == 2 and np.array_equal(small[:, 0], result[:, 0])
    # Renaming all ten colors must rename the result, including the background.
    permutation = np.array([7, 4, 9, 2, 0, 3, 5, 8, 1, 6], dtype=np.int8)
    assert np.array_equal(solve(permutation[small]), permutation[result])
    puzzle = generate_puzzle(0, 2, 1)
    # Corrupt an observable admitted echo; the validator must reject the label.
    bad = {s: [{k: _grid(v) for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    info = _read_input(bad["test"][0]["input"])
    q = next(q for g in info["groups"] for p, q in zip(g["source"], g["dest"])
             if _lit(p) != _lit(q) and not _inside(info, q))
    corrupted = np.rot90(bad["test"][0]["output"], info["side"]).copy()
    corrupted[q] = info["background"]
    bad["test"][0]["output"] = np.rot90(corrupted, -info["side"]).copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt echo accepted")
    return dict(seeds=42, pairs=210, shapes=len(shapes), normalized_inputs=len(structures),
                silhouettes=len(silhouettes), group_counts=sorted(counts), bar_lengths=sorted(lengths),
                geometry=dict(geometries), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
