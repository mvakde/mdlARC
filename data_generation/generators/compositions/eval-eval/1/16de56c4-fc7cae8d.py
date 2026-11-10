"""Keyed serpentine embroidery. Only NumPy and the Python standard library.

The input decoder has no access to seeds or construction metadata. Generation
paints labels from sampled path coordinates, then checks the visible input.
No function here writes files, logs, traces, or retained seed corpora.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


SIDES = ("T", "R", "B", "L")
MAPS = {
    "identity": ("T", "R", "B", "L"),
    "r90": ("L", "T", "R", "B"),
    "r180": ("B", "L", "T", "R"),
    "r270": ("R", "B", "L", "T"),
    "flip_lr": ("T", "L", "B", "R"),
    "flip_ud": ("B", "R", "T", "L"),
    "transpose": ("L", "B", "R", "T"),
    "anti_transpose": ("R", "T", "L", "B"),
}


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _transform(a, name):
    if name == "identity":
        result = a
    elif name.startswith("r"):
        result = np.rot90(a, int(name[1:]) // 90)
    elif name == "flip_lr":
        result = np.fliplr(a)
    elif name == "flip_ud":
        result = np.flipud(a)
    elif name == "transpose":
        result = a.T
    elif name == "anti_transpose":
        result = np.rot90(a.T, 2)
    else:
        raise ValueError("Unknown symmetry")
    return result.copy()


def _inverse(name):
    return {"r90": "r270", "r270": "r90"}.get(name, name)


def _side(a, side):
    return {"T": a[0], "R": a[:, -1], "B": a[-1], "L": a[:, 0]}[side]


def _adjacent(corner):
    return (SIDES[corner], SIDES[(corner + 1) % 4])


def _corner(sides):
    return next(i for i in range(4) if set(_adjacent(i)) == set(sides))


def _mapped_side(name, side):
    return MAPS[name][SIDES.index(side)]


def _unfold(a, snake=True):
    result = a.copy()
    if snake:
        result[1::2] = result[1::2, ::-1]
    return result.reshape(-1)


def _fold(flat, shape, snake=True):
    result = flat.reshape(shape).copy()
    if snake:
        result[1::2] = result[1::2, ::-1]
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.uint8, copy=False).tobytes()


def _guides(grid, bg, outer):
    bars = []
    for color in outer.values():
        positions = np.argwhere(grid[1:-1, 1:-1] == color) + 1
        if len(positions) < 4:
            raise ValueError("Each marker needs an interior guide")
        rr, cc = positions.T
        horizontal = len(set(rr)) == 1
        vertical = len(set(cc)) == 1
        if horizontal == vertical:
            raise ValueError("A guide must be one straight bar")
        axis = cc if horizontal else rr
        if int(axis.max() - axis.min() + 1) != len(axis):
            raise ValueError("A guide must be contiguous")
        bars.append((horizontal, int(color), int(rr.min()), int(cc.min()), len(axis)))
    if sum(b[0] for b in bars) != 1:
        raise ValueError("Need one horizontal and one vertical guide")
    hbar = next(b for b in bars if b[0])
    vbar = next(b for b in bars if not b[0])
    _, hc, hr, left, width = hbar
    _, vc, top, vcol, height = vbar
    if hr not in (top - 1, top + height) or vcol not in (left - 1, left + width):
        raise ValueError("Guides must border adjacent sides of one rectangle")
    guides = {"T" if hr == top - 1 else "B": hc,
              "L" if vcol == left - 1 else "R": vc}
    r0, r1 = top - ("T" in guides), top + height + ("B" in guides)
    c0, c1 = left - ("L" in guides), left + width + ("R" in guides)
    if min(r0, c0) < 2 or r1 > grid.shape[0] - 2 or c1 > grid.shape[1] - 2:
        raise ValueError("Board and guides need a clear moat")
    payload = grid[top:top + height, left:left + width].copy()
    if set(np.unique(payload)) & set(outer.values()):
        raise ValueError("Guide markers cannot be board ink")
    # Reconstruct all permitted non-payload cells; no hidden decorations.
    expected = np.full(grid.shape, bg, dtype=np.int8)
    for s, color in outer.items():
        _side(expected, s)[1:-1] = color
    expected[top:top + height, left:left + width] = payload
    _paint_guides(expected, guides, (top, left, height, width))
    if not np.array_equal(expected, grid):
        raise ValueError("Unexplained ink outside the guide-defined board")
    return guides, (top, left, height, width), payload


def _decode(input_grid):
    grid = _grid(input_grid)
    if min(grid.shape) < 9:
        raise ValueError("Canvas too small for a four-cell board and guides")
    corners = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
    if len(set(map(int, corners))) != 1:
        raise ValueError("The four canvas corners must show background")
    bg = int(corners[0])
    outer = {}
    for side in SIDES:
        core = _side(grid, side)[1:-1]
        if len(np.unique(core)) != 1:
            raise ValueError("Outer side cores must be complete")
        if int(core[0]) != bg:
            outer[side] = int(core[0])
    if len(outer) != 2 or len(set(outer.values())) != 2:
        raise ValueError("Need two distinct outer marker identities")
    if not any(set(outer) == set(_adjacent(i)) for i in range(4)):
        raise ValueError("Outer controls must be adjacent")
    guides, bounds, source = _guides(grid, bg, outer)
    matches = [name for name in MAPS
               if all(outer.get(_mapped_side(name, s)) == c for s, c in guides.items())]
    if len(matches) != 1:
        raise ValueError("Need exactly one marker-matched symmetry")
    name = matches[0]
    board = _transform(source, name)
    threads, noise = _threads(_unfold(board), bg)
    return dict(grid=grid, background=bg, outer=outer, guides=guides, bounds=bounds,
                source=source, transform=name, board=board, threads=threads, noise=noise,
                arrangement=(_corner(outer), _corner(guides)))


def _threads(flat, bg, strict=True):
    counts = Counter(int(c) for c in flat if int(c) != bg)
    if any(n > 2 for n in counts.values()):
        raise ValueError("Reference hues occur twice; other ink occurs once")
    refs = sorted(c for c, n in counts.items() if n == 2)
    if not 1 <= len(refs) <= 3:
        raise ValueError("Need one to three distinct reference pairs")
    singletons = [int(i) for i, c in enumerate(flat) if counts.get(int(c)) == 1]
    threads, claimed, occupied = [], set(), set()
    for color in refs:
        a, b = map(int, np.flatnonzero(flat == color))
        period = b - a
        full = list(range(a % period, len(flat), period))
        aligned = [i for i in singletons if (i - a) % period == 0]
        if len(aligned) > 1:
            raise ValueError("A thread has competing terminals")
        terminal = aligned[0] if aligned else None
        if terminal is not None and terminal in claimed:
            raise ValueError("A singleton belongs to two threads")
        if terminal is None:
            support, paint = full, color
            if strict and len(full) < 3:
                raise ValueError("A boundary thread must visibly extrapolate")
        else:
            lo, hi = min(a, terminal), max(b, terminal)
            support = [i for i in full if lo <= i <= hi]
            paint = int(flat[terminal])
            claimed.add(terminal)
        if strict:
            if occupied & set(support):
                raise ValueError("Threads cannot have overlapping painted supports")
            allowed = {bg, color, paint}
            if any(int(flat[i]) not in allowed for i in support):
                raise ValueError("A thread would overwrite unrelated ink")
        occupied.update(support)
        threads.append(dict(color=color, a=a, b=b, period=period, terminal=terminal,
                            paint=paint, support=support, full=full))
    return threads, sorted(set(singletons) - claimed)


def _complete(board, bg, *, snake=True, policy="normal", strict=True):
    flat = _unfold(board, snake)
    threads, _ = _threads(flat, bg, strict=strict)
    result = flat.copy()
    for s in threads:
        support, paint = s["support"], s["paint"]
        if policy == "ignore_terminal":
            support, paint = s["full"], s["color"]
        elif policy == "unbounded_terminal":
            support = s["full"]
        elif policy == "all_singletons":
            singles = [int(i) for i, c in enumerate(flat)
                       if c != bg and np.count_nonzero(flat == c) == 1]
            if singles:
                t = min(singles, key=lambda i: (min(abs(i - s["a"]), abs(i - s["b"])), i))
                lo, hi = min(s["a"], t), max(s["b"], t)
                support = [i for i in s["full"] if lo <= i <= hi]
                paint = int(flat[t])
        result[support] = paint
    return _fold(result, board.shape, snake)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from the input, using the family's fixed snake convention."""
    info = _decode(input_grid)
    return _complete(info["board"], info["background"])


def _paint_guides(grid, guides, bounds):
    r, c, h, w = bounds
    for side, color in guides.items():
        if side == "T":
            grid[r - 1, c:c + w] = color
        elif side == "B":
            grid[r + h, c:c + w] = color
        elif side == "L":
            grid[r:r + h, c - 1] = color
        else:
            grid[r:r + h, c + w] = color


def _layout(rng, canonical, answer, palette, inner_corner, name):
    bg, ma, mb = palette[:3]
    source = _transform(canonical, _inverse(name))
    h, w = source.shape
    guides = dict(zip(_adjacent(inner_corner), (ma, mb)))
    outer = {_mapped_side(name, s): color for s, color in guides.items()}
    ext_h, ext_w = h + 1, w + 1
    H = int(rng.integers(ext_h + 4, min(30, ext_h + 11) + 1))
    W = int(rng.integers(ext_w + 4, min(30, ext_w + 11) + 1))
    r0 = int(rng.integers(2, H - ext_h - 1))
    c0 = int(rng.integers(2, W - ext_w - 1))
    r, c = r0 + ("T" in guides), c0 + ("L" in guides)
    grid = np.full((H, W), bg, dtype=np.int8)
    for side, color in outer.items():
        _side(grid, side)[1:-1] = color
    grid[r:r + h, c:c + w] = source
    _paint_guides(grid, guides, (r, c, h, w))
    return {"input": grid, "output": answer.copy()}


def _swap_controls(pair):
    result = {k: v.copy() for k, v in pair.items()}
    info = _decode(pair["input"])
    colors = list(info["outer"].values())
    for side, color in info["outer"].items():
        _side(result["input"], side)[1:-1] = colors[1 - colors.index(color)]
    return result


def _alternatives(info):
    bg, board = info["background"], info["board"]
    result = {"copy": board}
    for policy in ("ignore_terminal", "unbounded_terminal", "all_singletons"):
        result[policy] = _complete(board, bg, policy=policy, strict=False)
    result["raster"] = _complete(board, bg, snake=False, strict=False)
    result["source_first"] = _transform(_complete(info["source"], bg, strict=False), info["transform"])
    return result


def _teaching_pair(rng, palette, inner, name):
    # Finite enumeration supplies a constructive contrast, not lucky rejection
    # of a whole puzzle. Fewer than 18,000 index triples are examined.
    h, w = int(rng.integers(6, 12)), int(rng.integers(6, 13))
    N = h * w
    bg, ref, terminal_color = palette[0], palette[3], palette[4]
    outer_sides = {_mapped_side(name, s) for s in _adjacent(inner)}
    relative = "transpose" if outer_sides in ({"T", "L"}, {"B", "R"}) else "anti_transpose"
    # Position labels give the exact index conversion under the key reversal.
    labels = _fold(np.arange(N), (h, w))
    reversed_labels = _unfold(_transform(labels, relative))
    index_map = np.argsort(reversed_labels)
    candidates = []
    for a in range(1, N - 2):
        b = a + 1
        aa, bb = sorted((int(index_map[a]), int(index_map[b])))
        d = bb - aa
        if d <= 1 or aa < d or bb + d >= N:
            continue
        for t in range(1, N - 1):
            if min(abs(t - a), abs(t - b)) < w + 2:
                continue
            if (int(index_map[t]) - aa) % d == 0:
                continue
            candidates.append((a, b, t, aa, bb))
    if not candidates:
        raise RuntimeError("No teaching geometry in the bounded enumeration")
    # Shuffle once; examine a bounded list until every promised shortcut has a
    # visible counterexample in these two demonstrations.
    for index in rng.permutation(len(candidates))[:256]:
        a, b, t, aa, bb = candidates[int(index)]
        flat = np.full(N, bg, dtype=np.int8)
        flat[[a, b]] = ref
        flat[t] = terminal_color
        out = flat.copy()
        out[min(a, t):max(b, t) + 1] = terminal_color
        canonical, answer = _fold(flat, (h, w)), _fold(out, (h, w))
        first = _layout(rng, canonical, answer, palette, inner, name)
        second = _swap_controls(first)
        board2 = _transform(canonical, relative)
        out2 = _unfold(board2)
        out2[aa % (bb - aa)::bb - aa] = ref
        second["output"] = _fold(out2, board2.shape)
        infos = [_decode(first["input"]), _decode(second["input"])]
        if all(any(not np.array_equal(_alternatives(info)[kind], p["output"])
                       for info, p in zip(infos, (first, second)))
               for kind in _alternatives(infos[0])):
            return first, second
    raise RuntimeError("Teaching contrast exhausted its bounded candidates")


def _random_pair(rng, palette, inner, name, count, mixed=False):
    h, w = int(rng.integers(4, 25)), int(rng.integers(4, 25))
    if rng.random() < .18:
        w = h
    N, bg = h * w, palette[0]
    base = int(rng.integers(count, min(count + 4, N // 3) + 1))
    residues = [int(x) for x in rng.choice(base, count, replace=False)]
    flags = [bool(rng.integers(2)) for _ in range(count)]
    if mixed:
        flags[:2] = [True, False]
        rng.shuffle(flags)
    refs = palette[3:3 + count]
    free = list(palette[3 + count:])
    flat = np.full(N, bg, dtype=np.int8)
    geometry = []
    for ref, residue, has_terminal in zip(refs, residues, flags):
        max_multiple = max(1, min(5, (N - 1 - residue) // (2 * base)))
        multiple = int(rng.integers(1, max_multiple + 1))
        d = base * multiple
        phases = [residue + base * q for q in range(multiple)
                  if len(range(residue + base * q, N, d)) >= 3]
        phase = int(_pick(rng, phases))
        positions = list(range(phase, N, d))
        pair_index = int(rng.integers(len(positions) - 1))
        a, b = positions[pair_index:pair_index + 2]
        flat[[a, b]] = ref
        if has_terminal:
            t = int(_pick(rng, [i for i in positions if i not in (a, b)]))
            paint = free.pop(0)
            flat[t] = paint
            support = [i for i in positions if min(a, t) <= i <= max(b, t)]
        else:
            t, paint, support = None, ref, positions
        geometry.append((a, b, d, t, paint, support))
    # Off-phase singletons cannot masquerade as any thread's terminal.
    noise_positions = [i for i in range(N) if all((i - a) % d for a, _, d, _, _, _ in geometry)]
    noise_count = int(rng.integers(min(3, len(free), len(noise_positions)) + 1))
    for i, color in zip(rng.choice(noise_positions, noise_count, replace=False), free):
        flat[int(i)] = color
    out = flat.copy()
    for _, _, _, _, paint, support in geometry:
        out[support] = paint
    return _layout(rng, _fold(flat, (h, w)), _fold(out, (h, w)), palette, inner, name)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    stable = bool(rng.random() < .25)
    inner = int(rng.integers(4))
    name = str(_pick(rng, ("r90", "r270")))
    first, second = _teaching_pair(rng, palette, inner, name)
    occupied = _decode(first["input"])["arrangement"]
    held = _pick(rng, [(o, i) for o in range(4) for i in range(4) if (o, i) != occupied])
    puzzle = {"train": [first, second], "test": []}
    seen = {_key(first["input"]), _key(second["input"])}
    maximum_train_count = 1
    for split, count in (("train", nt - 2), ("test", ne)):
        for index in range(count):
            if split == "train":
                signals = 2 if index == 0 else int(rng.integers(1, 4))
                maximum_train_count = max(maximum_train_count, signals)
            else:
                signals = int(rng.integers(1, maximum_train_count + 1))
            for _ in range(64):
                local = palette if stable else [int(c) for c in rng.permutation(10)]
                arrangement = held if split == "test" else _pick(
                    rng, [(o, i) for o in range(4) for i in range(4) if (o, i) != held])
                o, i = arrangement
                names = [n for n in MAPS if {_mapped_side(n, s) for s in _adjacent(i)} == set(_adjacent(o))]
                n = str(_pick(rng, names))
                pair = _random_pair(rng, local, i, n, signals, mixed=split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("Distinct-input retry cap reached")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate returned grids, family semantics, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "Duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            expected = _complete(info["board"], info["background"])
            assert np.array_equal(out, expected), "Incorrect folded-path completion"
            assert not np.array_equal(out, info["board"]), "Omitting A must change every answer"
            assert out.shape == info["board"].shape, "Guide-defined empty edges must survive"
            flat_in, flat_out = _unfold(info["board"]), _unfold(out)
            changed_support = set()
            for s in info["threads"]:
                assert all(flat_out[i] == s["paint"] for i in s["support"])
                assert any(flat_in[i] != flat_out[i] for i in s["support"])
                changed_support.update(s["support"])
            untouched = [i for i in range(len(flat_in)) if i not in changed_support]
            assert np.array_equal(flat_in[untouched], flat_out[untouched]), "Off-phase ink changed"
            infos[split].append(info)
    a, b = infos["train"][:2]
    pa, pb = puzzle["train"][:2]
    assert a["arrangement"] == b["arrangement"]
    assert a["bounds"] == b["bounds"] and a["guides"] == b["guides"]
    assert np.array_equal(a["source"], b["source"])
    assert np.array_equal(_swap_controls(pa)["input"], pb["input"]), "Need an exact identity reversal"
    assert a["transform"] in ("r90", "r270")
    assert b["transform"] not in ("identity", "r90", "r180", "r270")
    assert len(a["threads"]) == len(b["threads"]) == 1
    sa, sb = a["threads"][0], b["threads"][0]
    assert sa["period"] == 1 and sa["terminal"] is not None
    assert sb["period"] > 1 and sb["terminal"] is None and len(b["noise"]) == 1
    assert sa["support"][0] > 0 and sa["support"][-1] < a["board"].size - 1
    assert len(sa["support"]) > a["board"].shape[1] + 2, "Need a visible snake turn"
    assert sb["full"][0] < sb["a"] and sb["full"][-1] > sb["b"]
    assert not np.array_equal(pa["output"], pb["output"]), "Ignoring B must change the answer"
    alternatives = [_alternatives(a), _alternatives(b)]
    for kind in alternatives[0]:
        assert any(not np.array_equal(alt[kind], p["output"])
                   for alt, p in zip(alternatives, (pa, pb))), f"Training fails to refute {kind}"
    train_arrangements = {i["arrangement"] for i in infos["train"]}
    maximum = max(len(i["threads"]) for i in infos["train"])
    for i in infos["test"]:
        assert i["arrangement"] not in train_arrangements, "Test arrangement must be held out"
        assert len(i["threads"]) <= maximum, "Test thread count lacks a training witness"
    if maximum > 1:
        assert any(len(i["threads"]) > 1 and 0 < sum(s["terminal"] is not None for s in i["threads"])
                   < len(i["threads"]) for i in infos["train"]), "Need a mixed multithread witness"


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming tests. Return compact aggregate measurements."""
    signatures, shapes, board_shapes, transforms, backgrounds = set(), set(), set(), set(), set()
    periods, thread_counts, terminal_counts, geometries = set(), set(), set(), set()
    heterogeneous = noncollinear = blank_edges = pairs = 0
    reference = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        if seed == 0:
            reference = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                pairs += 1
                shapes.add(info["grid"].shape)
                board_shapes.add(info["board"].shape)
                transforms.add(info["transform"])
                backgrounds.add(info["background"])
                thread_counts.add(len(info["threads"]))
                terminal_counts.add(sum(s["terminal"] is not None for s in info["threads"]))
                heterogeneous += len({s["period"] for s in info["threads"]}) > 1
                blank_edges += any(np.all(_side(info["board"], s) == info["background"]) for s in SIDES)
                for s in info["threads"]:
                    periods.add(s["period"])
                    pos = np.argwhere(info["board"] == s["color"])
                    noncollinear += bool(pos[0, 0] != pos[1, 0] and pos[0, 1] != pos[1, 1])
                    geometries.add((info["board"].shape, s["a"], s["b"], s["terminal"]))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 50 and len(board_shapes) >= 50
    assert transforms == set(MAPS) and backgrounds == set(range(10))
    assert thread_counts == {1, 2, 3} and {0, 1, 2} <= terminal_counts
    assert len(periods) >= 20 and max(periods) >= 20
    assert heterogeneous >= 15 and noncollinear >= 30 and blank_edges >= 50
    assert len(geometries) >= 200
    # Exercise all orientations using the same canonical embroidery, so this
    # also tests transforms not selected in a particular random seed sample.
    rng = np.random.default_rng(93277)
    flat = np.full(63, 9, dtype=np.int8)
    flat[[12, 17]] = 0
    flat[37] = 3
    out = flat.copy()
    out[12:38:5] = 3
    for inner in range(4):
        for name in MAPS:
            pair = _layout(rng, _fold(flat, (7, 9)), _fold(out, (7, 9)),
                           [9, 1, 2, 0, 3, 4, 5, 6, 7, 8], inner, name)
            assert np.array_equal(solve(pair["input"]), pair["output"])
            permutation = np.array([6, 9, 1, 0, 7, 2, 8, 4, 5, 3], dtype=np.int8)
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    # Family members outside the sampler's shared residue partition: a solid
    # bounded thread beside a sparse boundary thread, and coprime periods with
    # disjoint bounded spans. The other thread's references can be on the full
    # phase but outside the bounded painted support.
    manual_cases = (
        ((6, 7, 17, 0, 3), (0, 20, None, 4, 4)),
        ((4, 6, 14, 0, 3), (33, 36, 45, 4, 5)),
        ((1, 31, None, 0, 0),),
    )
    for case in manual_cases:
        flat = np.full(63, 9, dtype=np.int8)
        for a, b, t, color, paint in case:
            flat[[a, b]] = color
            if t is not None:
                flat[t] = paint
        out = flat.copy()
        for a, b, t, color, paint in case:
            d = b - a
            if t is None:
                out[a % d::d] = color
            else:
                out[min(a, t):max(b, t) + 1:d] = paint
        pair = _layout(rng, _fold(flat, (7, 9)), _fold(out, (7, 9)),
                       [9, 1, 2, 0, 3, 4, 5, 6, 7, 8], 2, "anti_transpose")
        assert np.array_equal(solve(pair["input"]), pair["output"])
    # Exceptional associations must fail before any overwrite order is used.
    for points in ({0: 3, 2: 3, 4: 5, 6: 7},
                   {0: 3, 2: 3, 8: 4, 10: 4},
                   {0: 3, 3: 3, 6: 5, 9: 3}):
        flat = np.full(63, 9, dtype=np.int8)
        for position, color in points.items():
            flat[position] = color
        try:
            _threads(flat, 9)
        except ValueError:
            pass
        else:
            raise AssertionError("Ambiguous or overlapping threads accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123 + nt * 7 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for mutation in ("cell", "crop", "guide", "contrast"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in reference[s]] for s in reference}
        p = bad["test"][0]
        if mutation == "cell":
            p["output"][0, 0] = (int(p["output"][0, 0]) + 1) % 10
        elif mutation == "crop":
            p["output"] = p["output"][1:].copy()
        elif mutation == "contrast":
            bad["train"][1] = {k: v.copy() for k, v in bad["train"][0].items()}
        else:
            info = _decode(p["input"])
            color = next(iter(info["guides"].values()))
            rr, cc = np.argwhere(p["input"][1:-1, 1:-1] == color)[1] + 1
            p["input"][rr, cc] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"Corruption {mutation} accepted")
    return dict(seeds=42, pairs=pairs, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                board_shapes=len(board_shapes), symmetries=sorted(transforms),
                periods=sorted(periods), thread_counts=sorted(thread_counts),
                terminal_counts=sorted(terminal_counts), heterogeneous_period_pairs=heterogeneous,
                noncollinear_reference_pairs=noncollinear, blank_edge_boards=blank_edges,
                distinct_thread_geometries=len(geometries), independent_solver=False)
