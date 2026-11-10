"""Reunion lines: rotational eligibility, exact anchors, minimum substitutions.

Only NumPy and the standard library are used. All solving conventions are
visible in the input or fixed here; generation metadata is never read by solve.
"""
from collections import Counter
from functools import lru_cache
import math

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _trim(mask):
    rr, cc = np.nonzero(mask)
    if not len(rr):
        raise ValueError("Empty glyph")
    return np.asarray(mask, bool)[rr.min():rr.max()+1, cc.min():cc.max()+1].copy()


def _key(mask):
    m = _trim(mask)
    return (m.shape, m.tobytes())


def _orbit(mask):
    result, seen = [], set()
    for k in range(4):
        m = _trim(np.rot90(mask, k))
        key = _key(m)
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


@lru_cache(maxsize=8192)
def _canonical_key(key):
    shape, data = key
    mask = np.frombuffer(data, dtype=bool).reshape(shape)
    return min((m.shape, m.tobytes()) for m in (np.rot90(mask, k) for k in range(4)))


def _canonical(mask):
    # This cache memoizes a pure geometric function, never examples or answers.
    return _canonical_key(_key(mask))


def _center(mask, size, background, ink):
    m = _trim(mask)
    a = np.full((size, size), background, np.int8)
    r, c = ((size - d) // 2 for d in m.shape)
    a[r:r+m.shape[0], c:c+m.shape[1]][m] = ink
    return a


def _token(block, background):
    colors = np.unique(block[block != background])
    if len(colors) != 1:
        raise ValueError("Every slot must contain a nonempty monochromatic glyph")
    ink = int(colors[0])
    mask = _trim(block != background)
    if not np.array_equal(block, _center(mask, len(block), background, ink)):
        raise ValueError("Glyphs must use the visible slot's centered convention")
    return {"mask": mask, "ink": ink, "exact": (_key(mask), ink),
            "family": _canonical(mask)}


def _slice(info, position):
    r, c = divmod(position, info["n"])
    s, b = info["s"], info["board0"]
    return (slice(b+1+r*(s+1), b+1+r*(s+1)+s),
            slice(1+c*(s+1), 1+c*(s+1)+s))


def _parse(value):
    g = _grid(value)
    if min(g.shape) < 3:
        raise ValueError("Missing framed lattice")
    bg, highlight, divider = int(g[0, 1]), int(g[0, 0]), int(g[1, 0])
    if len({bg, highlight, divider}) != 3:
        raise ValueError("Background, swatch, and frame colors must differ")
    bars = [r for r in range(1, len(g)) if np.all(g[r] == divider)]
    if len(bars) < 3 or bars[0] != 1:
        raise ValueError("Missing tray dividers")
    s = bars[1] - 2
    if not 3 <= s <= 5 or (g.shape[1]-1) % (s+1):
        raise ValueError("Invalid slot footprint")
    n = (g.shape[1]-1) // (s+1)
    board0 = s+4
    if (not 3 <= n <= 5 or g.shape[0] != s+5+n*(s+1)
            or bars != [1, s+2] + [board0+r*(s+1) for r in range(n+1)]):
        raise ValueError("Invalid board/tray geometry")
    expected = np.full(g.shape, bg, np.int8)
    expected[0, 0] = highlight
    expected[1:s+3, ::s+1] = divider
    expected[[1, s+2], :] = divider
    expected[board0:, ::s+1] = divider
    expected[board0::s+1, :] = divider
    info = {"grid": g, "bg": bg, "highlight": highlight, "divider": divider,
            "s": s, "n": n, "board0": board0}
    board, lookup = [], {}
    for p in range(n*n):
        rs, cs = _slice(info, p)
        block = g[rs, cs]
        t = _token(block, bg)
        if t["ink"] in (highlight, divider) or t["exact"] in lookup:
            raise ValueError("Ink roles must be separate and exact board tokens unique")
        lookup[t["exact"]] = p
        board.append(t)
        expected[rs, cs] = block
    query, anchors = [], []
    for c in range(n):
        cs = slice(1+c*(s+1), 1+c*(s+1)+s)
        block = g[2:s+2, cs]
        t = _token(block, bg)
        if t["exact"] not in lookup:
            raise ValueError("Each query must have an exact board anchor")
        query.append(t)
        anchors.append(lookup[t["exact"]])
        expected[2:s+2, cs] = block
    if len({t["family"] for t in query}) != n:
        raise ValueError("Invitations must name distinct rotational families")
    if not np.array_equal(g, expected):
        raise ValueError("Unexpected cells outside glyphs, marker, and lattice")
    info.update(board=board, query=query, anchors=anchors)
    return info


def _relation(token, kind):
    if kind == "rotation":
        return token["family"]
    if kind == "exact":
        return token["exact"]
    if kind == "reflection":
        return min(token["family"], _canonical(np.fliplr(token["mask"])))
    if kind == "area":
        return int(token["mask"].sum())
    raise ValueError("Unknown relation")


def _rank(info, relation="rotation", scoring="exact"):
    n = info["n"]
    requested = Counter(_relation(t, relation) for t in info["query"])
    families = [_relation(t, relation) for t in info["board"]]
    lines = [("row", r, tuple(r*n+c for c in range(n))) for r in range(n)]
    lines += [("col", c, tuple(r*n+c for r in range(n))) for c in range(n)]
    candidates = [line for line in lines if Counter(families[p] for p in line[2]) == requested]
    anchors = set(info["anchors"])
    query_shapes = {_key(t["mask"]) for t in info["query"]}

    def retained(line):
        if scoring == "exact":
            return len(anchors.intersection(line[2]))
        if scoring == "colorless":
            return sum(_key(info["board"][p]["mask"]) in query_shapes for p in line[2])
        if scoring == "none":
            return 0
        raise ValueError("Unknown scoring rule")

    best = max((retained(line) for line in candidates), default=-1)
    winners = [line for line in candidates if retained(line) == best]
    return {"candidates": candidates, "winners": winners,
            "positions": {p for line in winners for p in line[2]},
            "cost": n-best if candidates else None}


def _paint_result(info, positions):
    out = info["grid"].copy()
    for p in positions:
        block = out[_slice(info, p)]
        block[block == info["bg"]] = info["highlight"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Determine every output cell from this input alone."""
    info = _parse(input_grid)
    return _paint_result(info, _rank(info)["positions"])


def _random_mask(rng, size):
    h, w = map(int, rng.integers(1, size+1, size=2))
    kind = int(rng.integers(7))
    if kind == 0:
        m = np.ones((h, w), bool)
    elif kind == 1:
        m = np.ones((h, w), bool)
        if h >= 3 and w >= 3:
            m[1:-1, 1:-1] = False
    elif kind == 2:
        m = np.zeros((h, w), bool)
        m[:, 0] = True
        m[-1, :] = True
    elif kind == 3:
        m = np.indices((h, w)).sum(0) % 2 == 0
    else:
        m = rng.random((h, w)) < float(rng.uniform(.25, .85))
        if not m.any():
            m[int(rng.integers(h)), int(rng.integers(w))] = True
    return _trim(m)


def _sample_mask(rng, size, used, chiral=False, asymmetric=False):
    def permitted(m):
        key = _canonical(m)
        mirror = _canonical(np.fliplr(m))
        return (key not in used and mirror not in used
                and (not chiral or key != mirror)
                and (not asymmetric or len(_orbit(m)) == 4))
    for _ in range(160):
        m = _random_mask(rng, size)
        if permitted(m):
            return m
    # Finite, small exhaustive fallback; all sizes admit a 3-by-3 glyph.
    for bits in range(1, 512):
        m = _trim(np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3))
        if permitted(m):
            return m
    raise RuntimeError("Shape vocabulary exhausted")


def _abstract(rng, n, mode, axis, max_cost=None):
    """Build class/anchor relations, independently of their eventual pixels."""
    filler_count = int(rng.integers(1, min(7, n*n-n)+1))
    labels = rng.integers(n, n+filler_count, size=(n, n)).astype(int)
    forced = {}
    if mode in ("teach", "parallel", "dense_parallel", "exact"):
        k = 2 if mode in ("teach", "parallel") else n if mode == "dense_parallel" else 1
        indices = list(map(int, rng.choice(n, k, replace=False)))
        lines = []
        common_order = list(map(int, rng.permutation(n)))
        for index in indices:
            order = common_order if mode == "dense_parallel" else list(map(int, rng.permutation(n)))
            line = {}
            for j, cls in enumerate(order):
                rc = (index, j) if axis == "row" else (j, index)
                labels[rc] = cls
                line[cls] = rc
            lines.append(line)
        anchors = [lines[0][cls] for cls in range(n)]
        if k >= 2:
            anchors[-1] = lines[1][n-1]
        if mode == "teach":
            for cls in range(n-1):
                forced[lines[0][cls]] = (0, 0)
                forced[lines[1][cls]] = (0, 1)
            forced[lines[1][n-1]] = (0, 0)
            forced[lines[0][n-1]] = (1, 1)
        elif mode == "dense_parallel":
            # Different dispersed anchor arrangements, but a unique winner.
            for _ in range(64):
                trial = [lines[int(rng.integers(k))][cls] for cls in range(n)]
                scores = [sum(rc in line.values() for rc in trial) for line in lines]
                if (max(scores) < n and scores.count(max(scores)) == 1
                        and (max_cost is None or n-max(scores) <= max_cost)):
                    anchors = trial
                    break
        return labels, anchors, forced
    if mode == "cross":
        r, c = map(int, rng.integers(n, size=2))
        labels[r, c] = 0
        row_cells = [(r, j) for j in range(n) if j != c]
        col_cells = [(j, c) for j in range(n) if j != r]
        row_order, col_order = rng.permutation(n-1)+1, rng.permutation(n-1)+1
        row_members, col_members = {}, {}
        for rc, cls in zip(row_cells, row_order):
            labels[rc] = int(cls)
            row_members[int(cls)] = rc
        for rc, cls in zip(col_cells, col_order):
            labels[rc] = int(cls)
            col_members[int(cls)] = rc
        anchors = [(r, c), row_members[1], col_members[2]]
        outside = [(i, j) for i in range(n) for j in range(n) if i != r and j != c]
        rng.shuffle(outside)
        for cls in range(3, n):
            rc = outside.pop()
            labels[rc] = cls
            anchors.append(rc)
        return labels, anchors, forced
    if mode == "weave":
        rr, cc, symbols = rng.permutation(n), rng.permutation(n), rng.permutation(n)
        labels = symbols[(rr[:, None]+cc[None, :]) % n].astype(int)
        anchors = []
        for _ in range(64):
            anchors = []
            for cls in range(n):
                cells = [tuple(map(int, rc)) for rc in np.argwhere(labels == cls)]
                anchors.append(cells[int(rng.integers(len(cells)))])
            retained = max(max(Counter(r for r, c in anchors).values()),
                           max(Counter(c for r, c in anchors).values()))
            if retained < n and (max_cost is None or n-retained <= max_cost):
                break
        else:
            # Avoid a zero-cost answer without making a hidden rule exception.
            anchors = [tuple(map(int, np.argwhere(labels == cls)[0])) for cls in range(n)]
            cls = n-1
            anchors[cls] = tuple(map(int, np.argwhere(labels == cls)[1]))
        return labels, anchors, forced
    if mode in ("mirror", "scatter"):
        # A near-complete line is a useful negative, with its missing family's
        # exact invitation elsewhere. The replacement is a mirror or unrelated.
        index = int(rng.integers(n))
        order = list(map(int, rng.permutation(n)))
        anchors = [None]*n
        for j, cls in enumerate(order):
            rc = (index, j) if axis == "row" else (j, index)
            labels[rc] = cls if cls < n-1 else n
            if cls < n-1:
                anchors[cls] = rc
        outside = [(r, c) for r in range(n) for c in range(n)
                   if (r != index if axis == "row" else c != index)]
        rc = outside[int(rng.integers(len(outside)))]
        labels[rc] = n-1
        anchors[n-1] = rc
        return labels, anchors, forced
    raise ValueError("Unknown sampling mode")


def _make_pair(rng, n, s, mode, axis, stable_colors=None, max_cost=None):
    labels, anchor_cells, forced = _abstract(rng, n, mode, axis, max_cost)
    counts = Counter(map(int, labels.ravel()))
    bases, used = {}, set()
    for cls in range(n):
        base = _sample_mask(rng, s, used, chiral=mode == "mirror" and cls == n-1,
                            asymmetric=(mode == "teach" and cls == n-1)
                            or counts[cls] > 7
                            or (mode == "mirror" and cls == n-1 and counts[n] > 7))
        bases[cls] = base
        used.add(_canonical(base))
    if mode == "mirror":
        bases[n] = np.fliplr(bases[n-1]).copy()
        used.add(_canonical(bases[n]))
    for cls in sorted(counts):
        if cls not in bases:
            base = _sample_mask(rng, s, used, asymmetric=counts[cls] > 7)
            bases[cls] = base
            used.add(_canonical(base))
    colors = stable_colors if stable_colors is not None else tuple(map(int, rng.permutation(10)))
    bg, divider, highlight = colors[:3]
    minimum = max(math.ceil(counts[cls]/len(_orbit(bases[cls]))) for cls in counts)
    ink_count = max(minimum, int(rng.integers(2, 8)))
    inks = colors[3:3+ink_count]
    assert ink_count <= 7
    tokens = {}
    for cls in sorted(counts):
        orbit = _orbit(bases[cls])
        cells = [tuple(map(int, rc)) for rc in np.argwhere(labels == cls)]
        reserved = {forced[rc] for rc in cells if rc in forced}
        choices = [(rot, ink) for rot in range(len(orbit)) for ink in range(ink_count)
                   if (rot, ink) not in reserved]
        rng.shuffle(choices)
        for rc in cells:
            rot, ink = forced[rc] if rc in forced else choices.pop()
            tokens[rc] = _center(orbit[rot], s, bg, inks[ink])
    height, width = s+5+n*(s+1), n*(s+1)+1
    inp = np.full((height, width), bg, np.int8)
    inp[0, 0] = highlight
    inp[1:s+3, ::s+1] = divider
    inp[[1, s+2], :] = divider
    board0 = s+4
    inp[board0:, ::s+1] = divider
    inp[board0::s+1, :] = divider
    for (r, c), token in tokens.items():
        inp[board0+1+r*(s+1):board0+1+r*(s+1)+s,
            1+c*(s+1):1+c*(s+1)+s] = token
    for slot, cls in enumerate(rng.permutation(n)):
        inp[2:s+2, 1+slot*(s+1):1+slot*(s+1)+s] = tokens[anchor_cells[int(cls)]]
    # Labels provide a construction-time target; the exported solver instead
    # reconstructs masks, identities, and candidate lines from the pixels.
    anchors = {r*n+c for r, c in anchor_cells}
    lines = [tuple(r*n+c for c in range(n)) for r in range(n)]
    lines += [tuple(r*n+c for r in range(n)) for c in range(n)]
    eligible = [line for line in lines if sorted(int(labels.ravel()[p]) for p in line) == list(range(n))]
    best = max((len(anchors.intersection(line)) for line in eligible), default=-1)
    positions = {p for line in eligible if len(anchors.intersection(line)) == best for p in line}
    out = inp.copy()
    for p in positions:
        r, c = divmod(p, n)
        block = out[board0+1+r*(s+1):board0+1+r*(s+1)+s, 1+c*(s+1):1+c*(s+1)+s]
        block[block == bg] = highlight
    assert np.array_equal(out, solve(inp)), "Sampled class plan disagrees with visible geometry"
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(name + " is outside its permitted integer range")
    return int(value)


def _layout(rng, n):
    sizes = [s for s in (3, 4, 5) if s+5+n*(s+1) <= 30]
    return n, int(rng.choice(sizes))


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10))) if rng.random() < .3 else None
    n0 = int(rng.choice([3, 4, 5]))
    layout0 = _layout(rng, n0)
    layout1 = _layout(rng, int(rng.choice([n for n in (3, 4, 5) if n != n0]))) if nt >= 4 else layout0
    axis = str(rng.choice(["row", "col"]))
    puzzle, seen = {"train": [], "test": []}, set()

    def append(split, layout, mode, local_axis, max_cost=None):
        for _ in range(30):
            pair = _make_pair(rng, *layout, mode, local_axis, palette, max_cost)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                return
        raise RuntimeError("Distinct scene sampling budget exhausted")

    append("train", layout0, "teach", axis)
    append("train", layout0, "mirror", axis)
    if nt >= 3:
        append("train", layout0, "cross", axis)
    if nt >= 4:
        append("train", layout1, "exact", "col" if axis == "row" else "row")
    for _ in range(4, nt):
        mode = str(rng.choice(["parallel", "cross", "exact", "mirror", "scatter"]))
        append("train", [layout0, layout1][int(rng.integers(2))], mode,
               str(rng.choice(["row", "col"])))
    positive_layouts = [layout0, layout1] if nt >= 4 else [layout0]
    demonstrated_cost = max(_rank(_parse(pair["input"]))["cost"] or 0
                            for pair in puzzle["train"])
    for i in range(nq):
        layout = positive_layouts[int(rng.integers(len(positive_layouts)))]
        if i == 0:
            mode = "dense_parallel" if nt == 2 else "weave"
        else:
            mode = str(rng.choice(["weave", "parallel", "cross", "exact", "mirror", "scatter"]))
        if mode == "cross":
            layout = layout0  # This tied cross's substitution count is witnessed.
        append("test", layout, mode, axis if nt == 2 else str(rng.choice(["row", "col"])),
               max_cost=demonstrated_cost)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    puzzle["test"] = [puzzle["test"][int(i)] for i in rng.permutation(nq)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(info, relation="rotation", scoring="exact"):
    return _paint_result(info, _rank(info, relation, scoring)["positions"])


def validate_generated_puzzle(puzzle) -> None:
    """Check pixels, family constraints, teaching witnesses, and structural shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    infos, ranks, seen = {}, {}, set()
    for split in ("train", "test"):
        infos[split], ranks[split] = [], []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _parse(inp)
            rank = _rank(info)
            assert np.array_equal(out, _paint_result(info, rank["positions"])), "Incorrect winner painting"
            # Strong preservation: only neutral board-slot cells can change.
            immutable = inp != info["bg"]
            assert np.array_equal(out[immutable], inp[immutable])
            assert np.array_equal(out[:info["board0"]], inp[:info["board0"]])
            actual = set()
            for p in range(info["n"]**2):
                a, b = inp[_slice(info, p)], out[_slice(info, p)]
                neutral = a == info["bg"]
                if np.any(b[neutral] == info["highlight"]):
                    assert np.all(b[neutral] == info["highlight"]), "Partial slot highlight"
                    actual.add(p)
            # Fully filled glyphs have no paintable cells; compare only observable slots.
            observable = {p for p in rank["positions"]
                          if np.any(inp[_slice(info, p)] == info["bg"])}
            assert actual == observable
            infos[split].append(info)
            ranks[split].append(rank)
    train, test = infos["train"], infos["test"]
    train_ranks, test_ranks = ranks["train"], ranks["test"]
    def contradict(relation="rotation", scoring="exact"):
        return any(not np.array_equal(_alternative(i, relation, scoring), solve(i["grid"])) for i in train)
    assert contradict("exact"), "Ignoring A must change a demonstrated answer"
    assert contradict(scoring="none"), "Ignoring B's anchor identity must change a demonstrated answer"
    assert contradict(scoring="colorless"), "Exact anchor colors must be demonstrated"
    assert any(not r["candidates"] and _rank(i, "reflection")["positions"]
               and not np.array_equal(_alternative(i, "reflection"), solve(i["grid"]))
               for i, r in zip(train, train_ranks)), "Mirror-only false line must be demonstrated"
    assert contradict("area"), "Area-only companionship must be contradicted"
    assert any(r["cost"] is not None and r["cost"] > 0 and len(r["candidates"]) > len(r["winners"])
               for r in train_ranks)
    train_axes = {line[0] for rank in train_ranks for line in rank["winners"]}
    test_axes = {line[0] for rank in test_ranks for line in rank["winners"]}
    assert test_axes <= train_axes, "Winning axes must be demonstrated"
    if any(len(r["winners"]) > 1 for r in test_ranks):
        assert any(len(r["winners"]) > 1 for r in train_ranks), "Tied union must be taught"
    witnessed_layouts = {(i["n"], i["s"]) for i, r in zip(train, train_ranks) if r["winners"]}
    assert all((i["n"], i["s"]) in witnessed_layouts for i in test)
    assert max(r["cost"] or 0 for r in test_ranks) <= max(r["cost"] or 0 for r in train_ranks), "Untaught substitution count"
    assert max(len(r["candidates"]) for r in test_ranks) > max(len(r["candidates"]) for r in train_ranks), "Missing dense-candidate test shift"


def _puzzle_key(puzzle):
    return tuple((_fingerprint(p["input"]), _fingerprint(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded 42-seed suite, meaningful diversity, metamorphic/negative checks."""
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                rank = _rank(info)
                records.append((split, info, rank))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert {i["n"] for _, i, _ in records} == {3, 4, 5}
    assert {i["s"] for _, i, _ in records} == {3, 4, 5}
    assert len({i["grid"].shape for _, i, _ in records}) >= 6
    assert len({(i["bg"], i["divider"], i["highlight"]) for _, i, _ in records}) >= 30
    assert len({tuple(t["family"] for t in i["board"]) for _, i, _ in records}) >= 150
    assert len({len(r["candidates"]) for _, i, r in records}) >= 5
    assert {r["cost"] for _, i, r in records} >= {None, 0, 1, 2}
    assert any(len(r["winners"]) > 1 for s, i, r in records if s == "test")
    assert any(len(r["winners"]) == 1 for s, i, r in records if s == "test")
    areas = {int(t["mask"].sum()) for _, i, _ in records for t in i["board"]}
    assert {1, 3, 5, 8, 9} <= areas and max(areas) >= 15
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1900+a*10+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, "1"), (None, 1)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    baseline = puzzles[0]
    # Arbitrary palette bijections preserve every semantic role.
    remap = np.random.default_rng(91).permutation(10)
    recolored = {split: [{k: remap[np.asarray(v)] for k, v in p.items()} for p in baseline[split]]
                 for split in baseline}
    validate_generated_puzzle(recolored)
    # Query order changes no selected board slots.
    for pair in baseline["train"]+baseline["test"]:
        info = _parse(pair["input"])
        inp, expected = info["grid"].copy(), pair["output"].copy()
        for c in range(info["n"]):
            dest = slice(1+c*(info["s"]+1), 1+c*(info["s"]+1)+info["s"])
            src_c = (c+1) % info["n"]
            src = slice(1+src_c*(info["s"]+1), 1+src_c*(info["s"]+1)+info["s"])
            inp[2:info["s"]+2, dest] = pair["input"][2:info["s"]+2, src]
            expected[2:info["s"]+2, dest] = inp[2:info["s"]+2, dest]
        assert np.array_equal(solve(inp), expected)
        # Transpose the board and each query glyph together: rows and columns
        # exchange roles, with the same anchor scores and reflected vocabulary.
        inp, expected = pair["input"].copy(), pair["output"].copy()
        inp[info["board0"]:] = inp[info["board0"]:].T.copy()
        expected[info["board0"]:] = expected[info["board0"]:].T.copy()
        for c in range(info["n"]):
            cs = slice(1+c*(info["s"]+1), 1+c*(info["s"]+1)+info["s"])
            inp[2:info["s"]+2, cs] = inp[2:info["s"]+2, cs].T.copy()
            expected[2:info["s"]+2, cs] = inp[2:info["s"]+2, cs]
        assert np.array_equal(solve(inp), expected)
    for defect in ("ink", "frame", "winner", "duplicate"):
        bad = {split: [{k: np.asarray(v).copy() for k, v in p.items()} for p in baseline[split]]
               for split in baseline}
        p = bad["test"][0]
        info = _parse(p["input"])
        if defect == "duplicate":
            bad["test"][0] = {k: v.copy() for k, v in bad["train"][0].items()}
        elif defect == "frame":
            p["output"][1, 0] = info["bg"]
        elif defect == "ink":
            rs, cs = _slice(info, 0)
            block = p["output"][rs, cs]
            rc = tuple(np.argwhere(p["input"][rs, cs] != info["bg"])[0])
            block[rc] = info["bg"]
        else:
            rank = _rank(info)
            p["output"] = _paint_result(info, set(range(info["n"]**2))-rank["positions"])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Corrupted puzzle accepted: " + defect)
    return {"seeds": 42, "pairs": len(records), "board_sizes": sorted({i["n"] for _, i, _ in records}),
            "slot_sizes": sorted({i["s"] for _, i, _ in records}),
            "layouts": len({i["grid"].shape for _, i, _ in records}),
            "role_palettes": len({(i["bg"], i["divider"], i["highlight"]) for _, i, _ in records}),
            "candidate_counts": sorted({len(r["candidates"]) for _, i, r in records}),
            "maximum_glyph_area": max(areas)}


if __name__ == "__main__":
    print(self_test())
