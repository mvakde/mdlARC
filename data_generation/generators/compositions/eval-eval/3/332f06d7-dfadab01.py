"""Keyed cargo: a blueprint's actual ink footprint travels along a guide rail.

Structural colors are fixed: 0 wall, 1 floor, 2 rail, 3 destination.
Colors 4..9 encode a visible, puzzle-specific code-to-ink dictionary.
No parent generator, example fixture, or generation state is used by solve.
"""

from collections import deque
import copy
import json

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a


def _neighbors(p, shape, diagonal=False):
    r, c = p
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        offsets = tuple((dr, dc) for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1) if dr or dc)
    for dr, dc in offsets:
        q = r + dr, c + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, queue = [start], deque([start])
        while queue:
            p = queue.popleft()
            for q in _neighbors(p, mask.shape, diagonal):
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    queue.append(q)
        components.append(cells)
    return components


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _book_key(book):
    return tuple((code, ink, _key(mask)) for code, (ink, mask) in sorted(book.items()))


def _decode(inp):
    """Read every convention from this one input, including all blueprints."""
    inp = _grid(inp)
    book, singleton, prototype_cells = {}, [], set()
    for component in _components(inp >= 4, diagonal=True):
        if len(component) == 1:
            singleton.append(component[0])
            continue
        r = min(p[0] for p in component)
        c = min(p[1] for p in component)
        _require((r, c) in component, "prototype label must occupy its upper-left corner")
        code = int(inp[r, c])
        ink_cells = [p for p in component if p != (r, c)]
        inks = {int(inp[p]) for p in ink_cells}
        _require(len(inks) == 1 and code not in inks, "prototype needs one label and one ink")
        ink = next(iter(inks))
        _require(min(p[0] for p in ink_cells) == r + 1
                 and min(p[1] for p in ink_cells) == c + 1,
                 "prototype ink starts diagonally below its label")
        h = max(p[0] for p in ink_cells) - r
        w = max(p[1] for p in ink_cells) - c
        _require(2 <= h <= 5 and 2 <= w <= 5, "blueprint extent outside 2..5")
        mask = np.zeros((h, w), dtype=bool)
        for a, b in ink_cells:
            mask[a-r-1, b-c-1] = True
        _require(mask[0, 0] and len(_components(mask)) == 1, "blueprint must be connected with filled origin")
        _require(code not in book, "duplicate blueprint code")
        book[code] = ink, mask
        prototype_cells.update(component)
    _require(1 <= len(book) <= 3, "one through three blueprints required")
    codes, inks = set(book), {v[0] for v in book.values()}
    _require(len(inks) == len(book) and codes.isdisjoint(inks), "code and ink roles must be disjoint")
    _require(len(singleton) == 1, "exactly one cargo request required")
    start = singleton[0]
    code = int(inp[start])
    _require(code in book, "cargo code lacks a blueprint")
    goals = list(map(tuple, np.argwhere(inp == 3)))
    _require(len(goals) == 1, "exactly one goal required")
    target = goals[0]
    rail = (inp == 2) | (inp == 3)
    rail[start] = True
    points = set(map(tuple, np.argwhere(rail)))
    _require(len(points) >= 4, "rail too short")
    for p in points:
        degree = sum(q in points for q in _neighbors(p, inp.shape))
        _require(degree == (1 if p in (start, target) else 2), "rail must be a simple chain")
    path, previous, current = [start], None, start
    while current != target:
        options = [q for q in _neighbors(current, inp.shape) if q in points and q != previous]
        _require(len(options) == 1, "ambiguous rail successor")
        previous, current = current, options[0]
        _require(current not in path, "rail loops")
        path.append(current)
    _require(len(path) == len(points), "unvisited rail component")
    allowed = (inp == 1) | rail
    _require(len(_components(allowed)) == 1, "floor must form one corridor")
    # A wall moat separates informational prototypes from the motion scene.
    _require(not any(allowed[q] for p in prototype_cells
                     for q in _neighbors(p, inp.shape, diagonal=True)), "prototype touches corridor")
    return book, code, path, allowed, prototype_cells


def _footprint(p, mask):
    return {(p[0] + int(r), p[1] + int(c)) for r, c in np.argwhere(mask)}


def _fits(allowed, p, mask):
    h, w = mask.shape
    r, c = p
    if r < 0 or c < 0 or r + h > allowed.shape[0] or c + w > allowed.shape[1]:
        return False
    return bool(np.all(allowed[r:r+h, c:c+w][mask]))


def _last_valid(allowed, path, mask):
    _require(_fits(allowed, path[0], mask), "cargo does not fit at its source")
    for i, p in enumerate(path[1:], 1):
        if not _fits(allowed, p, mask):
            return i - 1
    return len(path) - 1


def _render(inp, path, index, ink, mask):
    out = np.array(inp, dtype=np.int8, copy=True)
    out[path[0]] = 2
    for p in _footprint(path[index], mask):
        out[p] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Materialize the requested motif at the last consecutive legal rail pose."""
    inp = _grid(input_grid)
    book, code, path, allowed, _ = _decode(inp)
    ink, mask = book[code]
    return _render(inp, path, _last_valid(allowed, path, mask), ink, mask)


def _turns(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _sample_mask(rng, gap_guarantee=False):
    low = 3 if gap_guarantee else 2
    h, w = int(rng.integers(low, 6)), int(rng.integers(low, 6))
    mask = np.zeros((h, w), dtype=bool)
    family = 0 if gap_guarantee else int(rng.integers(4))
    if family == 0:
        # Open corner/comb: its leading empty spaces are mechanically useful.
        mask[0, :] = mask[:, 0] = True
        for r in range(1, h-1):
            mask[r, :int(rng.integers(1, w))] = True
    elif family == 1 and h >= 3 and w >= 3:
        mask[0] = mask[-1] = True
        mask[:, 0] = mask[:, -1] = True
    else:
        r = c = 0
        mask[0, 0] = True
        while r < h-1 or c < w-1:
            if r == h-1 or (c < w-1 and rng.random() < .5):
                c += 1
            else:
                r += 1
            mask[r, c] = True
        for _ in range(int(rng.integers(h*w//2 + 1))):
            points = list(map(tuple, np.argwhere(mask)))
            p = points[int(rng.integers(len(points)))]
            neighbors = list(_neighbors(p, mask.shape))
            q = neighbors[int(rng.integers(len(neighbors)))]
            mask[q] = True
    if not gap_guarantee and mask.all():
        # A full 2x2 is still an ordinary legal blueprint; other masks teach gaps.
        return mask
    return mask


def _expand(nodes):
    path = [nodes[0]]
    for a, b in zip(nodes, nodes[1:]):
        dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
        while path[-1] != b:
            r, c = path[-1]
            path.append((r+dr, c+dc))
    return path


def _sample_rail(rng, shape, extent, turns):
    def axis(length, size):
        offset = int(rng.integers(1, 5))
        coordinates = [offset]
        stop = length - size - int(rng.integers(1, 4))
        while coordinates[-1] + 3 <= stop:
            step = int(rng.integers(3, min(6, stop-coordinates[-1]) + 1))
            coordinates.append(coordinates[-1] + step)
        return coordinates
    rows, cols = axis(shape[0], extent[0]), axis(shape[1], extent[1])
    if min(len(rows), len(cols)) < 2:
        return None
    for _ in range(80):
        node = int(rng.integers(len(rows))), int(rng.integers(len(cols)))
        walk, used, directions = [node], {node}, []
        desired_length = turns + int(rng.integers(2, 5))
        while len(walk) < desired_length:
            options = [q for q in _neighbors(walk[-1], (len(rows), len(cols))) if q not in used]
            options = [q for q in options if (sum(a != b for a, b in zip(directions, directions[1:]))
                       + int(bool(directions) and (q[0]-walk[-1][0], q[1]-walk[-1][1]) != directions[-1])) <= turns]
            if not options:
                break
            q = options[int(rng.integers(len(options)))]
            directions.append((q[0]-walk[-1][0], q[1]-walk[-1][1]))
            walk.append(q)
            used.add(q)
        nodes = [(rows[r], cols[c]) for r, c in walk]
        path = _expand(nodes)
        if _turns(path) == turns and len(path) >= 7:
            return path
    return None


def _paint_footprint(allowed, p, mask):
    r, c = p
    h, w = mask.shape
    allowed[r:r+h, c:c+w] |= mask


def _halo(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[dr:dr+h, dc:dc+w]
                                 for dr in range(3) for dc in range(3)])


def _place_blueprints(rng, inp, book):
    occupied = inp != 0
    order = list(book)
    rng.shuffle(order)
    for code in order:
        ink, mask = book[code]
        h, w = mask.shape
        candidates = [(r, c) for r in range(inp.shape[0]-h)
                      for c in range(inp.shape[1]-w)]
        rng.shuffle(candidates)
        forbidden = _halo(occupied)
        for r, c in candidates:
            # Protect the entire label/blueprint rectangle, not just ink cells.
            if forbidden[r:r+h+1, c:c+w+1].any():
                continue
            inp[r, c] = code
            block = inp[r+1:r+h+1, c+1:c+w+1]
            block[mask] = ink
            occupied[r:r+h+1, c:c+w+1] = True
            break
        else:
            return False
    return True


def _construct_pair(rng, book, code, blocked, turns, gap_case, used):
    ink, mask = book[code]
    extent = max(m.shape[0] for _, m in book.values()), max(m.shape[1] for _, m in book.values())
    for attempt in range(160):
        # More space on later attempts makes all vocabulary sizes pack reliably.
        low = 20 if attempt < 80 else 27
        shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        path = _sample_rail(rng, shape, extent, turns)
        if path is None:
            continue
        footprints = [_footprint(p, mask) for p in path]
        allowed = np.zeros(shape, dtype=bool)
        for p in path:
            _paint_footprint(allowed, p, mask)
        # Source chamber teaches that a full box can fit initially, then fail later.
        full = np.ones(mask.shape, dtype=bool)
        for p in path[:2]:
            _paint_footprint(allowed, p, full)
        # Independent chamber widenings, separated from the thin guide geometry.
        if rng.random() < .65:
            for i in rng.choice(len(path), int(rng.integers(1, min(5, len(path))+1)), replace=False):
                _paint_footprint(allowed, path[int(i)], full)
        stop_index = len(path)-1
        if blocked:
            prefix = set().union(*footprints[:2])
            choices = []
            rail_cells, goal_cells = set(path), footprints[-1]
            for i in range(2, len(path)-1):
                frontier = footprints[i] - prefix - rail_cells - goal_cells
                if frontier:
                    choices.append((i, sorted(frontier)))
                prefix.update(footprints[i])
            if not choices:
                continue
            i, frontier = choices[int(rng.integers(len(choices)))]
            rng.shuffle(frontier)
            for q in frontier[:int(rng.integers(1, min(3, len(frontier))+1))]:
                allowed[q] = False
            stop_index = i-1
        if len(_components(allowed)) != 1:
            continue
        if _last_valid(allowed, path, mask) != stop_index:
            continue
        if blocked and path[-1] in footprints[stop_index]:
            continue
        if gap_case:
            rectangle_index = _last_valid(allowed, path, full)
            if rectangle_index >= stop_index or rectangle_index >= len(path)-2:
                continue
        inp = np.zeros(shape, dtype=np.int8)
        inp[allowed] = 1
        for p in path:
            inp[p] = 2
        inp[path[0]], inp[path[-1]] = code, 3
        if not _place_blueprints(rng, inp, book):
            continue
        if _key(inp) in used:
            continue
        # The expected destination comes from the constructed first obstruction,
        # separately from the parser and sequential collision check in solve.
        out = _render(inp, path, stop_index, ink, mask)
        _require(np.array_equal(solve(inp), out), "construction/parser disagreement")
        used.add(_key(inp))
        return {"input": inp, "output": out}
    raise RuntimeError("bounded rail/blueprint construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    count = int(rng.integers(1, min(3, nt)+1))
    palette = rng.permutation(np.arange(4, 10))
    book = {int(palette[i]): (int(palette[count+i]), _sample_mask(rng, i == 0))
            for i in range(count)}
    codes = list(book)
    # Two irreducible demonstrations survive every legal count request.
    specs = [(codes[0], False, int(rng.integers(1, 3)), True),
             (codes[min(1, count-1)], True, int(rng.integers(0, 3)), False)]
    for i in range(2, nt):
        code = codes[i] if i < count else int(rng.choice(codes))
        specs.append((code, bool(rng.integers(2)), int(rng.integers(0, 3)), False))
    rng.shuffle(specs)
    puzzle, used = {"train": [], "test": []}, set()
    for code, blocked, turns, gap in specs:
        puzzle["train"].append(_construct_pair(rng, book, code, blocked, turns, gap, used))
    for _ in range(nq):
        code = int(rng.choice(codes))
        puzzle["test"].append(_construct_pair(rng, book, code, bool(rng.integers(2)),
                                               int(rng.integers(3, 7)), False, used))
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(inp):
    book, code, path, allowed, prototypes = _decode(inp)
    ink, mask = book[code]
    index = _last_valid(allowed, path, mask)
    full = np.ones(mask.shape, dtype=bool)
    # All generated source chambers fit a full box. Other valid family inputs
    # can start with walls in empty motif cells, which solve also supports.
    rectangle_index = _last_valid(allowed, path, full) if _fits(allowed, path[0], full) else -1
    return {"book": book, "code": code, "ink": ink, "mask": mask, "path": path,
            "allowed": allowed, "prototype_cells": prototypes, "index": index,
            "success": index == len(path)-1, "turns": _turns(path),
            "rectangle_index": rectangle_index, "shape": inp.shape}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, dictionary, analyses = set(), None, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            current = _book_key(info["book"])
            if dictionary is None:
                dictionary = current
            _require(current == dictionary, "blueprint meanings vary within puzzle")
            _require(np.array_equal(solve(inp), out), "output disagrees with input-only rule")
            index, path, mask = info["index"], info["path"], info["mask"]
            destination = _footprint(path[index], mask)
            _require(all(info["allowed"][p] for p in destination), "cargo overlaps wall")
            changed = set(map(tuple, np.argwhere(inp != out)))
            _require(changed <= destination | {path[0]}, "scenery or blueprint changed")
            _require(all(out[p] == info["ink"] for p in destination), "wrong cargo material")
            _require(np.count_nonzero((out == info["ink"]) & info["allowed"]) == int(mask.sum()),
                     "cargo mass or trail is incorrect")
            if not info["success"]:
                _require(not _fits(info["allowed"], path[index+1], mask), "not first collision")
                _require(out[path[-1]] == 3, "blocked cargo altered distant goal")
                _require(_fits(info["allowed"], path[-1], mask), "goal needs a valid downstream pose")
            else:
                _require(out[path[-1]] == info["ink"], "success failed to replace goal")
            analyses[split].append(info)
    train = analyses["train"]
    _require(any(v["success"] and v["index"] > 0 for v in train), "missing moving success")
    _require(any(v["success"] and v["turns"] >= 1 for v in train), "missing successful bend demonstration")
    _require(any(not v["success"] and v["index"] > 0 for v in train), "missing moving collision")
    _require(any(v["success"] and 1 <= v["rectangle_index"] < v["index"]
                 and v["rectangle_index"] < len(v["path"])-2 for v in train),
             "training does not show useful negative space")
    taught = {v["code"] for v in train}
    _require(taught == set(train[0]["book"]), "every blueprint must be activated in training")
    _require(all(v["turns"] <= 2 for v in train), "training sampling contract")
    _require(all(3 <= v["turns"] <= 6 for v in analyses["test"]), "missing longer folded test rail")


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded checks: 42 seeds, all explicit counts, ablations and corruption."""
    puzzles, rows, books, structures = [], [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                books.add(_book_key(info["book"]))
                rows.append((split, info))
                roles = {0: 0, 1: 1, 2: 2, 3: 3}
                for j, (code, (ink, _)) in enumerate(info["book"].items()):
                    roles[code], roles[ink] = 4 + 2*j, 5 + 2*j
                normalized = np.array([[roles[int(x)] for x in row] for row in pair["input"]], np.int8)
                structures.add(_key(normalized))
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed diversity")
    _require(len(books) >= 35 and len(structures) >= 180, "dictionary/geometry diversity")
    _require({len(v["book"]) for _, v in rows} == {1, 2, 3}, "vocabulary size diversity")
    _require(len({v["shape"] for _, v in rows}) >= 45, "canvas geometry diversity")
    _require(len({v["mask"].shape for _, v in rows}) >= 10, "motif extent diversity")
    _require({v["turns"] for _, v in rows} == set(range(7)), "route turn diversity")
    _require({v["success"] for split, v in rows if split == "test"} == {False, True}, "test outcomes")
    _require({v["code"] for _, v in rows} == set(range(4, 10)), "code palette coverage")
    _require({v["ink"] for _, v in rows} == set(range(4, 10)), "ink palette coverage")
    headings = {(v["path"][1][0]-v["path"][0][0], v["path"][1][1]-v["path"][0][1])
                for _, v in rows}
    _require(len(headings) == 4, "heading diversity")
    _require(_fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0)), "nondeterminism")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(192+n*10+m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "ignored counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (None, 1), (4, None), (2.0, 1), (True, 1), (3, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Direct expected example: an open corner clears walls in its empty cells.
    inp = np.zeros((12, 18), dtype=np.int8)
    mask = np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)
    inp[0, 0] = 4
    inp[1:4, 1:4][mask] = 5
    path = [(6, c) for c in range(1, 12)]
    inp[6, 1:14] = 1
    inp[7:9, 1:12] = 1
    inp[6, 1:12] = 2
    inp[6, 1], inp[6, 11] = 4, 3
    expected = inp.copy()
    expected[6, 1] = 2
    expected[6, 11:14] = 5
    expected[7:9, 11] = 5
    _require(np.array_equal(solve(inp), expected), "hand-specified gap clearance")
    # An earlier wall stops the cargo even though its final goal pose still fits.
    inp[8, 7] = 0
    expected = inp.copy()
    expected[6, 1] = 2
    expected[6, 6:9] = 5
    expected[7:9, 6] = 5
    _require(np.array_equal(solve(inp), expected), "hand-specified first obstruction")
    # Five wrong outputs must fail: no A, no B, trail, scenery, deleted cargo.
    for mutation in ("always_goal", "rectangle", "trail", "wall", "erase"):
        bad = copy.deepcopy(puzzles[0])
        if mutation == "always_goal":
            pair = next(p for p in bad["train"] if not _analyze(p["input"])["success"])
            v = _analyze(pair["input"])
            pair["output"] = _render(pair["input"], v["path"], len(v["path"])-1, v["ink"], v["mask"])
        elif mutation == "rectangle":
            pair = next(p for p in bad["train"] if _analyze(p["input"])["rectangle_index"] >= 1
                        and _analyze(p["input"])["rectangle_index"] < _analyze(p["input"])["index"])
            v = _analyze(pair["input"])
            pair["output"] = _render(pair["input"], v["path"], v["rectangle_index"],
                                      v["ink"], np.ones(v["mask"].shape, bool))
        else:
            pair = bad["test"][0]
            v = _analyze(pair["input"])
            if mutation == "trail":
                p = next(p for p in v["path"] if pair["output"][p] != v["ink"])
                pair["output"][p] = v["ink"]
            elif mutation == "wall":
                p = tuple(np.argwhere(pair["input"] == 0)[0])
                pair["output"][p] = 1
            else:
                pair["output"] = pair["input"].copy()
                pair["output"][v["path"][0]] = 2
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + mutation)
    return {"seeds": 42, "pairs": len(rows), "dictionaries": len(books),
            "structures": len(structures), "shapes": len({v["shape"] for _, v in rows}),
            "turn_counts": sorted({v["turns"] for _, v in rows}),
            "semantic_verification": "generator-owned invariants and explicit hand cases"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
