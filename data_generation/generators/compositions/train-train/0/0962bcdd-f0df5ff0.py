"""Cross-coded, simultaneous two-color growth with conflict inhibition.

Only the original hub-colored pixels emit.  The unique bicolor cross in
each input supplies the hub/spoke roles.  No generation state is used by solve.
"""

from collections import deque
import numpy as np


CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SPOKE_OFFSETS = tuple((d * r, d * c) for d in (1, 2) for r, c in CARDINAL)
HUB_OFFSETS = tuple((d * r, d * c) for d in (1, 2) for r, c in DIAGONAL)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= s <= 30 for s in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(int, copy=False)


def _keys(g, background):
    """Find literal 3x3 bicolor crosses, with empty diagonal corners."""
    found = []
    height, width = g.shape
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            hub, spoke = int(g[row, col]), int(g[row - 1, col])
            if hub == background or spoke in (hub, background):
                continue
            if (all(g[row + dr, col + dc] == spoke for dr, dc in CARDINAL)
                    and all(g[row + dr, col + dc] == background for dr, dc in DIAGONAL)):
                found.append((row, col, hub, spoke))
    return found


def _decode(g):
    counts = np.bincount(g.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] * 2 <= g.size:
        raise ValueError("the background must occupy more than half of the input")
    keys = _keys(g, background)
    if len(keys) != 1:
        raise ValueError("expected exactly one bicolor cross key")
    row, col, hub, spoke = keys[0]
    return background, hub, spoke, (row, col)


def _claims(g, hub):
    """Boolean claims: multiplicity of one color is irrelevant."""
    hmask = np.zeros(g.shape, dtype=bool)
    smask = np.zeros(g.shape, dtype=bool)
    height, width = g.shape
    sources = np.argwhere(g == hub)
    for offsets, mask in ((HUB_OFFSETS, hmask), (SPOKE_OFFSETS, smask)):
        for dr, dc in offsets:
            rr, cc = sources[:, 0] + dr, sources[:, 1] + dc
            keep = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            mask[rr[keep], cc[keep]] = True
    return hmask, smask


def _apply(g, background, hub, spoke):
    hmask, smask = _claims(g, hub)
    out = g.copy()
    writable = g == background
    out[writable & hmask & ~smask] = hub
    out[writable & smask & ~hmask] = spoke
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible key and perform one masked, simultaneous update."""
    g = _grid(input_grid)
    background, hub, spoke, _ = _decode(g)
    return _apply(g, background, hub, spoke)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, queue = {start}, deque([start])
        while queue:
            row, col = queue.popleft()
            for dr, dc in CARDINAL:
                p = (row + dr, col + dc)
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    queue.append(p)
        result.append(component)
    return result


def _point_choice(rng, points):
    points = sorted(points)
    if not points:
        raise ValueError("no available witness or placement")
    return points[int(rng.integers(len(points)))]


def _polyomino(rng, size, style=None):
    if style is None:
        style = ("line", "stair", "grow", "compact")[int(rng.integers(4))]
        if size == 8 and rng.random() < .35:
            style = "ring"
        elif size == 9 and rng.random() < .25:
            style = "square"
    if style == "line":
        cells = {(0, c) for c in range(size)}
    elif style == "stair":
        cells, row, col = {(0, 0)}, 0, 0
        for i in range(1, size):
            if i % 2:
                col += 1
            else:
                row += 1
            cells.add((row, col))
    elif style == "ring":
        cells = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    elif style == "square":
        side = 2 if size == 4 else 3
        cells = {(r, c) for r in range(side) for c in range(side)}
    else:
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = {(r + dr, c + dc) for r, c in cells for dr, dc in CARDINAL} - cells
            if style == "compact":
                distances = {p: abs(p[0]) + abs(p[1]) for p in frontier}
                best = min(distances.values())
                frontier = {p for p in frontier if distances[p] == best}
            cells.add(_point_choice(rng, frontier))
    # A sampled dihedral transform changes the geometry without changing rules.
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    transformed = set()
    for row, col in cells:
        if reflect:
            col = -col
        for _ in range(turns):
            row, col = -col, row
        transformed.add((row, col))
    minr, minc = min(r for r, _ in transformed), min(c for _, c in transformed)
    return {(r - minr, c - minc) for r, c in transformed}


def _shape(rng):
    mode = int(rng.integers(6))
    if mode == 0:
        return int(rng.integers(7, 10)), int(rng.integers(18, 31))
    if mode == 1:
        return int(rng.integers(18, 31)), int(rng.integers(7, 10))
    if mode == 2:
        return 30, 30
    return int(rng.integers(11, 31)), int(rng.integers(11, 31))


def _place(rng, shape, cells, sources, key, *, clear_key=False, margin=0, edge=False):
    height, width = shape
    sh, sw = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
    candidates = []
    for row in range(margin, height - sh - margin + 1):
        for col in range(margin, width - sw - margin + 1):
            placed = {(row + r, col + c) for r, c in cells}
            if edge and not any(r in (0, height - 1) or c in (0, width - 1) for r, c in placed):
                continue
            if any(max(abs(r - key[0]), abs(c - key[1])) <= (4 if clear_key else 2)
                   for r, c in placed):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1
                   for r, c in placed for rr, cc in sources):
                continue
            candidates.append((row, col))
    row, col = _point_choice(rng, candidates)
    return {(row + r, col + c) for r, c in cells}


def _roles(rng, base, vary_background, split, index):
    background, hub, spoke = base
    if split == "train" and index == 1:
        hub, spoke = spoke, hub
    elif not (split == "train" and index == 0):
        mode = int(rng.integers(3))
        if mode == 1:
            hub, spoke = spoke, hub
        elif mode == 2:
            hub, spoke = (int(x) for x in rng.choice([c for c in range(10) if c != background],
                                                   size=2, replace=False))
    if vary_background and not (split == "train" and index == 0):
        background = int(rng.choice([c for c in range(10) if c not in (hub, spoke, base[0])]))
    return background, hub, spoke


def _make_pair(rng, base, vary_background, split, index):
    shape = _shape(rng)
    height, width = shape
    background, hub, spoke = _roles(rng, base, vary_background, split, index)
    exemplar = split == "train" and index == 0
    edge_demo = split == "train" and index == 1
    margin = 2 if exemplar or edge_demo or rng.random() < .7 else 1
    key = (int(rng.integers(margin, height - margin)),
           int(rng.integers(margin, width - margin)))
    sources = {key}
    if exemplar:
        payloads = [_polyomino(rng, 2, "line")]
    elif edge_demo:
        payloads = [_polyomino(rng, 3, "stair"), {(0, 0)}]
    else:
        count = int(rng.integers(1, 5))
        sizes = [int(rng.integers(5, 10)) if split == "test" else int(rng.integers(1, 5))]
        sizes += [int(rng.integers(1, 10 if split == "test" else 5)) for _ in range(count - 1)]
        payloads = [_polyomino(rng, n) for n in sizes]
    for i, cells in enumerate(payloads):
        sources |= _place(rng, shape, cells, sources, key,
                          clear_key=exemplar, margin=2 if exemplar else 0,
                          edge=edge_demo and i == 1)
    g = np.full(shape, background, dtype=int)
    for p in sources:
        g[p] = hub
    for dr, dc in CARDINAL:
        g[key[0] + dr, key[1] + dc] = spoke

    hmask, smask = _claims(g, hub)
    empty = g == background
    pure_h = set(map(tuple, np.argwhere(empty & hmask & ~smask)))
    pure_s = set(map(tuple, np.argwhere(empty & smask & ~hmask)))
    conflicts = set(map(tuple, np.argwhere(empty & hmask & smask)))
    outside = set(map(tuple, np.argwhere(empty & ~(hmask | smask))))
    protected = {(key[0] + dr, key[1] + dc)
                 for dr in range(-1, 2) for dc in range(-1, 2)}
    if exemplar:
        protected |= {(key[0] + dr, key[1] + dc)
                      for dr in range(-2, 3) for dc in range(-2, 3)}
        # Guarantee a visible counterexample to a second growth round. A future
        # H-only destination fed by a protected key diagonal remains such a
        # witness when later passive marks remove other possible H emitters.
        preview = _apply(g, background, hub, spoke)
        next_h, next_s = _claims(preview, hub)
        key_h = {key} | {(key[0] + dr, key[1] + dc) for dr, dc in HUB_OFFSETS}
        future = {tuple(p) for p in np.argwhere((preview == background) & next_h & ~next_s)
                  if any((p[0] - dr, p[1] - dc) in key_h for dr, dc in HUB_OFFSETS)}
        protected.add(_point_choice(rng, future))
    protected |= {_point_choice(rng, pure_h), _point_choice(rng, pure_s),
                  _point_choice(rng, outside)}
    if exemplar or conflicts:
        protected.add(_point_choice(rng, conflicts))
    if edge_demo:
        # Teach that conflicting colors inhibit even with unequal multiplicity.
        unequal = set()
        for p in conflicts:
            hc = sum((p[0] - dr, p[1] - dc) in sources for dr, dc in HUB_OFFSETS)
            sc = sum((p[0] - dr, p[1] - dc) in sources for dr, dc in SPOKE_OFFSETS)
            if hc != sc:
                unequal.add(p)
        protected.add(_point_choice(rng, unequal))
        edge_sources = {p for p in sources if p[0] in (0, height - 1) or p[1] in (0, width - 1)}
        edge_growth = {p for p in pure_h | pure_s
                       if any(max(abs(p[0] - r), abs(p[1] - c)) <= 2 for r, c in edge_sources)}
        protected.add(_point_choice(rng, edge_growth))

    available_colors = [c for c in range(10) if c not in (background, hub, spoke)]
    palette = [int(x) for x in rng.choice(available_colors, size=int(rng.integers(1, 8)), replace=False)]
    # The first demonstration explicitly has an intervening mark and paint beyond.
    if exemplar:
        bridges = []
        for row, col in sorted(sources - {key}):
            for dr, dc in CARDINAL:
                near, far = (row + dr, col + dc), (row + 2 * dr, col + 2 * dc)
                if near in pure_s - protected and far in pure_s:
                    bridges.append((near, far))
        if not bridges:
            raise ValueError("cannot expose non-shadowing mask behavior")
        near, far = bridges[int(rng.integers(len(bridges)))]
        protected.add(far)
        g[near] = palette[0]
    else:
        blocker = _point_choice(rng, (pure_h | pure_s) - protected)
        g[blocker] = palette[0]

    # Short passive patches and single marks both act as destination masks.
    for _ in range(int(rng.integers(4))):
        row, col = int(rng.integers(height)), int(rng.integers(width))
        patch = ((0, 0), (0, 1), (1, 0), (1, 1)) if rng.random() < .5 else ((0, 0), (1, 0), (2, 0))
        color = int(rng.choice(palette))
        for dr, dc in patch:
            p = row + dr, col + dc
            if 0 <= p[0] < height and 0 <= p[1] < width and g[p] == background and p not in protected:
                g[p] = color
    free = sorted(p for p in map(tuple, np.argwhere(g == background)) if p not in protected)
    rng.shuffle(free)
    cursor = 0
    for color in palette:
        if not np.any(g == color):
            if cursor >= len(free):
                raise ValueError("no room for the passive palette")
            g[free[cursor]] = color
            cursor += 1
    density = float(rng.uniform(.035, .18) if exemplar else rng.uniform(.05, .39))
    target = max(len(palette), int(g.size * density))
    already = int(np.count_nonzero((g != background) & (g != hub) & (g != spoke)))
    limit = (g.size - 1) // 2 - len(sources) - 4
    add = max(0, min(target, limit) - already)
    for p in free[cursor:cursor + add]:
        g[p] = int(rng.choice(palette))
    if _decode(g) != (background, hub, spoke, key):
        raise ValueError("accidental second key")
    return {"input": g, "output": solve(g)}


def _key(g):
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = tuple(int(c) for c in rng.permutation(10)[:3])
    vary_background = bool(rng.random() < .65)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(amount):
            for _ in range(160):
                try:
                    pair = _make_pair(rng, base, vary_background, split, index)
                except ValueError:
                    continue
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct, informative pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(g, out):
    """Destination-centric specification check, separate from solve's masks."""
    background, hub, spoke, key = _decode(g)
    sources = set(map(tuple, np.argwhere(g == hub)))
    _require(np.count_nonzero(g == spoke) == 4, "spoke markers occur only in the key")
    source_components = _components(sources - {key})
    sizes = sorted(len(c) for c in source_components)
    _require(1 <= len(sizes) <= 4 and all(1 <= n <= 9 for n in sizes), "invalid source island count/size")
    _require(all(max(abs(r - key[0]), abs(c - key[1])) >= 3 for r, c in sources - {key}),
             "payload islands are too close to the key")
    _require(all(max(abs(r - rr), abs(c - cc)) >= 2
                 for i, a in enumerate(source_components) for b in source_components[i + 1:]
                 for r, c in a for rr, cc in b), "distinct source islands touch diagonally")
    _require(out.shape == g.shape, "canvas dimensions must be preserved")
    expected = g.copy()
    pure_h, pure_s, conflicts, blocked, outside = set(), set(), set(), set(), set()
    for row in range(g.shape[0]):
        for col in range(g.shape[1]):
            hc = any((row - dr, col - dc) in sources for dr, dc in HUB_OFFSETS)
            sc = any((row - dr, col - dc) in sources for dr, dc in SPOKE_OFFSETS)
            p = row, col
            if g[p] != background:
                if hc != sc and g[p] not in (hub, spoke):
                    blocked.add(p)
                continue
            if hc and sc:
                conflicts.add(p)
            elif hc:
                expected[p] = hub
                pure_h.add(p)
            elif sc:
                expected[p] = spoke
                pure_s.add(p)
            else:
                outside.add(p)
    _require(np.array_equal(out, expected), "output violates simultaneous agreement-only growth")
    _require(pure_h and pure_s, "every pair must expose both growth colors")
    _require(blocked, "every pair must expose preservation of a targeted passive mark")
    _require(outside, "every pair needs untouched empty space outside all claims")
    _require(np.array_equal(g[g != background], out[g != background]), "original marks changed")
    return {"background": background, "hub": hub, "spoke": spoke, "key": key,
            "sources": sources, "sizes": sizes, "h_growth": pure_h,
            "s_growth": pure_s, "conflicts": conflicts, "blocked": blocked,
            "outside": outside,
            "edge": any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1) for r, c in sources)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "expected train/test only")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            info = _inspect(g, out)
            _require(np.array_equal(solve(g), out), "solve disagrees with specification check")
            if split == "train":
                _require(max(info["sizes"]) <= 4, "training islands must have at most four cells")
            else:
                _require(max(info["sizes"]) >= 5, "test must include a larger connected island")
            infos[split].append(info)
    first, second = infos["train"][:2]
    _require(first["hub"] == second["spoke"] and first["spoke"] == second["hub"],
             "first demonstrations must exchange literal hub/spoke roles")
    _require(first["sizes"] == [2] and first["conflicts"], "first pair must expose domino interference")
    _require(second["edge"], "clipping needs a training demonstration")
    second_grid = _grid(puzzle["train"][1]["input"])
    edge_sources = {p for p in second["sources"]
                    if p[0] in (0, second_grid.shape[0] - 1) or p[1] in (0, second_grid.shape[1] - 1)}
    _require(any(max(abs(r - rr), abs(c - cc)) <= 2
                 for r, c in second["h_growth"] | second["s_growth"] for rr, cc in edge_sources),
             "boundary source needs exposed in-bounds growth")
    _require(any(sum((p[0] - dr, p[1] - dc) in second["sources"] for dr, dc in HUB_OFFSETS)
                 != sum((p[0] - dr, p[1] - dc) in second["sources"] for dr, dc in SPOKE_OFFSETS)
                 for p in second["conflicts"]), "training must distinguish inhibition from majority voting")
    g, out = map(_grid, (puzzle["train"][0]["input"], puzzle["train"][0]["output"]))
    row, col = first["key"]
    _require(2 <= row < g.shape[0] - 2 and 2 <= col < g.shape[1] - 2, "clear key must fit")
    _require(all(max(abs(r - row), abs(c - col)) >= 5 for r, c in first["sources"] - {(row, col)}),
             "first key's motif must be isolated")
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            expected = (first["hub"] if (dr, dc) == (0, 0) else
                        first["spoke"] if (dr, dc) in CARDINAL else first["background"])
            _require(g[row + dr, col + dc] == expected, "first key must have a clear radius-two window")
    for dr, dc in HUB_OFFSETS:
        _require(out[row + dr, col + dc] == first["hub"], "clear key must show diagonal growth")
    for dr, dc in SPOKE_OFFSETS:
        _require(out[row + dr, col + dc] == first["spoke"], "clear key must show cardinal growth")
    # A-blind ablation: B's original 3x3 active-color stamp.
    naive = g.copy()
    for r, c in first["sources"]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == first["background"]:
                    naive[rr, cc] = first["hub"]
    _require(not np.array_equal(naive, out), "omitting A must change the answer")
    # B-blind ablation: erase the masking constraint at a singly claimed mark.
    _require(first["blocked"], "omitting B must overwrite a passive witness")
    shadow_witness = False
    for r, c in first["sources"]:
        for dr, dc in CARDINAL:
            near, far = (r + dr, c + dc), (r + 2 * dr, c + 2 * dc)
            if near in first["blocked"] and far in first["s_growth"]:
                shadow_witness = True
    _require(shadow_witness, "masking must be distinguished from blocked rays")
    _require(not np.array_equal(_apply(out, first["background"], first["hub"], first["spoke"]), out),
             "first demonstration must distinguish one step from recursive growth")


def self_test():
    """Bounded 42-seed diversity, invariants, metamorphic checks, and counts."""
    shapes, roles, sizes, backgrounds, layouts, structures = set(), set(), set(), set(), set(), set()
    passive_counts, edge_splits, concave = set(), set(), False
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(_grid(p["input"])) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g, out = pair["input"], pair["output"]
                f = _inspect(g, out)
                shapes.add(g.shape)
                backgrounds.add(f["background"])
                roles.add((f["hub"], f["spoke"]))
                sizes.update(f["sizes"])
                layouts.add((g.shape, tuple(sorted(f["sources"]))))
                normalized, mapping = [], {}
                for cell in g.flat:
                    normalized.append(mapping.setdefault(int(cell), len(mapping)))
                structures.add((g.shape, tuple(normalized)))
                passive_counts.add(len(set(map(int, g.flat)) - {f["background"], f["hub"], f["spoke"]}))
                if f["edge"]:
                    edge_splits.add(split)
                concave |= any(len(c) == 8 and len({r for r, _ in c}) == 3 and len({col for _, col in c}) == 3
                               for c in _components(f["sources"] - {f["key"]}))
        # Rule is equivariant to geometry and to arbitrary literal color names.
        sample = puzzle["test"][0]
        _require(np.array_equal(solve(np.rot90(sample["input"])), np.rot90(sample["output"])),
                 "rotation changes the rule")
        _require(np.array_equal(solve(sample["input"][:, ::-1]), sample["output"][:, ::-1]),
                 "reflection changes the rule")
        permutation = np.random.default_rng(10000 + seed).permutation(10)
        _require(np.array_equal(solve(permutation[sample["input"]]), permutation[sample["output"]]),
                 "literal color names change the rule")
    _require(len(fingerprints) == 42, "seeds must produce distinct puzzles")
    _require(len(shapes) >= 65 and (30, 30) in shapes and min(min(s) for s in shapes) == 7,
             "insufficient canvas variation")
    _require(len(backgrounds) == 10 and len(roles) >= 30, "insufficient role variation")
    _require(sizes == set(range(1, 10)) and passive_counts == set(range(1, 8)), "counts lack coverage")
    _require(len(layouts) >= 180 and len(structures) >= 180, "variation must go beyond recoloring")
    _require(edge_splits == {"train", "test"} and concave, "boundary/concavity variation missing")
    # Exercise every legal explicit count combination, plus representative type errors.
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(42000 + train * 10 + test, train, test)
            _require((len(p["train"]), len(p["test"])) == (train, test), "wrong explicit counts")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, again = generate_puzzle(12345, 2, 1), generate_puzzle(12345, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in ("train", "test")
                 for a, b in zip(first[s], again[s]) for k in ("input", "output")), "nondeterminism")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts, geometry/color equivariance)")
