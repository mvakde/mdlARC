"""ARC family: region boundaries cut connectivity before area-four inking.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = DIR4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _components(mask, directions=DIR4):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], set()
        while stack:
            r, c = stack.pop()
            cells.add((int(r), int(c)))
            for dr, dc in directions:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _regions(h, w, inset=2):
    rr, cc = np.indices((h, w))
    z = 2 * (rr >= h // 2) + (cc >= w // 2)
    z[(rr >= inset) & (rr < h - inset) & (cc >= inset) & (cc < w - inset)] = 4
    return z.astype(np.int8)


def _read_input(value):
    """Read enclosing multicolored cycles, including floating interior shapes."""
    a = _grid(value)
    bg = int(a[0, 0])  # A guaranteed exterior blank, not a majority heuristic.
    boards = []
    for comp in _components(a != bg):
        r0 = min(r for r, _ in comp)
        r1 = max(r for r, _ in comp) + 1
        c0 = min(c for _, c in comp)
        c1 = max(c for _, c in comp) + 1
        h, w = r1 - r0 - 2, c1 - c0 - 2
        if min(h, w) < 6 or h % 2 or w % 2:
            continue
        b = a[r0:r1, c0:c1]
        corners = tuple(int(b[r, c]) for r, c in
                        ((0, 0), (0, -1), (-1, 0), (-1, -1)))
        center = int(b[0, 1])
        if len(set((*corners, center, bg))) != 6:
            continue
        if not (np.all(b[0, 1:-1] == center) and np.all(b[-1, 1:-1] == center)
                and np.all(b[1:-1, 0] == center) and np.all(b[1:-1, -1] == center)):
            continue
        interior = b[1:-1, 1:-1]
        sources = set(map(int, np.unique(interior))) - {bg}
        if len(sources) != 1:
            raise ValueError("each framed interior needs exactly one source color")
        source = next(iter(sources))
        if source in (*corners, center):
            raise ValueError("source and inks must differ")
        boards.append((r0, c0, h, w, (*corners, center), source))
    if not 1 <= len(boards) <= 4:
        raise ValueError("expected one to four intact frames")
    claimed = np.zeros(a.shape, dtype=bool)
    for r, c, h, w, _, _ in boards:
        if np.any(claimed[r:r+h+2, c:c+w+2]):
            raise ValueError("frames overlap or nest")
        claimed[r:r+h+2, c:c+w+2] = True
    if np.any((a != bg) & ~claimed):
        raise ValueError("unframed non-background cells")
    for i, (r, c, h, w, _, _) in enumerate(boards):
        for s, t, u, v, _, _ in boards[i+1:]:
            if not (r+h+2 < s or s+u+2 < r or c+w+2 < t or t+v+2 < c):
                raise ValueError("frames must have a blank separating row or column")
    return a, bg, sorted(boards)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every ink, region, and fragment from this input alone."""
    a, _, boards = _read_input(input_grid)
    out = a.copy()
    for r, c, h, w, inks, source in boards:
        work = a[r+1:r+h+1, c+1:c+w+1]
        target = out[r+1:r+h+1, c+1:c+w+1]
        zones = _regions(h, w)
        for z in range(5):
            for fragment in _components((work == source) & (zones == z)):
                if len(fragment) >= 4:
                    for y, x in fragment:
                        target[y, x] = inks[z]
    return out


def _normalize(cells):
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _orientations(groups):
    seen, result = set(), []
    for swap in (False, True):
        for sr in (-1, 1):
            for sc in (-1, 1):
                transformed = [[(sr*(c if swap else r), sc*(r if swap else c))
                                for r, c in group] for group in groups]
                r0 = min(r for g in transformed for r, _ in g)
                c0 = min(c for g in transformed for _, c in g)
                key = tuple(tuple(sorted((r-r0, c-c0) for r, c in g)) for g in transformed)
                if key not in seen:
                    seen.add(key)
                    result.append(key)
    return result


def _touches(cells, occupied):
    return bool(cells & occupied) or any((r+dr, c+dc) in occupied
                                        for r, c in cells for dr, dc in DIR4)


MOTIFS = {
    # A single seven-cell component split 4 / 3 at the inset.
    "boundary": (((0, 0), (0, 1), (1, 0), (1, 1)),
                 ((2, 0), (2, 1), (3, 0))),
    # A four-cell square split 2 / 2 at a quadrant midpoint.
    "split": (((0, 0), (1, 0)), ((0, 1), (1, 1))),
    # Two disconnected two-cell corner arms rejoin through four center cells.
    "reentry": (((1, 3), (1, 4)), ((3, 1), (4, 1)),
                ((2, 3), (3, 3), (3, 2), (2, 2))),
    # Equal bounding boxes, different area; contact is only diagonal.
    "diagonal": (((0, 0), (1, 0), (1, 1)),
                 ((2, 2), (2, 3), (3, 2), (3, 3))),
}


def _place_motif(rng, occupied, zones, name):
    h, w = zones.shape
    choices = _orientations(MOTIFS[name])
    for oi in rng.permutation(len(choices)):
        groups = choices[int(oi)]
        sh = 1 + max(r for g in groups for r, _ in g)
        sw = 1 + max(c for g in groups for _, c in g)
        if sh > h or sw > w:
            continue
        positions = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
        for pi in rng.permutation(len(positions)):
            top, left = positions[int(pi)]
            shifted = [{(r+top, c+left) for r, c in group} for group in groups]
            labels = [{int(zones[r, c]) for r, c in g} for g in shifted]
            if any(len(v) != 1 for v in labels):
                continue
            zs = [next(iter(v)) for v in labels]
            if name == "boundary":
                valid = zs[0] < 4 and zs[1] == 4
            elif name == "split":
                valid = zs[0] < 4 and zs[1] < 4 and zs[0] != zs[1]
            elif name == "reentry":
                valid = zs[0] == zs[1] < 4 and zs[2] == 4
            else:
                valid = zs == [4, 4]
            cells = set().union(*shifted)
            if valid and not _touches(cells, occupied):
                occupied.update(cells)
                return True
    return False


def _polyomino(rng, area):
    style = int(rng.integers(5))
    if style == 0:
        cells = {(0, c) for c in range(area)}
    elif style == 1 and area >= 3:
        arm = int(rng.integers(2, area))
        cells = {(0, c) for c in range(arm)} | {(r, 0) for r in range(area-arm+1)}
    elif style == 2 and area == 4:
        cells = {(0, 0), (0, 1), (0, 2), (1, 1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIR4} - cells)
            if style == 3:
                weights = np.array([1+4*sum((r+dr, c+dc) in cells for dr, dc in DIR4)
                                    for r, c in frontier], dtype=float) ** 2
                index = int(rng.choice(len(frontier), p=weights/weights.sum()))
            else:
                index = int(rng.integers(len(frontier)))
            cells.add(frontier[index])
    oriented = _orientations([cells])
    return oriented[int(rng.integers(len(oriented)))][0]


def _place_shape(rng, occupied, zones, area, region=None):
    h, w = zones.shape
    for _ in range(18):
        shape = _polyomino(rng, area)
        sh, sw = 1+max(r for r, _ in shape), 1+max(c for _, c in shape)
        if sh > h or sw > w:
            continue
        positions = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
        for i in rng.permutation(len(positions)):
            top, left = positions[int(i)]
            cells = {(r+top, c+left) for r, c in shape}
            if region is not None and any(zones[r, c] != region for r, c in cells):
                continue
            if not _touches(cells, occupied):
                occupied.update(cells)
                return True
    return False


def _populate(rng, h, w, role):
    zones = _regions(h, w)
    if role in ("cut", "connect"):
        for _ in range(20):
            occupied = set()
            names = ("boundary", "split") if role == "cut" else ("reentry", "diagonal")
            if not all(_place_motif(rng, occupied, zones, name) for name in names):
                continue
            # Expose every ink; positions and silhouettes are freshly sampled.
            success = True
            for z in rng.permutation(5):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in occupied:
                    mask[r, c] = zones[r, c] == z
                if any(len(f) >= 4 for f in _components(mask)):
                    continue
                if not _place_shape(rng, occupied, zones, int(rng.integers(4, 7)), int(z)):
                    success = False
                    break
            if success:
                # A larger accepted fragment rules out an exact-area-four rule.
                has_large = False
                for z in range(5):
                    mask = np.zeros((h, w), dtype=bool)
                    for r, c in occupied:
                        mask[r, c] = zones[r, c] == z
                    has_large |= any(len(f) >= 5 for f in _components(mask))
                if not has_large and not _place_shape(rng, occupied, zones, int(rng.integers(5, 9)), 4):
                    continue
                for _ in range(int(rng.integers(4))):
                    _place_shape(rng, occupied, zones, int(rng.integers(1, 10)))
                return occupied
        raise RuntimeError("teaching motifs did not fit")
    occupied = set()
    if rng.random() < .72:
        names = list(MOTIFS)
        rng.shuffle(names)
        for name in names[:int(rng.integers(1, 3))]:
            _place_motif(rng, occupied, zones, name)
    count = int(rng.integers(1, min(11, h*w//12+2)))
    mode = int(rng.integers(6))
    for _ in range(count):
        area = int(rng.integers(1, 4) if mode == 0 else
                   rng.integers(4, 15) if mode == 1 else rng.integers(1, 15))
        _place_shape(rng, occupied, zones, area)
    if not occupied:
        occupied.add((int(rng.integers(h)), int(rng.integers(w))))
    return occupied


def _even(rng, low, high):
    options = list(range(low, high+1, 2))
    return int(rng.choice(options))


def _teaching_layout(rng, previous_main=None, force_multi=False):
    count = 2 if force_multi or rng.random() < .40 else 1
    for _ in range(100):
        h, w = _even(rng, 12, 20), _even(rng, 12, 20)
        if previous_main is not None and (h == previous_main[0] or w == previous_main[1]):
            continue
        rects = [(0, 0, h, w, True)]
        if count == 2:
            sh, sw = _even(rng, 6, 10), _even(rng, 6, 10)
            gap = int(rng.integers(1, 4))
            if rng.random() < .5:
                rects = [(0, int(rng.integers(max(w, sw)-w+1)), h, w, True),
                         (h+2+gap, int(rng.integers(max(w, sw)-sw+1)), sh, sw, False)]
            else:
                rects = [(int(rng.integers(max(h, sh)-h+1)), 0, h, w, True),
                         (int(rng.integers(max(h, sh)-sh+1)), w+2+gap, sh, sw, False)]
        bh = max(r+h+2 for r, c, h, w, _ in rects)
        bw = max(c+w+2 for r, c, h, w, _ in rects)
        if max(bh, bw) <= 29:
            # Mirror arrangements freely; corner ink roles remain absolute.
            if rng.random() < .5:
                rects = [(bh-r-h-2, c, h, w, main) for r, c, h, w, main in rects]
            if rng.random() < .5:
                rects = [(r, bw-c-w-2, h, w, main) for r, c, h, w, main in rects]
            top = int(rng.integers(1, min(3, 30-bh)+1))
            left = int(rng.integers(1, min(3, 30-bw)+1))
            shape = (bh+top+int(rng.integers(min(3, 30-bh-top)+1)),
                     bw+left+int(rng.integers(min(3, 30-bw-left)+1)))
            return shape, [(r+top, c+left, h, w, main) for r, c, h, w, main in rects]
    raise RuntimeError("teaching layout failed")


def _general_layout(rng, unseen=None):
    for attempt in range(40):
        count = int(rng.choice((1, 2, 3, 4), p=(.38, .31, .19, .12))) if attempt < 30 else 1
        height, width = int(rng.integers(18, 31)), int(rng.integers(18, 31))
        ah, aw = height-1, width-1
        gap = int(rng.integers(1, 4))
        if count == 1:
            slots = [(1, 1, ah, aw)]
        elif count == 2:
            vertical = bool(rng.integers(2))
            extent = ah if vertical else aw
            if extent < 16+gap:
                continue
            cut = int(rng.integers(8, extent-gap-7))
            slots = ([(1, 1, cut, aw), (1+cut+gap, 1, ah-cut-gap, aw)] if vertical else
                     [(1, 1, ah, cut), (1, 1+cut+gap, ah, aw-cut-gap)])
        else:
            if min(ah, aw) < 16+gap:
                continue
            rh = int(rng.integers(8, ah-gap-7))
            cw = int(rng.integers(8, aw-gap-7))
            slots = [(1, 1, rh, cw), (1, 1+cw+gap, rh, aw-cw-gap),
                     (1+rh+gap, 1, ah-rh-gap, cw),
                     (1+rh+gap, 1+cw+gap, ah-rh-gap, aw-cw-gap)]
            if count == 3:
                slots.pop(int(rng.integers(4)))
        rects = []
        for r, c, sh, sw in slots:
            h, w = _even(rng, 6, sh-2), _even(rng, 6, sw-2)
            rects.append((r+int(rng.integers(sh-h-1)), c+int(rng.integers(sw-w-1)), h, w, False))
        if unseen is None or any((h, w) not in unseen for _, _, h, w, _ in rects):
            return (height, width), rects
    # There are 121 possible single-board size pairs and at most 40 training boards.
    options = [(h, w) for h in range(6, 27, 2) for w in range(6, 27, 2) if (h, w) not in unseen]
    h, w = options[int(rng.integers(len(options)))]
    return (h+3, w+3), [(1, 1, h, w, False)]


def _make_pair(rng, palette, role, unseen=None, previous_main=None, force_multi=False):
    shape, rects = (_teaching_layout(rng, previous_main, force_multi) if role in ("cut", "connect")
                    else _general_layout(rng, unseen))
    bg, source, *inks = palette
    a = np.full(shape, bg, dtype=np.int8)
    for r, c, h, w, main in rects:
        a[r:r+h+2, c:c+w+2] = inks[4]
        a[r+1:r+h+1, c+1:c+w+1] = bg
        for (y, x), ink in zip(((r, c), (r, c+w+1), (r+h+1, c), (r+h+1, c+w+1)), inks):
            a[y, x] = ink
        cells = _populate(rng, h, w, role if main else "general")
        for y, x in cells:
            a[r+1+y, c+1+x] = source
    return {"input": a, "output": solve(a)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base = tuple(map(int, rng.choice(10, 7, replace=False)))
    variable_palette = bool(rng.random() < .55)
    roles = ["cut", "connect"]
    rng.shuffle(roles)
    puzzle, seen, training_sizes = {"train": [], "test": []}, set(), set()
    first_main = None
    first_count = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            role = roles[index] if split == "train" and index < 2 else "general"
            palette = (tuple(map(int, rng.choice(10, 7, replace=False)))
                       if variable_palette and (split != "train" or index != 0) else base)
            for _ in range(12):
                pair = _make_pair(rng, palette, role, training_sizes if split == "test" else None,
                                  first_main if split == "train" and index == 1 else None,
                                  split == "train" and index == 1 and first_count == 1)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not make a distinct input")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                pair_sizes = [(h, w) for _, _, h, w, _, _ in _read_input(pair["input"])[2]]
                training_sizes.update(pair_sizes)
                if index == 0:
                    first_main = next((h, w) for h, w in pair_sizes if min(h, w) >= 12)
                    first_count = len(pair_sizes)
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _oracle_and_facts(a):
    """Separate union-find oracle: adjacent cells join only with equal zone IDs."""
    a, bg, boards = _read_input(a)
    out, facts = a.copy(), []
    for r, c, h, w, inks, source in boards:
        work = a[r+1:r+h+1, c+1:c+w+1]
        zones = _regions(h, w)
        cells = set(map(tuple, np.argwhere(work == source)))
        parent = {p: p for p in cells}
        def root(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p
        for y, x in sorted(cells):
            for p in ((y-1, x), (y, x-1)):
                if p in cells and zones[p] == zones[y, x]:
                    parent[root(p)] = root((y, x))
        counts = Counter(root(p) for p in cells)
        fragment_areas = []
        for rt, area in counts.items():
            fragment_areas.append((int(zones[rt]), area))
        for y, x in cells:
            if counts[root((y, x))] >= 4:
                out[r+1+y, c+1+x] = inks[int(zones[y, x])]
        facts.append({"size": (h, w), "fragments": fragment_areas,
                      "objects": len(_components(work == source)), "inks": inks,
                      "source": source, "bg": bg})
    return out, facts


def _alternative(value, mode):
    """Explicit plausible competing rules, used only for demonstration checks."""
    a, _, boards = _read_input(value)
    out = a.copy()
    for r, c, h, w, inks, source in boards:
        work = a[r+1:r+h+1, c+1:c+w+1]
        target = out[r+1:r+h+1, c+1:c+w+1]
        zones = _regions(h, w, 1 if mode == "inset1" else 3 if mode == "inset3" else 2)
        if mode == "quadrants":
            rr, cc = np.indices((h, w))
            zones = 2*(rr >= h//2) + (cc >= w//2)
        if mode == "no_cut":
            groups = _components(work == source)
        elif mode == "no_threshold":
            groups = [set(map(tuple, np.argwhere(work == source)))]
        elif mode == "pool":
            groups = [set(map(tuple, np.argwhere((work == source) & (zones == z)))) for z in range(5)]
        elif mode == "object_zone":
            groups = [{p for p in comp if zones[p] == z} for comp in _components(work == source) for z in range(5)]
        else:
            groups = [f for z in range(5) for f in _components(
                (work == source) & (zones == z), DIR8 if mode == "eight" else DIR4)]
        for cells in groups:
            if len(cells) >= 4 or mode == "no_threshold":
                for y, x in cells:
                    target[y, x] = inks[int(zones[y, x])]
    return out


ALTERNATIVES = ("no_cut", "no_threshold", "eight", "pool", "object_zone", "quadrants", "inset1", "inset3")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_sizes, test_sizes = set(), set(), []
    exposed, areas, distinguished = set(), set(), set()
    multi_board_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape
            key = _key(a)
            assert key not in seen
            seen.add(key)
            expected, facts = _oracle_and_facts(a)
            assert np.array_equal(b, expected), "fragment rule violated"
            assert np.array_equal(b, solve(a)), "input-only solver disagrees"
            # Only source material may change; ink and empty cells are immutable.
            _, _, boards = _read_input(a)
            mutable = np.zeros(a.shape, dtype=bool)
            sizes = set()
            for r, c, h, w, inks, source in boards:
                work = a[r+1:r+h+1, c+1:c+w+1]
                mutable[r+1:r+h+1, c+1:c+w+1] = work == source
                sizes.add((h, w))
            assert np.array_equal(a[~mutable], b[~mutable])
            if split == "train":
                multi_board_demo |= len(facts) >= 2
                train_sizes.update(sizes)
                for fact in facts:
                    for z, area in fact["fragments"]:
                        areas.add(area)
                        if area >= 4:
                            exposed.add(z)
                for mode in ALTERNATIVES:
                    if not np.array_equal(_alternative(a, mode), b):
                        distinguished.add(mode)
            else:
                test_sizes.append(sizes)
    assert {3, 4} <= areas and any(n >= 5 for n in areas), "threshold evidence missing"
    assert multi_board_demo, "multiple-board independence needs a demonstration"
    assert len({h for h, _ in train_sizes}) >= 2 and len({w for _, w in train_sizes}) >= 2
    assert exposed == set(range(5)), "all five chambers must be demonstrated"
    assert distinguished == set(ALTERNATIVES), f"undistinguished alternatives: {set(ALTERNATIVES)-distinguished}"
    assert all(sizes-train_sizes for sizes in test_sizes), "test needs a novel board dimension pair"
    return True


def _fingerprint(puzzle):
    return tuple((split, _key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test():
    """Bounded: 42 diverse seeds, count boundaries, determinism, and ablations."""
    assert np.array_equal(_regions(6, 6), np.array([
        [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1],
        [0, 0, 4, 4, 1, 1], [2, 2, 4, 4, 3, 3],
        [2, 2, 2, 3, 3, 3], [2, 2, 2, 3, 3, 3]]))
    # Hand-authored expected answers, independent of sampling and region helpers.
    boundary4 = {(0, 3), (0, 4), (1, 3), (1, 4)}
    center3 = {(2, 3), (2, 4), (3, 3)}
    arms = {(1, 3), (1, 4), (3, 1), (4, 1)}
    center4 = {(2, 3), (3, 3), (3, 2), (2, 2)}
    tri = {(3, 3), (4, 3), (4, 4)}
    square = {(5, 5), (5, 6), (6, 5), (6, 6)}
    split = {(0, 5), (0, 6), (1, 5), (1, 6)}
    for source_cells, colored, color in ((boundary4 | center3, boundary4, 6),
                                         (arms | center4, center4, 4),
                                         (tri | square, square, 4), (split, set(), 6)):
        a = np.zeros((15, 15), dtype=np.int8)
        a[1:15, 1:15] = 4
        a[2:14, 2:14] = 0
        a[1, 1], a[1, 14], a[14, 1], a[14, 14] = 6, 1, 2, 3
        for r, c in source_cells:
            a[r+2, c+2] = 8
        expected, original = a.copy(), a.copy()
        for r, c in colored:
            expected[r+2, c+2] = color
        assert np.array_equal(solve(a), expected)
        assert np.array_equal(a, original)
    shapes, sizes, counts, profiles, palettes, structures = set(), set(), set(), set(), set(), set()
    fingerprints, unchanged, partial = set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                _, facts = _oracle_and_facts(a)
                shapes.add(a.shape)
                counts.add(len(facts))
                sizes.update(f["size"] for f in facts)
                profiles.add(tuple(sorted((z, n) for f in facts for z, n in f["fragments"])))
                palettes.update((f["bg"], f["source"], *f["inks"]) for f in facts)
                # Normalize literal colors while preserving geometry and marker roles.
                roles = {}
                normalized = np.array([roles.setdefault(int(v), len(roles)) for v in a.flat], dtype=np.int8).reshape(a.shape)
                structures.add(_key(normalized))
                unchanged += np.array_equal(a, b)
                partial += not np.array_equal(_alternative(a, "no_cut"), b)
    assert len(fingerprints) == 42 and len(shapes) >= 40 and len(sizes) >= 35
    assert counts == {1, 2, 3, 4} and len(profiles) >= 130
    assert len(palettes) >= 42 and len(structures) >= 190 and partial >= 84
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            p = generate_puzzle(9000+10*tr+te, tr, te)
            assert (len(p["train"]), len(p["test"])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "canvas_shapes": len(shapes), "board_sizes": len(sizes),
            "board_counts": sorted(counts), "fragment_profiles": len(profiles),
            "palettes": len(palettes), "structures": len(structures),
            "pairs_with_cut_effect": partial, "unchanged_pairs": unchanged}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
