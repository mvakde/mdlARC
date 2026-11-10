"""Register a temporary corner frame, then draw metric-clipped quarter rings.

All solving conventions are visible in the input. No parent implementation,
generation coordinates, seed, examples, files, or persistent cache are consulted.
"""
from collections import Counter, defaultdict, deque
import hashlib
from itertools import permutations

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "bad grid colors")
    return a.astype(np.int8, copy=False)


def _groups(points):
    todo = set(points)
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _normalize(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}, (r0, c0)


def _shape(cells):
    return max(r for r, c in cells)+1, max(c for r, c in cells)+1


def _fingerprint(cells):
    normal, _ = _normalize(cells)
    return tuple((r, c, v) for (r, c), v in sorted(normal.items()))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _corners(h, w):
    return ((0, 0), (0, w-1), (h-1, w-1), (h-1, 0))


def _reconstruct(input_grid):
    """Exact whole-signature registration; reject unresolved translation systems."""
    grid = _grid(input_grid)
    counts = Counter(int(v) for v in grid.flat)
    background, count = counts.most_common(1)[0]
    _require(count * 2 > grid.size, "background must occupy more than half the input")
    foreground_counts = {v: n for v, n in counts.items() if v != background}
    _require(bool(foreground_counts), "no fragments")
    largest = max(foreground_counts.values())
    modes = [v for v, n in foreground_counts.items() if n == largest]
    _require(len(modes) == 1, "outline color must be the unique foreground mode")
    scaffold = modes[0]
    parts = []
    for group in _groups(zip(*np.nonzero(grid != background))):
        part, _ = _normalize({p: int(grid[p]) for p in group})
        parts.append(part)
    _require(2 <= len(parts) <= 6, "need two to six separated fragments")
    occurrences = defaultdict(list)
    for i, part in enumerate(parts):
        keys = []
        for group in _groups(p for p, v in part.items() if v != scaffold):
            if len(group) == 1:
                continue  # A marked corner, once the frame has been registered.
            cells = {p: part[p] for p in group}
            _require(len(group) >= 3 and len(set(cells.values())) >= 2,
                     "a registration signature must be multicolor")
            signature = _fingerprint(cells)
            _, origin = _normalize(cells)
            occurrences[signature].append((i, origin))
            keys.append(signature)
        _require(bool(keys), "every fragment needs a signature")
        _require(len(set(keys)) == len(keys), "repeated signature within one fragment")
    adjacency = [[] for _ in parts]
    repeated = 0
    for locations in occurrences.values():
        if len(locations) > 1:
            repeated += 1
        i, a = locations[0]
        for j, b in locations[1:]:
            delta = a[0]-b[0], a[1]-b[1]
            adjacency[i].append((j, delta))
            adjacency[j].append((i, (-delta[0], -delta[1])))
    offsets = {0: (0, 0)}
    queue = deque([0])
    while queue:
        i = queue.popleft()
        for j, delta in adjacency[i]:
            position = offsets[i][0]+delta[0], offsets[i][1]+delta[1]
            if j in offsets:
                _require(offsets[j] == position, "inconsistent signature cycle")
            else:
                offsets[j] = position
                queue.append(j)
    _require(len(offsets) == len(parts), "disconnected signature graph")
    union = {}
    for i, part in enumerate(parts):
        dr, dc = offsets[i]
        for (r, c), v in part.items():
            p = r+dr, c+dc
            _require(p not in union or union[p] == v, "conflicting overlap")
            union[p] = v
    union, origin = _normalize(union)
    h, w = _shape(union)
    _require(9 <= h <= 30 and 9 <= w <= 30, "invalid reconstructed domain")
    corner_set = set(_corners(h, w))
    perimeter = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h-1) or c in (0, w-1)}
    _require(perimeter <= union.keys(), "incomplete rectangular outline")
    sources = [(r, c, union[r, c]) for r, c in _corners(h, w)
               if union[r, c] != scaffold]
    _require(2 <= len(sources) <= 4, "need two to four corner sources")
    signature_cells = set()
    signature_colors = set()
    assembled_signatures = set()
    for group in _groups(p for p, v in union.items() if v != scaffold):
        if len(group) == 1:
            _require(group <= corner_set, "non-corner singleton")
        else:
            _require(not (group & corner_set), "signature touches a corner")
            cells = {p: union[p] for p in group}
            fp = _fingerprint(cells)
            _require(fp not in assembled_signatures, "signature reused at different scene positions")
            assembled_signatures.add(fp)
            signature_cells.update(group)
            signature_colors.update(cells.values())
    _require(assembled_signatures == set(occurrences),
             "signatures must be complete in every view")
    _require(not ({v for r, c, v in sources} & signature_colors),
             "source and signature palettes must be disjoint")
    _require(all(p in perimeter or p in signature_cells for p in union),
             "unexplained interior material")
    return {"background": background, "scaffold": scaffold, "shape": (h, w),
            "sources": sources, "union": union, "parts": parts,
            "offsets": [(offsets[i][0]-origin[0], offsets[i][1]-origin[1])
                        for i in range(len(parts))],
            "signature_count": len(assembled_signatures), "repeated": repeated,
            "duplicate_cells": sum(map(len, parts))-len(union)}


def _paint(shape, sources, background, ownership="manhattan", contours="chebyshev",
           include_ties=False):
    rows, cols = np.indices(shape)
    axial = [(abs(rows-r), abs(cols-c)) for r, c, v in sources]
    distances = np.stack([a+b if ownership == "manhattan" else np.maximum(a, b)
                          for a, b in axial])
    minimum = distances.min(axis=0)
    unique = (distances == minimum).sum(axis=0) == 1
    winner = distances.argmin(axis=0)
    output = np.full(shape, background, dtype=np.int8)
    for i, ((r, c, color), (a, b)) in enumerate(zip(sources, axial)):
        radius = np.maximum(a, b) if contours == "chebyshev" else a+b
        mask = (winner == i) & (radius % 2 == 0)
        if not include_ties:
            mask &= unique
        output[mask] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _reconstruct(input_grid)
    return _paint(info["shape"], info["sources"], info["background"])


def _has_same_color_tie(shape, sources):
    """A tie where collapsing source identities to colors would visibly paint."""
    for r in range(shape[0]):
        for c in range(shape[1]):
            ds = [abs(r-a)+abs(c-b) for a, b, v in sources]
            nearest = [s for s, d in zip(sources, ds) if d == min(ds)]
            if (len(nearest) > 1 and len({v for a, b, v in nearest}) == 1
                    and any(max(abs(r-a), abs(c-b)) % 2 == 0 for a, b, v in nearest)):
                return True
    return False


def _make_scene(rng, h, w, mask, background, scaffold, glyph_colors,
                source_colors):
    """Four corner arcs share arbitrary exact edge words with inward teeth."""
    scene = {(r, c): scaffold for r in range(h) for c in range(w)
             if r in (0, h-1) or c in (0, w-1)}
    keys = []
    intervals = []
    used = set()
    for edge, length in enumerate((w, h, w, h)):
        span = int(rng.integers(3, min(5, length-4)+1))
        low = max(2, (length-span)//2-1)
        high = min(length-span-2, (length-span)//2+1)
        start = int(rng.integers(low, high+1))
        stop = start+span-1
        intervals.append((start, stop))
        for attempt in range(80):
            word = [int(v) for v in rng.choice(glyph_colors, span)]
            if len(set(word)) < 2:
                continue
            key = {}
            for t, color in enumerate(word):
                u = start+t
                p = ((0, u), (u, w-1), (h-1, u), (u, 0))[edge]
                key[p] = color
            # Shape and colors vary independently. Teeth all point inward.
            if int(rng.integers(4)) != 0:
                tooth_positions = rng.choice(np.arange(1, span-1),
                                             int(rng.integers(1, span-1)), replace=False)
                for t in sorted(int(v) for v in tooth_positions):
                    u = start+t
                    p = ((1, u), (u, w-2), (h-2, u), (u, 1))[edge]
                    key[p] = int(rng.choice(glyph_colors))
            fp = _fingerprint(key)
            if fp not in used:
                used.add(fp)
                keys.append(key)
                scene.update(key)
                break
        else:
            raise RuntimeError("unique edge signatures exhausted")
    for corner, color in zip(mask, source_colors):
        scene[_corners(h, w)[corner]] = int(color)
    # Start with one connected L-shaped view per corner. Both views of an
    # edge include the WHOLE key and one scaffold cell beyond each end.
    arcs = [{} for _ in range(4)]
    for edge, length in enumerate((w, h, w, h)):
        start, stop = intervals[edge]
        for side in (0, 1):
            corner = ((0, 1), (1, 2), (3, 2), (0, 3))[edge][side]
            indices = range(0, stop+2) if side == 0 else range(start-1, length)
            for u in indices:
                p = ((0, u), (u, w-1), (h-1, u), (u, 0))[edge]
                arcs[corner][p] = scene[p]
            arcs[corner].update(keys[edge])
    return scene, arcs, keys


def _join(arcs, rng, primary_count):
    # Random cyclic origin; no individual piece is rotated or reflected.
    turn = int(rng.integers(4))
    order = [(turn+i) % 4 for i in range(4)]
    groups = [order[:2], order[2:]] if primary_count == 2 else (
        [order[:2], [order[2]], [order[3]]] if primary_count == 3 else
        [[i] for i in order])
    return [dict(item for i in group for item in arcs[i].items()) for group in groups]


def _pack(parts, background, rng, redundant=None):
    parts = [_normalize(p)[0] for p in parts]
    shapes = [_shape(p) for p in parts]
    candidates = []
    # At most 4! * 2**3 = 192 tiny shelf arrangements, no file intermediates.
    for order in permutations(range(len(parts))):
        for breaks in range(1 << (len(parts)-1)):
            rows = [[order[0]]]
            for j, i in enumerate(order[1:]):
                if breaks & (1 << j):
                    rows.append([])
                rows[-1].append(i)
            height = sum(max(shapes[i][0] for i in row) for row in rows)+len(rows)-1
            width = max(sum(shapes[i][1] for i in row)+len(row)-1 for row in rows)
            if height <= 30 and width <= 30:
                candidates.append((height, width, rows))
    _require(bool(candidates), "fragments do not fit")
    for attempt in range(40):
        mh, mw, rows = candidates[int(rng.integers(len(candidates)))]
        h, w = int(rng.integers(mh, 31)), int(rng.integers(mw, 31))
        material = sum(map(len, parts)) + (len(redundant) if redundant else 0)
        if h*w <= 2*material:
            continue
        grid = np.full((h, w), background, dtype=np.int8)
        y = int(rng.integers(h-mh+1))
        for row in rows:
            row_height = max(shapes[i][0] for i in row)
            row_width = sum(shapes[i][1] for i in row)+len(row)-1
            x = int(rng.integers(w-row_width+1))
            for i in row:
                dy = int(rng.integers(row_height-shapes[i][0]+1))
                for (r, c), v in parts[i].items():
                    grid[y+dy+r, x+c] = v
                x += shapes[i][1]+1
            y += row_height+1
        if redundant:
            piece = _normalize(redundant)[0]
            ph, pw = _shape(piece)
            free = np.ones((h, w), dtype=bool)
            for r, c in zip(*np.nonzero(grid != background)):
                free[max(0, r-1):min(h, r+2), max(0, c-1):min(w, c+2)] = False
            spots = [(r0, c0) for r0 in range(h-ph+1) for c0 in range(w-pw+1)
                     if all(free[r+r0, c+c0] for r, c in piece)]
            if not spots:
                continue
            r0, c0 = spots[int(rng.integers(len(spots)))]
            for (r, c), v in piece.items():
                grid[r+r0, c+c0] = v
        return grid
    raise ValueError("bounded packing attempts exhausted")


def _counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                    ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(name+" must be an integer")
            if not low <= int(value) <= high:
                raise ValueError(name+" out of range")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test+1, 11))
    elif num_train is None:
        num_train = int(rng.integers(int(num_test)+1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train)-1)+1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _generate_pair(rng, split, index, preferred_background, used_geometry):
    for attempt in range(160):
        if split == "train" and index == 0:
            h = int(rng.integers(9, 16))
            w = int(rng.choice([9, 11, 13, 15]))
            turn = int(rng.integers(4))
            if turn % 2:
                h, w = w, h
            mask = tuple(sorted((turn, (turn+1) % 4)))
            primary_count = 2
        elif split == "train" and index == 1:
            h, w = int(rng.integers(9, 16)), int(rng.integers(9, 16))
            missing = int(rng.integers(4))
            mask = tuple(i for i in range(4) if i != missing)
            primary_count = 3
        else:
            high = 16 if split == "train" else 22
            h, w = int(rng.integers(9, high)), int(rng.integers(9, high))
            bits = int(rng.choice([3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]))
            mask = tuple(i for i in range(4) if bits & (1 << i))
            primary_count = int(rng.integers(2, 4)) if split == "train" else 4
        geometry = h, w, mask
        if geometry in used_geometry:
            continue
        palette = [int(v) for v in rng.permutation(10)]
        background = preferred_background if rng.random() < .7 else palette[0]
        palette = [v for v in palette if v != background]
        scaffold = palette[0]
        glyph_colors = palette[1:4]
        choices = palette[4:]
        source_colors = choices[:len(mask)]
        if split == "train" and index == 1:
            source_colors = [choices[0], choices[0], choices[1]]
            rng.shuffle(source_colors)
        elif not (split == "train" and index == 0) and rng.random() < .28:
            source_colors[-1] = source_colors[0]
        scene, arcs, signatures = _make_scene(rng, h, w, mask, background,
                                              scaffold, glyph_colors, source_colors)
        pieces = _join(arcs, rng, primary_count)
        # Absorption is demonstrated before it is used in a test.
        redundant = signatures[int(rng.integers(4))] if (
            split == "train" and index == 0 or rng.random() < .3) else None
        try:
            inp = _pack(pieces, background, rng, redundant)
            info = _reconstruct(inp)
        except ValueError:
            continue
        if info["scaffold"] != scaffold or info["shape"] != (h, w) or info["union"] != scene:
            continue
        output = _paint((h, w), info["sources"], background)
        if split == "train" and index == 0:
            if np.array_equal(output, _paint((h, w), info["sources"], background, include_ties=True)):
                continue
        if split == "train" and index == 1:
            if not _has_same_color_tie((h, w), info["sources"]):
                continue
        used_geometry.add(geometry)
        return {"input": inp, "output": output}
    raise RuntimeError("bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle. None counts are also supported as a convenience."""
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    preferred_background = int(rng.integers(10))
    puzzle = {"train": [], "test": []}
    used_geometry, seen = set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            for retry in range(5):
                pair = _generate_pair(rng, split, index, preferred_background, used_geometry)
                k = _key(pair["input"])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input attempts exhausted")
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _reconstruct(inp)
    h, w = info["shape"]
    _require(out.shape == (h, w), "crop must follow the reconstructed outline")
    _require(info["duplicate_cells"] > 0 and info["repeated"] >= 1,
             "registration must merge genuine overlaps")
    # Cellwise validation also checks ties between SAME-COLOR sources by
    # source identity, rather than counting the set of nearest colors.
    even = odd = ties = 0
    for r in range(h):
        for c in range(w):
            distances = [abs(r-a)+abs(c-b) for a, b, v in info["sources"]]
            d = min(distances)
            owners = [i for i, v in enumerate(distances) if v == d]
            color = info["background"]
            if len(owners) == 1:
                a, b, v = info["sources"][owners[0]]
                if max(abs(r-a), abs(c-b)) % 2 == 0:
                    color = v
                    even += 1
                else:
                    odd += 1
            else:
                ties += 1
            _require(int(out[r, c]) == color, "incorrect territory, tie, or contour")
    _require(even > len(info["sources"]) and odd > 0, "insufficient contour evidence")
    _require(set(np.unique(out)) == {info["background"]} | {v for r, c, v in info["sources"]},
             "construction colors must disappear")
    info["ties"] = ties
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "bad splits")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            k = _key(pair["input"])
            _require(k not in seen, "duplicate input")
            seen.add(k)
            records[split].append(info)
    # These are demonstration checks, not hidden solution conventions.
    train = list(zip(puzzle["train"], records["train"]))
    for name, kwargs in (("tie exclusion", {"include_ties": True}),
                         ("Manhattan ownership", {"ownership": "chebyshev"}),
                         ("Chebyshev contours", {"contours": "manhattan"})):
        _require(any(not np.array_equal(np.asarray(pair["output"]),
                     _paint(info["shape"], info["sources"], info["background"], **kwargs))
                     for pair, info in train), "training fails to distinguish "+name)
    _require(any(len({v for r, c, v in i["sources"]}) >= 2 for i in records["train"]),
             "training must propagate multiple colors")
    _require(any(len({v for r, c, v in i["sources"]}) < len(i["sources"])
                 for i in records["train"]), "training must show repeated source colors")
    _require(any(_has_same_color_tie(i["shape"], i["sources"]) for i in records["train"]),
             "training must distinguish source-identity ties from color ties")
    _require(any(any(all(v != i["scaffold"] for v in p.values()) for p in i["parts"])
                 for i in records["train"]), "training must show signature absorption")
    return None


def self_test():
    """42 seeds, streamed metrics, bounded counterfactual and count regressions."""
    input_shapes, output_shapes, backgrounds, scaffolds = set(), set(), set(), set()
    masks, fragments, fingerprints, digests, metric_shapes = set(), set(), set(), set(), set()
    tied = untied = same_color = zero_source = two_dimensional = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _reconstruct(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                for role in ("input", "output"):
                    k = _key(pair[role])
                    digest.update(bytes(k[0]))
                    digest.update(k[1])
                input_shapes.add(tuple(pair["input"].shape))
                output_shapes.add(info["shape"])
                backgrounds.add(info["background"])
                scaffolds.add(info["scaffold"])
                cs = _corners(*info["shape"])
                masks.add(tuple(cs.index((r, c)) for r, c, v in info["sources"]))
                fragments.add(len(info["parts"]))
                colors = [v for r, c, v in info["sources"]]
                same_color += len(set(colors)) < len(colors)
                zero_source += 0 in colors
                for group in _groups(p for p, v in info["union"].items() if v != info["scaffold"]):
                    if len(group) > 1:
                        key = {p: info["union"][p] for p in group}
                        fingerprints.add(_fingerprint(key))
                        metric_shapes.add(_shape(_normalize(key)[0]))
                tie_changed = not np.array_equal(pair["output"], _paint(
                    info["shape"], info["sources"], info["background"], include_ties=True))
                tied += tie_changed
                untied += not tie_changed
                two_dimensional += (len({r for r, c in info["offsets"]}) > 1 and
                                    len({c for r, c in info["offsets"]}) > 1)
        digests.add(digest.digest())
    assert len(digests) == 42
    assert len(input_shapes) >= 30 and len(output_shapes) >= 50
    assert len(backgrounds) == 10 and len(scaffolds) == 10
    assert len(masks) >= 10 and len(fragments) >= 3
    assert len(fingerprints) >= 100 and len(metric_shapes) >= 10
    assert tied >= 42 and untied >= 5 and same_color >= 42 and zero_source >= 5
    assert two_dimensional >= 100
    # A second packing has identical content and a different physical layout.
    pair = generate_puzzle(0)["train"][0]
    info = _reconstruct(pair["input"])
    primary = [p for p in info["parts"] if any(v == info["scaffold"] for v in p.values())]
    for seed in (1001, 1002):
        repacked = _pack(primary, info["background"], np.random.default_rng(seed))
        assert np.array_equal(solve(repacked), pair["output"])
    # Full-example transpose and arbitrary palette permutation preserve the rule.
    assert np.array_equal(solve(pair["input"].T), pair["output"].T)
    palette = np.array([7, 2, 9, 0, 4, 8, 1, 6, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(palette[pair["input"]]), palette[pair["output"]])
    a, b = generate_puzzle(990, 2, 1), generate_puzzle(990, 2, 1)
    assert all(_key(p[x]) == _key(q[x]) for s in a for p, q in zip(a[s], b[s])
               for x in ("input", "output"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(9000+nt*10+nq, nt, nq))
    for nt, nq in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        validate_generated_puzzle(generate_puzzle(909, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, False), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [dict(p) for p in a[s]] for s in a}
    bad["train"][0]["output"] = a["train"][0]["output"].copy()
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds plus bounded count and invariance regressions")
