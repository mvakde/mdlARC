"""A queried rectangle supplies the missing axis of a corner-scattering cross.

No parent code, fixtures, files, or generation metadata are needed by solve.
Experiments stream puzzles in memory; this module writes no files or traces.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or value < low or
            (high is not None and value > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape) or
            not np.issubdtype(g.dtype, np.integer) or
            np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found = [start]
        for r, c in found:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    found.append(p)
        result.append(found)
    return result


def _separate(a, b):
    # Exclusive stops: strict inequality leaves at least one background cell.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _analyze(input_grid):
    g = _grid(input_grid)
    if min(g.shape) < 9:
        raise ValueError("the flag and interior records need a larger canvas")
    rim = np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))
    bg = int(np.bincount(rim, minlength=10).argmax())
    flags = []
    # Rotate into a view whose flag is on top. Rotation is undone on return.
    for turn in range(4):
        v = np.rot90(g, turn)
        positions = np.flatnonzero(v[0] != bg)
        if len(positions) != 3 or np.any(np.diff(positions) != 1):
            continue
        c = int(positions[1])
        if not 3 <= c < v.shape[1]-3:
            continue
        query = frozenset(int(x) for x in v[0, c-1:c+2])
        stem = int(v[1, c])
        if len(query) == 3 and stem not in query and stem != bg:
            flags.append((turn, c, query, stem))
    if len(flags) != 1 or np.count_nonzero(rim != bg) != 3:
        raise ValueError("expected one three-color border flag")
    turn, c, query, stem = flags[0]
    v = np.rot90(g, turn)
    if np.count_nonzero(v == stem) != 1:
        raise ValueError("the stem color is reserved for guides")
    mask = v != bg
    mask = mask.copy()
    mask[0, c-1:c+2] = False
    mask[1, c] = False
    records = []
    for cells in _components(mask):
        rs, cs = zip(*cells)
        box = tuple(map(int, (min(rs), max(rs)+1, min(cs), max(cs)+1)))
        r0, r1, c0, c1 = box
        if (len(cells) != (r1-r0)*(c1-c0) or r0 < 3 or
                r1 > v.shape[0]-2 or c0 < 2 or c1 > v.shape[1]-2):
            raise ValueError("records must be solid, separated interior rectangles")
        patch = v[r0:r1, c0:c1]
        counts = Counter(map(int, patch.flat))
        records.append({"box": box, "counts": counts,
                        "match": query <= counts.keys()})
    if not 2 <= len(records) <= 12:
        raise ValueError("expected 2..12 records")
    if any(not _separate(a["box"], b["box"])
           for i, a in enumerate(records) for b in records[i+1:]):
        raise ValueError("records need background separation")
    keys = [o for o in records if o["match"]]
    if len(keys) != 1:
        raise ValueError("exactly one rectangle must contain the query")
    key = keys[0]
    r0, r1, c0, c1 = key["box"]
    if (r1-r0) % 2 != 1 or (c1-c0) % 2 != 1 or (c0+c1-1)//2 != c:
        raise ValueError("the matching rectangle needs an odd, aligned center")
    r = (r0+r1-1)//2
    surviving = mask.copy()
    surviving[r0:r1, c0:c1] = False
    return {"view": v, "turn": turn, "background": bg, "stem": stem,
            "query": query, "row": r, "col": c, "records": records,
            "key": key, "surviving": surviving}


def _groups(info):
    groups = {(a, b): [] for a in (0, 1) for b in (0, 1)}
    r, c = info["row"], info["col"]
    for y, x in np.argwhere(info["surviving"]):
        if y != r and x != c:
            groups[(int(y > r), int(x > c))].append((int(y), int(x)))
    return groups


def _scatter(info, separate=False, keep=False):
    """Transform in flag-on-top coordinates; optional mutants are test-only."""
    v, bg, stem = info["view"], info["background"], info["stem"]
    r, c = info["row"], info["col"]
    h, w = v.shape
    out = np.full(v.shape, bg, dtype=v.dtype)
    out[:, c] = stem
    out[r, :] = stem
    axial = info["surviving"].copy()
    axial[:r, :c] = False
    axial[:r, c+1:] = False
    axial[r+1:, :c] = False
    axial[r+1:, c+1:] = False
    out[axial] = v[axial]
    out[0, c-1:c+2] = v[0, c-1:c+2]
    for (bottom, right), cells in _groups(info).items():
        if not cells:
            continue
        parts = [cells]
        if separate:
            mask = np.zeros(v.shape, dtype=bool)
            for p in cells:
                mask[p] = True
            parts = _components(mask)
        for part in parts:
            ys, xs = zip(*part)
            dy = h-1-max(ys) if bottom else -min(ys)
            dx = w-1-max(xs) if right else -min(xs)
            if keep:
                dy = dx = 0
            for y, x in part:
                target = (y+dy, x+dx)
                if not separate:
                    assert out[target] == bg, "quadrants must not collide with scaffold"
                out[target] = v[y, x]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the query, erase its landmark, and scatter using its center."""
    info = _analyze(input_grid)
    return np.rot90(_scatter(info), -info["turn"]).copy()


def _find_box(rng, boxes, shapes, h, w, predicate=lambda b: True):
    # A bounded random placement search, with exhaustive small-grid fallback.
    def admissible(b):
        return predicate(b) and all(_separate(b, old) for old in boxes)
    for _ in range(180):
        bh, bw = shapes[int(rng.integers(len(shapes)))]
        y = int(rng.integers(3, h-1-bh))
        x = int(rng.integers(2, w-1-bw))
        b = (y, y+bh, x, x+bw)
        if admissible(b):
            return b
    candidates = []
    for bh, bw in shapes:
        for y in range(3, h-1-bh):
            for x in range(2, w-1-bw):
                b = (y, y+bh, x, x+bw)
                if admissible(b):
                    candidates.append(b)
    return candidates[int(rng.integers(len(candidates)))] if candidates else None


def _negative_marks(rng, query, extras, missing, area, dense=False):
    available = [q for j, q in enumerate(query) if j != missing]
    if rng.random() < .15:
        available = available[:int(rng.integers(3))]
    available += list(rng.permutation(extras)[:int(rng.integers(len(extras)+1))])
    if not available:
        return []
    target = int(rng.integers(len(available), max(len(available),
                 min(area-1, int(area*(.8 if dense else .5))))+1))
    marks = list(available)
    while len(marks) < target:
        marks.append(int(rng.choice(available)))
    return marks[:area-1]


def _make_pair(rng, colors, query, turn, package, count, thin):
    bg, stem, filler = colors[:3]
    extras = [x for x in colors[3:] if x not in query]
    q = query
    for _ in range(24):
        h, w = (int(x) for x in rng.integers(23, 31, size=2))
        r = int(rng.integers(8, h-7))
        c = int(rng.integers(8, w-7))
        kh, kw = int(rng.choice((3, 5))), int(rng.choice((3, 5, 7)))
        key = (r-kh//2, r+kh//2+1, c-kw//2, c+kw//2+1)
        boxes = [key]
        # The same-size, same-axis twin prevents a purely geometric key rule.
        twin = _find_box(rng, boxes, [(kh, kw)], h, w,
                         lambda b: (b[2]+b[3]-1)//2 == c)
        if twin is None:
            continue
        boxes.append(twin)
        horizontal = _find_box(rng, boxes,
                    [(a, b) for a in (2, 3, 4, 5) for b in (3, 4, 5, 6) if a*b >= 10],
                    h, w, lambda b: b[0] < r < b[1]-1)
        if horizontal is None:
            continue
        boxes.append(horizontal)
        # These two straddlers share a quadrant; the opposite quadrant needs
        # its own witness so all four corner destinations are demonstrated.
        missing_bottom = twin[1] <= r
        missing_right = horizontal[3] <= c
        def opposite(b):
            return ((b[0] > r if missing_bottom else b[1] <= r) and
                    (b[2] > c if missing_right else b[3] <= c))
        diagonal = _find_box(rng, boxes, [(a, b) for a in (2, 3, 4)
                             for b in (3, 4, 5) if a*b >= 9], h, w, opposite)
        if diagonal is None:
            continue
        boxes.append(diagonal)
        for index in range(4, count):
            if index == 4:
                shapes = [(1, k) for k in range(2, 8)]+[(k, 1) for k in range(2, 8)]
            elif thin and rng.random() < .65:
                shapes = [(1, k) for k in range(4, 10)]+[(k, 1) for k in range(4, 10)]
            else:
                shapes = [(a, b) for a in range(2, 7) for b in range(2, 8)]
            box = _find_box(rng, boxes, shapes, h, w)
            if box is None:
                break
            boxes.append(box)
        if len(boxes) == count:
            break
    else:
        raise RuntimeError("bounded rectangle packing exhausted")
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[0, c-1:c+2] = rng.permutation(q)
    grid[1, c] = stem
    for index, (y0, y1, x0, x1) in enumerate(boxes):
        area = (y1-y0)*(x1-x0)
        if index == 0:
            marks = list(q)
            if package != "identity":
                # Reserve room for the triple, an extra, and both repetitions.
                used = extras[:int(rng.integers(1, min(len(extras), area-6)+1))]
                marks += used+[q[0], used[0]]
        elif index == 1:
            marks = [q[1], q[2], extras[0]]
            if package != "identity":
                marks += [q[1], extras[0]]
        elif index == 2:
            marks = [q[0], q[2], extras[0], extras[1], q[0], extras[0]]
        elif index == 3:
            marks = [q[0], q[1], extras[1], extras[2], q[1], extras[1]]
        elif index == 4:
            marks = []
        else:
            # The identity teaching scene has a selector-only counterfactual:
            # only its twin contains {q1,q2,extra0}.
            missing = int(rng.integers(1 if package == "identity" else 0, 3))
            marks = _negative_marks(rng, q, extras, missing, area, thin)
        patch = grid[y0:y1, x0:x1]
        patch[:] = filler
        positions = rng.choice(area, len(marks), replace=False)
        for p, color in zip(positions, rng.permutation(marks)):
            patch[int(p)//(x1-x0), int(p)%(x1-x0)] = color
    grid = np.rot90(grid, -turn).copy()
    return {"input": grid, "output": solve(grid)}


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(10)]
    vary_colors = bool(rng.integers(2))
    orientations = [int(x) for x in rng.choice(4, 2, replace=False)]
    mixed = rng.random() < .7
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            if vary_colors:
                colors = [int(x) for x in rng.permutation(10)]
            q = [int(x) for x in rng.choice(colors[3:], 3, replace=False)]
            turn = orientations[index % 2] if mixed else orientations[0]
            package = ("identity" if index == 0 else "extras") if split == "train" else "free"
            count = int(rng.integers(6, 8)) if split == "train" else int(rng.integers(8, 11))
            for _ in range(8):
                pair = _make_pair(rng, colors, q, turn, package, count,
                                  split == "test" or index % 2 == 1)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    v = info["view"]
    r, c = info["row"], info["col"]
    groups = _groups(info)
    fragments = []
    for cells in groups.values():
        mask = np.zeros(v.shape, dtype=bool)
        for p in cells:
            mask[p] = True
        fragments.append(len(_components(mask)))
    others = [o for o in info["records"] if not o["match"]]
    return {"shape": list(v.shape), "orientation": info["turn"],
            "records": len(info["records"]), "axes": [r, c],
            "key_shape": [info["key"]["box"][1]-info["key"]["box"][0],
                          info["key"]["box"][3]-info["key"]["box"][2]],
            "fragments": fragments, "quadrants": sum(bool(g) for g in groups.values()),
            "vertical_cut": any(o["box"][2] < c < o["box"][3]-1 for o in others),
            "horizontal_cut": any(o["box"][0] < r < o["box"][1]-1 for o in others)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_grid(pair[k]) for k in ("input", "output"))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _analyze(inp)
            infos[split].append(info)
            empty_colors = {next(iter(o["counts"])) for o in info["records"] if len(o["counts"]) == 1}
            assert len(empty_colors) == 1
            filler = next(iter(empty_colors))
            assert filler not in info["query"] | {info["background"], info["stem"]}
            assert all(filler in o["counts"] for o in info["records"])
            info["filler"] = filler
            expected = solve(inp)
            assert np.array_equal(expected, out), "incorrect scatter or deletion"
            assert not np.array_equal(inp, out)
            v, y = info["view"], np.rot90(out, info["turn"])
            r, c, stem, bg = (info[k] for k in ("row", "col", "stem", "background"))
            # Every surviving cell keeps its literal color. Only the selected
            # record loses cells; the guide is the only new non-background ink.
            before, after = Counter(map(int, inp.flat)), Counter(map(int, out.flat))
            for color in set(before) | set(after):
                if color not in (bg, stem):
                    assert after[color] == before[color]-info["key"]["counts"][color]
            assert np.array_equal(y[0, c-1:c+2], v[0, c-1:c+2])
            for row, col in [(j, c) for j in range(1, len(v))]+[(r, j) for j in range(v.shape[1])]:
                assert y[row, col] == (v[row, col] if info["surviving"][row, col] else stem)
            f = _features(info)
            assert f["quadrants"] == 4 and max(f["fragments"]) >= 2
            assert f["vertical_cut"] and f["horizontal_cut"]
            # Neither deleting the key alone nor scattering each component
            # separately explains the generated training data.
            assert not np.array_equal(_scatter(info, keep=True), y)
            assert not np.array_equal(_scatter(info, separate=True), y)
    teaching = infos["train"]
    # An exact-query key and a same-geometry, same-mark-count hard negative.
    def dimensions(o):
        y0, y1, x0, x1 = o["box"]
        return y1-y0, x1-x0
    assert any(sum(i["key"]["counts"][q] for q in i["query"]) == 3 and
               len(set(i["key"]["counts"]) - i["query"]) == 1 and
               any(not o["match"] and dimensions(o) == dimensions(i["key"])
                   and sum(n for color, n in o["counts"].items() if color != i["filler"]) == 3
                   and (o["box"][2]+o["box"][3]-1)//2 == i["col"]
                   and len(o["counts"]) == 4 for o in i["records"]) for i in teaching)
    assert any(len(i["key"]["counts"]) > 4 and
               any(i["key"]["counts"][q] > 1 for q in i["query"]) and
               any(n > 1 for color, n in i["key"]["counts"].items()
                   if color not in i["query"] | {i["filler"]}) for i in teaching)
    for i in teaching:
        for q in i["query"]:
            assert any(not o["match"] and i["query"]-o["counts"].keys() == {q}
                       for o in i["records"])
        assert any(len(o["counts"]) == 1 for o in i["records"])
        assert any(len(o["counts"])-1 >= 4 and not o["match"] for o in i["records"])
    assert max(len(i["records"]) for i in teaching) < min(len(i["records"]) for i in infos["test"])
    assert {i["turn"] for i in infos["test"]} <= {i["turn"] for i in teaching}


def self_test():
    """Stream 42 seeds; report measurements without retaining grids or files."""
    shapes, structures, layouts, orientations, key_shapes, record_counts = (set() for _ in range(6))
    signatures = set()
    counterfactuals = 0
    max_fragments = 0
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _analyze(inp)
                f = _features(info)
                pairs += 1
                shapes.add(inp.shape)
                orientations.add(info["turn"])
                key_shapes.add(tuple(f["key_shape"]))
                record_counts.add(f["records"])
                max_fragments = max(max_fragments, max(f["fragments"]))
                layout = (info["view"].shape, f["axes"][0], f["axes"][1],
                          tuple(o["box"] for o in info["records"]))
                layouts.add(layout)
                roles = {}
                normalized = bytes(roles.setdefault(int(x), len(roles)) for x in inp.flat)
                structures.add(hashlib.sha256(normalized).hexdigest())
                digest.update(str(inp.shape).encode())
                digest.update(inp.astype(np.int8).tobytes())
                digest.update(pair["output"].astype(np.int8).tobytes())
                # Whole-scene rotation, reflection and arbitrary recoloring
                # preserve the rule, including flags on all four sides.
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(pair["output"]))
                assert np.array_equal(solve(inp[:, ::-1]), pair["output"][:, ::-1])
                palette = np.random.default_rng(seed+100).permutation(10)
                assert np.array_equal(solve(palette[inp]), palette[pair["output"]])
        signatures.add(digest.hexdigest())
        # Change only the flag on the identity scene: the same-size twin is
        # now deleted, and the perpendicular axis moves to its different row.
        for pair in puzzle["train"]:
            info = _analyze(pair["input"])
            if len(info["key"]["counts"]) != 4:
                continue
            r, c = info["row"], info["col"]
            choices = [o for o in info["records"] if not o["match"] and
                       len(o["counts"]) == 4 and
                       (o["box"][2]+o["box"][3]-1)//2 == c and
                       (o["box"][1]-o["box"][0]) % 2]
            for twin in choices:
                # The common filler is the sole non-query color of this key.
                filler = next(x for x in info["key"]["counts"] if x not in info["query"])
                new_query = sorted(set(twin["counts"])-{filler})
                if len(new_query) != 3:
                    continue
                changed = info["view"].copy()
                changed[0, c-1:c+2] = new_query
                try:
                    revised = _analyze(changed)
                except ValueError:
                    continue
                assert revised["row"] != r
                assert revised["key"]["box"] == twin["box"]
                assert not np.array_equal(solve(changed)[2:], _scatter(info)[2:])
                counterfactuals += 1
                break
            else:
                continue
            break
        else:
            raise AssertionError("missing selector-only counterfactual")
        if seed < 3:
            again = generate_puzzle(seed)
            for s in ("train", "test"):
                for a, b in zip(puzzle[s], again[s]):
                    assert all(np.array_equal(a[k], b[k]) for k in ("input", "output"))
    assert len(signatures) == 42 and len(structures) >= 200 and len(layouts) >= 200
    assert len(shapes) >= 40 and len(key_shapes) == 6 and len(orientations) == 4
    assert record_counts == {6, 7, 8, 9, 10} and max_fragments >= 4
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(7103, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Corruption must be caught by full output and conservation checks.
    bad = generate_puzzle(2)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "color_normalized_inputs": len(structures), "layouts": len(layouts),
            "grid_shapes": len(shapes), "key_shapes": sorted(key_shapes),
            "flag_sides": len(orientations), "record_counts": sorted(record_counts),
            "max_quadrant_components": max_fragments,
            "selector_only_counterfactuals": counterfactuals}
