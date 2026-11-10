"""Containment-scoped anomaly filtering: compare before pruning, then redraw.

Only NumPy and the Python standard library are used. All semantic information
comes from the visible connected components, holes, and sibling arrangements.
"""
from collections import Counter
import json

import numpy as np


_B = ("B",)


def _c(axis, *children):
    return ("C", int(axis) if len(children) > 1 else 0, tuple(children))


def _g(axis, *children):
    return ("G", int(axis) if len(children) > 1 else 0, tuple(children))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _join(parts, axis, gaps=None):
    """Center aligned odd-sized tiles, in their given cardinal order."""
    if len(parts) == 1:
        return parts[0].copy()
    gaps = [1] * (len(parts) - 1) if gaps is None else gaps
    shape = [0, 0]
    shape[axis] = sum(p.shape[axis] for p in parts) + sum(gaps)
    shape[1-axis] = max(p.shape[1-axis] for p in parts)
    out = np.zeros(shape, dtype=bool)
    start = 0
    for i, p in enumerate(parts):
        cross = (shape[1-axis] - p.shape[1-axis]) // 2
        y, x = (start, cross) if axis == 0 else (cross, start)
        out[y:y+p.shape[0], x:x+p.shape[1]] |= p
        start += p.shape[axis] + (gaps[i] if i < len(gaps) else 0)
    return out


def _frame(content, padding=2):
    out = np.pad(content, padding)
    out[0, :] = out[-1, :] = True
    out[:, 0] = out[:, -1] = True
    return out


def _normal(node):
    """Complete, UNPRUNED canonical bitmap of an abstract construction."""
    if node[0] == "B":
        return np.ones((1, 1), dtype=bool)
    content = _join([_normal(ch) for ch in node[2]], node[1])
    return _frame(content) if node[0] == "C" else content


def _minority(keys):
    counts = Counter(keys)
    if len(keys) < 3 or len(counts) != 2:
        raise ValueError("a voting contour needs one exceptional equivalence class")
    common = [k for k, n in counts.items() if n == len(keys)-1]
    if len(common) != 1:
        raise ValueError("no unique n-1 sibling consensus")
    return next(i for i, k in enumerate(keys) if k != common[0])


def _filter_abstract(node):
    """Construction-side target, using the complete pre-deletion child motifs."""
    if node[0] == "B":
        return node
    children = node[2]
    if node[0] == "C" and len(children) >= 3:
        index = _minority([_key(_normal(ch)) for ch in children])
        children = (children[index],)
    result = tuple(_filter_abstract(ch) for ch in children)
    return _c(node[1], *result) if node[0] == "C" else _g(node[1], *result)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    h, w = mask.shape
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        pending = [(int(r), int(c))]
        seen[r, c] = True
        cells = []
        while pending:
            y, x = pending.pop()
            cells.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    pending.append((yy, xx))
        groups.append(np.array(cells, dtype=int))
    return groups


def _holes(mask):
    padded = np.pad(mask, 1)
    outside = np.zeros_like(padded)
    pending = [(0, 0)]
    outside[0, 0] = True
    h, w = padded.shape
    while pending:
        y, x = pending.pop()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < h and 0 <= xx < w and not padded[yy, xx] and not outside[yy, xx]:
                outside[yy, xx] = True
                pending.append((yy, xx))
    return (~padded & ~outside)[1:-1, 1:-1]


def _parse(input_grid, background=None):
    grid = np.asarray(input_grid)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("expected an integer 2D grid")
    values = set(map(int, np.unique(grid)))
    if len(values) != 2 or not values <= set(range(10)):
        raise ValueError("expected exactly two colors")
    if background is None:
        background = int(grid[0, 0])
        if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
                and np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)):
            raise ValueError("input needs a complete background border")
    foreground = next(iter(values - {background}))
    records = []
    labels = np.zeros(grid.shape, dtype=int)
    for cells in _components(grid == foreground):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        mask = np.zeros(grid.shape, dtype=bool)
        mask[cells[:, 0], cells[:, 1]] = True
        local = mask[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
        hole = np.zeros_like(mask)
        local_hole = _holes(local)
        hole[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = local_hole
        if local_hole.any():
            if len(_components(local_hole)) != 1:
                raise ValueError("a contour must enclose exactly one connected region")
            kind = "C"
        else:
            if len(cells) * 2 < local.size:
                raise ValueError("terminal blobs must be compact and filled")
            kind = "B"
        records.append(dict(kind=kind, lo=lo, hi=hi, mask=mask,
                            local=local, hole=hole, cells=cells))
        labels[mask] = len(records)
    if not records:
        raise ValueError("empty diagram")
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:]),
                 (labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        if np.any((a > 0) & (b > 0) & (a != b)):
            raise ValueError("distinct components must not touch, even diagonally")
    parents = []
    for i, obj in enumerate(records):
        candidates = []
        for j, outer in enumerate(records):
            if j == i or outer["kind"] != "C":
                continue
            inside = outer["hole"][obj["cells"][:, 0], obj["cells"][:, 1]]
            if inside.any() and not inside.all():
                raise ValueError("partially enclosed component")
            if inside.all():
                candidates.append(j)
        parents.append(min(candidates, key=lambda j: int(records[j]["hole"].sum()))
                       if candidates else None)
    children = [[] for _ in records]
    roots = []
    for i, p in enumerate(parents):
        (roots if p is None else children[p]).append(i)

    def order(ids):
        if len(ids) == 1:
            return 0, list(ids)
        choices = []
        for axis in (0, 1):
            cross = 1-axis
            centers = {int(records[i]["lo"][cross]+records[i]["hi"][cross]) for i in ids}
            ordered = sorted(ids, key=lambda i: int(records[i]["lo"][axis]))
            if len(centers) == 1 and all(
                records[b]["lo"][axis] - records[a]["hi"][axis] >= 2
                for a, b in zip(ordered, ordered[1:])
            ):
                choices.append((axis, ordered))
        if len(choices) != 1:
            raise ValueError("siblings need one unambiguous centered row or column")
        return choices[0]

    for i, obj in enumerate(records):
        if bool(children[i]) != (obj["kind"] == "C"):
            raise ValueError("empty contour or blob containing another object")
        if children[i]:
            obj["axis"], children[i] = order(children[i])
        else:
            obj["axis"] = 0
    if any(records[i]["kind"] != "C" for i in roots):
        raise ValueError("all top-level objects must be contours")
    root_axis, roots = order(roots)
    return dict(grid=grid, background=background, foreground=foreground,
                records=records, parents=parents, children=children,
                roots=roots, root_axis=root_axis)


def _analyze(input_grid, background=None):
    """Pixel-derived normalization, local votes, and recursively pruned tiles."""
    info = _parse(input_grid, background)
    records, children = info["records"], info["children"]
    full, filtered, votes = {}, {}, []

    def visit(i):
        obj = records[i]
        if obj["kind"] == "B":
            full[i] = filtered[i] = np.ones((1, 1), dtype=bool)
            return
        ids, axis = children[i], obj["axis"]
        for j in ids:
            visit(j)
        full[i] = _frame(_join([full[j] for j in ids], axis))
        kept = ids
        if len(ids) >= 3:
            keys = [_key(full[j]) for j in ids]
            index = _minority(keys)
            selected = ids[index]
            kept = [selected]
            common = next(j for j in ids if j != selected)
            # Features reconstructed from the actual returned pixels.
            votes.append(dict(container=i, selected=selected, common=common,
                              count=len(ids), axis=axis, keys=keys,
                              raw_keys=[_key(records[j]["local"]) for j in ids],
                              equal_cells=int(full[selected].sum()) == int(full[common].sum()),
                              common_key=_key(full[common])))
        filtered[i] = _frame(_join([filtered[j] for j in kept], axis))

    for i in info["roots"]:
        visit(i)
    info["full"], info["filtered"], info["votes"] = full, filtered, votes
    info["answer_mask"] = _join([filtered[i] for i in info["roots"]], info["root_axis"])
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """No seed, examples, sampled metadata, or generator state is consulted."""
    info = _analyze(input_grid)
    return np.where(info["answer_mask"], info["foreground"], info["background"]).astype(np.int8)


def _outline(content, padding, rng, minimal):
    """Move independently chosen boundary sections inward in safe corridors."""
    out = _frame(content, padding)
    h, w = out.shape
    cap = padding - 2
    if not cap:
        return out
    sides = [int(rng.integers(4))] if minimal else list(map(int, rng.choice(4, int(rng.integers(1, 5)), replace=False)))
    for side in sides:
        # Operate on a temporary orientation in which the chosen edge is top.
        oriented = np.rot90(out, side).copy()
        width = oriented.shape[1]
        # Leave the side corridors disjoint even when the content is a point.
        # A one-point content still has room for a genuine inset section.
        low, high = padding-1, width-padding
        if high <= low:
            continue
        a = low if minimal else int(rng.integers(low, high))
        b = high if minimal else int(rng.integers(a+1, high+1))
        depth = 1 if minimal else int(rng.integers(1, cap+1))
        oriented[0, a+1:b] = False
        oriented[depth, a:b+1] = True
        oriented[:depth+1, a] = oriented[:depth+1, b] = True
        # For a one-column interval the move adds nubs only; retain actual
        # displaced sections by drawing a sufficiently wide interval instead.
        if b-a < 2:
            continue
        out = np.rot90(oriented, -side)
    return out


def _draw(node, rng, *, minimal=False, widen_path=None, path=()):
    if node[0] == "B":
        if minimal or rng.random() < .70:
            return np.ones((1, 1), dtype=bool)
        shape = [(3, 3), (1, 3), (3, 1)][int(rng.integers(3))]
        blob = np.ones(shape, dtype=bool)
        if shape == (3, 3):
            style = int(rng.integers(3))
            if style == 1:
                blob[0, 0] = blob[0, -1] = blob[-1, 0] = blob[-1, -1] = False
            elif style == 2:
                blob[int(rng.choice([0, 2])), int(rng.choice([0, 2]))] = False
        return blob
    parts = [_draw(ch, rng, minimal=minimal, widen_path=widen_path, path=path+(i,))
             for i, ch in enumerate(node[2])]
    gaps = [1 if minimal or rng.random() < .82 else int(rng.choice([3, 5])) for _ in parts[:-1]]
    content = _join(parts, node[1], gaps)
    if node[0] == "G":
        return content
    padding = (3 if path == widen_path else 2) if minimal else int(rng.choice([2, 3, 4], p=[.74, .23, .03]))
    return _outline(content, padding, rng, minimal)


def _paths(node, path=()):
    result = [path] if node[0] == "C" else []
    if node[0] != "B":
        for i, ch in enumerate(node[2]):
            result.extend(_paths(ch, path+(i,)))
    return result


_TOKENS = (
    _B, _c(0, _B), _c(0, _B, _B), _c(1, _B, _B),
    _c(0, _c(0, _B)),
    _c(0, _B, _c(0, _B)), _c(0, _c(0, _B), _B),
    _c(1, _B, _c(0, _B)), _c(1, _c(0, _B), _B),
)


def _vote(rng, master, exception, axis, count=3):
    children = [master] * (count-1)
    children.insert(int(rng.integers(count)), exception)
    return _c(axis, *children)


def _random_token(rng, depth=3):
    """Compose unary/binary motifs rather than choose from an output catalog."""
    if depth == 0 or rng.random() < .28:
        return _B
    count = int(rng.choice([1, 2], p=[.38, .62]))
    return _c(int(rng.integers(2)), *[_random_token(rng, depth-1) for _ in range(count)])


def _random_vote(rng):
    # Rich tokens can be minorities without requiring three large copies.
    for _ in range(32):
        master = _TOKENS[int(rng.choice(4, p=[.43, .25, .16, .16]))]
        exception = _TOKENS[int(rng.integers(len(_TOKENS)))] if rng.random() < .25 else _random_token(rng)
        if _key(_normal(master)) == _key(_normal(exception)):
            continue
        axis, count = int(rng.integers(2)), int(rng.integers(3, 6))
        candidate = _c(axis, *([master]*(count-1)+[exception]))
        if max(_normal(candidate).shape) <= 26:
            return _vote(rng, master, exception, axis, count)
    # A finite geometry enumeration is only a fallback for exhausted sampling.
    choices = []
    for master in _TOKENS[:4]:
        for exception in _TOKENS:
            if _key(_normal(master)) == _key(_normal(exception)):
                continue
            for axis in (0, 1):
                for count in (3, 4, 5):
                    node = _c(axis, *([master]*(count-1)+[exception]))
                    if max(_normal(node).shape) <= 26:
                        choices.append((master, exception, axis, count))
    args = choices[int(rng.integers(len(choices)))]
    return _vote(rng, *args)


def _balanced_witness(rng):
    axis = int(rng.integers(2))
    # Repeated tokens have their short dimension along the comparison row.
    master, exception = _c(1-axis, _B, _B), _c(axis, _B, _B)
    return _g(0, _vote(rng, master, exception, axis))


def _context_witness(rng):
    axis = int(rng.integers(2))
    # The inner consensus differs from the outer consensus. Filtering the
    # inner chamber turns it into precisely the OUTER common motif, C(B).
    # Comparing already-filtered children would erase the outer anomaly.
    inner = _vote(rng, _c(axis, _B, _B), _B, 1-axis)
    outer = _vote(rng, _c(0, _B), inner, axis)
    inner_index = inner[2].index(_B)
    outer_index = outer[2].index(inner)
    if inner_index == outer_index:
        # Evidence conditioning only: eliminate a constant-position shortcut
        # within this training witness. Other scope positions remain arbitrary.
        target = (outer_index + int(rng.integers(1, 3))) % 3
        siblings = [_c(0, _B)] * 3
        siblings[target] = inner
        outer = _c(axis, *siblings)
    return _g(0, outer)


def _nested_scene(rng):
    for _ in range(48):
        inner = _random_vote(rng)
        selected = inner[2][_minority([_key(_normal(ch)) for ch in inner[2]])]
        if selected in (_B, _c(0, _B)):
            continue
        outer = _vote(rng, _B, inner, 1-inner[1], int(rng.choice([3, 4])))
        if max(_normal(outer).shape) <= 26:
            if rng.random() < .30 and max(_normal(outer).shape) <= 22:
                outer = _c(0, outer)
            return _g(0, outer)
    # Bounded construction with variable orientations and exceptional positions.
    axis = int(rng.integers(2))
    inner = _vote(rng, _B, _c(1-axis, _B, _B), axis)
    return _g(0, _vote(rng, _B, inner, 1-axis))


def _scene(rng):
    node = _random_vote(rng)
    if rng.random() < .3 and max(_normal(node).shape) <= 22:
        node = _c(0, node)
    if rng.random() < .3:
        for _ in range(8):
            other = _random_vote(rng)
            axes = list(map(int, rng.permutation(2)))
            for axis in axes:
                scene = _g(axis, node, other)
                if max(_normal(scene).shape) <= 26:
                    return scene
    return _g(0, node)


def _has_nonrect(info):
    return any(o["kind"] == "C" and not (
        o["local"][0].all() and o["local"][-1].all()
        and o["local"][:, 0].all() and o["local"][:, -1].all()
    ) for o in info["records"])


def _geometry_witness(info):
    for v in info["votes"]:
        common = [raw for key, raw in zip(v["keys"], v["raw_keys"]) if key == v["common_key"]]
        if len(set(common)) > 1:
            return True
    return False


def _make_pair(rng, scene, palette, geometry_witness=False, require_nonrect=True):
    background, foreground = palette
    expected = np.where(_normal(_filter_abstract(scene)), foreground, background).astype(np.int8)
    fallback_paths = [None] + _paths(scene)
    # Stochastic metric realizations, followed by a finite path enumeration.
    for attempt in range(64+len(fallback_paths)):
        minimal = attempt >= 64
        widened = fallback_paths[attempt-64] if minimal else None
        drawing = _draw(scene, rng, minimal=minimal, widen_path=widened)
        if max(drawing.shape) > 28:
            continue
        # All inputs visibly identify background with at least a one-cell border.
        extra_y = int(rng.integers(min(4, 29-drawing.shape[0])))
        extra_x = int(rng.integers(min(4, 29-drawing.shape[1])))
        top, left = int(rng.integers(extra_y+1)), int(rng.integers(extra_x+1))
        mask = np.pad(drawing, ((1+top, 1+extra_y-top), (1+left, 1+extra_x-left)))
        grid = np.where(mask, foreground, background).astype(np.int8)
        try:
            info = _analyze(grid)
            if not np.array_equal(np.where(info["answer_mask"], foreground, background), expected):
                continue
            if (require_nonrect and not _has_nonrect(info)) or (geometry_witness and not _geometry_witness(info)):
                continue
        except ValueError:
            continue
        return {"input": grid, "output": expected}
    raise RuntimeError("no fitting, unambiguous metric realization")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(name + " is outside its legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    stable_palette = bool(rng.random() < .25)
    base_palette = tuple(map(int, rng.choice(10, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(24):
                if split == "test":
                    scene = _nested_scene(rng)
                elif i == 0:
                    scene = _balanced_witness(rng)
                elif i == 1:
                    scene = _context_witness(rng)
                else:
                    scene = _scene(rng)
                palette = base_palette if stable_palette else tuple(map(int, rng.choice(10, 2, replace=False)))
                pair = _make_pair(rng, scene, palette, geometry_witness=split == "train" and i == 0,
                                  require_nonrect=not (split == "train" and i == 1))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce a distinct input")
        rng.shuffle(puzzle[split])
    return puzzle


def _tree(info, i):
    obj = info["records"][i]
    return _B if obj["kind"] == "B" else _c(obj["axis"], *[_tree(info, j) for j in info["children"][i]])


def _nested_votes(info):
    containers = {v["container"] for v in info["votes"]}
    for i in containers:
        p = info["parents"][i]
        while p is not None:
            if p in containers:
                return True
            p = info["parents"][p]
    return False


def _comparison_collision(info):
    return any(len({_key(info["filtered"][j]) for j in info["children"][v["container"]]}) == 1
               for v in info["votes"])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert info["votes"], "every example must actually filter a consensus group"
            assert np.array_equal(out, solve(inp))
            assert set(map(int, np.unique(out))) == {info["background"], info["foreground"]}
            assert out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]
            original = _g(info["root_axis"], *[_tree(info, i) for i in info["roots"]])
            output_info = _parse(out, info["background"])
            actual = _g(output_info["root_axis"], *[_tree(output_info, i) for i in output_info["roots"]])
            assert actual == _filter_abstract(original), "selected topology and cardinal order must survive"
            assert np.array_equal(_normal(actual), out == info["foreground"]), "minimal canonical crop and gaps"
            for obj in output_info["records"]:
                if obj["kind"] == "B":
                    assert len(obj["cells"]) == 1
                else:
                    h, w = obj["local"].shape
                    assert len(obj["cells"]) == 2*(h+w)-4, "all output contours are thin rectangles"
            if split == "train":
                training.append(info)
            else:
                assert _nested_votes(info), "held-out examples combine voting scopes recursively"
    assert any(_geometry_witness(i) for i in training), "raw bitmap equality must fail in a demonstration"
    assert any(_has_nonrect(i) for i in training), "demonstrate displaced contour geometry"
    assert any(_comparison_collision(i) for i in training), "comparison must precede recursive deletion"
    assert any(len({i["children"][v["container"]].index(v["selected"]) for v in i["votes"]}) > 1
               for i in training), "a constant selected position must fail in training"
    assert any(v["equal_cells"] for i in training for v in i["votes"]), "color totals cannot identify every exception"
    assert any(len({v["common_key"] for v in i["votes"]}) > 1 for i in training), "local contexts must differ"
    assert any(i["records"][v["selected"]]["kind"] == "B"
               and i["records"][v["common"]]["kind"] == "C"
               for i in training for v in i["votes"]), "some exceptions are simpler than their consensus"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded structural, metamorphic, ablation, mutation, and diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, normalized, votes, palettes, blobs, depths = set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                normalized.add(_key(pair["output"] == info["foreground"]))
                palettes.add((info["background"], info["foreground"]))
                for v in info["votes"]:
                    votes.add((v["count"], v["axis"], v["common_key"], v["keys"][info["children"][v["container"]].index(v["selected"]) ]))
                for i, obj in enumerate(info["records"]):
                    if obj["kind"] == "B":
                        blobs.add((obj["local"].shape, len(obj["cells"])))
                    depth, p = 0, info["parents"][i]
                    while p is not None:
                        depth += 1
                        p = info["parents"][p]
                    depths.add(depth)
                # Removing B (consensus filtering) keeps the full normalized tree.
                whole = _join([info["full"][i] for i in info["roots"]], info["root_axis"])
                assert not np.array_equal(whole, info["answer_mask"])
        p = puzzle["test"][0]
        bg = int(p["input"][0, 0])
        # Translation, color roles, and metric deformation preserve the answer.
        translated = np.pad(p["input"], ((1, 2), (2, 1)), constant_values=bg)
        assert np.array_equal(solve(translated), p["output"])
        lookup = np.roll(np.arange(10), 3)
        assert np.array_equal(solve(lookup[p["input"]]), lookup[p["output"]])
        stretched = np.repeat(np.repeat(p["input"], 2, axis=0), 2, axis=1)
        assert np.array_equal(solve(stretched), p["output"])
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    assert len(shapes) >= 35 and len(normalized) >= 25 and len(votes) >= 35
    assert len(palettes) >= 50 and len(blobs) >= 4 and max(depths) >= 3
    assert {v[0] for v in votes} >= {3, 4, 5}
    assert {v[1] for v in votes} == {0, 1}
    # Exercise the compact fallback directly, instead of relying on the random
    # metric search to reach it. Enlarge one common token in each orientation.
    for seed in range(4):
        rng = np.random.default_rng(seed+902)
        scene = _balanced_witness(rng)
        chamber = scene[2][0]
        signatures = [_key(_normal(ch)) for ch in chamber[2]]
        common_key = Counter(signatures).most_common(1)[0][0]
        common_index = signatures.index(common_key)
        drawing = np.pad(_draw(scene, rng, minimal=True, widen_path=(0, common_index)), 1)
        assert max(drawing.shape) <= 30
        info = _analyze(drawing.astype(np.int8))
        assert _has_nonrect(info) and _geometry_witness(info)
        assert np.array_equal(info["answer_mask"], _normal(_filter_abstract(scene)))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+10*a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    # Explicit A ablation: common canonical motifs have distinct raw drawings.
    # Without normalization there is no n-1 raw consensus at that contour.
    witness = next(_analyze(p["input"]) for p in puzzles[0]["train"] if _geometry_witness(_analyze(p["input"])))
    assert any(max(Counter(v["raw_keys"]).values()) < v["count"]-1 for v in witness["votes"])
    # A nested counterexample: choosing only the outer anomaly retains an inner
    # consensus group. Recursive filtering produces a smaller, different answer.
    nested = _analyze(puzzles[0]["test"][0]["input"])
    root = nested["roots"][0]
    outer_votes = [v for v in nested["votes"] if v["container"] == root]
    vote = outer_votes[0] if outer_votes else next(v for v in nested["votes"] if nested["parents"][v["container"]] == root)
    outer_only = _frame(nested["full"][vote["selected"]])
    assert not np.array_equal(outer_only, nested["filtered"][vote["container"]])
    # Corrupted labels, absent exceptional structure, and broken boundaries fail.
    for defect in ("output", "consensus", "open"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        pair = next(p for p in bad["train"] if len(_analyze(p["input"])["votes"]) == 1)
        info = _analyze(pair["input"])
        if defect == "output":
            pair["output"][0, 0] = info["background"]
        elif defect == "consensus":
            v = info["votes"][0]
            obj = info["records"][v["selected"]]
            y, x = obj["lo"]
            yy, xx = obj["hi"]
            pair["input"][y:yy+1, x:xx+1] = info["background"]
        else:
            obj = info["records"][info["roots"][0]]
            boundary = obj["mask"]
            y, x = next((int(y), int(x)) for y, x in obj["cells"]
                        if 0 < y < boundary.shape[0]-1 and 0 < x < boundary.shape[1]-1
                        and boundary[y, x-1] and boundary[y, x+1]
                        and not boundary[y-1, x] and not boundary[y+1, x])
            pair["input"][y, x] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("mutation accepted: " + defect)
    return dict(seeds=42, pairs=210, input_shapes=len(shapes), normalized_outputs=len(normalized),
                local_vote_types=len(votes), ordered_palettes=len(palettes), blob_types=len(blobs),
                max_component_depth=max(depths), independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
