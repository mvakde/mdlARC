"""Synthetic rectangular-panel registration puzzles."""

import numpy as np


SHAPES = {
    "dot": ((0, 0),),
    "h2": ((0, 0), (0, 1)),
    "v2": ((0, 0), (1, 0)),
    "h3": ((0, 0), (0, 1), (0, 2)),
    "v3": ((0, 0), (1, 0), (2, 0)),
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "l": ((0, 0), (1, 0), (1, 1)),
    "corner": ((0, 0), (0, 1), (1, 0)),
    "t": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "z": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "plus": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "u": ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    "ring": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(array):
    return array.shape, array.tobytes()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = {}
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            todo = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while todo:
                r, c = todo.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        todo.append((rr, cc))
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            shape = tuple(sorted((r-r0, c-c0) for r, c in cells))
            result.setdefault(shape, []).append((r0, c0))
    for origins in result.values():
        origins.sort()
    return result


def _gap(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah < br or br + bh < ar or ac + aw < bc or bc + bw < ac


def _extract(grid):
    h, w = grid.shape
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(np.all(border == border[0]), "input border must be uniform")
    bg = int(border[0])
    occupied = grid != bg
    seen = np.zeros(grid.shape, dtype=bool)
    panels, boxes = [], []
    for sr in range(h):
        for sc in range(w):
            if not occupied[sr, sc] or seen[sr, sc]:
                continue
            todo = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while todo:
                r, c = todo.pop()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and occupied[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        todo.append((rr, cc))
            r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
            c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
            panel = grid[r0:r1+1, c0:c1+1].copy()
            _require(panel.size == len(cells) and np.all(panel != bg), "panel is not a solid rectangle")
            panels.append(panel)
            boxes.append((r0, c0, *panel.shape))
    _require(len(panels) >= 2, "missing panels")
    return bg, panels, boxes


def _interpret_input(grid):
    """Infer every legal assembled answer using the visible panels alone."""
    bg, panels, boxes = _extract(grid)
    color_sets = [set(map(int, np.unique(p))) for p in panels]
    _require(all(len(s) == 2 for s in color_sets), "each panel needs two colors")
    shared = set.intersection(*color_sets)
    _require(len(shared) == 1, "registration color is not unique")
    reg = shared.pop()
    private = [next(iter(s-{reg})) for s in color_sets]
    _require(bg not in set(private) | {reg} and len(set(private)) == len(private), "color roles collide")
    areas = [p.size for p in panels]
    root = int(np.argmax(areas))
    _require(areas.count(areas[root]) == 1, "foundation is not uniquely largest")
    groups = [_components(p == reg) for p in panels]
    _require(all(g for g in groups), "registration absent from a panel")
    candidates = [[] for _ in panels]
    for child in range(len(panels)):
        if child == root:
            continue
        ch, cw = panels[child].shape
        for parent in range(len(panels)):
            if areas[parent] <= areas[child]:
                continue
            ph, pw = panels[parent].shape
            for shape in groups[child].keys() & groups[parent].keys():
                a, b = groups[child][shape], groups[parent][shape]
                if len(a) != len(b):
                    continue
                a0, b0 = a[0], b[0]
                if {(r-a0[0], c-a0[1]) for r, c in a} != {(r-b0[0], c-b0[1]) for r, c in b}:
                    continue
                dr, dc = b0[0]-a0[0], b0[1]-a0[1]
                if 0 <= dr <= ph-ch and 0 <= dc <= pw-cw:
                    candidate = (parent, dr, dc)
                    if candidate not in candidates[child]:
                        candidates[child].append(candidate)
        _require(candidates[child], "a panel has no complete registration match")
    order = sorted((i for i in range(len(panels)) if i != root), key=lambda i: -areas[i])
    positions = {root: (0, 0)}
    answers = {}
    rh, rw = panels[root].shape

    def search(k):
        if len(answers) > 1:
            return
        if k == len(order):
            answer = panels[root].copy()
            answer[answer == reg] = private[root]
            for i in order:
                r, c = positions[i]
                h, w = panels[i].shape
                answer[r:r+h, c:c+w] = panels[i]
            answers[_key(answer)] = (answer, positions.copy())
            return
        i = order[k]
        for parent, dr, dc in candidates[i]:
            if parent not in positions:
                continue
            pr, pc = positions[parent]
            r, c = pr+dr, pc+dc
            h, w = panels[i].shape
            if r+h > rh or c+w > rw:
                continue
            positions[i] = (r, c)
            search(k+1)
            del positions[i]
    search(0)
    _require(len(answers) == 1, "assembly has multiple answers")
    answer, inferred_positions = next(iter(answers.values()))
    return answer, {"background": bg, "registration": reg,
        "panels": panels, "boxes": boxes, "groups": groups, "root": root,
        "candidates": candidates, "private": private, "positions": inferred_positions}


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _counts(seed, rng, num_train, num_test):
    tr = _count(num_train, "num_train", 2, 10)
    te = _count(num_test, "num_test", 1, 4)
    if tr is not None and te is not None:
        if tr <= te:
            raise ValueError("num_train must exceed num_test")
    elif tr is None and te is None:
        te = 1 + seed % 4
        tr = max(te+1, 2 + (seed//4) % 9)
    elif tr is None:
        tr = int(rng.integers(te+1, 11))
    else:
        te = int(rng.integers(1, min(4, tr-1)+1))
    return tr, te


def _topology(rng, kind):
    if kind == "simple":
        rh, rw = int(rng.integers(5, 18)), int(rng.integers(5, 19))
        ch, cw = int(rng.integers(3, rh)), int(rng.integers(3, rw))
        return [(rh, rw), (ch, cw)], [-1, 0]
    if kind == "chain":
        rh, rw = int(rng.integers(9, 20)), int(rng.integers(9, 20))
        mh, mw = int(rng.integers(6, min(13, rh))), int(rng.integers(6, min(13, rw)))
        sh, sw = int(rng.integers(3, min(7, mh))), int(rng.integers(3, min(7, mw)))
        return [(rh, rw), (mh, mw), (sh, sw)], [-1, 0, 1]
    if kind == "branch":
        rh, rw = int(rng.integers(11, 21)), int(rng.integers(11, 21))
        n = int(rng.integers(2, 4))
        dims = [(rh, rw)] + [(int(rng.integers(3, min(9, rh-1))), int(rng.integers(3, min(9, rw-1)))) for _ in range(n)]
        return dims, [-1] + [0]*n
    rh, rw = int(rng.integers(13, 21)), int(rng.integers(13, 21))
    mh, mw = int(rng.integers(7, min(12, rh))), int(rng.integers(7, min(12, rw)))
    sh, sw = int(rng.integers(3, min(7, mh))), int(rng.integers(3, min(7, mw)))
    bh, bw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    return [(rh, rw), (mh, mw), (sh, sw), (bh, bw)], [-1, 0, 1, 0]


def _signature(rng, dim, name, multi):
    shape = SHAPES[name]
    sh = max(r for r, _ in shape)+1
    sw = max(c for _, c in shape)+1
    h, w = dim
    if sh > h or sw > w:
        return None
    cells = set()
    choices = [(r, c) for r in range(h-sh+1) for c in range(w-sw+1)]
    rng.shuffle(choices)
    target = int(rng.integers(2, 5)) if multi else 1
    origins = []
    for r, c in choices:
        new = {(r+dr, c+dc) for dr, dc in shape}
        if any(abs(a-x)+abs(b-y) <= 1 for a, b in new for x, y in cells):
            continue
        cells.update(new)
        origins.append((r, c))
        if len(origins) == target:
            break
    if len(origins) != target:
        return None
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _build_geometry(rng, dims, parents, keys, triple):
    n = len(dims)
    children = [[i for i in range(1, n) if parents[i] == p] for p in range(n)]
    for _ in range(180):
        offsets = [(0, 0)]*n
        okay = True
        for p in range(n):
            placed = []
            ph, pw = dims[p]
            for child in children[p]:
                ch, cw = dims[child]
                possibilities = [(r, c, ch, cw) for r in range(ph-ch+1) for c in range(pw-cw+1)]
                rng.shuffle(possibilities)
                box = next((b for b in possibilities if all(_gap(b, q) for q in placed)), None)
                if box is None:
                    okay = False
                    break
                offsets[child] = box[:2]
                placed.append(box)
            if not okay:
                break
        if not okay:
            continue
        masks = [np.zeros(d, dtype=bool) for d in dims]
        expected = [set() for _ in dims]
        for child in range(1, n):
            sig = keys[child]
            kh, kw = max(r for r, _ in sig)+1, max(c for _, c in sig)+1
            ch, cw = dims[child]
            sr = int(rng.integers(0, ch-kh+1))
            sc = int(rng.integers(0, cw-kw+1))
            p = parents[child]
            for dr, dc in sig:
                a = (sr+dr, sc+dc)
                b = (offsets[child][0]+a[0], offsets[child][1]+a[1])
                masks[child][a] = True
                masks[p][b] = True
                expected[child].add(a)
                expected[p].add(b)
        if triple and parents[2] == 1 and keys[2] == ((0, 0),):
            # The child's dot also coincides with a foundation dot.
            dot = next(iter(expected[2]))
            pos = (offsets[1][0]+offsets[2][0]+dot[0], offsets[1][1]+offsets[2][1]+dot[1])
            masks[0][pos] = True
            expected[0].add(pos)
        if all(set(zip(*np.where(m))) == e and 0 < len(e) < m.size for m, e in zip(masks, expected)):
            # A key's connected components must remain intact, without merger
            # with another key in the same panel.
            valid = True
            for i, mask in enumerate(masks):
                actual = _components(mask)
                for shape, origins in actual.items():
                    if shape not in SHAPES.values():
                        # Repeated signatures use one catalog component shape.
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return masks, offsets
    return None


def _pack(rng, panels, bg):
    n = len(panels)
    min_h = max(p.shape[0] for p in panels)+2
    min_w = max(p.shape[1] for p in panels)+2
    for attempt in range(90):
        h = 30 if attempt > 45 or rng.random() < .22 else int(rng.integers(max(12, min_h), 31))
        w = 30 if attempt > 45 or rng.random() < .22 else int(rng.integers(max(12, min_w), 31))
        positions = []
        for panel in panels:
            ph, pw = panel.shape
            places = [(r, c, ph, pw) for r in range(1, h-ph) for c in range(1, w-pw)]
            rng.shuffle(places)
            box = next((b for b in places if all(_gap(b, q) for q in positions)), None)
            if box is None:
                break
            positions.append(box)
        if len(positions) == n:
            grid = np.full((h, w), bg, dtype=np.int8)
            for panel, (r, c, ph, pw) in zip(panels, positions):
                grid[r:r+ph, c:c+pw] = panel
            return grid
    return None


def _make_pair(rng, kind, force_multi, palette):
    for _ in range(600):
        dims, parents = _topology(rng, kind)
        names = list(SHAPES)
        rng.shuffle(names)
        keys = [None]*len(dims)
        for i in range(1, len(dims)):
            want_multi = force_multi if i == 1 and force_multi is not None else rng.random() < .48
            options = names[:]
            # Dot remains available for a three-way registration in chains.
            if kind == "chain" and i == 2 and rng.random() < .25:
                options = ["dot"]
            key = None
            for name in options:
                if name not in names:
                    continue
                trial = _signature(rng, dims[i], name, want_multi)
                if trial is not None:
                    key = trial
                    names.remove(name)
                    break
            if key is None:
                break
            keys[i] = key
        if any(k is None for k in keys[1:]):
            continue
        triple = kind == "chain" and keys[2] == ((0, 0),) and rng.random() < .7
        geometry = _build_geometry(rng, dims, parents, keys, triple)
        if geometry is None:
            continue
        masks, offsets = geometry
        colors = [int(x) for x in rng.choice(palette, size=len(dims)+2, replace=False)]
        bg, reg = colors[:2]
        private = colors[2:]
        panels = []
        for dim, mask, color in zip(dims, masks, private):
            panel = np.full(dim, color, dtype=np.int8)
            panel[mask] = reg
            panels.append(panel)
        output = panels[0].copy()
        output[output == reg] = private[0]
        positions = [(0, 0)]*len(panels)
        for i in range(1, len(panels)):
            p = parents[i]
            positions[i] = (positions[p][0]+offsets[i][0], positions[p][1]+offsets[i][1])
        for i in sorted(range(1, len(panels)), key=lambda x: -panels[x].size):
            r, c = positions[i]
            ph, pw = panels[i].shape
            output[r:r+ph, c:c+pw] = panels[i]
        input_grid = _pack(rng, panels, bg)
        if input_grid is None:
            continue
        try:
            inferred, _ = _interpret_input(input_grid)
        except AssertionError:
            continue
        if np.array_equal(inferred, output):
            return {"input": input_grid, "output": output}
    raise RuntimeError("could not construct a uniquely solvable panel puzzle")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    palette = rng.permutation(10).astype(np.int8)
    pair_seeds = rng.integers(0, 2**63-1, size=ntrain+ntest, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_roles = []
    for split, count in (("train", ntrain), ("test", ntest)):
        for j in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[(0 if split == "train" else ntrain)+j]))
            if split == "train" and j == 0:
                kind, multi = "simple", False
            elif split == "train" and j == 1:
                kind, multi = "chain", True
            else:
                kind = str(pair_rng.choice(("simple", "chain", "branch", "mixed"), p=(.23, .28, .34, .15) if split == "train" else (.17, .25, .38, .20)))
                multi = None
            for _ in range(12):
                pair = _make_pair(pair_rng, kind, multi, palette)
                key = _key(pair["input"])
                if split == "train":
                    _, detail = _interpret_input(pair["input"])
                    role = (detail["background"], detail["registration"])
                    if j == 1 and (role[0] == training_roles[0][0] or role[1] == training_roles[0][1]):
                        continue
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        training_roles.append(role)
                    break
            else:
                raise RuntimeError("duplicate input generation")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, "pair counts")
    inputs, pairs = set(), set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), "grid type")
                _require(all(1 <= x <= 30 for x in grid.shape) and np.all((0 <= grid) & (grid <= 9)), "grid bounds")
            ik = _key(pair["input"])
            pk = (ik, _key(pair["output"]))
            _require(ik not in inputs and pk not in pairs, "duplicate pair")
            inputs.add(ik)
            pairs.add(pk)
            _require(not (pair["input"].shape == pair["output"].shape and np.array_equal(pair["input"], pair["output"])), "unchanged pair")
            predicted, detail = _interpret_input(pair["input"])
            _require(np.array_equal(predicted, pair["output"]), "wrong assembly")
            _require(pair["output"].shape == detail["panels"][detail["root"]].shape, "wrong crop")
            _require(all(_gap(a, b) for i, a in enumerate(detail["boxes"]) for b in detail["boxes"][i+1:]), "panels touch")
            root = detail["root"]
            reg = detail["registration"]
            for r, c in zip(*np.where(detail["panels"][root] == reg)):
                _require(any(i != root and pos[0] <= r < pos[0]+detail["panels"][i].shape[0]
                    and pos[1] <= c < pos[1]+detail["panels"][i].shape[1]
                    for i, pos in detail["positions"].items()), "unused foundation guide")
            details[split].append(detail)
    _require(len({(d["background"], d["registration"]) for d in details["train"]}) >= 2, "training color roles do not vary")
    _require(details["train"][0]["background"] != details["train"][1]["background"]
        and details["train"][0]["registration"] != details["train"][1]["registration"], "initial training roles do not vary")
    _require(any(len(d["panels"]) == 2 for d in details["train"]), "training needs simple example")
    _require(any(any(len(v) > 1 for g in d["groups"] for v in g.values()) for d in details["train"]), "training needs a multi-component match")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(i) for i in range(42)]
    topologies, component_counts, role_pairs, root_widths = set(), set(), set(), set()
    full_canvas = broad_placement = large_child = three_way = 0
    pair_shapes = set()
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                _, d = _interpret_input(pair["input"])
                panels, root = d["panels"], d["root"]
                areas = [x.size for x in panels]
                parents = {}
                for i in range(len(panels)):
                    if i == root:
                        continue
                    possible = [parent for parent, dr, dc in d["candidates"][i]
                        if parent in d["positions"] and
                        (d["positions"][parent][0]+dr, d["positions"][parent][1]+dc) == d["positions"][i]]
                    parents[i] = min(possible, key=lambda x: areas[x])
                depth = {root: 0}
                for i in sorted(parents, key=lambda x: -areas[x]):
                    depth[i] = depth[parents[i]]+1
                branch = sum(parent == root for parent in parents.values()) >= 2
                nested = max(depth.values()) >= 2
                topologies.add((branch, nested))
                component_counts.update(len(v) for g in d["groups"] for v in g.values())
                role_pairs.add((d["background"], d["registration"]))
                root_widths.add(panels[root].shape[1])
                pair_shapes.add((pair["input"].shape, pair["output"].shape))
                full_canvas += pair["input"].shape == (30, 30)
                large_child += max(areas[i] for i in parents) / areas[root] >= .5
                small_centers = [(r+h/2, c+w/2) for i, (r,c,h,w) in enumerate(d["boxes"]) if i != root]
                if len(small_centers) >= 2:
                    broad_placement += (max(x for x,y in small_centers)-min(x for x,y in small_centers) >= 2
                        and max(y for x,y in small_centers)-min(y for x,y in small_centers) >= 2)
                dot = ((0, 0),)
                placed_dots = [(d["positions"][i][0]+r, d["positions"][i][1]+c)
                    for i, group in enumerate(d["groups"]) for r,c in group.get(dot, [])]
                if len(placed_dots) >= 3 and len(set(placed_dots)) < len(placed_dots)-1:
                    three_way += 1
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({_key(p["train"][0]["input"]) for p in puzzles}) == 42, "seed diversity")
    _require(topologies == {(False, False), (False, True), (True, False), (True, True)}, "topology diversity")
    _require(component_counts.issuperset({1, 2, 3, 4}), "component arrangement diversity")
    _require(len(role_pairs) >= 25 and len(root_widths) >= 12, "color or proportion diversity")
    _require(len(pair_shapes) >= 80 and full_canvas >= 10, "canvas diversity")
    _require(broad_placement >= 30 and large_child >= 20 and three_way >= 2, "placement diversity")
    _require(_equal(generate_puzzle(7), generate_puzzle(7)), "nondeterministic output")
    for tr, te in ((2, 1), (10, 1), (10, 4), (6, 3), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(283, tr, te)
        validate_generated_puzzle(p)
        if tr is not None:
            _require(len(p["train"]) == tr, "train count ignored")
        if te is not None:
            _require(len(p["test"]) == te, "test count ignored")
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(3, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
