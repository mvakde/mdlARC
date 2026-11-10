"""Rhythm-switched color keys. Standard library and NumPy only.

The solver reads the scene; it never uses the sampler or a saved puzzle.
Coordinates in sampling are horizontal; transposition supplies the other axis.
"""
from __future__ import annotations

import numpy as np


def _perimeter(h, w):
    return ([(0, c) for c in range(w)]
            + [(r, w - 1) for r in range(1, h)]
            + [(h - 1, c) for c in range(w - 2, -1, -1)]
            + [(r, 0) for r in range(h - 2, 0, -1)])


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _border(a):
    p = _perimeter(*a.shape)
    values = np.array([a[q] for q in p])
    colors, counts = np.unique(values, return_counts=True)
    if len(colors) != 2 or counts[0] == counts[1]:
        raise ValueError("border must contain background and one complete seed")
    bg = int(colors[np.argmax(counts)])
    accent = int(colors[np.argmin(counts)])
    seed = values == accent
    starts = np.flatnonzero(seed & ~np.roll(seed, 1))
    k = int(seed.sum())
    if len(starts) != 1 or len(p) % (2 * k) or len(p) // (2 * k) < 2:
        raise ValueError("seed must tile the cyclic border with equal gaps")
    start = int(starts[0])
    active = (np.arange(len(p)) - start) % (2 * k) < k
    return bg, accent, p, k, start, seed, active


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, part = [(r, c)], []
        while stack:
            r, c = stack.pop()
            part.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                        and not seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(part)
    return result


def _parse(input_grid):
    """Recover roles, key associations, crop order and live keys from input only."""
    original = _grid(input_grid)
    if min(original.shape) < 11:
        raise ValueError("insufficient space for two key rails and silhouettes")
    bg, accent, _, _, _, _, _ = _border(original)
    if np.any(original[1:-1, 1:-1] == accent):
        raise ValueError("the rhythm color is reserved for the border")
    candidates = []
    for color in np.unique(original):
        if int(color) in (bg, accent):
            continue
        parts = _components(original == color)
        if 2 <= len(parts) <= 5 and all(len(p) >= 5 for p in parts):
            candidates.append(int(color))
    if len(candidates) != 1:
        raise ValueError("a unique carrier must form 2..5 substantial objects")
    carrier = candidates[0]
    interpretations = []
    for axis in (1, 0):
        a = original if axis == 1 else original.T
        h, w = a.shape
        _, _, perimeter, k, start, seed, active = _border(a)
        index = {q: j for j, q in enumerate(perimeter)}
        objects = []
        valid = True
        for part in _components(a == carrier):
            rows, cols = zip(*part)
            top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
            crop = a[top:bottom + 1, left:right + 1]
            if (not (3 <= crop.shape[0] <= 5 and 3 <= crop.shape[1] <= 5)
                    or top < 4 or bottom > h - 5 or left < 2 or right > w - 3
                    or np.any((crop != bg) & (crop != carrier))):
                valid = False
                break
            objects.append({"box": (top, bottom, left, right), "mask": crop == carrier})
        if not valid or len({o["mask"].shape for o in objects}) != 1:
            continue
        objects.sort(key=lambda o: o["box"][2])
        if any(b["box"][2] - a0["box"][3] < 2
               for a0, b in zip(objects, objects[1:])):
            continue
        marker_mask = (a != bg) & (a != carrier) & (a != accent)
        positions = [tuple(map(int, q)) for q in np.argwhere(marker_mask)]
        if len(positions) != 2 * len(objects) or any(r not in (2, h - 3) for r, c in positions):
            continue
        used = set()
        for obj in objects:
            top, bottom, left, right = obj["box"]
            keys = []
            for rail in (2, h - 3):
                locations = [q for q in positions if q[0] == rail and left <= q[1] <= right]
                if len(locations) != 1:
                    valid = False
                    break
                q = locations[0]
                used.add(q)
                port = (0 if rail == 2 else h - 1, q[1])
                keys.append({"position": q, "port": index[port], "color": int(a[q]),
                             "active": bool(active[index[port]])})
            if not valid:
                break
            if keys[0]["color"] == keys[1]["color"] or keys[0]["active"] == keys[1]["active"]:
                valid = False
                break
            obj["keys"] = keys
            obj["winner"] = 0 if keys[0]["active"] else 1
        if not valid or len(used) != len(positions):
            continue
        interpretations.append({"canonical": a, "axis": axis, "background": bg,
                                "accent": accent, "carrier": carrier, "k": k,
                                "start": start, "perimeter": perimeter, "seed": seed,
                                "active": active, "objects": objects})
    if len(interpretations) != 1:
        raise ValueError("keys must determine one exclusive, unambiguous switching scene")
    return interpretations[0]


def _render(info, opposite=False):
    tiles = [np.where(obj["mask"], obj["keys"][obj["winner"] ^ int(opposite)]["color"],
                      info["background"]).astype(np.int8) for obj in info["objects"]]
    strip = np.concatenate(tiles, axis=1)
    return strip if info["axis"] == 1 else strip.T.copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete the border mentally, select live keys, recolor and pack tight crops."""
    return _render(_parse(input_grid))


def _mask(rng, h, w):
    mode = float(rng.random())
    if mode < 0.15:
        m = np.ones((h, w), dtype=bool)
        m[1:-1, 1:-1] = False
        return m
    if mode < 0.30:
        m = np.zeros((h, w), dtype=bool)
        m[int(rng.integers(h)), :] = True
        m[:, int(rng.integers(w))] = True
        return m
    for _ in range(40):
        anchors = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                   (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(anchors)
        m = np.zeros((h, w), dtype=bool)
        m[anchors[0]] = True
        for tr, tc in anchors[1:]:
            cells = np.argwhere(m)
            distances = np.abs(cells - [tr, tc]).sum(axis=1)
            closest = cells[distances == distances.min()]
            r, c = map(int, closest[int(rng.integers(len(closest)))])
            while (r, c) != (tr, tc):
                moves = []
                if r != tr:
                    moves.append((r + (1 if tr > r else -1), c))
                if c != tc:
                    moves.append((r, c + (1 if tc > c else -1)))
                r, c = moves[int(rng.integers(len(moves)))]
                m[r, c] = True
        target = min(h * w - 1, max(int(m.sum()), int(rng.uniform(.40, .86) * h * w)))
        if m.all():
            continue
        while m.sum() < target:
            frontier = set()
            for rr, cc in np.argwhere(m):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r, c = int(rr) + dr, int(cc) + dc
                    if 0 <= r < h and 0 <= c < w and not m[r, c]:
                        frontier.add((r, c))
            choices = sorted(frontier)
            m[choices[int(rng.integers(len(choices)))]] = True
        return m
    raise RuntimeError("mask construction exhausted")


def _draw_seed(a, bg, accent, start, k):
    p = _perimeter(*a.shape)
    for q in p:
        a[q] = bg
    for j in range(k):
        a[p[(start + j) % len(p)]] = accent


def _seed_bends(h, w, k, start):
    length = 2 * (h + w - 2)
    corners = {0, w - 1, w + h - 2, 2 * w + h - 3}
    return any((start + j) % length in corners for j in range(1, k - 1))


def _make_scene(rng, count, axis, *, anchor=False, allowed_k=None, excluded_k=()):
    """Construct labels from known switches; labels are not obtained by solve()."""
    for _ in range(1400):
        th = int(rng.integers(3, 6))
        max_tw = min(5, (30 - 4 - (count - 1)) // count)
        tw = int(rng.integers(3, max_tw + 1))
        gaps = [int(rng.integers(1, 4)) for _ in range(count - 1)]
        left_margin, right_margin = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        w = left_margin + right_margin + count * tw + sum(gaps)
        if w > 30:
            continue
        h = int(rng.integers(th + 8, 31))
        ks = [k for k in range(1, min(13, (h + w - 2) // 2) + 1)
              if (h + w - 2) % k == 0 and k not in excluded_k
              and (not anchor or 3 <= k <= 9)
              and (allowed_k is None or k in allowed_k)]
        if not ks:
            continue
        k = ks[int(rng.integers(len(ks)))]
        p = _perimeter(h, w)
        index = {q: j for j, q in enumerate(p)}
        boxes = []
        cursor = left_margin
        for j in range(count):
            top = int(rng.integers(4, h - th - 3))
            boxes.append((top, cursor))
            cursor += tw + (gaps[j] if j < len(gaps) else 0)
        starts = [s for s in range(len(p)) if not anchor or _seed_bends(h, w, k, s)]
        rng.shuffle(starts)
        for start in starts[:30]:
            active = (np.arange(len(p)) - start) % (2 * k) < k
            switches = []
            for top, left in boxes:
                choices = [(c, d) for c in range(left, left + tw) for d in range(left, left + tw)
                           if active[index[(0, c)]] != active[index[(h - 1, d)]]]
                if not choices:
                    break
                c, d = choices[int(rng.integers(len(choices)))]
                winner = 0 if active[index[(0, c)]] else 1
                switches.append((c, d, winner))
            if len(switches) != count or {x[2] for x in switches} != {0, 1}:
                continue
            live_ports = [index[(0, c)] if winner == 0 else index[(h - 1, d)]
                          for c, d, winner in switches]
            if not any((q - start) % len(p) >= k for q in live_ports):
                continue
            # At least one winner must be the more distant key: proximity cannot
            # substitute for cyclic phase. Distances refer to tight crop edges.
            distance_differences = [
                (top - 2 - (h - 3 - (top + th - 1))) * (1 if win == 0 else -1)
                for (top, left), (c, d, win) in zip(boxes, switches)]
            if not any(d > 0 for d in distance_differences):
                continue
            if anchor and not any(d < 0 for d in distance_differences):
                continue
            palette = [int(x) for x in rng.permutation(10)]
            bg, accent, carrier = palette[:3]
            a = np.full((h, w), bg, dtype=np.int8)
            _draw_seed(a, bg, accent, start, k)
            masks, color_pairs, tiles = [], [], []
            for j, ((top, left), (c, d, winner)) in enumerate(zip(boxes, switches)):
                m = masks[0].copy() if j == 1 and rng.random() < .20 else _mask(rng, th, tw)
                masks.append(m)
                colors = [int(x) for x in rng.choice(palette[3:], size=2, replace=False)]
                color_pairs.append(colors)
                a[top:top + th, left:left + tw][m] = carrier
                a[2, c], a[h - 3, d] = colors
                tiles.append(np.where(m, colors[winner], bg).astype(np.int8))
            # Retain literal diversity among live colors, including repeated
            # hues when they arise. Opposing labels are always different.
            if len({colors[sw[2]] for colors, sw in zip(color_pairs, switches)}) < 2:
                continue
            if anchor and len({colors[1 - sw[2]] for colors, sw in zip(color_pairs, switches)}) < 2:
                continue
            out = np.concatenate(tiles, axis=1)
            pair = {"input": a if axis == 1 else a.T.copy(),
                    "output": out if axis == 1 else out.T.copy()}
            try:
                _parse(pair["input"])
            except ValueError:
                continue
            return pair
    raise RuntimeError("could not construct an exclusive switching scene")


def _complement(pair):
    info = _parse(pair["input"])
    a = info["canonical"].copy()
    _draw_seed(a, info["background"], info["accent"], info["start"] + info["k"], info["k"])
    return {"input": a if info["axis"] == 1 else a.T.copy(),
            "output": _render(info, opposite=True)}


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _normalized(a):
    roles = {}
    return np.asarray(a).shape, tuple(roles.setdefault(int(x), len(roles)) for x in np.asarray(a).flat)


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    first = _make_scene(rng, 3, axis, anchor=True)
    puzzle = {"train": [first, _complement(first)], "test": []}
    first_k = _parse(first["input"])["k"]
    seen = {_normalized(p["input"]) for p in puzzle["train"]}
    training_limit = int(rng.integers(3, 5))
    for j in range(2, num_train):
        for _ in range(30):
            pair = _make_scene(rng, int(rng.integers(2, training_limit + 1)),
                               1 - axis if j == 2 else int(rng.integers(2)),
                               excluded_k=(first_k,) if j == 2 else ())
            key = _normalized(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("training structure collision")
        seen.add(key)
        puzzle["train"].append(pair)
    training = [_parse(p["input"]) for p in puzzle["train"]]
    lengths = {i["k"] for i in training}
    axes = sorted({i["axis"] for i in training})
    max_count = max(len(i["objects"]) for i in training)
    for _ in range(num_test):
        # With two demos only one k is taught. Additional demos teach variable
        # k explicitly; then tests may recombine an unseen compatible length.
        allowed = lengths if len(lengths) == 1 or rng.random() < .65 else None
        for attempt in range(30):
            pair = _make_scene(rng, int(rng.integers(max_count + 1, 6)),
                               axes[int(rng.integers(len(axes)))], allowed_k=allowed)
            key = _normalized(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("test structure collision")
        seen.add(key)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    info = _parse(a)
    objects, k = info["objects"], info["k"]
    p, active = info["perimeter"], info["active"]
    # Check the complete cyclic signal's runs, not just equality with solve.
    transitions = np.flatnonzero(active != np.roll(active, 1))
    runs = np.diff(np.r_[transitions, transitions[0] + len(p)])
    assert np.all(runs == k) and active.sum() * 2 == len(p)
    assert all(active[j] for j in np.flatnonzero(info["seed"]))
    assert {o["winner"] for o in objects} == {0, 1}
    assert len({o["keys"][o["winner"]]["color"] for o in objects}) >= 2
    assert any(not info["seed"][o["keys"][o["winner"]]["port"]] for o in objects)
    strip = out if info["axis"] == 1 else out.T
    th, tw = objects[0]["mask"].shape
    assert strip.shape == (th, tw * len(objects))
    farther = False
    h = info["canonical"].shape[0]
    for j, obj in enumerate(objects):
        m = obj["mask"]
        assert not m.all() and len(_components(m)) == 1
        assert m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any()
        tile = strip[:, j * tw:(j + 1) * tw]
        color = obj["keys"][obj["winner"]]["color"]
        assert np.all(tile[m] == color) and np.all(tile[~m] == info["background"])
        top, bottom, left, right = obj["box"]
        distances = (top - 2, h - 3 - bottom)
        farther |= distances[obj["winner"]] > distances[1 - obj["winner"]]
    assert farther, "nearest-key shortcut must fail"
    assert info["accent"] not in out and info["carrier"] not in out
    assert np.array_equal(out, solve(a))
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, structures = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _signature(_grid(pair["input"]))
            structure = _normalized(pair["input"])
            assert key not in seen and structure not in structures
            seen.add(key)
            structures.add(structure)
            analyses[split].append(_check_pair(pair))
    first, second = analyses["train"][:2]
    a, b = first["canonical"], second["canonical"]
    assert first["axis"] == second["axis"] and a.shape == b.shape
    assert np.array_equal(a[1:-1, 1:-1], b[1:-1, 1:-1])
    assert first["k"] == second["k"]
    assert (second["start"] - first["start"]) % len(first["perimeter"]) == first["k"]
    assert np.array_equal(first["active"], ~second["active"])
    assert _seed_bends(*a.shape, first["k"], first["start"])
    assert all(x["winner"] != y["winner"] for x, y in zip(first["objects"], second["objects"]))
    if len(analyses["train"]) >= 3:
        third = analyses["train"][2]
        assert third["axis"] != first["axis"] and third["k"] != first["k"]
    train_axes = {i["axis"] for i in analyses["train"]}
    train_lengths = {i["k"] for i in analyses["train"]}
    largest = max(len(i["objects"]) for i in analyses["train"])
    for info in analyses["test"]:
        assert info["axis"] in train_axes and len(info["objects"]) > largest
        if len(train_lengths) == 1:
            assert info["k"] in train_lengths


def self_test():
    """Bounded 42-seed diversity, controlled counterfactuals and API checks."""
    # Explicit wraparound seed: (1,0), (0,0), (0,1). The probes at
    # top/bottom column 2 are respectively inactive/active. Duplicating each
    # corner would add three steps on the way to the bottom probe and lose it.
    corner_case = np.zeros((11, 15), dtype=np.int8)
    _draw_seed(corner_case, 0, 8, 47, 3)
    _, _, perimeter, k, start, _, active = _border(corner_case)
    assert len(perimeter) == len(set(perimeter)) == 48
    assert (k, start) == (3, 47)
    assert not active[perimeter.index((0, 2))]
    assert active[perimeter.index((10, 2))]
    assert (37 + 3) % 6 >= 3  # wrong answer with three duplicated corners
    for k in (1, 2, 3, 6, 12):
        _draw_seed(corner_case, 0, 8, 47, k)
        info = _border(corner_case)
        assert info[3] == k and int(info[6].sum()) == 24
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [_check_pair(p) for task in puzzles for split in ("train", "test") for p in task[split]]
    shapes = {tuple(i["canonical"].shape) for i in infos}
    layouts = {(i["canonical"].shape, i["axis"], i["k"], i["start"],
                tuple(o["box"] for o in i["objects"])) for i in infos}
    silhouettes = {(o["mask"].shape, o["mask"].tobytes()) for i in infos for o in i["objects"]}
    assert len(shapes) >= 35 and len(layouts) >= 160 and len(silhouettes) >= 150
    assert {i["axis"] for i in infos} == {0, 1}
    assert {len(i["objects"]) for i in infos} == {2, 3, 4, 5}
    assert {i["background"] for i in infos} == set(range(10))
    assert {i["carrier"] for i in infos} == set(range(10))
    assert len({i["k"] for i in infos}) >= 8
    assert max(i["k"] for i in infos) >= 9
    # Parent-A ablation: flip phase alone, retaining all Parent-B information.
    # Parent-B ablation: filled rectangles destroy every mask's blank cells;
    # retaining carrier color disagrees with each occupied output cell.
    for task in puzzles:
        first, second = task["train"][:2]
        assert not np.array_equal(first["output"], second["output"])
        for pair in task["train"]:
            info = _parse(pair["input"])
            assert all((~o["mask"]).any() for o in info["objects"])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            task = generate_puzzle(100 + a * 5 + b, a, b)
            assert len(task["train"]) == a and len(task["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    one, two = generate_puzzle(12345, 5, 2), generate_puzzle(12345, 5, 2)
    assert all(_signature(p[k]) == _signature(q[k]) for split in one
               for p, q in zip(one[split], two[split]) for k in ("input", "output"))
    for pair in one["train"] + one["test"]:
        a = pair["input"]
        before = a.copy()
        out = solve(a)
        assert np.array_equal(a, before) and np.array_equal(out, pair["output"])
        # Global transpose is a symmetry, with the output strip transposed too.
        assert np.array_equal(solve(a.T), out.T)
    return {"seeds": 42, "pairs": len(infos), "canvas_shapes": len(shapes),
            "layouts": len(layouts), "silhouettes": len(silhouettes),
            "run_lengths": sorted({i["k"] for i in infos})}
