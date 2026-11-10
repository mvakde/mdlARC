"""Folded rulers: sort oriented span templates along a visible simple spine.

Only NumPy and the standard library are used. No parent code or saved examples
are consulted by generation or solving.
"""
from collections import Counter
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(p):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in _DIRECTIONS]


def _chain(cells):
    """Recognize a bent monochrome induced chain, oriented by its endpoints."""
    adjacency = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    endpoints = [p for p in cells if len(adjacency[p]) == 1]
    if (len(endpoints) != 2 or endpoints[0][0] == endpoints[1][0]
            or any(len(v) not in (1, 2) for v in adjacency.values())):
        return None
    start = min(endpoints, key=lambda p: p[0])
    path, previous, current = [], None, start
    visited = set()
    while current not in visited:
        path.append(current)
        visited.add(current)
        choices = [q for q in adjacency[current] if q != previous]
        if not choices:
            break
        if len(choices) != 1:
            return None
        previous, current = current, choices[0]
    if len(path) != len(cells):
        return None
    steps = {(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])}
    return path if len(steps) >= 2 else None


def _normal(path, i):
    if i == 0 or i == len(path) - 1:
        return None
    a, p, b = path[i - 1:i + 2]
    before = (p[0] - a[0], p[1] - a[1])
    after = (b[0] - p[0], b[1] - p[1])
    if before != after:
        return None
    # Positive offset means LEFT while walking: east -> up, south -> east.
    dr, dc = before
    return -dc, dr


def _cells_for(path, record, lo, hi):
    dr, dc = record["normal"]
    return {(path[i][0] + k * dr, path[i][1] + k * dc)
            for i in record["indices"] for k in range(lo, hi + 1) if k}


def _safe_envelopes(path, records, lo, hi, shape=None):
    """Every template must fit every station, with no route/other-bar contact."""
    route = set(path)
    used = set()
    for rec in records:
        anchors = {path[i] for i in rec["indices"]}
        cells = _cells_for(path, rec, lo, hi)
        if cells & route or cells & used:
            return False
        if shape and any(not (0 <= r < shape[0] and 0 <= c < shape[1])
                         for r, c in cells):
            return False
        for p in cells:
            neighbors = set(_neighbors(p))
            if neighbors & (route - anchors) or neighbors & used:
                return False
        used.update(cells)
    return True


def _paint(shape, background, carrier, path, records, templates):
    out = np.full(shape, background, dtype=np.int8)
    for p in path:
        out[p] = carrier
    for rec, (lo, hi) in zip(records, templates):
        for p in _cells_for(path, rec, lo, hi):
            assert 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
            assert out[p] == background, "bar overlap"
            out[p] = rec["color"]
    return out


def _analyze(input_grid):
    """Recover the spine, intrinsic frames, stations and full spans from pixels."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(counts.argmax())
    assert counts[background] > grid.size // 2, "background must be a strict majority"
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        cells = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        path = _chain(cells)
        if path is not None:
            candidates.append((color, path))
    assert len(candidates) == 1, "one uniquely bent monochrome spine is required"
    carrier, path = candidates[0]
    records = []
    for i, (r, c) in enumerate(path):
        normal = _normal(path, i)
        if normal is None:
            continue
        dr, dc = normal
        sides = []
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            color = int(grid[rr, cc]) if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] else background
            if color in (background, carrier):
                sides.append((None, 0))
                continue
            length = 0
            while (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]
                   and grid[rr, cc] == color):
                length += 1
                rr, cc = rr + sign * dr, cc + sign * dc
            sides.append((color, length))
        colors = {color for color, length in sides if length}
        if not colors:
            continue
        assert len(colors) == 1, "both halves of a station share a color"
        color = colors.pop()
        lo, hi = -sides[0][1], sides[1][1]
        if (records and records[-1]["indices"][-1] == i - 1
                and (records[-1]["color"], records[-1]["lo"], records[-1]["hi"], records[-1]["normal"])
                == (color, lo, hi, normal)):
            records[-1]["indices"].append(i)
        else:
            records.append({"indices": [i], "normal": normal, "color": color,
                            "lo": lo, "hi": hi})
    assert 2 <= len(records) <= 8
    templates = {}
    for j, rec in enumerate(records):
        rec["width"] = len(rec["indices"])
        rec["length"] = rec["hi"] - rec["lo"] + 1
        assert 1 <= rec["width"] <= 3
        assert rec["length"] >= 2
        assert templates.setdefault(rec["length"], (rec["lo"], rec["hi"])) == (rec["lo"], rec["hi"]), "different-offset length ties"
        if j:
            assert records[j - 1]["indices"][-1] + 1 < rec["indices"][0], "stations require a bare spine gap"
    assert _safe_envelopes(path, records, min(r["lo"] for r in records),
                           max(r["hi"] for r in records), grid.shape)
    rebuilt = _paint(grid.shape, background, carrier, path, records,
                     [(r["lo"], r["hi"]) for r in records])
    assert np.array_equal(grid, rebuilt), "unexplained cells or nonsolid ribs"
    return {"background": background, "carrier": carrier, "path": path,
            "records": records, "shape": grid.shape}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    spans = sorted(((r["lo"], r["hi"]) for r in info["records"]),
                   key=lambda span: span[1] - span[0] + 1)
    return _paint(info["shape"], info["background"], info["carrier"],
                  info["path"], info["records"], spans)


def _coarse_path(rng, edges):
    """Generate topology first; long segments supply room for normal ribs."""
    for _ in range(100):
        path = [(0, 0)]
        occupied = set(path)
        for _ in range(edges):
            candidates = []
            for q in _neighbors(path[-1]):
                if q in occupied or sum(p in occupied for p in _neighbors(q)) != 1:
                    continue
                rows = [p[0] for p in path] + [q[0]]
                cols = [p[1] for p in path] + [q[1]]
                if max(rows) - min(rows) <= 3 and max(cols) - min(cols) <= 3:
                    candidates.append(q)
            if not candidates:
                break
            q = candidates[int(rng.integers(len(candidates)))]
            path.append(q)
            occupied.add(q)
        if len(path) != edges + 1:
            continue
        if _chain(set(path)) is not None:
            break
    else:
        # Induced 4x4 serpent; the random walk is the main geometry vocabulary.
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
                (2, 2), (2, 1), (2, 0), (3, 0)][:edges + 1]
    for _ in range(int(rng.integers(4))):
        path = [(c, -r) for r, c in path]
    if rng.random() < 0.5:
        path = [(r, -c) for r, c in path]
    if path[0][0] == path[-1][0]:
        path = [(c, r) for r, c in path]
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _expand(coarse, scale):
    path = [(coarse[0][0] * scale, coarse[0][1] * scale)]
    for a, b in zip(coarse, coarse[1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        for _ in range(scale):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    return path


def _place_stations(rng, path, n, cap, wide):
    candidates = []
    for width in (1, 2, 3):
        for start in range(1, len(path) - width):
            indices = list(range(start, start + width))
            normal = _normal(path, start)
            if normal is None or any(_normal(path, i) != normal for i in indices):
                continue
            record = {"indices": indices, "normal": normal}
            if _safe_envelopes(path, [record], -cap, cap):
                candidates.append(record)
    # Reserve one thick slot, then select independent strips greedily.
    for _ in range(30):
        records = []
        target_widths = [2] + [int(rng.integers(1, wide + 1)) for _ in range(n - 1)]
        for width in target_widths:
            choices = [r for r in candidates if len(r["indices"]) == width]
            rng.shuffle(choices)
            for rec in choices:
                if any(set(rec["indices"]) & set(r["indices"]) or
                       abs(rec["indices"][0] - r["indices"][-1]) <= 1 or
                       abs(r["indices"][0] - rec["indices"][-1]) <= 1 for r in records):
                    continue
                if _safe_envelopes(path, records + [rec], -cap, cap):
                    records.append(rec.copy())
                    break
            else:
                break
        if len(records) == n:
            return sorted(records, key=lambda r: r["indices"][0])
    return None


def _sample_spans(rng, n, cap, witness):
    if witness == 1:
        lengths = [2 * cap + 1, 2, 2 * cap]
        templates = {2 * cap + 1: (-cap, cap), 2: (0, 1),
                     2 * cap: (-cap, cap - 1)}
    elif witness == 2:
        lengths = [4, 3, 2, 4] + ([5] if n == 5 else [])
        templates = {2: (-1, 0), 3: (-2, 0),
                     4: (-1, 2) if cap == 2 else (0, 3), 5: (-2, 2)}
    else:
        lengths = [int(x) for x in rng.integers(2, 2 * cap + 2, size=n)]
        if len(set(lengths)) == 1:
            lengths[0] = 2 if lengths[0] != 2 else 3
        mode = int(rng.integers(5))
        if mode == 0:
            lengths.sort()
        elif mode == 1:
            lengths.sort(reverse=True)
        templates = {}
        for length in sorted(set(lengths)):
            lo = int(rng.integers(max(-cap, 1 - length), min(0, cap + 1 - length) + 1))
            templates[length] = (lo, lo + length - 1)
    return [templates[length] for length in lengths]


def _embed(rng, path, records, spans, cap, roles):
    envelope = set(path)
    for rec in records:
        envelope.update(_cells_for(path, rec, -cap, cap))
    min_r, min_c = min(p[0] for p in envelope), min(p[1] for p in envelope)
    bh = max(p[0] for p in envelope) - min_r + 1
    bw = max(p[1] for p in envelope) - min_c + 1
    if max(bh, bw) > 30:
        return None
    h = min(30, bh + int(rng.integers(0, 6)))
    w = min(30, bw + int(rng.integers(0, 6)))
    largest = max(hi - lo + 1 for lo, hi in spans)
    maximum_foreground = len(path) + sum(len(r["indices"]) * (largest - 1) for r in records)
    while h * w <= 2 * maximum_foreground:
        if h <= w and h < 30 or w == 30:
            h += 1
        else:
            w += 1
    top = int(rng.integers(h - bh + 1)) - min_r
    left = int(rng.integers(w - bw + 1)) - min_c
    path = [(r + top, c + left) for r, c in path]
    bg, carrier = roles
    inp = _paint((h, w), bg, carrier, path, records, spans)
    ordered = sorted(spans, key=lambda p: p[1] - p[0])
    out = _paint((h, w), bg, carrier, path, records, ordered)
    return {"input": inp, "output": out}


def _features(info):
    records, path = info["records"], info["path"]
    spans = [(r["lo"], r["hi"]) for r in records]
    assigned = sorted(spans, key=lambda s: s[1] - s[0])
    raster = sorted(range(len(records)), key=lambda j: path[records[j]["indices"][0]])
    column = sorted(range(len(records)), key=lambda j: path[records[j]["indices"][0]][::-1])
    def order_diff(order):
        target = [None] * len(records)
        for j, span in zip(order, assigned):
            target[j] = span
        return target != assigned
    area = sorted(range(len(records)), key=lambda j: records[j]["width"] * (records[j]["length"] - 1))
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    moved_turn = False
    for j, span in enumerate(assigned):
        if span == spans[j] or span[0] == -span[1]:
            continue
        sources = [r for r in records if (r["lo"], r["hi"]) == span]
        if any(r["normal"] != records[j]["normal"] for r in sources):
            moved_turn = True
    return {"changed": spans != assigned, "reversal_diff": spans[::-1] != assigned,
            "area_diff": [spans[j] for j in area] != assigned,
            "raster_diff": order_diff(raster), "column_diff": order_diff(column),
            "turned_transfer": moved_turn, "count": len(records),
            "widths": tuple(r["width"] for r in records),
            "lengths": tuple(r["length"] for r in records),
            "spans": tuple(spans), "normals": tuple(r["normal"] for r in records),
            "ties": len(set(spans)) < len(spans),
            "repeat_color": len({r["color"] for r in records}) < len(records),
            "turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "above": min(r for r, _ in path) < path[0][0],
            "path_geometry": tuple((r - path[0][0], c - path[0][1]) for r, c in path),
            "layout": (tuple((r - path[0][0], c - path[0][1]) for r, c in path),
                       tuple(tuple(r["indices"]) for r in records), tuple(spans))}


def _sample_pair(rng, n, cap, roles, witness=0, test=False,
                 challenge=False, allow_above=True):
    for _ in range(180):
        edges = int(rng.integers(max(4, n), min(9, max(6, n + 2)) + 1))
        coarse = _coarse_path(rng, edges)
        if coarse is None:
            continue
        scale = int(rng.integers(2 * cap + 2, 9))
        path = _expand(coarse, scale)
        records = _place_stations(rng, path, n, cap, 3 if test else 2)
        if records is None:
            continue
        spans = _sample_spans(rng, n, cap, witness)
        if witness == 1:
            # Put the long narrow / short thick conflict in the first witness.
            records[0]["indices"] = records[0]["indices"][:1]
            if len(records[2]["indices"]) != 2:
                continue
        pool = [c for c in range(10) if c not in roles]
        palette = [int(c) for c in rng.choice(pool, int(rng.integers(2, min(n, 6) + 1)), replace=False)]
        colors = [int(c) for c in rng.choice(palette, n)]
        if witness == 2:
            colors[2] = colors[0]
            colors[1] = next(c for c in palette if c != colors[0])
        for rec, color in zip(records, colors):
            rec["color"] = color
        pair = _embed(rng, path, records, spans, cap, roles)
        if pair is None:
            continue
        info = _analyze(pair["input"])
        f = _features(info)
        if not allow_above and f["above"]:
            continue
        if witness == 1 and not (f["area_diff"] and f["turned_transfer"] and
                                  f["raster_diff"] and f["column_diff"]):
            continue
        if witness == 2 and not (f["changed"] and f["turned_transfer"]):
            continue
        if test and not (f["raster_diff"] and f["column_diff"]):
            continue
        if challenge and not (f["changed"] and f["reversal_diff"] and f["turned_transfer"]):
            continue
        return pair
    raise RuntimeError("bounded folded-ruler construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                  ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    cap = int(rng.integers(2, 4))
    shared_roles = bool(rng.integers(2))
    roles = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    second_count = int(rng.integers(4, 6))
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            witness = i + 1 if split == "train" and i < 2 else 0
            n = (3 if witness == 1 else second_count if witness == 2 else
                 int(rng.integers(6, 9)) if split == "test" else int(rng.integers(2, 6)))
            local_roles = roles if shared_roles else tuple(int(c) for c in rng.choice(10, 2, replace=False))
            allow_above = (split == "train" or any(
                _features(_analyze(p["input"]))["above"] for p in puzzle["train"]))
            for _ in range(20):
                pair = _sample_pair(rng, n, cap, local_roles, witness, split == "test",
                                    split == "test" and i == 0, allow_above)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = (_grid(pair[k]) for k in ("input", "output"))
    assert inp.shape == out.shape
    a, b = _analyze(inp), _analyze(out)
    for key in ("background", "carrier", "path", "shape"):
        assert a[key] == b[key]
    fixed = lambda info: [(tuple(r["indices"]), r["color"], r["normal"], r["width"])
                          for r in info["records"]]
    assert fixed(a) == fixed(b), "fixed stations changed"
    spans = lambda info: [(r["lo"], r["hi"]) for r in info["records"]]
    assert Counter(spans(a)) == Counter(spans(b)), "oriented span multiset changed"
    lengths = [r["length"] for r in b["records"]]
    assert all(x <= y for x, y in zip(lengths, lengths[1:])), "not sorted along route"
    return a, _features(a)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info, f = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            features[split].append(f)
    tr = features["train"]
    assert sum(f["changed"] for f in tr) >= 2
    assert any(f["area_diff"] and f["reversal_diff"] for f in tr)
    assert any(f["raster_diff"] and f["column_diff"] for f in tr)
    assert any(f["turned_transfer"] for f in tr)
    assert any(f["ties"] and f["repeat_color"] for f in tr)
    assert len({f["count"] for f in tr}) >= 2
    assert min(f["count"] for f in features["test"]) > max(f["count"] for f in tr)
    assert features["test"][0]["changed"] and features["test"][0]["turned_transfer"]
    if any(f["above"] for f in features["test"]):
        assert any(f["above"] for f in tr), "upper endpoint versus topmost cell needs teaching"
    return None


def self_test():
    """Bounded, pixel-based checks and structural measurements across 42 seeds."""
    # A compatible member outside the sampler's lattice/length cap: unequal
    # straight-section lengths and nine-cell intervals at four stations.
    path = [(4, 2)]
    for (dr, dc), steps in (((0, 1), 18), ((1, 0), 21), ((0, -1), 16)):
        for _ in range(steps):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    records = [{"indices": indices, "normal": _normal(path, indices[0]), "color": color}
               for indices, color in (([6, 7], 2), ([25, 26, 27], 4), ([34], 2), ([49, 50], 7))]
    spans = [(-4, 4), (-1, 1), (-4, 3), (0, 4)]
    fixture = {"input": _paint((30, 27), 0, 6, path, records, spans),
               "output": _paint((30, 27), 0, 6, path, records,
                                [(-1, 1), (0, 4), (-4, 3), (-4, 4)])}
    _check_pair(fixture)
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    assert np.array_equal(solve(np.fliplr(fixture["input"])), np.fliplr(fixture["output"]))
    puzzles, infos, features = [], [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(pair["input"]) for split in puzzle for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info, f = _check_pair(pair)
                assert np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"])
                infos.append(info)
                features.append(f)
    assert len(set(puzzles)) == 42
    assert len({i["shape"] for i in infos}) >= 35
    assert len({f["path_geometry"] for f in features}) >= 75
    assert len({f["layout"] for f in features}) >= 150
    assert {f["count"] for f in features} == set(range(2, 9))
    assert {w for f in features for w in f["widths"]} == {1, 2, 3}
    assert {l for f in features for l in f["lengths"]} >= set(range(2, 8))
    assert {normal for f in features for normal in f["normals"]} == set(_DIRECTIONS)
    assert {i["background"] for i in infos} == set(range(10))
    assert {i["carrier"] for i in infos} == set(range(10))
    assert len({f["turns"] for f in features}) >= 4
    assert any(f["above"] for f in features)
    # Mutation checks distinguish missing A, missing B, area and recentering.
    rejected = set()
    for info in infos:
        records = info["records"]
        spans = [(r["lo"], r["hi"]) for r in records]
        lengths = [r["length"] for r in records]
        sorted_spans = sorted(spans, key=lambda s: s[1] - s[0])
        raster = sorted(range(len(records)), key=lambda j: info["path"][records[j]["indices"][0]])
        wrong_raster = [None] * len(records)
        for j, span in zip(raster, sorted_spans):
            wrong_raster[j] = span
        alternatives = {"omit_A": spans, "omit_B": wrong_raster,
                        "area": [spans[j] for j in sorted(range(len(records)),
                                 key=lambda j: records[j]["width"] * (lengths[j] - 1))],
                        "reverse": spans[::-1],
                        "recenter": [(-(length // 2), (length - 1) // 2)
                                     for length in sorted(lengths)]}
        for name, templates in alternatives.items():
            if name in rejected or templates == sorted_spans:
                continue
            try:
                bad_out = _paint(info["shape"], info["background"], info["carrier"],
                                 info["path"], records, templates)
                inp = _paint(info["shape"], info["background"], info["carrier"],
                             info["path"], records, spans)
                _check_pair({"input": inp, "output": bad_out})
            except AssertionError:
                rejected.add(name)
            else:
                raise AssertionError(f"accepted {name} shortcut")
        if len(rejected) == 5:
            break
    assert len(rejected) == 5
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + 10 * a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first = generate_puzzle(0)
    again = generate_puzzle(0)
    assert all(_key(a[k]) == _key(b[k]) for s in first
               for a, b in zip(first[s], again[s]) for k in ("input", "output"))
    return {"seeds": 42, "pairs": len(infos),
            "shapes": len({i["shape"] for i in infos}),
            "path_geometries": len({f["path_geometry"] for f in features}),
            "layouts": len({f["layout"] for f in features}),
            "shortcut_mutations_rejected": sorted(rejected)}
