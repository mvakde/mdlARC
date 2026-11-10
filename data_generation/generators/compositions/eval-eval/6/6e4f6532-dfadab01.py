"""Compass stamp interference: registration, local D4 frames, and odd coverage.

No parent modules, files, seed state, or demonstration labels enter solve().
The visible multicolor prototypes are the dictionary for the current input.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


_FOUR = ((-1, 0), (0, 1), (1, 0), (0, -1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_DIRECTIONS = dict(zip(("top", "right", "bottom", "left"), _FOUR))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(points, diagonal=True):
    remaining = set(points)
    result = []
    neighbors = _EIGHT if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, part = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    todo.append(q)
        result.append(part)
    return result


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _point(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return r, c


def _transform(cells, t):
    return {_point(p, t): v for p, v in cells.items()}


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0): v for (r, c), v in cells.items()}


def _shift(points, delta):
    return {(r + delta[0], c + delta[1]) for r, c in points}


def _bounds(points):
    return (min(r for r, _ in points), max(r for r, _ in points),
            min(c for _, c in points), max(c for _, c in points))


def _window(points):
    r0, r1, c0, c1 = _bounds(points)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _inside(p, region):
    r0, r1, c0, c1 = region["rect"]
    return r0 <= p[0] < r1 and c0 <= p[1] < c1


def _label_directions(cells, bands):
    core = {p for p, v in cells.items() if v not in bands}
    r0, r1, c0, c1 = _bounds(core)
    directions = {}
    for (r, c), value in cells.items():
        if value in bands:
            ds = [d for d, valid in (("top", r < r0), ("bottom", r > r1),
                                     ("left", c < c0), ("right", c > c1)) if valid]
            assert len(ds) == 1, "a compass label must face exactly one side"
            assert value not in directions or directions[value] == ds[0]
            directions[value] = ds[0]
    assert len(directions) == 2
    a, b = [_DIRECTIONS[d] for d in directions.values()]
    assert a[0] * b[0] + a[1] * b[1] == 0, "compass labels must be adjacent"
    return directions


def _parse(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1
    H, W = g.shape
    rows = [r for r in range(H) if g[r, 0] != bg and np.all(g[r] == g[r, 0])]
    assert rows and rows[0] == 0 and rows[-1] == H - 1
    active = [r for r in range(H) if r not in rows]
    cols = [c for c in range(W) if len(active) and g[active[0], c] != bg
            and np.all(g[active, c] == g[active[0], c])]
    assert cols and cols[0] == 0 and cols[-1] == W - 1
    assert all(b - a >= 4 for a, b in zip(rows, rows[1:]))
    assert all(b - a >= 4 for a, b in zip(cols, cols[1:]))
    hs = [int(g[r, 0]) for r in rows]
    vs = [int(g[active[0], c]) for c in cols]
    assert len(set(hs)) == len(set(vs)) == 2 and set(hs).isdisjoint(vs)
    assert all(hs[i] == hs[i % 2] for i in range(len(hs)))
    assert all(vs[i] == vs[i % 2] for i in range(len(vs)))
    bands = set(hs + vs)
    structural = np.zeros(g.shape, dtype=bool)
    structural[rows, :] = True
    structural[:, cols] = True
    regions, region_map = [], np.full(g.shape, -1, dtype=int)
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        for j, (c, d) in enumerate(zip(cols, cols[1:])):
            rect = (a + 1, b, c + 1, d)
            regions.append({"rect": rect, "sides": dict(zip(
                ("top", "bottom", "left", "right"), (hs[i], hs[i + 1], vs[j], vs[j + 1])))})
            region_map[a + 1:b, c + 1:d] = len(regions) - 1
    objects = _components(map(tuple, np.argwhere((g != bg) & ~structural)))
    sources, targets = [], []
    for obj in objects:
        values = {int(g[p]) for p in obj}
        ids = {int(region_map[p]) for p in obj}
        assert len(ids) == 1 and -1 not in ids
        if len(values) == 1:
            targets.append((next(iter(values)), obj, next(iter(ids))))
        else:
            sources.append({p: int(g[p]) for p in obj})
    codes = {code for code, _, _ in targets}
    assert 1 <= len(codes) <= 3 and codes.isdisjoint(bands | {bg})
    assert len(sources) == len(codes)
    inks = set().union(*(set(s.values()) for s in sources)) - codes - bands - {bg}
    assert len(inks) == 1
    ink = next(iter(inks))
    book = {}
    for source in sources:
        keys = set(source.values()) & codes
        assert len(keys) == 1
        code = next(iter(keys))
        assert code not in book and ink in source.values()
        labels = _label_directions(source, bands)
        assert len(set(labels) & set(hs)) == len(set(labels) & set(vs)) == 1
        assert len(_components(source, diagonal=False)) == 1
        marks = {p for p, v in source.items() if v == code}
        assert 1 <= len(marks) <= 4 and len(_components(marks)) == 1
        book[code] = source
    assert set(book) == codes
    return {"background": bg, "ink": ink, "bands": bands, "structural": structural,
            "regions": regions, "book": book, "targets": targets,
            "axes": (len(rows) - 1, len(cols) - 1)}


def _oriented(source, region, bands):
    candidates = []
    for t in range(8):
        cells = _transform(source, t)
        directions = _label_directions(cells, bands)
        if all(region["sides"][d] == color for color, d in directions.items()):
            candidates.append((t, cells))
    assert len(candidates) == 1, "two adjacent compass identities determine one D4 frame"
    return candidates[0]


def _coverage(g, info, *, union=False, orient=True, one_per_code=False):
    coverage = np.zeros(g.shape, dtype=np.int16)
    records, used = [], set()
    for code, target, index in info["targets"]:
        if one_per_code and code in used:
            continue
        used.add(code)
        region = info["regions"][index]
        source = info["book"][code]
        t, cells = _oriented(source, region, info["bands"]) if orient else (0, source)
        marks = {p for p, v in cells.items() if v == code}
        delta = tuple(a - b for a, b in zip(min(target), min(marks)))
        if orient:
            assert _shift(marks, delta) == target, "registration shapes must match exactly"
        body = _shift({p for p, v in cells.items() if v == info["ink"]}, delta)
        visible = {p for p in body if _inside(p, region)}
        for p in visible:
            coverage[p] += 1
        records.append({"code": code, "region": index, "transform": t,
                        "body": body, "visible": visible, "marks": target,
                        "window": _window(body | target), "clipped": body != visible})
    out = np.full(g.shape, info["background"], dtype=np.int8)
    out[info["structural"]] = g[info["structural"]]
    out[coverage > 0 if union else coverage % 2 == 1] = info["ink"]
    return out, coverage, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read prototypes and sockets, orient locally, align, clip, then XOR ink."""
    g = _grid(input_grid)
    return _coverage(g, _parse(g))[0]


def _sample_template(rng, dense=False):
    h = int(rng.integers(5, 7) if dense else rng.integers(3, 7))
    w = int(rng.integers(5, 7) if dense else rng.integers(3, 7))
    patterns = ({(0, 0)}, {(0, 0), (0, 1)}, {(0, 0), (1, 1)},
                {(0, 0), (1, 0), (0, 1)}, {(0, 0), (0, 1), (1, 0), (1, 1)})
    marks = patterns[int(rng.integers(len(patterns)))]
    ph, pw = max(r for r, _ in marks) + 1, max(c for _, c in marks) + 1
    offset = (int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1)))
    marks = _shift(marks, offset)
    # Adjacent label patches lie just beyond the core envelope, not at corners.
    top_c, right_r = int(rng.integers(w)), int(rng.integers(h))
    top = {(-1, c) for c in range(top_c, min(w, top_c + int(rng.integers(1, 3))))}
    right = {(r, w) for r in range(right_r, min(h, right_r + int(rng.integers(1, 3))))}
    keep = marks | {(0, c) for _, c in top} | {(r, w - 1) for r, _ in right}
    # Preserve one support on each boundary so empty padding never encodes size.
    keep |= {(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
             (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)}
    core = {(r, c) for r in range(h) for c in range(w)}
    desired = int(h * w * float(rng.uniform(.82, .96) if dense else rng.uniform(.45, .86)))
    remove = sorted(core - keep)
    rng.shuffle(remove)
    for p in remove:
        if len(core) <= desired:
            break
        trial = core - {p}
        if len(_components(trial, diagonal=False)) == 1:
            core = trial
    return {"body": core - marks, "marks": marks, "top": top, "right": right}


def _template(rng, dense=False):
    for _ in range(100):
        template = _sample_template(rng, dense)
        body = {p: 1 for p in template["body"]}
        poses = {tuple(sorted(_normalize(_transform(body, t)))) for t in range(8)}
        if len(poses) == 8:
            return template
    raise RuntimeError("failed to sample a visibly asymmetric stamp")


def _material(template, code, ink, top_color, right_color):
    return {**{p: ink for p in template["body"]},
            **{p: code for p in template["marks"]},
            **{p: top_color for p in template["top"]},
            **{p: right_color for p in template["right"]}}


def _axis(rng, n):
    total = int(rng.integers(28, 31) if n == 3 else rng.integers(22, 31))
    lengths = [8 if n == 3 else 9 if n == 2 else 16] * n
    for _ in range(total - n - 1 - sum(lengths)):
        lengths[int(rng.integers(n))] += 1
    bands = [0]
    for length in lengths:
        bands.append(bands[-1] + length + 1)
    return total, bands


def _scene(rng, axes, bg, colors):
    H, rows = _axis(rng, axes[0])
    W, cols = _axis(rng, axes[1])
    g = np.full((H, W), bg, dtype=np.int8)
    hc, vc = colors[:2], colors[2:]
    for j, c in enumerate(cols):
        g[:, c] = vc[j % 2]
    for i, r in enumerate(rows):
        g[r, :] = hc[i % 2]
    regions = []
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        for j, (c, d) in enumerate(zip(cols, cols[1:])):
            regions.append({"rect": (a + 1, b, c + 1, d), "sides": dict(zip(
                ("top", "bottom", "left", "right"), (hc[i % 2], hc[(i + 1) % 2],
                                                         vc[j % 2], vc[(j + 1) % 2])))})
    return g, regions


def _placements(source, code, ink, index, region, bands):
    t, cells = _oriented(source, region, bands)
    marks = {p for p, v in cells.items() if v == code}
    body = {p for p, v in cells.items() if v == ink}
    first = min(marks)
    marks = _shift(marks, (-first[0], -first[1]))
    body = _shift(body, (-first[0], -first[1]))
    r0, r1, c0, c1 = region["rect"]
    result = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            ms, bs = _shift(marks, (r, c)), _shift(body, (r, c))
            if not all(_inside(p, region) for p in ms):
                continue
            visible = {p for p in bs if _inside(p, region)}
            if not visible:
                continue
            result.append({"code": code, "region": index, "marks": ms,
                           "body": bs, "visible": visible, "origin": (r, c),
                           "transform": t, "clipped": bs != visible})
    return result


def _construct(rng, templates, codes, ink, bg, band_colors, axes, mode):
    colors = list(map(int, rng.permutation(band_colors)))
    base, regions = _scene(rng, axes, bg, colors)
    bands = set(colors)
    book = {}
    for code, template in zip(codes, templates):
        first_group = int(rng.integers(2))
        top_color = colors[2 * first_group + int(rng.integers(2))]
        right_color = colors[2 * (1 - first_group) + int(rng.integers(2))]
        book[code] = _material(template, code, ink, top_color, right_color)
    pools = {(code, i): _placements(source, code, ink, i, region, bands)
             for code, source in book.items() for i, region in enumerate(regions)}
    chosen, blocked, covered_windows = [], set(), set()

    def add(rec):
        chosen.append(rec)
        blocked.update(_halo(rec["marks"]))
        covered_windows.update(_window(rec["body"] | rec["marks"]))

    def choose(code, index=None, full=False, disjoint=False, corner=False):
        candidates = [rec for (key, i), pool in pools.items() if key == code
                      and (index is None or i == index) for rec in pool]
        rng.shuffle(candidates)
        for rec in candidates:
            if rec["marks"] & blocked or (full and rec["clipped"]):
                continue
            if disjoint and _window(rec["body"] | rec["marks"]) & covered_windows:
                continue
            if corner:
                region = regions[rec["region"]]
                r0, r1, c0, c1 = region["rect"]
                if not (any(r < r0 or r >= r1 for r, _ in rec["body"])
                        and any(c < c0 or c >= c1 for _, c in rec["body"])):
                    continue
            add(rec)
            return True
        return False

    if mode == "isolated":
        for code in codes:
            for index in range(len(regions)):
                if not choose(code, index, full=True, disjoint=True):
                    return None
    elif mode == "interference":
        code = codes[0]
        indices = list(map(int, rng.permutation(len(regions))))
        found = False
        for index in indices:
            pool = [r for r in pools[code, index] if not r["clipped"]]
            lookup = {r["origin"]: r for r in pool}
            rng.shuffle(pool)
            for first in pool:
                origin = first["origin"]
                for step in (2, 3, 4):
                    for sr, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                        second = lookup.get((origin[0] + step * sr, origin[1]))
                        third = lookup.get((origin[0], origin[1] + step * sc))
                        if second is None or third is None:
                            continue
                        group = [first, second, third]
                        if any(a["marks"] & _halo(b["marks"])
                               for j, a in enumerate(group) for b in group[j + 1:]):
                            continue
                        counts = Counter(p for rec in group for p in rec["visible"])
                        if 2 not in counts.values() or 3 not in counts.values():
                            continue
                        for rec in group:
                            add(rec)
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None
        other = next(i for i in indices if i != index)
        if not choose(code, other, corner=True):
            return None
        for code in codes[1:]:
            if not choose(code, full=True):
                return None
    else:
        for code in codes:
            repeats = int(rng.integers(5, 7) if mode == "crowded" and code == codes[0]
                          else rng.integers(3, 5) if mode == "crowded" else rng.integers(1, 5))
            for _ in range(repeats):
                if not choose(code, full=(len(chosen) == 0)):
                    return None
        if not choose(codes[int(rng.integers(len(codes)))], corner=True):
            return None

    inp = base.copy()
    for rec in chosen:
        for p in rec["marks"]:
            inp[p] = rec["code"]
    # Source locations are free and may lie in final ink footprints. Protect
    # input connectivity only; final stamps are deliberately allowed to overlap.
    for code in rng.permutation(codes):
        source = _normalize(_transform(book[int(code)], int(rng.integers(8))))
        h = max(r for r, _ in source) + 1
        w = max(c for _, c in source) + 1
        origins = [(r, c) for region in regions for r in range(region["rect"][0], region["rect"][1] - h + 1)
                   for c in range(region["rect"][2], region["rect"][3] - w + 1)]
        rng.shuffle(origins)
        for origin in origins:
            cells = {(r + origin[0], c + origin[1]): v for (r, c), v in source.items()}
            if cells.keys() & blocked:
                continue
            for p, v in cells.items():
                inp[p] = v
            blocked.update(_halo(cells))
            break
        else:
            return None
    # Construct the answer from the selected records, separately from parsing.
    count = Counter(p for rec in chosen for p in rec["visible"])
    out = base.copy()
    for p, n in count.items():
        if n % 2:
            out[p] = ink
    try:
        info = _parse(inp)
        predicted, cov, records = _coverage(inp, info)
        if len(records) != len(chosen) or not np.array_equal(predicted, out):
            return None
        if mode == "isolated" and np.array_equal(_coverage(inp, info, orient=False)[0], out):
            return None
        if mode == "interference" and not (np.any(cov == 2) and np.any(cov == 3)):
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    nq = _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    k = int(rng.integers(1, 4))
    ink, band_colors = palette[0], palette[1:5]
    codes, backgrounds = palette[5:5 + k], palette[5 + k:7 + k]
    templates = [_template(rng, dense=(i == 0)) for i in range(k)]
    first_axes = (1, 2) if rng.random() < .5 else (2, 1)
    regime = str(rng.choice(("crossed", "crowded", "extended")))
    p, seen = {"train": [], "test": []}, set()
    for split, n in (("train", nt), ("test", nq)):
        for i in range(n):
            if split == "train" and i == 0:
                axes, mode = first_axes, "isolated"
            elif split == "train" and i == 1:
                axes, mode = first_axes[::-1], "interference"
            elif split == "train":
                axes = ((1, 1), (1, 2), (2, 1), (2, 2))[int(rng.integers(4))]
                mode = "ordinary"
            elif regime == "crossed":
                axes, mode = (2, 2), "ordinary"
            elif regime == "extended":
                axes = ((3, 1), (1, 3), (3, 2), (2, 3))[int(rng.integers(4))]
                mode = "ordinary"
            else:
                axes, mode = first_axes, "crowded"
            bg = backgrounds[i] if split == "train" and i < 2 else int(rng.choice(backgrounds))
            for _ in range(120):
                pair = _construct(rng, templates, codes, ink, bg, band_colors, axes, mode)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded packing failed; no alternate rule or template substituted")
            seen.add(_key(pair["input"]))
            p[split].append(pair)
    validate_generated_puzzle(p)
    return p


def _book_signature(info):
    signatures = []
    for code, source in sorted(info["book"].items()):
        roles = {p: 0 if v == info["ink"] else 1 if v == code else 2 for p, v in source.items()}
        signature = min(tuple(sorted(_normalize(_transform(roles, t)).items())) for t in range(8))
        signatures.append((code, signature))
    return info["ink"], tuple(sorted(info["bands"])), tuple(signatures)


def _inspect(inp, out):
    info = _parse(inp)
    prediction, coverage, records = _coverage(inp, info)
    assert np.array_equal(prediction, out), "wrong orientation, registration, clipping, or parity"
    assert inp.shape == out.shape and not np.array_equal(inp, out)
    assert np.array_equal(inp[info["structural"]], out[info["structural"]])
    assert set(map(int, np.unique(out))) <= info["bands"] | {info["background"], info["ink"]}
    assert int(coverage.sum()) == sum(len(r["visible"]) for r in records)
    assert all(r["visible"] for r in records)
    return info, coverage, records


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, signature, full, poses = set(), None, Counter(), {}
    doubles = triples = clipping = no_a = no_b = no_union = False
    training_backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info, coverage, records = _inspect(inp, out)
            current = _book_signature(info)
            if signature is None:
                signature = current
            assert signature == current, "prototype vocabulary changes within the puzzle"
            if split == "train":
                training_backgrounds.add(info["background"])
                doubles |= bool(np.any(coverage == 2))
                triples |= bool(np.any(coverage == 3))
                for rec in records:
                    region = info["regions"][rec["region"]]
                    r0, r1, c0, c1 = region["rect"]
                    clipping |= (any(r < r0 or r >= r1 for r, _ in rec["body"])
                                 and any(c < c0 or c >= c1 for _, c in rec["body"]))
                no_a |= not np.array_equal(_coverage(inp, info, orient=False)[0], out)
                no_b |= not np.array_equal(_coverage(inp, info, one_per_code=True)[0], out)
                no_union |= not np.array_equal(_coverage(inp, info, union=True)[0], out)
                for j, rec in enumerate(records):
                    if not rec["clipped"] and all(not rec["window"] & other["window"]
                                                   for k, other in enumerate(records) if j != k):
                        full[rec["code"]] += 1
                    # Normalize the actual final silhouette, not transform IDs.
                    norm = tuple(sorted(_normalize({p: 1 for p in rec["body"]})))
                    poses.setdefault(rec["code"], set()).add(norm)
    assert len(training_backgrounds) == 2
    assert doubles and triples and clipping, "training must teach interference and region clipping"
    assert no_a and no_b and no_union, "the demonstrations must defeat all three omissions"
    assert all(full[code] >= 2 for code in poses)
    assert any(len(p) >= 2 for p in poses.values()), "local compass must change a silhouette"


def self_test():
    """Bounded 42-seed variation audit, counterfactuals, and corruption checks."""
    # A T registration is deliberately absent from the sampler's five shapes.
    # This fixture also checks overlap and top clipping with a scrambled source.
    fixture = np.zeros((20, 22), dtype=np.int8)
    fixture[0, :] = 3
    fixture[-1, :] = 4
    fixture[1:-1, 0] = 5
    fixture[1:-1, -1] = 6
    marks = {(1, 1), (1, 2), (1, 3), (2, 2)}
    body = {(r, c) for r in range(5) for c in range(5)} - marks - {(3, 3), (4, 2)}
    source = {**{p: 1 for p in body}, **{p: 2 for p in marks}, (-1, 2): 3, (2, 5): 6}
    source = _normalize(_transform(source, 5))
    for (r, c), v in source.items():
        fixture[r + 10, c + 13] = v
    expected = fixture.copy()
    expected[1:-1, 1:-1] = 0
    counts = Counter()
    for origin in ((3, 4), (3, 8), (0, 1)):
        for p in _shift(marks, origin):
            fixture[p] = 2
        for r, c in _shift(body, origin):
            if 1 <= r < 19 and 1 <= c < 21:
                counts[r, c] += 1
    for p, n in counts.items():
        if n % 2:
            expected[p] = 1
    assert np.array_equal(solve(fixture), expected)
    shapes, axes, vocabularies, body_shapes, registrations = set(), set(), set(), set(), set()
    target_counts, backgrounds, signatures = set(), set(), set()
    canceled = triples = clipped = old_new = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info, cov, records = _inspect(inp, out)
                shapes.add(inp.shape)
                axes.add(info["axes"])
                vocabularies.add(len(info["book"]))
                backgrounds.add(info["background"])
                target_counts.add(len(records))
                canceled += bool(np.any((cov > 0) & (cov % 2 == 0)))
                triples += bool(np.any(cov >= 3))
                clipped += any(r["clipped"] for r in records)
                old = set().union(*(set(source) for source in info["book"].values()))
                old_new += any(out[p] == info["ink"] for p in old)
                for code, source in info["book"].items():
                    core = {p for p, v in source.items() if v not in info["bands"]}
                    a, b, c, d = _bounds(core)
                    body_shapes.add((b - a + 1, d - c + 1))
                    marks = {p: 1 for p, v in source.items() if v == code}
                    registrations.add(min(tuple(sorted(_normalize(_transform(marks, t)))) for t in range(8)))
    assert len(signatures) == 42 and len(shapes) >= 20
    assert vocabularies == {1, 2, 3} and len(body_shapes) >= 8 and len(registrations) >= 5
    assert {(1, 2), (2, 1), (2, 2)} <= axes and any(3 in a for a in axes)
    assert len(target_counts) >= 8 and len(backgrounds) >= 8
    assert min(canceled, triples, clipped, old_new) >= 42
    p = generate_puzzle(0, 2, 1)
    q = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    generate_puzzle(777, 10, 4)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Output corruption must be rejected without redefining live sockets.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    inp = corrupted["test"][0]["input"]
    info = _parse(inp)
    out = corrupted["test"][0]["output"]
    r, c = map(int, np.argwhere(out == info["ink"])[0])
    out[r, c] = info["background"]
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("deleted ink accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "region_axes": sorted(axes), "vocabulary_sizes": sorted(vocabularies),
            "body_extents": len(body_shapes), "registration_geometries": len(registrations),
            "target_counts": sorted(target_counts), "backgrounds": sorted(backgrounds),
            "pairs_with_cancellation": canceled, "pairs_with_triple_coverage": triples,
            "pairs_with_clipping": clipped, "pairs_with_ink_on_old_prototypes": old_new}
