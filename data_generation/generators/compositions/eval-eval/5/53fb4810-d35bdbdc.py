"""Evolving echoes: each whole seed-tile copy advances one relation step.

The isolated 3x3 tokens are an immutable, total body-to-center lookup table.
All parsing and solving uses only the supplied input pixels. No parent code,
generation metadata, files, latent palette, or random state is used by solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0,147964504,521776206,555193616,688238767,817700408,808609063,835323869,998797109,179836945,990972352,850462823,659809147,676634561,956696153,758789200,372164239,538190298,114618592,496897784,2573719,825468998,664003641,442077817,317684148,736881990,664712993,96776933,719160427,315566691,405219905,105997673,796505776,458276126,510684873,775122232,735585842,148216670,99316923,106741921,793503150,484797824)


def _array(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape"
    assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9)), "invalid colors"
    return g.astype(np.int8, copy=False)


def _key(g):
    return tuple(g.shape), np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _components(cells):
    todo, parts = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _box(cells):
    rr, cc = zip(*cells)
    return int(min(rr)), int(min(cc)), int(max(rr)-min(rr)+1), int(max(cc)-min(cc)+1)


def _rect(box):
    t, l, h, w = box
    return {(r, c) for r in range(t, t+h) for c in range(l, l+w)}


def _collar(cells):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _mask(h, w, kind, arms=None):
    a = np.ones((h, w), dtype=bool)
    if kind == "clipped":
        a[0, 0] = a[0, -1] = a[-1, 0] = a[-1, -1] = False
    elif kind == "cross":
        th, tw = arms or (1 if h % 2 else 2, 1 if w % 2 else 2)
        a[:] = False
        a[(h-th)//2:(h+th)//2, :] = True
        a[:, (w-tw)//2:(w+tw)//2] = True
    return a


def _shape_kind(cells, box):
    t, l, h, w = box
    if not (3 <= h <= 7 and 3 <= w <= 7):
        return None
    for kind in ("rectangle", "clipped"):
        mask = _mask(h, w, kind)
        if cells == {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}:
            return "cross" if kind == "clipped" and h == w == 3 else kind
    for th in range(1, h-1):
        for tw in range(1, w-1):
            if (h-th) % 2 == (w-tw) % 2 == 0:
                mask = _mask(h, w, "cross", (th, tw))
                if cells == {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}:
                    return "cross"
    return None


def _distance(shape, box, direction):
    t, l, h, w = box
    return (t, shape[0]-t-h, l, shape[1]-l-w)[direction]


def _slice(box, direction, width, distance):
    t, l, h, w = box
    if direction < 2:
        r = t-distance if direction == 0 else t+h-1+distance
        return [(r, l+(w-width)//2+j) for j in range(width)]
    c = l-distance if direction == 2 else l+w-1+distance
    return [(t+(h-width)//2+j, c) for j in range(width)]


def _parse(input_grid):
    g = _array(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1, "background must be uniquely most frequent"
    parts = _components(zip(*np.nonzero(g != bg)))
    tokens, objects = [], []
    mapping = {}
    for cells in parts:
        box = _box(cells)
        t, l, h, w = box
        if (h, w) == (3, 3) and len(cells) == 9:
            center = t+1, l+1
            bodies = {int(g[p]) for p in cells if p != center}
            assert len(bodies) == 1, "token ring must have a uniform body"
            body, value = bodies.pop(), int(g[center])
            assert body not in mapping, "table keys must be unique"
            mapping[body] = value
            tokens.append(dict(cells=cells, center=center, body=body, value=value))
        else:
            objects.append(cells)
    assert 3 <= len(tokens) <= 8 and 1 <= len(objects) <= 4, "missing table or emitters"
    assert set(mapping.values()) <= set(mapping), "table must be total on its identities"
    hub_colors = set(map(int, np.unique(g))) - {bg} - set(mapping)
    assert len(hub_colors) == 1, "one geometrically separate hub color is required"
    hub_color = hub_colors.pop()
    assert hub_color not in mapping.values()
    emitters = []
    for cells in objects:
        hub = {p for p in cells if int(g[p]) == hub_color}
        assert hub and len(_components(hub)) == 1
        box = _box(hub)
        kind = _shape_kind(hub, box)
        assert kind is not None, "malformed hub"
        tail = cells - hub
        assert tail and all(int(g[p]) in mapping for p in tail)
        assert tail == _rect(_box(tail)), "seed strip must be a filled rectangle"
        candidates = []
        t, l, h, w = box
        for d in range(4):
            transverse = w if d < 2 else h
            for width in range(1, min(3, transverse-2)+1):
                if (transverse-width) % 2:
                    continue
                length = len(tail)//width
                maximum = _distance(g.shape, box, d)
                if not (1 <= length <= maximum and length*width == len(tail)):
                    continue
                points = {p for z in range(1, length+1) for p in _slice(box, d, width, z)}
                if points != tail:
                    continue
                dr, dc = _D4[d]
                if not all((r-dr, c-dc) in hub for r, c in _slice(box, d, width, 1)):
                    continue
                word = [tuple(int(g[p]) for p in _slice(box, d, width, z)) for z in range(1, length+1)]
                candidates.append(dict(box=box, hub=hub, kind=kind, cells=tail, direction=d,
                                       width=width, word=word, length=length, maximum=maximum,
                                       complete=length == maximum))
        assert len(candidates) == 1, "strip must have a unique centered attachment"
        emitters.extend(candidates)
    # Strong separation is part of the grammar, not a hidden nearest-neighbor rule.
    for i, cells in enumerate(parts):
        assert all(not (_collar(cells) & other) for other in parts[i+1:]), "objects need empty collars"
    return g, dict(background=bg, hub_color=hub_color, tokens=tokens, mapping=mapping, emitters=emitters)


def _predict(input_grid, *, mode="intended"):
    g, a = _parse(input_grid)
    f, emitters = a["mapping"], a["emitters"]
    result, writes, crossings = g.copy(), {}, set()
    protected = set().union(*(t["cells"] for t in a["tokens"]))
    protected |= set().union(*(e["hub"] | (e["cells"] if not e["complete"] else set()) for e in emitters))
    completed = set().union(*(e["cells"] for e in emitters if e["complete"]))
    for e in emitters:
        if e["complete"]:
            continue
        word = e["word"]
        length = e["length"]
        if mode == "shortest_period":
            length = next(p for p in range(1, len(word)+1) if len(word) % p == 0
                          and all(row == word[i % p] for i, row in enumerate(word)))
        for distance in range(e["length"]+1, e["maximum"]+1):
            phase, generation = (distance-1) % length, (distance-1)//length
            if mode == "plain":
                generation = 0
            elif mode == "once":
                generation = 1
            elif mode == "per_slice":
                generation = distance - e["length"]
            values = list(word[0 if mode == "no_phase" else phase])
            for _ in range(generation):
                values = [f[v] for v in values]
            for p, value in zip(_slice(e["box"], e["direction"], e["width"], distance), values):
                assert p not in protected, "echo hits table, hub, or another original seed"
                assert p not in writes, "new echoes must not intersect"
                if int(g[p]) != a["background"]:
                    assert p in completed, "unknown obstruction"
                    crossings.add(p)
                writes[p] = value
                result[p] = value
    a["writes"], a["crossings"] = writes, crossings
    return result, a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Repeat the full visible seed, advancing all its colors once per copy."""
    return _predict(input_grid)[0]


def _new_spec(rng, shape, width, length, complete, f, special=None, minimum=0, dynamic=False):
    direction = int(rng.integers(4))
    transverse = width + 2*int(rng.integers(1, 3))
    longitudinal = int(rng.integers(3, 7))
    h, w = (longitudinal, transverse) if direction < 2 else (transverse, longitudinal)
    if max(h, w) > 7 or h > shape[0] or w > shape[1]:
        return None
    t, l = int(rng.integers(shape[0]-h+1)), int(rng.integers(shape[1]-w+1))
    box = t, l, h, w
    maximum = _distance(shape, box, direction)
    if maximum < (max(length+1, minimum) if not complete else max(1, minimum)):
        return None
    kind = str(rng.choice(["rectangle", "clipped", "cross"]))
    ths = [x for x in range(1, h-1) if (h-x) % 2 == 0 and (direction < 2 or x >= width)]
    tws = [x for x in range(1, w-1) if (w-x) % 2 == 0 and (direction >= 2 or x >= width)]
    mask = _mask(h, w, kind, (int(rng.choice(ths)), int(rng.choice(tws))))
    hub = {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}
    palette = list(f)
    if complete:
        length = maximum
    word = rng.choice(palette, size=(length, width)).astype(np.int8)
    if special == "clock":
        word[:] = palette[0]  # AA is one entire two-slice seed, not period one.
    elif special == "phase":
        word[0] = palette[0]
        word[0, -1] = palette[3]
        word[1] = palette[1]
        word[1, -1] = palette[0]
        word[2] = word[0]  # ABA is a complete word, not a prefix of ABAB.
    elif complete and length >= 5:
        # A visible early ABAB cycle followed by an exception; leave it authoritative.
        word[2] = word[0]
        word[3] = word[1]
        word[-1, 0] = next(v for v in palette if v != int(word[(length-1) % 2, 0]))
    if dynamic:
        moving = [v for v in f if f[v] != v and f[f[v]] != f[v]]
        word[0, 0] = int(rng.choice(moving))
    original = {p for z in range(1, length+1) for p in _slice(box, direction, width, z)}
    extension = {p for z in range(length+1, maximum+1) for p in _slice(box, direction, width, z)}
    return dict(box=box, hub=hub, kind=kind, direction=direction, width=width, length=length,
                maximum=maximum, complete=complete, word=word, original=original, extension=extension)


def _compatible(a, previous, allow_crossing):
    if a is None:
        return False
    for b in previous:
        if _collar(a["hub"] | a["original"]) & (b["hub"] | b["original"]):
            return False
        if _collar(a["hub"]) & b["extension"] or _collar(b["hub"]) & a["extension"]:
            return False
        if a["extension"] & b["extension"]:
            return False
        for x, y in ((a, b), (b, a)):
            if x["extension"] & y["original"]:
                if not allow_crossing or not y["complete"] or x["direction"]//2 == y["direction"]//2:
                    return False
    return True


def _place_tokens(rng, shape, specs, n):
    forbidden = set().union(*(_collar(s["hub"] | s["original"] | s["extension"]) for s in specs))
    boxes = []
    candidates = [(r, c, 3, 3) for r in range(shape[0]-2) for c in range(shape[1]-2)]
    for index in rng.permutation(len(candidates)):
        box = candidates[int(index)]
        cells = _rect(box)
        if not cells & forbidden:
            boxes.append(box)
            forbidden |= _collar(cells)
            if len(boxes) == n:
                return boxes
    return None


def _paint(shape, specs, boxes, f, bg, hub_color):
    g = np.full(shape, bg, dtype=np.int8)
    # Relational evidence is preserved in its original form in both grids.
    for box, (body, value) in zip(boxes, f.items()):
        t, l, _, _ = box
        g[t:t+3, l:l+3] = body
        g[t+1, l+1] = value
    for e in specs:
        for p in e["hub"]:
            g[p] = hub_color
        for z, row in enumerate(e["word"], 1):
            for p, value in zip(_slice(e["box"], e["direction"], e["width"], z), row):
                g[p] = value
    output = g.copy()
    # Constructor advances a full mutable block at each block boundary.
    # The solver instead computes the generation exponent for each slice.
    for e in specs:
        if e["complete"]:
            continue
        block = e["word"].copy()
        start = e["length"]+1
        while start <= e["maximum"]:
            block = np.array([[f[int(v)] for v in row] for row in block], dtype=np.int8)
            for j, row in enumerate(block):
                z = start+j
                if z > e["maximum"]:
                    break
                for p, value in zip(_slice(e["box"], e["direction"], e["width"], z), row):
                    output[p] = value
            start += e["length"]
    return {"input": g, "output": output}


def _table(rng, identities, special):
    values = list(map(int, rng.choice(identities, size=len(identities))))
    a, b, c = identities[:3]
    if special == "clock":
        values[:3] = [b, c, a]
    elif special == "phase":
        d = identities[3]
        values[:4] = [b, c, c, b]
    return dict(zip(identities, values))


def _fallback(rng, f, bg, hub_color, count, primary_width, primary_length, special, complete, dynamic=False):
    """Finite spacious escape; no retries, external state, or hidden semantics."""
    shape = (30, 30)
    specs = []
    # Three/four width-three lanes leave row 23 onward for up to eight tokens.
    for i in range(count):
        width = primary_width if i == 0 else 1
        length = primary_length if i == 0 else 2
        t, l = 1+6*i, 1
        h, w = width+2, 3
        box = t, l, h, w
        mask = _mask(h, w, "clipped")
        hub = {(t+int(r), l+int(c)) for r, c in zip(*np.nonzero(mask))}
        maximum = 26
        is_complete = bool(complete and i == count-1 and i != 0)
        if is_complete:
            length = maximum
        word = rng.choice(list(f), size=(length, width)).astype(np.int8)
        if i == 0 and special == "clock":
            word[:] = list(f)[0]
        elif i == 0 and special == "phase":
            word[0] = list(f)[0]
            word[0, -1] = list(f)[3]
            word[1] = list(f)[1]
            word[1, -1] = list(f)[0]
            word[2] = word[0]
        elif is_complete:
            word[2] = word[0]
            word[3] = word[1]
            word[-1, 0] = next(v for v in f if v != int(word[(length-1) % 2, 0]))
        if i == 0 and dynamic:
            word[0, 0] = int(rng.choice([v for v in f if f[v] != v and f[f[v]] != f[v]]))
        specs.append(dict(box=box, hub=hub, kind="clipped", direction=3, width=width,
                          length=length, maximum=maximum, complete=is_complete, word=word,
                          original={p for z in range(1, length+1) for p in _slice(box, 3, width, z)},
                          extension={p for z in range(length+1, maximum+1) for p in _slice(box, 3, width, z)}))
    slots = [(r, c, 3, 3) for r in (23, 27) for c in (1, 5, 9, 13, 17, 21, 25)]
    rng.shuffle(slots)
    return _paint(shape, specs, slots[:len(f)], f, bg, hub_color)


def _make_pair(rng, config, special=None, testing=False):
    regime, intensity, stable_roles, roles, crossing_enabled = config
    colors = list(map(int, rng.permutation(10)))
    if stable_roles:
        bg, hub_color = roles
        colors = [bg, hub_color] + [c for c in colors if c not in roles]
    bg, hub_color = colors[:2]
    n = int(rng.integers(4 if special == "phase" else 3, 7+int(intensity > 0)))
    identities = colors[2:2+n]
    f = _table(rng, identities, special)
    if testing and not any(f[v] != v and f[f[v]] != f[v] for v in f):
        # Every test must actually exercise elapsed copy count as well as lookup.
        a, b, c = identities[:3]
        f.update({a: b, b: c, c: a})
    count = (2 if special == "clock" else int(rng.integers(1, 3))) if special else int(rng.integers(1, 3+intensity))
    if testing and regime == "count":
        count = int(rng.integers(3, 5))
    elif regime == "count":
        count = min(count, 2)
    primary_width = 1 if special == "clock" else 2 if special == "phase" else int(rng.integers(1, 3))
    primary_length = 2 if special == "clock" else 3 if special == "phase" else int(rng.integers(1, 4))
    if testing and regime == "width":
        primary_width = 3
    if testing and regime == "length":
        primary_length = 4
    # Completed material is useful as a preservation witness, never a color key.
    include_complete = count > 1 and (special == "clock" or rng.random() < .42)
    allow_crossing = crossing_enabled
    for scene in range(10):
        low = 14 if count <= 2 else 22
        if scene < 5:
            shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        else:
            shape = (30, 30)
        specs = []
        # Complete strips are placed first so an incomplete ray can cross them.
        order = list(range(count))
        if include_complete:
            order = [count-1] + list(range(count-1))
        for i in order:
            is_complete = include_complete and i == count-1
            for _ in range(100):
                width = primary_width if i == 0 else int(rng.integers(1, 3))
                length = primary_length if i == 0 else int(rng.integers(1, 4))
                minimum = 3*length+1 if special and i == 0 else length+1
                if special == "clock" and is_complete:
                    minimum = 7
                if testing and i == 0:
                    minimum = max(minimum, 2*length+1)
                s = _new_spec(rng, shape, width, length, is_complete, f,
                              special=special if i == 0 else None, minimum=minimum,
                              dynamic=testing and i == 0)
                if _compatible(s, specs, allow_crossing):
                    specs.append(s)
                    break
            else:
                break
        if len(specs) != count:
            continue
        boxes = _place_tokens(rng, shape, specs, n)
        if boxes is None:
            continue
        pair = _paint(shape, specs, boxes, f, bg, hub_color)
        try:
            predicted, a = _predict(pair["input"])
            assert np.array_equal(predicted, pair["output"])
            if special == "clock":
                assert _witnesses(pair["input"])["clock"]
        except AssertionError:
            continue
        break
    else:
        pair = _fallback(rng, f, bg, hub_color, count, primary_width, primary_length, special,
                         include_complete, dynamic=testing)
    rotation = int(rng.integers(4))
    mirror = bool(rng.integers(2))
    return {k: np.ascontiguousarray(np.fliplr(np.rot90(v, rotation)) if mirror else np.rot90(v, rotation))
            for k, v in pair.items()}


def _counts(num_train, num_test):
    for name, v, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    # Sampling settings never alter the visible input-only transformation.
    config = (str(rng.choice(["width", "length", "count"])), int(rng.integers(2)),
              bool(rng.integers(2)), tuple(map(int, rng.choice(10, 2, replace=False))), False)
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            special = ("clock", "phase")[i] if split == "train" and i < 2 else None
            for _ in range(32):
                pair = _make_pair(rng, config, special, testing=split == "test")
                key = _key(pair["input"])
                if key not in used:
                    break
            else:
                raise RuntimeError("bounded uniqueness attempts exhausted")
            used.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(g):
    predicted, a = _predict(g)
    f = a["mapping"]
    clock = phase = merge = False
    for e in a["emitters"]:
        word, p = e["word"], e["length"]
        if e["complete"]:
            continue
        if p == 2 and word[0] == word[1] and e["maximum"] >= 3*p:
            clock |= any(f[v] != v and f[f[v]] not in (v, f[v]) for v in word[0])
        if p == 3 and word[0] == word[2] and word[0] != word[1] and e["maximum"] >= 2*p:
            phase = True
        used = {v for row in word for v in row}
        merge |= any(x != y and f[x] == f[y] for x in used for y in used)
    preserved_complete = any(e["complete"] and all(predicted[p] == g[p] for p in e["cells"]-a["crossings"])
                             for e in a["emitters"])
    irregular = any(e["complete"] and len(e["word"]) >= 5
                    and e["word"][0] == e["word"][2] and e["word"][1] == e["word"][3]
                    and any(row != e["word"][i % 2] for i, row in enumerate(e["word"]) if i >= 4)
                    for e in a["emitters"])
    return dict(clock=bool(clock), phase=bool(phase), merge=bool(merge),
                preserved_complete=bool(preserved_complete), irregular_complete=bool(irregular))


def puzzle_features(puzzle):
    """Compact, grid-derived structural evidence; no hidden construction labels."""
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            g = _array(pair["input"])
            _, a = _predict(g)
            es, f = a["emitters"], a["mapping"]
            active = [e for e in es if not e["complete"]]
            records.append(dict(split=split, shape=list(g.shape), background=a["background"],
                                hub_color=a["hub_color"], identities=len(f), emitters=len(es),
                                widths=sorted({e["width"] for e in active}),
                                lengths=sorted({e["length"] for e in active}),
                                directions=sorted({e["direction"] for e in es}),
                                hub_shapes=sorted({e["kind"] for e in es}),
                                completed=sum(e["complete"] for e in es),
                                extensions=sorted(e["maximum"]-e["length"] for e in active),
                                generations=sorted((e["maximum"]-1)//e["length"] for e in active),
                                clipped=any(e["maximum"] % e["length"] for e in active),
                                fixed=sum(k == v for k, v in f.items()),
                                merging=len(set(f.values())) < len(f), crossings=len(a["crossings"]),
                                **_witnesses(g)))
    return records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _array(pair["input"]), _array(pair["output"])
            assert g.shape == out.shape
            key = _key(g)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, a = _predict(g)
            assert np.array_equal(expected, out), "incorrect echo, phase, lookup, or preservation"
            changed = out != g
            allowed = np.zeros(g.shape, dtype=bool)
            for p in a["writes"]:
                allowed[p] = True
            assert not np.any(changed & ~allowed), "changed protected cells"
            assert all(out[p] == g[p] for t in a["tokens"] for p in t["cells"]), "table mutated"
            analyses[split].append(a)
    witnesses = [_witnesses(_array(p["input"])) for p in puzzle["train"]]
    for name in ("clock", "phase", "merge", "preserved_complete", "irregular_complete"):
        assert any(w[name] for w in witnesses), f"missing training witness: {name}"
    # Directly check the two parent omissions and the temporal shortcut, not hashes.
    for mode in ("plain", "no_phase", "once", "per_slice", "shortest_period"):
        assert any(not np.array_equal(_predict(_array(p["input"]), mode=mode)[0], p["output"])
                   for p in puzzle["train"]), f"training does not distinguish {mode}"
    for pair in puzzle["test"]:
        for mode in ("plain", "once"):
            assert not np.array_equal(_predict(_array(pair["input"]), mode=mode)[0], pair["output"]), f"test omits {mode} contribution"
    if any(a["crossings"] for a in analyses["test"]):
        assert any(a["crossings"] for a in analyses["train"]), "untaught crossing"
    train_es = [e for a in analyses["train"] for e in a["emitters"] if not e["complete"]]
    test_es = [e for a in analyses["test"] for e in a["emitters"] if not e["complete"]]
    assert test_es
    assert (max(e["width"] for e in test_es) > max(e["width"] for e in train_es)
            or max(e["length"] for e in test_es) > max(e["length"] for e in train_es)
            or max(len(a["emitters"]) for a in analyses["test"]) > max(len(a["emitters"]) for a in analyses["train"])), "missing structural train/test shift"


def self_test():
    """Bounded 42-seed checks. Retains metrics and hashes, never a seed corpus."""
    records, hashes = [], set()
    for seed in _HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        h = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                for role in ("input", "output"):
                    key = _key(pair[role])
                    h.update(bytes(key[0]))
                    h.update(key[1])
        hashes.add(h.hexdigest())
        records.extend(puzzle_features(p))
    assert len(hashes) == 42
    assert len({tuple(r["shape"]) for r in records}) >= 50
    assert {d for r in records for d in r["directions"]} == set(range(4))
    assert {w for r in records for w in r["widths"]} == {1, 2, 3}
    assert {p for r in records for p in r["lengths"]} >= {1, 2, 3, 4}
    assert {r["emitters"] for r in records} == {1, 2, 3, 4}
    assert {r["identities"] for r in records} >= {3, 4, 5, 6, 7}
    assert {r["background"] for r in records} == set(range(10))
    assert len({(r["background"], r["hub_color"]) for r in records}) >= 30
    assert any(r["fixed"] for r in records) and any(r["merging"] for r in records)
    assert any(r["clipped"] for r in records)
    assert {x for r in records for x in r["hub_shapes"]} == {"rectangle", "clipped", "cross"}
    # Exercise the escape construction directly, including its widest clocks.
    f = {1: 2, 2: 3, 3: 1, 4: 2, 6: 6, 7: 4, 8: 7, 9: 8}
    for width in (1, 2, 3):
        for length in (1, 2, 3, 4):
            for count in (1, 4):
                pair = _fallback(np.random.default_rng(900+width*10+length), f, 0, 5,
                                 count, width, length, None, True, dynamic=True)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                for mode in ("plain", "once"):
                    assert not np.array_equal(_predict(pair["input"], mode=mode)[0], pair["output"])
    clock = _fallback(np.random.default_rng(5), f, 0, 5, 2, 1, 2, "clock", True)
    assert clock["output"][2, 4:12].tolist() == [1, 1, 2, 2, 3, 3, 1, 1]
    assert _witnesses(clock["input"])["irregular_complete"]
    fm = {1: 2, 2: 3, 3: 3, 4: 2, 6: 6, 7: 4, 8: 7, 9: 8}
    phase = _fallback(np.random.default_rng(6), fm, 0, 5, 2, 2, 3, "phase", True)
    assert np.array_equal(solve(phase["input"]), phase["output"])
    assert _witnesses(phase["input"])["phase"] and _witnesses(phase["input"])["merge"]
    # Explicit count and rejection checks supplement the host verifier.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(7601, nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    # D4 equivariance and arbitrary role permutations hold for all family inputs.
    permutation = np.array([7, 3, 9, 0, 6, 1, 8, 4, 2, 5], dtype=np.int8)
    for pair in p["train"] + p["test"]:
        for turns in range(4):
            g = np.fliplr(np.rot90(pair["input"], turns))
            expected = np.fliplr(np.rot90(pair["output"], turns))
            assert np.array_equal(solve(g), expected)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    token = _parse(bad["train"][0]["input"])[1]["tokens"][0]
    at = token["center"]
    bad["train"][0]["output"][at] = (int(bad["train"][0]["output"][at])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("mutated donor accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({tuple(r["shape"]) for r in records}),
            "role_pairs": len({(r["background"], r["hub_color"]) for r in records}), "valid": True}
