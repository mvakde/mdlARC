"""Turning corner loom: cyclic color instructions on a shrinking rectangle.

Only the visible input is used by solve. No parent generator is imported.
Run this module directly for the bounded 42-seed self-test.
"""

from collections import Counter
import json

import numpy as np


# Clockwise order, independent of literal colors and canvas position.
_DIR = ((-1, -1), (-1, 1), (1, 1), (1, -1))
_COMBOS = tuple((s, d) for s in range(4) for d in range(4) if s != d)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.tobytes()


def _geometry(top, left, height, width, corner):
    dr, dc = _DIR[corner]
    r = top + (height - 1 if dr > 0 else 0)
    c = left + (width - 1 if dc > 0 else 0)
    center = (r + dr, c + dc)
    socket = frozenset(((r + 2 * dr, c), (r, c + 2 * dc),
                        (r + 2 * dr, c + 2 * dc)))
    return center, socket


def _decode(value):
    """Read the key, hollow frame, and unique source/socket from the input."""
    a = _grid(value)
    if a.shape[0] < 7:
        raise ValueError("missing header or chamber")
    bg = int(a[2, 0])
    if not np.all(a[2] == bg):
        raise ValueError("row 2 must be the blank header/chamber gap")
    key_columns = np.flatnonzero(a[0] != bg)
    k = len(key_columns)
    if not 2 <= k <= 6 or not np.array_equal(key_columns, np.arange(k)):
        raise ValueError("key must occupy the first 2..6 cells of row 0")
    key = tuple(int(x) for x in a[0, :k])
    sep = int(a[1, 0])
    if (len(set(key)) < 2 or sep == bg or sep in key
            or not np.all(a[1, :k] == sep) or not np.all(a[1, k:] == bg)):
        raise ValueError("invalid key/separator roles")
    coords = np.argwhere(a[3:] == sep)
    if not len(coords):
        raise ValueError("missing rectangular frame")
    top, left = (int(x) for x in coords.min(axis=0))
    bottom, right = (int(x) for x in coords.max(axis=0))
    top += 3
    bottom += 3
    h, w = bottom - top - 1, right - left - 1
    if h < 2 or w < 2:
        raise ValueError("chamber interior must be at least 2 by 2")
    clean = np.full_like(a, bg)
    clean[0, :k] = key
    clean[1, :k] = sep
    clean[top, left:right + 1] = sep
    clean[bottom, left:right + 1] = sep
    clean[top:bottom + 1, left] = sep
    clean[top:bottom + 1, right] = sep
    # Everything not explained by the blank chamber, frame, or header is control.
    residual = np.argwhere(a != clean)
    if len(residual) != 4:
        raise ValueError("expected exactly four control cells")
    cells = frozenset(tuple(int(x) for x in p) for p in residual)
    colors = {int(a[p]) for p in cells}
    if len(colors) != 1:
        raise ValueError("control cells must share one color")
    marker = colors.pop()
    if marker in {bg, sep, *key} or any(clean[p] != bg for p in cells):
        raise ValueError("control role overlaps the key or frame")
    candidates = []
    for s, d in _COMBOS:
        source, _ = _geometry(top, left, h + 2, w + 2, s)
        destination, socket = _geometry(top, left, h + 2, w + 2, d)
        if cells == socket | {source}:
            candidates.append((s, d, destination))
    if len(candidates) != 1:
        raise ValueError("source/socket geometry must have one interpretation")
    s, d, destination = candidates[0]
    if not (0 <= destination[0] < a.shape[0]
            and 0 <= destination[1] < a.shape[1]) or a[destination] != bg:
        raise ValueError("socket center must be blank and in bounds")
    return a, dict(bg=bg, sep=sep, marker=marker, key=key, cells=cells,
                   top=top, left=left, h=h, w=w, source=s, destination=d,
                   turn=(d - s) % 4, landing=destination)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, m = _decode(input_grid)
    out = a.copy()
    for cell in m["cells"]:
        out[cell] = m["bg"]
    out[m["landing"]] = m["marker"]
    lo_r, hi_r = m["top"] + 1, m["top"] + m["h"]
    lo_c, hi_c = m["left"] + 1, m["left"] + m["w"]
    corner, step = m["source"], 0
    while lo_r <= hi_r and lo_c <= hi_c:
        dr, dc = _DIR[corner]
        row = lo_r if dr < 0 else hi_r
        col = lo_c if dc < 0 else hi_c
        color = m["key"][step % len(m["key"])]
        out[row, lo_c:hi_c + 1] = color
        out[lo_r:hi_r + 1, col] = color
        if dr < 0:
            lo_r += 1
        else:
            hi_r -= 1
        if dc < 0:
            lo_c += 1
        else:
            hi_c -= 1
        corner = (corner + m["turn"]) % 4
        step += 1
    return out


def _layers(h, w, source, turn):
    """Coordinate-set construction, also used to check the painted partition."""
    rows, cols = list(range(h)), list(range(w))
    corner = source
    while rows and cols:
        dr, dc = _DIR[corner]
        r, c = rows[0 if dr < 0 else -1], cols[0 if dc < 0 else -1]
        yield frozenset([(r, x) for x in cols] + [(y, c) for y in rows])
        rows.remove(r)
        cols.remove(c)
        corner = (corner + turn) % 4


def _paint(h, w, key, source, turn):
    out = np.full((h, w), -1, dtype=np.int8)
    for i, cells in enumerate(_layers(h, w, source, turn)):
        for cell in cells:
            out[cell] = key[i % len(key)]
    assert np.all(out >= 0)
    return out


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key_kind(key):
    if len(set(key)) == len(key):
        return "distinct"
    return "adjacent" if any(x == y for x, y in zip(key, key[1:])) else "separated"


def _labels(rng, k, kind):
    if kind == "distinct":
        return tuple(range(k))
    for _ in range(300):
        ncolors = int(rng.integers(2, min(k - 1, 4) + 1))
        key = [0, 1]
        for _ in range(k - 2):
            choices = [c for c in range(ncolors)
                       if kind != "separated" or c != key[-1]]
            key.append(int(rng.choice(choices)))
        # A primitive cycle makes the visible key length informative.
        primitive = all(any(key[i] != key[(i + shift) % k] for i in range(k))
                        for shift in range(1, k))
        if _key_kind(key) == kind and primitive:
            return tuple(key)
    raise RuntimeError("could not sample a primitive repeated key")


def _random_labels(rng, k):
    kind = "distinct" if k == 2 else str(rng.choice(("distinct", "adjacent", "separated")))
    return _labels(rng, k, kind)


def _axis(rng, minimum, maximum, cap, at_least=0):
    footprint = maximum - minimum + 1
    small = max(footprint, at_least)
    extent = small + int(rng.integers(0, min(5, cap - small) + 1))
    offset = int(rng.integers(0, extent - footprint + 1)) - minimum
    return extent, offset


def _make_pair(rng, h, w, labels, palette, combo, layout=None, reserve_all=False):
    bg, sep, marker, *payload = palette
    key = tuple(payload[x] for x in labels)
    s, d = combo
    if layout is None:
        if reserve_all:
            min_r, max_r, min_c, max_c = -2, h + 3, -2, w + 3
        else:
            source, _ = _geometry(0, 0, h + 2, w + 2, s)
            _, socket = _geometry(0, 0, h + 2, w + 2, d)
            points = list(socket) + [source, (0, 0), (h + 1, w + 1)]
            min_r, max_r = min(p[0] for p in points), max(p[0] for p in points)
            min_c, max_c = min(p[1] for p in points), max(p[1] for p in points)
        height, top = _axis(rng, min_r, max_r, 27)
        width, left = _axis(rng, min_c, max_c, 30, len(key))
        height, top = height + 3, top + 3
        layout = height, width, top, left
    height, width, top, left = layout
    a = np.full((height, width), bg, dtype=np.int8)
    a[0, :len(key)] = key
    a[1, :len(key)] = sep
    a[top, left:left + w + 2] = sep
    a[top + h + 1, left:left + w + 2] = sep
    a[top:top + h + 2, left] = sep
    a[top:top + h + 2, left + w + 1] = sep
    source, _ = _geometry(top, left, h + 2, w + 2, s)
    landing, socket = _geometry(top, left, h + 2, w + 2, d)
    controls = socket | {source}
    for cell in controls:
        a[cell] = marker
    out = a.copy()
    for cell in controls:
        out[cell] = bg
    out[landing] = marker
    out[top + 1:top + h + 1, left + 1:left + w + 1] = _paint(
        h, w, key, s, (d - s) % 4)
    return {"input": a, "output": out}, layout


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    constant_palette = bool(rng.integers(0, 2))
    shared = tuple(int(x) for x in rng.permutation(10))
    k = int(rng.integers(2, 7))
    labels = _random_labels(rng, k)
    short = 2 * k + 1 + int(rng.integers(0, 2))
    longer = short + int(rng.integers(0, 6))
    h, w = (short, longer) if rng.integers(0, 2) else (longer, short)
    s = int(rng.integers(0, 4))
    q = int(rng.choice((1, 3)))
    d1, d2 = (s + q) % 4, (s - q) % 4
    train_combos = [(s, d1), (s, d2), ((d1 + 2) % 4, d1)]
    puzzle = {"train": [], "test": []}
    seen, descriptions, training_minima = set(), [], []
    base_layout = None
    for i in range(ntrain):
        if i < 3:
            ph, pw, lab, palette, combo = h, w, labels, shared, train_combos[i]
        else:
            pk = int(rng.integers(2, min(6, short - 1) + 1))
            lab = _random_labels(rng, pk)
            minor = int(rng.integers(pk + 2, short + 2))
            major = min(21, minor + int(rng.integers(0, 7)))
            ph, pw = (minor, major) if rng.integers(0, 2) else (major, minor)
            palette = shared if constant_palette else tuple(int(x) for x in rng.permutation(10))
            combo = _COMBOS[int(rng.integers(0, len(_COMBOS)))]
        for _ in range(100):
            pair, layout = _make_pair(rng, ph, pw, lab, palette, combo,
                                      base_layout if i < 3 else None, i < 3)
            sig = _signature(pair["input"])
            if sig not in seen:
                break
        else:
            raise RuntimeError("could not construct a distinct training input")
        if i == 0:
            base_layout = layout
        puzzle["train"].append(pair)
        seen.add(sig)
        descriptions.append(lab)
        training_minima.append(min(ph, pw))
    turns = {(d - s) % 4 for s, d in train_combos[:min(ntrain, 3)]}
    # With two demonstrations the test reuses a demonstrated relation.
    candidates = (train_combos[:2] if ntrain == 2 else
                  [c for c in _COMBOS if (c[1] - c[0]) % 4 in turns])
    if ntrain >= 3:
        unseen = [c for c in candidates if c not in train_combos[:3]]
        first_test = unseen[int(rng.integers(0, len(unseen)))]
    else:
        first_test = candidates[int(rng.integers(0, len(candidates)))]
    used_combos = set()
    for i in range(ntest):
        available = [c for c in candidates if c not in used_combos]
        combo = first_test if i == 0 else available[int(rng.integers(0, len(available)))]
        used_combos.add(combo)
        lab = descriptions[int(rng.integers(0, len(descriptions)))]
        minor = int(rng.integers(max(training_minima) + 1,
                                 min(20, max(training_minima) + 3) + 1))
        major = minor + int(rng.integers(0, min(5, 21 - minor) + 1))
        ph, pw = (minor, major) if rng.integers(0, 2) else (major, minor)
        # Permuting payload identities changes key colors/order without adding
        # a repeated-color topology absent from the demonstrations.
        if constant_palette:
            palette = shared[:3] + tuple(int(x) for x in rng.permutation(shared[3:]))
        else:
            palette = tuple(int(x) for x in rng.permutation(10))
        for _ in range(100):
            pair, _ = _make_pair(rng, ph, pw, lab, palette, combo)
            sig = _signature(pair["input"])
            if sig not in seen:
                break
        else:
            raise RuntimeError("could not construct a distinct test input")
        puzzle["test"].append(pair)
        seen.add(sig)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check visible geometry, coverage, phase, controls, contrasts and shift."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, m = _decode(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape
            sig = _signature(a)
            assert sig not in seen
            seen.add(sig)
            t, l, h, w = m["top"], m["left"], m["h"], m["w"]
            patch = out[t + 1:t + h + 1, l + 1:l + w + 1]
            outside = np.ones(a.shape, dtype=bool)
            outside[t + 1:t + h + 1, l + 1:l + w + 1] = False
            expected_outside = a.copy()
            for cell in m["cells"]:
                expected_outside[cell] = m["bg"]
            expected_outside[m["landing"]] = m["marker"]
            assert np.array_equal(out[outside], expected_outside[outside])
            assert np.count_nonzero(out == m["marker"]) == 1
            covered, histogram = set(), Counter()
            layers = list(_layers(h, w, m["source"], m["turn"]))
            assert len(layers) == min(h, w)
            for i, cells in enumerate(layers):
                assert not covered.intersection(cells)
                assert len(cells) == (h - i) + (w - i) - 1
                color = m["key"][i % len(m["key"])]
                assert all(int(patch[cell]) == color for cell in cells)
                histogram[color] += len(cells)
                covered.update(cells)
            assert len(covered) == h * w
            assert histogram == Counter(int(x) for x in patch.flat)
            assert np.array_equal(solve(a), out)
            # Direct counterfactuals on the SAME chamber, key and source.
            assert not np.all(patch == m["key"][0])  # omit A's ordered program
            assert not np.array_equal(patch, _paint(h, w, m["key"], m["source"], 0))
            analyses[split].append(m)  # omit B's repeated turn
    train, test = analyses["train"], analyses["test"]
    a, b = train[:2]
    for name in ("key", "bg", "sep", "marker", "top", "left", "h", "w", "source"):
        assert a[name] == b[name]
    assert a["destination"] != b["destination"]
    assert {a["turn"], b["turn"]} == {1, 3}
    assert min(a["h"], a["w"]) >= 2 * len(a["key"]) + 1
    if nt >= 3:
        c = train[2]
        assert c["destination"] == a["destination"] and c["source"] != a["source"]
        assert c["turn"] == 2
        for name in ("key", "bg", "sep", "marker", "top", "left", "h", "w"):
            assert c[name] == a[name]
    train_turns = {m["turn"] for m in train}
    train_key_forms = {(len(m["key"]), _key_kind(m["key"])) for m in train}
    for m in test:
        assert min(m["h"], m["w"]) > max(min(x["h"], x["w"]) for x in train)
        assert m["turn"] in train_turns
        assert (len(m["key"]), _key_kind(m["key"])) in train_key_forms
        if nt == 2:
            assert (m["source"], m["destination"]) in {
                (x["source"], x["destination"]) for x in train}


def _puzzle_signature(puzzle):
    return tuple(_signature(_grid(p[key])) for split in ("train", "test")
                 for p in puzzle[split] for key in ("input", "output"))


def self_test():
    """Bounded semantic/diversity checks; returns a compact measurable account."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(p) for p in puzzles}) == 42
    shapes, sizes, keys, kinds, relations, endings, palettes = (set() for _ in range(7))
    rectangles = squares = fixed_turn_failures = stripe_failures = phase_failures = 0
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, m = _decode(pair["input"])
                h, w, key, s, q = m["h"], m["w"], m["key"], m["source"], m["turn"]
                patch = pair["output"][m["top"] + 1:m["top"] + h + 1,
                                       m["left"] + 1:m["left"] + w + 1]
                shapes.add(a.shape)
                sizes.add((h, w))
                keys.add(len(key))
                kinds.add(_key_kind(key))
                relations.add((s, m["destination"]))
                endings.add((h == w, min(h, w) % len(key)))
                palettes.add((m["bg"], m["sep"], m["marker"]))
                rectangles += h != w
                squares += h == w
                fixed_turn_failures += not np.array_equal(patch, _paint(h, w, key, s, 1))
                stripes = np.asarray(key)[np.arange(h) % len(key), None] * np.ones((1, w), dtype=int)
                stripe_failures += not np.array_equal(patch, stripes)
                # Wrong alternative: restart color phase whenever the corner returns.
                period = 2 if q == 2 else 4
                wrong = np.empty((h, w), dtype=np.int8)
                for i, cells in enumerate(_layers(h, w, s, q)):
                    for cell in cells:
                        wrong[cell] = key[(i % period) % len(key)]
                phase_failures += not np.array_equal(wrong, patch)
    assert keys == {2, 3, 4, 5, 6}
    assert kinds == {"distinct", "adjacent", "separated"}
    assert relations == set(_COMBOS)
    assert len(shapes) >= 60 and len(sizes) >= 50 and len(palettes) >= 30
    assert rectangles >= 100 and squares >= 10
    assert fixed_turn_failures >= 80 and stripe_failures == 210 and phase_failures >= 40
    assert len(endings) >= 8
    # Every legal count combination, including the maximum 14 distinct inputs.
    count_calls = 0
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain - 1) + 1):
            p = generate_puzzle(1000 + 10 * ntrain + ntest, ntrain, ntest)
            assert len(p["train"]) == ntrain and len(p["test"]) == ntest
            count_calls += 1
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (None, 1), (4, None), (True, 1), (4.0, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {train}/{test}")
    assert _puzzle_signature(generate_puzzle(827851175)) == _puzzle_signature(generate_puzzle(827851175))
    assert _puzzle_signature(generate_puzzle(23, 10, 4)) == _puzzle_signature(generate_puzzle(23, 10, 4))
    # Input-only solver is equivariant to every role-color permutation.
    permutation = np.array([7, 0, 9, 2, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        before = pair["input"].copy()
        assert np.array_equal(solve(permutation[before]), permutation[pair["output"]])
        assert np.array_equal(before, pair["input"])
    return dict(seeds=42, pairs=210, distinct_puzzles=42, grid_shapes=len(shapes),
                chamber_shapes=len(sizes), key_lengths=sorted(keys), key_kinds=sorted(kinds),
                corner_relations=len(relations), role_palettes=len(palettes),
                rectangles=rectangles, squares=squares, ending_forms=len(endings),
                fixed_clockwise_failures=fixed_turn_failures,
                straight_stripe_failures=stripe_failures, phase_reset_failures=phase_failures,
                legal_count_calls=count_calls, semantic_checks="generator-owned only")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
