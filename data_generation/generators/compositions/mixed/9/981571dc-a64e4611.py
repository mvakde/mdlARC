"""Repair two reflection folds, then mark every clear clipped 3x3 center.

Special convention: 3 is the highlight and is absent from inputs. Background
and erasure literals vary; each input uniquely identifies its repair and its
strict-majority repaired background. No parent module or saved data is loaded.
"""
from functools import lru_cache
from hashlib import sha256

import numpy as np


HIGHLIGHT = 3


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


@lru_cache(maxsize=32)
def _axis_pairs(length):
    pairs = []
    for total in range(1, 2 * length - 2):
        first = np.arange(max(0, total - length + 1), (total + 1) // 2)
        pairs.append((total, first, total - first))
    return tuple(pairs)


def _axes(a, erase, dimension):
    field = a if dimension == 0 else a.T
    result = []
    for total, first, second in _axis_pairs(field.shape[0]):
        x, y = field[first], field[second]
        visible = (x != erase) & (y != erase)
        if np.any(visible) and np.all(x[visible] == y[visible]):
            result.append(total)
    return result


@lru_cache(maxsize=128)
def _orbits(h, w, row_sum, col_sum):
    seen, groups = set(), []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = row_sum - r, col_sum - c
            rows = {r, rr} if 0 <= rr < h else {r}
            cols = {c, cc} if 0 <= cc < w else {c}
            group = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(group)
            groups.append(group)
    return tuple(groups)


def _repair_at(a, erase, row_sum, col_sum):
    repaired = a.copy()
    for group in _orbits(*a.shape, row_sum, col_sum):
        values = {int(a[p]) for p in group if a[p] != erase}
        if len(values) != 1:
            raise ValueError("reflection orbit has no unanimous surviving value")
        color = next(iter(values))
        for p in group:
            if a[p] == erase:
                repaired[p] = color
    return repaired


def _infer(a):
    """Exhaust all nontrivial in-bounds folds and every possible erasure color."""
    if np.any(a == HIGHLIGHT):
        raise ValueError("highlight color 3 is reserved and absent from inputs")
    answers = []
    for erase in map(int, np.unique(a)):
        row_axes = _axes(a, erase, 0)
        col_axes = _axes(a, erase, 1)
        for rs in row_axes:
            for cs in col_axes:
                try:
                    repaired = _repair_at(a, erase, rs, cs)
                except ValueError:
                    continue
                colors, counts = np.unique(repaired, return_counts=True)
                if len(colors) < 3 or counts.max() * 2 <= a.size:
                    continue
                bg = int(colors[np.argmax(counts)])
                answers.append((erase, bg, rs, cs, repaired))
                if len(answers) > 1:
                    raise ValueError("ambiguous erasure role or reflection folds")
    if len(answers) != 1:
        raise ValueError("input has no unique recoverable two-fold interpretation")
    return answers[0]


def _centers(a, bg):
    """The union of qualifying centers, never the union of their footprints."""
    h, w = a.shape
    padded = np.pad(a != bg, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + h, dc:dc + w]
    return ~near


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    _, bg, _, _, repaired = _infer(a)
    out = repaired.copy()
    out[_centers(repaired, bg)] = HIGHLIGHT
    return out


def _components(mask):
    todo = set(zip(*np.nonzero(mask)))
    result = []
    while todo:
        p = min(todo)
        todo.remove(p)
        stack, group = [p], [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    stack.append(q)
                    group.append(q)
        result.append(group)
    return result


def _set_orbits(field, mask, groups, color, rng=None, palette=None):
    for group in groups:
        if any(mask[p] for p in group):
            value = color if palette is None else int(rng.choice(palette))
            for p in group:
                field[p] = value


def _draw_field(rng, shape, sums, bg, palette):
    h, w = shape
    groups = _orbits(h, w, *sums)
    field = np.full(shape, bg, dtype=np.int8)
    density = float(rng.uniform(.22, .40))
    style = int(rng.integers(3))
    raw = rng.random(shape) < density
    if style == 1:  # Short irregular strokes, with no parity lattice.
        for _ in range(int(rng.integers(3, 9))):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            length = int(rng.integers(2, 5))
            if rng.random() < .5:
                raw[r:r + length, c] = True
            else:
                raw[r, c:c + length] = True
    elif style == 2:  # Correlated fragments mixed with fine texture.
        coarse = rng.random(((h + 1) // 2, (w + 1) // 2)) < density
        raw = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)[:h, :w]
        raw ^= rng.random(shape) < .09
    for group in groups:
        if raw[group[0]]:
            color = int(rng.choice(palette))
            for p in group:
                field[p] = color

    # Reflect open geometric fragments. These are not separate puzzle panels.
    rr, cc = np.indices(shape)
    for _ in range(int(rng.integers(1, 4))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        rh, cw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        kind = int(rng.integers(4))
        if kind == 0:
            opening = (abs(rr - r) <= rh // 2) & (abs(cc - c) <= cw // 2)
        elif kind == 1:
            opening = abs(rr - r) + abs(cc - c) <= max(rh, cw)
        elif kind == 2:
            opening = ((rr >= r) & (rr < r + rh + 2)
                       & (cc >= c) & (cc < c + cw + (rr - r) // 2))
        else:
            opening = ((abs(rr - r) <= 1) & (abs(cc - c) <= cw))
            opening |= ((abs(cc - c) <= 1) & (abs(rr - r) <= rh))
        _set_orbits(field, opening, groups, bg)

    # A visibly bounded 3x3 pocket has exactly one qualifying center. Its
    # reflected copies distinguish ALL centers from a size cutoff or one fit.
    eligible = [g for g in groups if len(g) == 4 and all(
        2 <= y < h - 2 and 2 <= x < w - 2
        and abs(2 * y - sums[0]) >= 6 and abs(2 * x - sums[1]) >= 6
        for y, x in g)]
    if not eligible:
        return None
    r, c = eligible[int(rng.integers(len(eligible)))][0]
    ring = (abs(rr - r) <= 2) & (abs(cc - c) <= 2)
    pocket = (abs(rr - r) <= 1) & (abs(cc - c) <= 1)
    _set_orbits(field, ring, groups, None, rng, palette)
    _set_orbits(field, pocket, groups, bg)

    # A short boundary opening explicitly exposes clipped-window behavior.
    edge = np.zeros(shape, dtype=bool)
    side = int(rng.integers(4))
    if side < 2:
        c = int(rng.integers(1, w - 3))
        edge[:2 if side == 0 else h, c:c + 3] = True
        if side == 1:
            edge[:-2] = False
    else:
        r = int(rng.integers(1, h - 3))
        edge[r:r + 3, :2 if side == 2 else w] = True
        if side == 3:
            edge[:, :-2] = False
    _set_orbits(field, edge, groups, bg)
    # Keep every selected payload literal present without changing diagnostics.
    visible_groups = [g for g in groups if field[g[0]] != bg]
    if len(visible_groups) < len(palette):
        return None
    for color, i in zip(palette, rng.permutation(len(visible_groups))):
        for p in visible_groups[int(i)]:
            field[p] = color
    return field


def _mask(rng, clean, bg, groups, hard):
    h, w = clean.shape
    mask = np.zeros(clean.shape, dtype=bool)
    kind = int(rng.integers(5))
    if kind in (0, 1, 4):
        for _ in range(2 if kind == 4 else 1):
            r, c = int(rng.integers(h)), int(rng.integers(w))
            rh, cw = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            if kind == 1:
                rh, cw = (1, int(rng.integers(5, 10))) if rng.random() < .5 else (
                    int(rng.integers(5, 10)), 1)
            mask[r:r + rh, c:c + cw] = True
    if kind == 2:
        r, c = int(rng.integers(h)), int(rng.integers(w))
        for _ in range(int(rng.integers(10, 25))):
            mask[r, c] = True
            dr, dc = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            r, c = int(np.clip(r + dr, 0, h - 1)), int(np.clip(c + dc, 0, w - 1))
    if kind in (3, 4):
        mask |= rng.random(clean.shape) < (.08 if hard else .035)
    for g in groups:
        if all(mask[p] for p in g):
            mask[g[int(rng.integers(len(g)))]] = False

    fg_four = [g for g in groups if len(g) == 4 and clean[g[0]] != bg]
    if len(fg_four) < 5:
        return None
    # Training has one or two deliberately triple-erased foreground orbits;
    # tests combine three to six. Every group retains its own visible witness.
    chosen = rng.permutation(len(fg_four))[:int(rng.integers(3, 7) if hard else rng.integers(1, 3))]
    for i in chosen:
        g = fg_four[int(i)]
        survivor = g[int(rng.integers(4))]
        for p in g:
            mask[p] = p != survivor
    # At least two restored payload colors make the erasure-role evidence rich.
    for color in np.unique(clean):
        if color == bg:
            continue
        options = [g for g in groups if len(g) > 1 and clean[g[0]] == color]
        if not options:
            return None
        g = options[int(rng.integers(len(options)))]
        if not any(mask[p] for p in g):
            mask[g[int(rng.integers(len(g)))]] = True
    centers = _centers(clean, bg)
    options = [g for g in groups if len(g) > 1 and any(centers[p] for p in g)]
    if not options:
        return None
    g = options[int(rng.integers(len(options)))]
    if not any(mask[p] for p in g):
        mask[g[int(rng.integers(len(g)))]] = True
    return mask


def _diagnostics(inp, clean, erase, bg, sums):
    mask = inp == erase
    marked = _centers(clean, bg)
    sizes = sorted(len(g) for g in _components(marked))
    as_bg = np.where(mask, bg, inp)
    false_empty = int(np.count_nonzero(_centers(as_bg, bg) & ~marked))
    blocked_empty = int(np.count_nonzero(marked & ~_centers(inp, bg)))
    groups = _orbits(*inp.shape, *sums)
    triples = sum(len(g) == 4 and sum(bool(mask[p]) for p in g) == 3 for g in groups)
    diagonal = 0
    h, w = clean.shape
    for r, c in zip(*np.nonzero(clean == bg)):
        local = clean[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)]
        cross = [clean[y, x] for y, x in ((r, c), (r - 1, c), (r + 1, c),
                                         (r, c - 1), (r, c + 1))
                 if 0 <= y < h and 0 <= x < w]
        diagonal += all(v == bg for v in cross) and np.any(local != bg)
    return {"components": len(sizes), "smallest": min(sizes, default=0),
            "largest": max(sizes, default=0), "highlighted": int(marked.sum()),
            "false_empty_without_repair": false_empty,
            "blocked_empty_without_repair": blocked_empty,
            "triple_erased_orbits": triples, "diagonal_blocked": int(diagonal),
            "erased": int(mask.sum()), "mask_components": len(_components(mask)),
            "edge_highlights": int(marked[0].sum() + marked[-1].sum()
                                   + marked[:, 0].sum() + marked[:, -1].sum())}


def _make_pair(rng, bg, erase, hard):
    for _ in range(160):
        low, high = (24, 31) if hard else (16, 25)
        shape = tuple(map(int, rng.integers(low, high, size=2)))
        offsets = [0 if rng.random() < .30 else int(rng.integers(-3, 4)) for _ in range(2)]
        sums = tuple(n - 1 + d for n, d in zip(shape, offsets))
        pool = [c for c in range(10) if c not in (bg, erase, HIGHLIGHT)]
        palette = list(map(int, rng.choice(pool, size=int(rng.integers(2, 6)), replace=False)))
        clean = _draw_field(rng, shape, sums, bg, palette)
        if clean is None or np.count_nonzero(clean == bg) * 2 <= clean.size:
            continue
        groups = _orbits(*shape, *sums)
        mask = _mask(rng, clean, bg, groups, hard)
        if mask is None:
            continue
        inp = clean.copy()
        inp[mask] = erase
        f = _diagnostics(inp, clean, erase, bg, sums)
        if (f["components"] < 2 or f["smallest"] != 1 or f["largest"] < 3
                or not f["false_empty_without_repair"] or not f["blocked_empty_without_repair"]
                or not f["diagonal_blocked"] or not f["edge_highlights"]):
            continue
        try:
            e, b, rs, cs, recovered = _infer(inp)
        except ValueError:
            continue
        if (e, b, rs, cs) != (erase, bg, *sums) or not np.array_equal(recovered, clean):
            continue
        out = clean.copy()
        out[_centers(clean, bg)] = HIGHLIGHT
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction exhausted; no ambiguous fallback is returned")


def _integer(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer("num_train", num_train, 2, 10)
    ns = _integer("num_test", num_test, 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg, erase = map(int, rng.choice([c for c in range(10) if c != HIGHLIGHT], size=2, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for _ in range(count):
            for _ in range(12):
                pair = _make_pair(rng, bg, erase, split == "test")
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, roles = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            erase, bg, rs, cs, clean = _infer(inp)
            roles.add((bg, erase))
            assert erase != bg and HIGHLIGHT not in (erase, bg)
            mask = inp == erase
            assert np.any(mask & (clean == bg))
            assert len(np.unique(clean[mask & (clean != bg)])) >= 2
            assert np.array_equal(clean[~mask], inp[~mask])
            # Check the output cell by cell, separately from vectorized centers.
            expected = clean.copy()
            h, w = clean.shape
            for r in range(h):
                for c in range(w):
                    if np.all(clean[max(0, r - 1):min(h, r + 2),
                                    max(0, c - 1):min(w, c + 2)] == bg):
                        expected[r, c] = HIGHLIGHT
            assert np.array_equal(out, expected), "wrong repair or empty-center highlight"
            assert not np.any(out == erase)
            assert np.array_equal(out[clean != bg], clean[clean != bg])
            f = _diagnostics(inp, clean, erase, bg, (rs, cs))
            assert f["components"] >= 2 and f["smallest"] == 1 and f["largest"] >= 3
            assert f["false_empty_without_repair"] and f["blocked_empty_without_repair"]
            assert f["triple_erased_orbits"] >= (3 if split == "test" else 1)
            assert f["diagonal_blocked"] and f["edge_highlights"]
    assert len(roles) == 1, "background and erasure roles must be puzzle-wide"


def _signature(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds; no payloads, logs or execution traces are written."""
    shapes, signs, parities, backgrounds, erasures, payload_sizes = (set() for _ in range(6))
    signatures, normalized, mask_counts = set(), set(), set()
    totals = {"pairs": 0, "false_empty_without_repair": 0,
              "blocked_empty_without_repair": 0, "triple_erased_orbits": 0,
              "pairs_with_singleton_highlight": 0, "edge_highlights": 0}
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                e, b, rs, cs, clean = _infer(inp)
                assert np.array_equal(solve(inp.tolist()), out)
                # An input-only spatial equivariance check has no training state.
                assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
                shapes.add(inp.shape)
                signs.add((int(np.sign(rs - inp.shape[0] + 1)), int(np.sign(cs - inp.shape[1] + 1))))
                parities.add((rs % 2, cs % 2))
                backgrounds.add(b)
                erasures.add(e)
                payload_sizes.add(len(np.unique(clean)) - 1)
                remap = {}
                normalized.add((inp.shape, bytes(remap.setdefault(int(v), len(remap)) for v in inp.flat)))
                f = _diagnostics(inp, clean, e, b, (rs, cs))
                mask_counts.add(f["mask_components"])
                totals["pairs"] += 1
                totals["pairs_with_singleton_highlight"] += 1
                for k in ("false_empty_without_repair", "blocked_empty_without_repair",
                          "triple_erased_orbits", "edge_highlights"):
                    totals[k] += f[k]
    assert len(signatures) == 42 and len(normalized) == 210
    assert len(shapes) >= 40 and len(signs) == 9 and len(parities) == 4
    assert len(backgrounds) >= 7 and len(erasures) >= 7 and payload_sizes == {2, 3, 4, 5}
    assert len(mask_counts) >= 8
    p = generate_puzzle(0)
    assert _signature(p) == _signature(generate_puzzle(0))
    for nt, ns in ((2, 1), (10, 1), (10, 4), (6, 3)):
        validate_generated_puzzle(generate_puzzle(1700 + nt, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in q.items()} for q in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "distinct_shapes": len(shapes), "normalized_inputs": len(normalized),
            "background_colors": sorted(backgrounds), "erasure_colors": sorted(erasures),
            "fold_sign_combinations": len(signs), "fold_parity_combinations": len(parities),
            "payload_color_counts": sorted(payload_sizes), "totals": totals}
