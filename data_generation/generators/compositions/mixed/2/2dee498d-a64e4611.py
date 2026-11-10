"""Stable column compaction followed by all-window negative-space highlighting.

Background is 0, highlight is 3, and every other input color is an obstacle.
Only NumPy and the standard library are used. No files or latent state are read.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


FOREGROUND_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(n < 1 or n > 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _keep(a):
    seen, indices = set(), []
    for j in range(a.shape[1]):
        key = tuple(int(v) for v in a[:, j])
        if key not in seen:
            seen.add(key)
            indices.append(j)
    return indices


def _compact(a):
    return a[:, _keep(a)].copy()


def _empty(a):
    """Outside-grid positions impose no condition: the window is clipped."""
    h, w = a.shape
    padded = np.pad(a != 0, 1, constant_values=False)
    near = np.zeros((h, w), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            near |= padded[dy:dy + h, dx:dx + w]
    return ~near


def _highlight(a):
    result = a.copy()
    result[_empty(a)] = 3
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Use only visible, original full-height column vectors, then their new space."""
    a = _grid(input_grid)
    if np.any(a == 3):
        raise ValueError("3 is reserved for output highlights")
    return _highlight(_compact(a))


def _reference(a):
    """Direct definition for validation; no generation variables or solve call."""
    columns = []
    for j in range(a.shape[1]):
        column = [int(a[r, j]) for r in range(a.shape[0])]
        if column not in columns:
            columns.append(column)
    base = np.array(columns, dtype=np.int8).T
    out = base.copy()
    h, w = base.shape
    for r in range(h):
        for c in range(w):
            if all(base[y, x] == 0
                   for y in range(max(0, r - 1), min(h, r + 2))
                   for x in range(max(0, c - 1), min(w, c + 2))):
                out[r, c] = 3
    return base, out


def _components(mask):
    h, w = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    sizes = []
    while remaining:
        start = remaining.pop()
        todo, size = [start], 0
        while todo:
            r, c = todo.pop()
            size += 1
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        sizes.append(size)
    return sizes


def _facts(inp, base, out):
    mask = out == 3
    h, k = base.shape
    orth = base != 0
    orth = orth.copy()
    orth[1:] |= base[:-1] != 0
    orth[:-1] |= base[1:] != 0
    orth[:, 1:] |= base[:, :-1] != 0
    orth[:, :-1] |= base[:, 1:] != 0
    keys = [tuple(int(v) for v in inp[:, j]) for j in range(inp.shape[1])]
    ids = list(dict.fromkeys(keys))
    arrivals = [keys.index(key) for key in ids]
    run_indices = [0] + [j for j in range(1, len(keys)) if keys[j] != keys[j - 1]]
    last_indices = [j for j, key in enumerate(keys) if key not in keys[j + 1:]]
    sorted_base = np.array(sorted(ids), dtype=np.int8).T
    return {
        "crop_fails": not np.array_equal(_highlight(inp[:, :k]), out),
        "run_fails": not np.array_equal(_highlight(inp[:, run_indices]), out),
        "last_fails": not np.array_equal(_highlight(inp[:, last_indices]), out),
        "sort_fails": not np.array_equal(_highlight(sorted_base), out),
        "order_fails": not np.array_equal(_compact(_highlight(inp)), out),
        "original_space_fails": not np.array_equal(_highlight(inp)[:, _keep(inp)], out),
        "early_repeat": arrivals[-1] >= k,
        "singleton": 1 in _components(mask),
        "multiple_components": len(_components(mask)) >= 2,
        "corner": bool(mask[0, 0] or mask[0, -1] or mask[-1, 0] or mask[-1, -1]),
        "diagonal": bool(np.any(~orth & ~mask)),
        "duplicate_rows": len({tuple(row) for row in base.tolist()}) < h,
        "width": k,
        "multiplicities": [keys.count(key) for key in ids],
    }


def _geometry(rng, h, k, mode):
    """Irregular obstacle fields with several types of locally empty geometry."""
    fg = rng.random((h, k)) < float(rng.uniform(.22, .48))

    def rectangle():
        ph = int(rng.integers(3, min(8, h) + 1))
        pw = int(rng.integers(3, min(7, k) + 1))
        r = int(rng.integers(0, h - ph + 1))
        c = int(rng.integers(0, k - pw + 1))
        fg[r:r + ph, c:c + pw] = False

    if mode == 0:
        rectangle()
    elif mode == 1:
        top = int(rng.integers(2, h - 4))
        breadth = int(rng.integers(3, min(6, h - top) + 1))
        for c in range(k):
            if rng.random() < .25:
                top = int(np.clip(top + int(rng.choice([-1, 1])), 1, h - breadth - 1))
            fg[top:top + breadth, c] = False
    elif mode == 2:
        left = int(rng.integers(0, k - 2))
        top, bottom = sorted(rng.choice(h, 2, replace=False).tolist())
        bottom = max(bottom, min(h, top + 5))
        for r in range(top, bottom):
            if rng.random() < .2:
                left = int(np.clip(left + int(rng.choice([-1, 1])), 0, k - 3))
            fg[r, left:left + 3] = False
    elif mode == 3:
        r, c = int(rng.integers(1, h - 3)), int(rng.integers(1, k - 3))
        fg[r:r + 3, :] = False
        fg[max(0, r - 4):min(h, r + 6), c:c + 3] = False
    elif mode == 4:
        for _ in range(int(rng.integers(2, 5))):
            rectangle()
    else:
        r, c = int(rng.integers(1, h - 2)), int(rng.integers(1, k - 2))
        for _ in range(int(rng.integers(4, 10))):
            fg[r - 1:r + 2, c - 1:c + 2] = False
            r = int(np.clip(r + int(rng.choice([-1, 0, 1])), 1, h - 2))
            c = int(np.clip(c + int(rng.choice([-1, 0, 1])), 1, k - 2))
    return fg


def _canonical(rng, h, k, mode, diagnostic, palette):
    fg = _geometry(rng, h, k, mode)
    if diagnostic is not None:
        # A bounded pocket whose only eligible cell is its center.
        r = int(rng.integers(5, h - 6))
        c = int(rng.integers(2, k - 2))
        fg[r - 2:r + 3, c - 2:c + 3] = True
        fg[r - 1:r + 2, c - 1:c + 2] = False
        # A clipped, isolated corner candidate; not a padded full 3x3 window.
        fg[:3, :3] = True
        fg[:2, :2] = False
        # A zero center with clear orthogonal neighbors but one diagonal obstacle.
        fg[:3, -3:] = False
        fg[0, -3] = True
        # A larger region, separated from the singleton by its obstacle rim.
        breadth = k if diagnostic == 0 else int(rng.integers(4, k + 1))
        fg[-4:, -breadth:] = False

    values = rng.choice(palette, (h, k)).astype(np.int8)
    base = np.where(fg, values, 0).astype(np.int8)
    # Flips reposition the diagnostic geometry without changing the column rule.
    if rng.random() < .5:
        base = base[::-1].copy()
    if rng.random() < .5:
        base = base[:, ::-1].copy()
    return base


def _indices(rng, k, layout):
    counts = np.full(k, 2, dtype=int)
    total = int(rng.integers(2 * k + 1, min(30, 4 * k) + 1))
    for _ in range(total - 2 * k):
        choices = np.flatnonzero(counts < 5)
        counts[int(rng.choice(choices))] += 1
    remaining = counts.copy()
    order = []
    for j in range(k):
        order.append(j)
        remaining[j] -= 1
        if layout == "woven" and j < k - 1 and (j == 1 or rng.random() < .65):
            choices = np.flatnonzero(remaining[:j + 1])
            chosen = int(rng.choice(choices))
            order.append(chosen)
            remaining[chosen] -= 1
    if layout in ("adjacent", "separated"):
        j = int(rng.integers(0, k - 2))
        position = j + 1 if layout == "adjacent" else int(rng.integers(j + 2, k))
        order.insert(position, j)
        remaining[j] -= 1
    tail = [j for j in range(k) for _ in range(int(remaining[j]))]
    if layout == "blocks":
        tail = []
        for wave in range(int(remaining.max())):
            ids = list(range(k))
            if rng.random() < .5:
                ids.reverse()
            tail.extend(j for j in ids if remaining[j] > wave)
    elif rng.random() < .7 or layout == "woven":
        rng.shuffle(tail)
    elif rng.random() < .5:
        tail.reverse()
    return order + tail


def _make_pair(rng, k, mode, diagnostic, layout, seen):
    for _ in range(160):
        h = int(rng.integers(16, 25)) if diagnostic is not None else int(rng.integers(10, 31))
        size = int(rng.integers(1, 5))
        palette = rng.choice(FOREGROUND_COLORS, size, replace=False)
        base = _canonical(rng, h, k, mode, diagnostic, palette)
        if len(_keep(base)) != k:
            continue
        inp = base[:, _indices(rng, k, layout)]
        key = (inp.shape, inp.tobytes())
        if key in seen:
            continue
        out = _highlight(base)
        if not (np.any(out == 3) and np.any(out == 0) and np.any(base != 0)):
            continue
        if diagnostic is not None:
            f = _facts(inp, base, out)
            needed = ("crop_fails", "run_fails", "last_fails", "sort_fails",
                      "order_fails", "original_space_fails", "early_repeat",
                      "singleton", "multiple_components", "corner", "diagonal")
            if not all(f[q] for q in needed):
                continue
        seen.add(key)
        return {"input": inp.copy(), "output": out}
    raise RuntimeError("bounded sampling failed to create a distinct diagnostic pair")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    ntrain = _integer(num_train, "num_train", 2, 10)
    ntest = _integer(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    modes = rng.permutation(6).tolist()
    puzzle, seen = {"train": [], "test": []}, set()
    for i in range(ntrain):
        diagnostic = i if i < 2 else None
        k = int(rng.integers(8, 11)) if i < 2 else int(rng.integers(6, 11))
        layout = ("adjacent" if i == 0 else "separated" if i == 1 else
                  str(rng.choice(("blocks", "adjacent", "separated", "woven"))))
        puzzle["train"].append(_make_pair(rng, k, modes[i % 6], diagnostic, layout, seen))
    for i in range(ntest):
        k = int(rng.integers(11, 15))
        puzzle["test"].append(_make_pair(rng, k, modes[(i + 1) % 6], None, "woven", seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate semantics from visible inputs and require diagnostic training evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen, training, widths = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert not np.any(inp == 3) and np.any(inp == 0) and np.any(inp != 0)
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "repeated input"
            seen.add(key)
            base, expected = _reference(inp)
            assert np.array_equal(out, expected), "label violates the stated rule"
            assert out.shape[0] == inp.shape[0] and out.shape[1] < inp.shape[1]
            assert np.any(out == 3) and np.any(out == 0)
            assert np.array_equal(out[base != 0], base[base != 0]), "obstacle changed"
            assert set(np.unique(out)) == set(np.unique(inp)) | {3}
            facts = _facts(inp, base, out)
            assert min(facts["multiplicities"]) >= 2
            widths[split].append(base.shape[1])
            if split == "train":
                training.append(facts)
    for q in ("crop_fails", "run_fails", "last_fails", "sort_fails", "order_fails",
              "original_space_fails", "early_repeat", "singleton", "multiple_components",
              "corner", "diagonal", "duplicate_rows"):
        assert any(f[q] for f in training), f"training lacks a {q} witness"
    assert max(widths["train"]) < min(widths["test"]), "missing width extrapolation"


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds and legal counts; save no puzzles, traces, or logs."""
    # Hand-worked edge, diagonal, and full-vector-equality cases.
    for inp, expected in (
        ([[0]], [[3]]),
        ([[2]], [[2]]),
        ([[0, 0], [0, 0]], [[3], [3]]),
        ([[0, 0, 2], [0, 0, 0], [4, 0, 5]], [[3, 0, 2], [0, 0, 0], [4, 0, 5]]),
        ([[0, 0, 0, 2, 0], [1, 1, 4, 0, 1]], [[0, 0, 2], [1, 4, 0]]),
    ):
        assert np.array_equal(solve(np.array(inp)), expected)
    signatures, shapes, structures, palettes, multiplicities, widths = set(), set(), set(), set(), set(), set()
    diagnostic_counts = {q: 0 for q in ("order_fails", "singleton", "corner", "diagonal", "multiple_components")}
    pairs, full_height, rectangles = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed)), "nondeterministic generation"
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                original = inp.copy()
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(inp, original), "solve mutated its input"
                base, _ = _reference(inp)
                f = _facts(inp, base, out)
                pairs += 1
                full_height += inp.shape[0] == 30
                rectangles += out.shape[0] != out.shape[1]
                shapes.add((inp.shape, out.shape))
                widths.add(out.shape[1])
                palettes.add(tuple(int(c) for c in np.unique(inp)))
                multiplicities.update(f["multiplicities"])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(norm).hexdigest()))
                for q in diagnostic_counts:
                    diagnostic_counts[q] += int(f[q])
    assert len(signatures) == 42 and len(shapes) >= 80 and len(structures) >= 190
    assert len(palettes) >= 50 and {2, 3, 4, 5} <= multiplicities
    assert {6, 7, 8, 9, 10, 11, 12, 13, 14} <= widths
    assert full_height and rectangles >= 150
    assert all(n >= 84 for n in diagnostic_counts.values())
    legal_calls = 0
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            puzzle = generate_puzzle(10000 + tr * 10 + te, tr, te)
            assert len(puzzle["train"]) == tr and len(puzzle["test"]) == te
            legal_calls += 1
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 1.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shape_combinations": len(shapes), "normalized_inputs": len(structures),
            "palettes": len(palettes), "multiplicities": sorted(multiplicities),
            "compact_widths": sorted(widths), "full_height_pairs": full_height,
            "rectangular_outputs": rectangles, "diagnostic_pairs": diagnostic_counts,
            "legal_count_combinations": legal_calls}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
