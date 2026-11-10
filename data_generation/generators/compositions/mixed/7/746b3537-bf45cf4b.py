"""Mask-controlled stamping followed by full-vector adjacent-run collapse.

Standard library + NumPy only. No files, sampled parameters, or demonstrations
are consulted by solve. Generation retains no execution traces or per-seed logs.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("grids must be integer rectangles, dimensions 1..30, colors 0..9")
    return grid.astype(np.int8, copy=True)


def _key(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def _bbox(binary):
    r, c = np.nonzero(binary)
    if not len(r):
        raise ValueError("empty encoding")
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def _crop(grid, box):
    r0, r1, c0, c1 = box
    return grid[r0:r1, c0:c1].copy()


def _touches_sides(binary):
    return all(np.any(side) for side in
               (binary[0], binary[-1], binary[:, 0], binary[:, -1]))


def _read_input(value):
    """Unique color-role interpretation; disconnected mask cells stay grouped."""
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must occupy a strict majority")
    candidates = []
    for raw in colors:
        signal = int(raw)
        if signal == bg or len(set(map(int, colors)) - {bg, signal}) < 2:
            continue
        mb = _bbox(grid == signal)
        sb = _bbox((grid != bg) & (grid != signal))
        # Exclusive bounds: a strict inequality leaves at least one blank line.
        if not (mb[1] < sb[0] or sb[1] < mb[0]
                or mb[3] < sb[2] or sb[3] < mb[2]):
            continue
        mask_crop, stamp = _crop(grid, mb), _crop(grid, sb)
        if not np.all((mask_crop == bg) | (mask_crop == signal)):
            continue
        if np.any(stamp == signal):
            continue
        mask = mask_crop == signal
        if any(m * s > 30 for m, s in zip(mask.shape, stamp.shape)):
            continue
        candidates.append({"background": bg, "signal": signal, "mask": mask,
                           "stamp": stamp, "mask_bbox": mb, "stamp_bbox": sb})
    if len(candidates) != 1:
        raise ValueError("expected exactly one separated mask/stamp interpretation")
    return candidates[0]


def _expand(mask, stamp, background):
    """Construct the actual intermediate, including all inactive blocks."""
    sh, sw = stamp.shape
    result = np.full((mask.shape[0] * sh, mask.shape[1] * sw), background, dtype=np.int8)
    for r, c in np.argwhere(mask):
        result[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = stamp
    return result


def _run_indices(grid):
    # Equality is over entire vectors in the ORIGINAL expanded grid, not cells.
    rows = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    cols = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return np.flatnonzero(rows), np.flatnonzero(cols)


def _collapse(grid):
    rows, cols = _run_indices(grid)
    return grid[np.ix_(rows, cols)].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    expanded = _expand(info["mask"], info["stamp"], info["background"])
    return _collapse(expanded)


def _transforms(grid):
    return [transform(np.rot90(grid, k)) for k in range(4)
            for transform in (lambda x: x, np.fliplr)]


def _asymmetric(grid):
    return all(not np.array_equal(grid, other) for other in _transforms(grid)[1:])


def _global_unique(grid):
    # Deliberately WRONG alternative: keeps first occurrences, including runs
    # separated by other vectors. It is used only to test demonstration coverage.
    rows = sorted(np.unique(grid, axis=0, return_index=True)[1].tolist())
    cols = sorted(np.unique(grid, axis=1, return_index=True)[1].tolist())
    return grid[np.ix_(rows, cols)]


def _alternatives(info):
    """Finite counterfactuals, not an independent solver or a blind predictor."""
    m, s, bg = info["mask"], info["stamp"], info["background"]
    e = _expand(m, s, bg)
    rows, cols = _run_indices(e)
    filled = s.copy()
    filled[filled == bg] = int(s[s != bg][0])
    alternatives = {
        "no_A": e,
        "rows_only": e[rows],
        "columns_only": e[:, cols],
        "global_unique": _global_unique(e),
        "mask_first": _expand(_collapse(m), s, bg),
        "stamp_first": _expand(m, _collapse(s), bg),
        "both_first": _expand(_collapse(m), _collapse(s), bg),
        "no_B": _collapse(np.where(m, info["signal"], bg).astype(np.int8)),
        "fill_stamp_holes": _collapse(_expand(m, filled, bg)),
        "invert_mask": _collapse(_expand(~m, s, bg)),
    }
    for k in range(1, 8):
        alternatives[f"mask_transform_{k}"] = _collapse(_expand(_transforms(m)[k], s, bg))
        alternatives[f"stamp_transform_{k}"] = _collapse(_expand(m, _transforms(s)[k], bg))
    return alternatives


def _positive_parts(rng, total, count):
    result = np.ones(count, dtype=int)
    for _ in range(total - count):
        result[int(rng.integers(count))] += 1
    return result


def _mask(rng, shape, kind=None):
    h, w = shape
    r, c = np.indices(shape)
    kind = kind or str(rng.choice(["random", "path", "frame", "cross", "full", "islands", "repeat"]))
    if kind == "full":
        m = np.ones(shape, dtype=bool)
    elif kind == "frame":
        m = (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    elif kind == "cross":
        m = (r == h // 2) | (c == w // 2)
    elif kind == "islands":
        m = (r % 2 == 0) & (c % 2 == 0)
    elif kind == "path":
        m = np.zeros(shape, dtype=bool)
        rr = cc = 0
        m[0, 0] = True
        while rr < h - 1 or cc < w - 1:
            choices = [(dr, dc) for dr, dc in ((1, 0), (0, 1), (1, 1))
                       if rr + dr < h and cc + dc < w]
            dr, dc = choices[int(rng.integers(len(choices)))]
            rr, cc = rr + dr, cc + dc
            m[rr, cc] = True
    elif kind == "repeat":
        bh, bw = min(h, int(rng.integers(1, 4))), min(w, int(rng.integers(1, 4)))
        core = rng.random((bh, bw)) < float(rng.uniform(.3, .8))
        m = np.repeat(np.repeat(core, _positive_parts(rng, h, bh), axis=0),
                      _positive_parts(rng, w, bw), axis=1)
    else:
        m = rng.random(shape) < float(rng.uniform(.2, .9))
    # Tight bounds, including for disconnected masks. This does not require
    # every row or column to contain an active cell.
    if not _touches_sides(m):
        m[0, 0] = m[-1, -1] = True
    if rng.integers(2):
        m = m[::-1]
    if rng.integers(2):
        m = m[:, ::-1]
    return m.copy()


def _stamp(rng, shape, bg, payload):
    h, w = shape
    for _ in range(32):
        kind = str(rng.choice(["random", "bands", "plaid", "rings", "stretch", "seams"]))
        r, c = np.indices(shape)
        if kind == "bands":
            labels = r if rng.integers(2) else c
        elif kind == "plaid":
            labels = r + c
        elif kind == "rings":
            labels = np.minimum.reduce([r, c, h - 1 - r, w - 1 - c])
        elif kind == "stretch":
            bh, bw = int(rng.integers(1, min(h, 4) + 1)), int(rng.integers(1, min(w, 4) + 1))
            core = rng.integers(len(payload), size=(bh, bw))
            labels = np.repeat(np.repeat(core, _positive_parts(rng, h, bh), axis=0),
                               _positive_parts(rng, w, bw), axis=1)
        else:
            labels = rng.integers(len(payload), size=shape)
        s = np.asarray(payload, dtype=np.int8)[labels % len(payload)].copy()
        if rng.random() < .5:
            s[rng.random(shape) < float(rng.uniform(.05, .25))] = bg
        if kind == "seams":
            if h >= 3:
                s[-1] = s[0]
            if w >= 3:
                s[:, -1] = s[:, 0]
        if len(set(map(int, s.flat)) - {bg}) >= 2 and _touches_sides(s != bg):
            return s
    # Small/thin shapes can make a chosen motif constant. A deterministic
    # two-color fallback preserves readable tight bounds and the same rule.
    return np.asarray(payload, dtype=np.int8)[np.arange(h * w).reshape(shape) % 2]


def _internal_witness(rng, bg, payload):
    """Internal repeated rows/columns, asymmetric objects, and a visible hole."""
    shape = (2, 3) if rng.integers(2) else (3, 2)
    choices = []
    for bits in itertools.product((False, True), repeat=6):
        core = np.array(bits, dtype=bool).reshape(shape)
        if _touches_sides(core) and np.any(~core) and _asymmetric(core):
            choices.append(core)
    core = choices[int(rng.integers(len(choices)))]
    rw, cw = np.ones(shape[0], dtype=int), np.ones(shape[1], dtype=int)
    rw[int(rng.integers(len(rw)))] = 2
    cw[int(rng.integers(len(cw)))] = 2
    m = np.repeat(np.repeat(core, rw, axis=0), cw, axis=1)
    for _ in range(32):
        core_s = rng.choice(payload, size=(3, 3)).astype(np.int8)
        core_s[1, 1] = bg
        if (_asymmetric(core_s) and _collapse(core_s).shape == core_s.shape
                and len(set(map(int, core_s.flat)) - {bg}) >= 2):
            break
    else:
        a, b, c = map(int, payload[:3])
        core_s = np.array([[a, b, a], [c, bg, b], [a, a, c]], dtype=np.int8)
    rw, cw = np.ones(3, dtype=int), np.ones(3, dtype=int)
    rw[int(rng.integers(3))] = int(rng.integers(2, 4))
    cw[int(rng.integers(3))] = int(rng.integers(2, 4))
    s = np.repeat(np.repeat(core_s, rw, axis=0), cw, axis=1)
    return m, s


def _seam_witness(rng, bg, payload):
    """Both seam directions, blank bands, and separated vector recurrence."""
    def occupancy():
        # At least one adjacent active pair and an internal blank band.
        vec = np.array([1, 1, 0, 1], dtype=bool)
        if rng.integers(2):
            vec = np.insert(vec, 2, False)
        if rng.integers(2):
            vec = vec[::-1]
        return vec
    m = occupancy()[:, None] & occupancy()[None, :]
    h, w = ((3, 3), (3, 4), (4, 3))[int(rng.integers(3))]
    for _ in range(32):
        s = rng.choice(payload, size=(h, w)).astype(np.int8)
        s[1, 1] = bg
        s[-1] = s[0]
        s[:, -1] = s[:, 0]
        if len(set(map(int, s.flat)) - {bg}) >= 2 and _collapse(s).shape == s.shape:
            return m, s
    a, b, c = map(int, payload[:3])
    return m, np.array([[a, b, a], [c, bg, c], [a, b, a]], dtype=np.int8)


def _compose_input(rng, bg, signal, mask, stamp):
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    gap = int(rng.integers(1, 6))
    rim = int(rng.integers(2))
    mh, mw = mask.shape
    sh, sw = stamp.shape
    bh = (max(mh, sh) if horizontal else mh + sh + gap) + 2 * rim
    bw = (mw + sw + gap if horizontal else max(mw, sw)) + 2 * rim
    if bh > 30 or bw > 30:
        raise ValueError("source rectangles do not fit")
    h = bh + int(rng.integers(min(30 - bh, 7) + 1))
    w = bw + int(rng.integers(min(30 - bw, 7) + 1))
    fg = int(mask.sum() + np.count_nonzero(stamp != bg))
    while h * w <= 2 * fg:
        if h < 30 and (h <= w or w == 30):
            h += 1
        elif w < 30:
            w += 1
        else:
            raise ValueError("no strict-majority background layout")
    if horizontal:
        start = rim + int(rng.integers(w - bw + 1))
        mc, sc = ((start + sw + gap, start) if reverse else (start, start + mw + gap))
        mr = rim + int(rng.integers(h - mh - 2 * rim + 1))
        sr = rim + int(rng.integers(h - sh - 2 * rim + 1))
    else:
        start = rim + int(rng.integers(h - bh + 1))
        mr, sr = ((start + sh + gap, start) if reverse else (start, start + mh + gap))
        mc = rim + int(rng.integers(w - mw - 2 * rim + 1))
        sc = rim + int(rng.integers(w - sw - 2 * rim + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[mr:mr + mh, mc:mc + mw] = np.where(mask, signal, bg)
    grid[sr:sr + sh, sc:sc + sw] = stamp
    return grid


def _random_shapes(rng, shift=None):
    if shift is None:
        m = tuple(map(int, rng.integers(1, 6, size=2)))
        options = [(h, w) for h in range(1, 6) for w in range(1, 6) if h * w >= 2]
        return m, options[int(rng.integers(len(options)))]
    # The visibly larger encoding can be on either axis. Products remain <=30.
    axis = int(rng.integers(2))
    large = int(rng.integers(6, 10))
    big = [int(rng.integers(1, 6)), int(rng.integers(1, 6))]
    big[axis] = large
    small = [int(rng.integers(1, min(5, 30 // size) + 1)) for size in big]
    m, s = (big, small) if shift == "mask" else (small, big)
    if s[0] * s[1] == 1:
        s[1 - axis] = 2
    return tuple(m), tuple(s)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .25)
    palette = rng.permutation(10)
    shift = str(rng.choice(["mask", "stamp"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(32):
                colors = palette if stable else rng.permutation(10)
                bg, signal = int(colors[0]), int(colors[1])
                payload = colors[2:2 + int(rng.integers(3, 6))]
                if split == "train" and index == 0:
                    mask, stamp = _internal_witness(rng, bg, payload)
                elif split == "train" and index == 1:
                    mask, stamp = _seam_witness(rng, bg, payload)
                else:
                    mshape, sshape = _random_shapes(rng, shift if split == "test" else None)
                    mask = _mask(rng, mshape)
                    stamp = _stamp(rng, sshape, bg, payload)
                inp = _compose_input(rng, bg, signal, mask, stamp)
                key = _key(inp)
                if key in seen:
                    continue
                # Check the displayed crops recover precisely what was drawn.
                try:
                    info = _read_input(inp)
                except ValueError:
                    continue
                if not (np.array_equal(info["mask"], mask) and np.array_equal(info["stamp"], stamp)):
                    continue
                out = _collapse(_expand(mask, stamp, bg))
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("bounded attempts exhausted constructing a unique readable pair")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _read_input(pair["input"])
    m, s, bg = info["mask"], info["stamp"], info["background"]
    e = _expand(m, s, bg)
    rows, cols = _run_indices(e)
    def recurrence(a, axis):
        vectors = a if axis == 0 else a.T
        return len({row.tobytes() for row in vectors}) < len(vectors)
    compressed = e[np.ix_(rows, cols)]
    return {"mask_shape": tuple(m.shape), "stamp_shape": tuple(s.shape),
            "input_shape": tuple(np.asarray(pair["input"]).shape), "output_shape": tuple(compressed.shape),
            "expanded_shape": tuple(e.shape), "background": bg, "signal": info["signal"],
            "active": int(m.sum()), "holes": int(np.count_nonzero(s == bg)),
            "empty_mask_row": bool(np.any(~m.any(axis=1))),
            "empty_mask_col": bool(np.any(~m.any(axis=0))),
            "internal_row_collapse": bool(np.any(np.all(s[1:] == s[:-1], axis=1))),
            "internal_col_collapse": bool(np.any(np.all(s[:, 1:] == s[:, :-1], axis=0))),
            "row_seam": bool(np.array_equal(s[0], s[-1]) and np.any(np.all(m[1:] == m[:-1], axis=1) & m[1:].any(axis=1))),
            "col_seam": bool(np.array_equal(s[:, 0], s[:, -1]) and np.any(np.all(m[:, 1:] == m[:, :-1], axis=0) & m[:, 1:].any(axis=0))),
            "rows_removed": int(e.shape[0] - len(rows)), "cols_removed": int(e.shape[1] - len(cols)),
            "row_recurrence": recurrence(compressed, 0), "col_recurrence": recurrence(compressed, 1)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, records = set(), {}, {"train": [], "test": []}
    required_wrong = {"no_A", "no_B", "rows_only", "columns_only", "global_unique",
                      "mask_first", "stamp_first", "both_first", "fill_stamp_holes", "invert_mask"}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(inp)
            m, s, bg = info["mask"], info["stamp"], info["background"]
            e = _expand(m, s, bg)
            rows, cols = _run_indices(e)
            assert np.array_equal(out, e[np.ix_(rows, cols)]), "wrong run-compressed expansion"
            assert np.array_equal(out, solve(inp.copy()))
            assert _collapse(out).shape == out.shape, "adjacent equal vectors survived"
            assert set(map(int, out.flat)) == (set(map(int, s.flat)) | ({bg} if np.any(~m) else set()))
            assert info["signal"] not in out, "mask color leaked"
            # Full-vector equality: each removed vector equals its run's first
            # representative, even if a crop side or stamp seam was involved.
            row_runs = np.cumsum(np.r_[True, np.any(e[1:] != e[:-1], axis=1)]) - 1
            col_runs = np.cumsum(np.r_[True, np.any(e[:, 1:] != e[:, :-1], axis=0)]) - 1
            assert np.array_equal(e, out[np.ix_(row_runs, col_runs)])
            wrong = {name for name, value in _alternatives(info).items() if not np.array_equal(out, value)}
            evidence.setdefault(split, []).append(wrong)
            records[split].append(_features(pair))
    defeated = set().union(*evidence["train"])
    assert required_wrong <= defeated, "demonstrations miss a required counterexample"
    # A symmetry may remain training-compatible only when it also leaves every
    # test answer unchanged. This checks actual compressed outputs, not shape labels.
    for wrong in evidence["test"]:
        assert wrong <= defeated, "a test needs an undemonstrated distinction"
    train = records["train"]
    assert any(r["internal_row_collapse"] and r["internal_col_collapse"] for r in train)
    assert any(r["row_seam"] and r["col_seam"] for r in train)
    assert any(r["empty_mask_row"] and r["empty_mask_col"] for r in train)
    assert any(r["row_recurrence"] and r["col_recurrence"] for r in train)
    assert len({r["mask_shape"] for r in train}) >= 2
    assert len({r["stamp_shape"] for r in train}) >= 2
    # The supplied distribution has one larger encoding in tests. This is a
    # distribution check; solve itself does not inspect train/test membership.
    max_m = max(max(r["mask_shape"]) for r in train)
    max_s = max(max(r["stamp_shape"]) for r in train)
    assert all(max(r["mask_shape"]) > max_m or max(r["stamp_shape"]) > max_s for r in records["test"])


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def _normalized_key(grid):
    roles, values = {}, []
    for value in grid.flat:
        value = int(value)
        values.append(roles.setdefault(value, len(roles)))
    return tuple(grid.shape), bytes(values)


def self_test():
    """Streaming 42-seed diagnostics; no writes, no retained puzzle collection."""
    # Exact hand calculation includes stamp internals AND block seam collapse.
    s = np.array([[2, 2, 3, 2], [2, 2, 3, 2], [4, 4, 0, 4], [2, 2, 3, 2]], dtype=np.int8)
    e = _expand(np.ones((2, 2), dtype=bool), s, 0)
    expected = np.array([[2, 3, 2, 3, 2], [4, 0, 4, 0, 4], [2, 3, 2, 3, 2],
                         [4, 0, 4, 0, 4], [2, 3, 2, 3, 2]], dtype=np.int8)
    assert np.array_equal(_collapse(e), expected)
    # Local equal cells do not make whole vectors equal; recurrence survives.
    example = np.array([[2, 2, 3], [2, 4, 3], [2, 2, 3]], dtype=np.int8)
    assert np.array_equal(_collapse(example), example)
    assert np.array_equal(_collapse(e), _collapse(e.T).T)
    rows, cols = _run_indices(e)
    assert np.array_equal(_collapse(e), _collapse(e[rows]))
    assert np.array_equal(_collapse(e), _collapse(e[:, cols]))
    digests, shapes, masks, stamps, outputs, backgrounds = set(), set(), set(), set(), set(), set()
    flags = {name: 0 for name in ("row_seam", "col_seam", "internal_row_collapse", "internal_col_collapse",
                                 "empty_mask_row", "empty_mask_col", "row_recurrence", "col_recurrence")}
    pairs = rows_removed = cols_removed = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        assert digest == _puzzle_digest(generate_puzzle(seed)), "nondeterminism"
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                feature = _features(pair)
                shapes.add(feature["input_shape"])
                masks.add(_key(info["mask"]))
                stamps.add(_normalized_key(info["stamp"]))
                outputs.add(feature["output_shape"])
                backgrounds.add(info["background"])
                rows_removed += feature["rows_removed"]
                cols_removed += feature["cols_removed"]
                pairs += 1
                for name in flags:
                    flags[name] += int(feature[name])
        # Literal color roles are read from the input, including color zero.
        recoloring = np.random.default_rng(seed + 1000).permutation(10)
        for pair in puzzle["test"]:
            assert np.array_equal(solve(recoloring[pair["input"]]), recoloring[pair["output"]])
        # Corrupt one output and require rejection; no duplicate saved payload.
        old = int(puzzle["test"][0]["output"][0, 0])
        puzzle["test"][0]["output"][0, 0] = (old + 1) % 10
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong output accepted")
    assert len(digests) == 42
    assert len(shapes) >= 50 and len(masks) >= 70 and len(stamps) >= 100 and len(outputs) >= 40
    assert backgrounds == set(range(10))
    assert all(value >= 42 for value in flags.values())
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(73, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
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
    return {"valid": True, "seeds": 42, "pairs": pairs, "distinct_puzzles": len(digests),
            "input_shapes": len(shapes), "mask_structures": len(masks),
            "normalized_stamp_structures": len(stamps), "output_shapes": len(outputs),
            "background_colors": sorted(backgrounds), "rows_removed": rows_removed,
            "cols_removed": cols_removed, "feature_counts": flags,
            "independent_solver": False, "blind_solve": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
