"""Damaged mirror quilt: symmetry of occupancy, unchanged local substitution.

The only solving evidence is the supplied input. No parent modules are imported.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np


_MODELS = ("H", "V", "HV", "D", "AD", "DA", "ALL")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=512)
def _orbits(shape, model):
    """Whole-coarse-board reflection groups, never pixel reflections of a tile."""
    h, w = shape
    _require(model in _MODELS and (h == w or model in ("H", "V", "HV")),
             "Diagonal reflections require a square coarse board")
    result = {}
    for r in range(h):
        for c in range(w):
            if model == "H":
                cells = {(r, c), (h - 1 - r, c)}
            elif model == "V":
                cells = {(r, c), (r, w - 1 - c)}
            elif model == "HV":
                cells = {(r, c), (h - 1 - r, c), (r, w - 1 - c),
                         (h - 1 - r, w - 1 - c)}
            elif model == "D":
                cells = {(r, c), (c, r)}
            elif model == "AD":
                cells = {(r, c), (h - 1 - c, h - 1 - r)}
            elif model == "DA":
                cells = {(r, c), (c, r), (h - 1 - c, h - 1 - r),
                         (h - 1 - r, h - 1 - c)}
            else:
                cells = {(r, c), (c, r), (h - 1 - r, c), (r, h - 1 - c),
                         (h - 1 - r, h - 1 - c), (h - 1 - c, r),
                         (c, h - 1 - r), (h - 1 - c, h - 1 - r)}
            canonical = min(cells)
            result.setdefault(canonical, set()).update(x * w + y for x, y in cells)
    return tuple(tuple(sorted(v)) for _, v in sorted(result.items()))


def _infer_occupancy(observed):
    """All sufficiently witnessed reflection models must agree on the query.

An orbit with k visible equal labels supplies k*(k-1)/2 unordered witnesses.
At least two occupied and two empty witness pairs are required per model.
"""
    models = _MODELS if observed.shape[0] == observed.shape[1] else _MODELS[:3]
    flat = observed.ravel()
    predictions, supported = [], []
    for model in models:
        prediction = np.full(flat.shape, -1, dtype=np.int8)
        witnesses = [0, 0]
        valid = True
        for orbit in _orbits(observed.shape, model):
            indices = np.asarray(orbit)
            visible = flat[indices]
            visible = visible[visible >= 0]
            if visible.size and np.any(visible != visible[0]):
                valid = False
                break
            if visible.size:
                label = int(visible[0])
                prediction[indices] = label
                witnesses[label] += len(visible) * (len(visible) - 1) // 2
        if valid and min(witnesses) >= 2:
            predictions.append(prediction)
            supported.append(model)
    _require(bool(predictions), "No visibly supported coarse reflection")
    predictions = np.stack(predictions)
    low = np.where(predictions >= 0, predictions, 2).min(axis=0)
    high = predictions.max(axis=0)
    _require(np.all(high >= 0), "An occupancy orbit has no surviving witness")
    _require(np.array_equal(low, high), "Supported reflections disagree")
    return high.reshape(observed.shape).astype(bool), tuple(supported)


def _read_input(input_grid):
    a = _grid(input_grid)
    rectangles = []
    for color in np.unique(a):
        rr, cc = np.nonzero(a == color)
        box = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
        r0, r1, c0, c1 = box
        if len(rr) >= 2 and len(rr) == (r1 - r0) * (c1 - c0):
            rectangles.append((int(color), box))
    _require(len(rectangles) == 1, "Erasure must be the unique rectangular color")
    erased, box = rectangles[0]
    known = a != erased
    corners = [int(a[r, c]) for r, c in
               ((0, 0), (0, a.shape[1] - 1), (a.shape[0] - 1, 0),
                (a.shape[0] - 1, a.shape[1] - 1)) if known[r, c]]
    _require(bool(corners) and len(set(corners)) == 1, "No unambiguous outer frame")
    line = corners[0]
    # An untouched complete tile row and column guarantee that these are
    # precisely the separator lines, even when the erasure crosses the rim.
    row_lines = np.flatnonzero(np.all((a == line) | ~known, axis=1) & known.any(axis=1))
    col_lines = np.flatnonzero(np.all((a == line) | ~known, axis=0) & known.any(axis=0))
    _require(len(row_lines) >= 4 and len(col_lines) >= 4, "Need at least 3x3 coarse cells")
    _require(row_lines[0] == col_lines[0] == 0
             and row_lines[-1] == a.shape[0] - 1
             and col_lines[-1] == a.shape[1] - 1, "Frame must bound the whole quilt")
    ph, pw = int(row_lines[1]), int(col_lines[1])
    _require(ph >= 3 and pw >= 3 and np.all(np.diff(row_lines) == ph)
             and np.all(np.diff(col_lines) == pw), "Separators must form a regular tile lattice")
    sh, sw = ph - 1, pw - 1
    mh, mw = len(row_lines) - 1, len(col_lines) - 1
    rr, cc = np.indices(a.shape)
    frame = (rr % ph == 0) | (cc % pw == 0)
    _require(np.all(a[known & frame] == line)
             and not np.any(a[known & ~frame] == line), "Separator color is exclusive to the lattice")
    patches, complete, uniform, motifs = [], [], [], []
    for r in range(mh):
        for c in range(mw):
            patch = a[r * ph + 1:(r + 1) * ph, c * pw + 1:(c + 1) * pw]
            patches.append(patch)
            intact = not np.any(patch == erased)
            complete.append(intact)
            if intact:
                (uniform if np.all(patch == patch[0, 0]) else motifs).append(patch)
    _require(len(uniform) >= 2 and len(motifs) >= 2,
             "Need two intact empty tiles and two intact motif tiles")
    background = int(uniform[0][0, 0])
    _require(all(np.all(p == background) for p in uniform), "Empty tiles must share one color")
    stamp = motifs[0].copy()
    _require(all(np.array_equal(p, stamp) for p in motifs), "All intact motifs must be unchanged copies")
    _require(len(set(map(int, np.unique(stamp))) - {background}) >= 2,
             "Motif must contain at least two payload colors")
    _require(len({line, background, erased}) == 3 and line not in stamp and erased not in stamp,
             "Frame, empty, erasure and payload roles must be disjoint")
    intact = np.asarray(complete).reshape(mh, mw)
    _require(np.any(intact.all(axis=0)) and np.any(intact.all(axis=1)),
             "A full tile row and column must survive")
    observed = np.full((mh, mw), -1, dtype=np.int8)
    for i, patch in enumerate(patches):
        visible = patch != erased
        fits_empty = np.all(patch[visible] == background)
        fits_stamp = np.array_equal(patch[visible], stamp[visible])
        _require(fits_empty or fits_stamp, "A tile disagrees with both allowed contents")
        if fits_empty != fits_stamp:
            observed.flat[i] = int(fits_stamp)
    occupancy, supported = _infer_occupancy(observed)
    return {"grid": a, "box": box, "erased": erased, "line": line,
            "background": background, "stamp": stamp, "observed": observed,
            "mask": occupancy, "supported": supported, "pitch": (ph, pw),
            "intact": intact}


def _patch_prediction(info, mask=None, stamp=None):
    """Direct coordinate indexing of the erased patch; no generated state."""
    mask = info["mask"] if mask is None else mask
    stamp = info["stamp"] if stamp is None else stamp
    ph, pw = info["pitch"]
    r0, r1, c0, c1 = info["box"]
    rr, cc = np.indices((r1 - r0, c1 - c0))
    rr, cc = rr + r0, cc + c0
    separators = (rr % ph == 0) | (cc % pw == 0)
    mr = np.minimum(rr // ph, mask.shape[0] - 1)
    mc = np.minimum(cc // pw, mask.shape[1] - 1)
    local_r, local_c = np.maximum(rr % ph - 1, 0), np.maximum(cc % pw - 1, 0)
    content = np.where(mask[mr, mc], stamp[local_r, local_c], info["background"])
    return np.where(separators, info["line"], content).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover only the erased fine-resolution window, entirely from input."""
    return _patch_prediction(_read_input(input_grid))


def _render(mask, stamp, background, line):
    """Generation ground truth: write full blocks rather than query indexing."""
    mh, mw = mask.shape
    sh, sw = stamp.shape
    board = np.full((mh * (sh + 1) + 1, mw * (sw + 1) + 1), line, dtype=np.int8)
    for r in range(mh):
        for c in range(mw):
            board[r * (sh + 1) + 1:(r + 1) * (sh + 1),
                  c * (sw + 1) + 1:(c + 1) * (sw + 1)] = stamp if mask[r, c] else background
    return board


def _asymmetric(stamp):
    return all(not np.array_equal(stamp, t) for t in
               (stamp[::-1], stamp[:, ::-1], stamp[::-1, ::-1],
                stamp.T, stamp.T[::-1], stamp.T[:, ::-1], stamp.T[::-1, ::-1]))


def _sample_stamp(rng, shape, bg, payload, witness, style):
    h, w = shape
    r, c = np.indices(shape)
    n = len(payload)
    for _ in range(48):
        if witness or style == "random":
            labels = rng.integers(n, size=shape)
        elif style == "bands":
            labels = (r if rng.integers(2) else c) // int(rng.integers(1, 3))
        elif style == "rings":
            labels = np.minimum.reduce((r, c, h - 1 - r, w - 1 - c))
        elif style == "diagonal":
            labels = r + int(rng.integers(1, n + 1)) * c
        elif style == "patches":
            labels = rng.integers(n, size=((h + 1) // 2, (w + 1) // 2))[r // 2, c // 2]
        else:
            labels = (abs(2 * r - h + 1) + abs(2 * c - w + 1)) // 2
        stamp = payload[(labels + int(rng.integers(n))) % n].copy()
        hole_style = int(rng.integers(4))
        if witness or hole_style:
            if hole_style == 3 and h >= 4 and w >= 4 and not witness:
                stamp[1:-1, 1:-1] = bg
            else:
                count = 1 if hole_style == 1 else int(rng.integers(1, max(2, stamp.size // 3)))
                stamp.flat[rng.choice(stamp.size, count, replace=False)] = bg
        if len(set(map(int, np.unique(stamp))) - {bg}) >= 2 and (not witness or _asymmetric(stamp)):
            return stamp
    # An explicit asymmetric motif with a background hole; no retry is unbounded.
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[0, 0] = payload[1]
    stamp[0, 1] = bg
    stamp[-1, 0] = payload[-1]
    return stamp


def _sample_mask(rng, shape, model, style):
    mask = np.zeros(shape, dtype=bool)
    density = float(rng.choice([.25, .4, .55, .7]))
    h, w = shape
    for orbit in _orbits(shape, model):
        r, c = divmod(orbit[0], w)
        if style == "random":
            value = rng.random() < density
        elif style == "bands":
            value = (r + c // 2 + int(h % 2)) % 3 != 0
        elif style == "rings":
            value = min(r, c, h - 1 - r, w - 1 - c) % 2 == 0
        elif style == "islands":
            value = (r + 2 * c) % 3 == 0
        else:
            value = (r // 2 + c // 2) % 2 == 0
        # Optional styles retain regular regions with occasional orbit changes.
        if style != "random" and rng.random() < .12:
            value = not value
        mask.flat[list(orbit)] = value
    return mask


def _shapes(rng, model, regime, train, witness, previous):
    low, high = ((4, 5) if train else (6, 7)) if regime == "scale" else (4, 7)
    candidates = []
    for mh in range(low, high + 1):
        for mw in range(low, high + 1):
            if model not in ("H", "V", "HV") and mh != mw:
                continue
            for sh in range(3 if witness else 2, min(6, 29 // mh - 1) + 1):
                for sw in range(3 if witness else 2, min(6, 29 // mw - 1) + 1):
                    if previous and ((mh, mw) == previous[0] or (sh, sw) == previous[1]):
                        continue
                    candidates.append(((mh, mw), (sh, sw)))
    _require(bool(candidates), "No feasible mask and motif shape")
    return candidates[int(rng.integers(len(candidates)))]


def _boxes(rng, mask, stamp_shape, train, regime):
    """Rectangles hide whole tiles plus optionally fragments and separators."""
    mh, mw = mask.shape
    sh, sw = stamp_shape
    ph, pw = sh + 1, sw + 1
    height, width = mh * ph + 1, mw * pw + 1
    boxes = []
    max_span = 2 if train else (3 if regime == "scale" else 2)
    for nr in range(1, min(max_span, mh - 1) + 1):
        for nc in range(1, min(max_span, mw - 1) + 1):
            if nr * nc < 2:
                continue
            for r in range(mh - nr + 1):
                for c in range(mw - nc + 1):
                    region = mask[r:r + nr, c:c + nc]
                    if np.all(region) or not np.any(region):
                        continue
                    r0, r1, c0, c1 = r * ph + 1, (r + nr) * ph, c * pw + 1, (c + nc) * pw
                    if not train and regime == "edge":
                        edges = []
                        if r == 0:
                            edges.append((0, r1, c0, c1))
                        if r + nr == mh:
                            edges.append((r0, height, c0, c1))
                        if c == 0:
                            edges.append((r0, r1, 0, c1))
                        if c + nc == mw:
                            edges.append((r0, r1, c0, width))
                        boxes.extend(edges)
                    else:
                        # Even aligned windows include the separator between
                        # adjacent hidden tiles. Phase windows also cut neighbors.
                        extend = (not train) if regime == "phase" else rng.random() < .45
                        if extend:
                            extra_min = 2 if regime == "phase" else 0
                            r0 = max(1, r0 - int(rng.integers(extra_min, ph)))
                            r1 = min(height - 1, r1 + int(rng.integers(extra_min, ph)))
                            c0 = max(1, c0 - int(rng.integers(extra_min, pw)))
                            c1 = min(width - 1, c1 + int(rng.integers(extra_min, pw)))
                        boxes.append((r0, r1, c0, c1))
    rng.shuffle(boxes)
    return boxes[:64]


def _make_pair(rng, params, train, index, previous, seen):
    witness = train and index == 0
    for _ in range(160):
        mshape, sshape = _shapes(rng, params["model"], params["regime"], train, witness, previous)
        palette = rng.permutation(10) if params["varied"] else params["palette"]
        line, bg, erased = map(int, palette[:3])
        payload = palette[3:3 + int(rng.integers(2, 6))].astype(np.int8)
        style = str(rng.choice(["random", "bands", "rings", "diagonal", "patches", "centered"]))
        stamp = _sample_stamp(rng, sshape, bg, payload, witness, style)
        mask = _sample_mask(rng, mshape, params["model"], str(rng.choice(
            ["random", "random", "bands", "rings", "islands", "patches"])))
        if min(np.count_nonzero(mask), np.count_nonzero(~mask)) < 4:
            continue
        board = _render(mask, stamp, bg, line)
        for box in _boxes(rng, mask, sshape, train, params["regime"]):
            r0, r1, c0, c1 = box
            if (r1 - r0) * (c1 - c0) > board.size * .46:
                continue
            a = board.copy()
            a[r0:r1, c0:c1] = erased
            if _key(a) in seen:
                continue
            try:
                info = _read_input(a)
            except ValueError:
                continue
            # Verify the intended model survived, and reject alternative
            # completions rather than hiding their disagreement in the label.
            if params["model"] not in info["supported"] or not np.array_equal(info["mask"], mask):
                continue
            unknown = info["observed"] < 0
            if not (np.any(unknown & mask) and np.any(unknown & ~mask)):
                continue
            b = board[r0:r1, c0:c1].copy()
            if not np.array_equal(_patch_prediction(info), b):
                continue
            return {"input": a, "output": b}, (mshape, sshape)
    raise RuntimeError("Could not construct a well-witnessed quilt within bounded attempts")


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    params = {"model": str(rng.choice(_MODELS)), "regime": str(rng.choice(["edge", "phase", "scale"])),
              "varied": bool(rng.random() < .75), "palette": rng.permutation(10)}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        previous = None
        for index in range(count):
            pair, previous = _make_pair(rng, params, split == "train", index, previous, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "train length", 2, 10)
    _count(nq, "test length", 1, 4)
    _require(nt > nq, "Need more demonstrations than tests")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            info = _read_input(a)
            _require(np.array_equal(_patch_prediction(info), b), "Wrong recovered crop")
            _require(info["erased"] not in b, "Erasure leaked into output")
            completed = _render(info["mask"], info["stamp"], info["background"], info["line"])
            visible = a != info["erased"]
            _require(np.array_equal(a[visible], completed[visible]), "Visible quilt is inconsistent")
            r0, r1, c0, c1 = info["box"]
            _require(np.array_equal(completed[r0:r1, c0:c1], b), "Block-rendered crop disagrees")
            unknown = info["observed"] < 0
            _require(np.any(unknown & info["mask"]) and np.any(unknown & ~info["mask"]),
                     "Each query must need recovery of both occupied and empty tiles")
            # Two concrete ablations of A must both fail, not just one.
            for fill in (False, True):
                shortcut = info["mask"].copy()
                shortcut[unknown] = fill
                _require(not np.array_equal(_patch_prediction(info, mask=shortcut), b),
                         "Ignoring reflected occupancy would give the same answer")
            # Omitting B's multicolor local substitution must also change it.
            solid = np.full_like(info["stamp"], info["stamp"].flat[0])
            _require(not np.array_equal(_patch_prediction(info, stamp=solid), b),
                     "Uniform enlargement would give the same answer")
            records[split].append(info)
    tr, te = records["train"], records["test"]
    _require(any(_asymmetric(i["stamp"]) and np.any(i["stamp"] == i["background"]) for i in tr),
             "Training must establish unchanged asymmetric motifs and background holes")
    for i in tr:
        if _asymmetric(i["stamp"]):
            correct = _patch_prediction(i)
            for reflected in (i["stamp"][::-1], i["stamp"][:, ::-1], i["stamp"][::-1, ::-1]):
                _require(not np.array_equal(_patch_prediction(i, stamp=reflected), correct),
                         "The motif orientation witness must affect the erased pixels")
    _require(len({i["mask"].shape for i in tr}) >= 2 and len({i["stamp"].shape for i in tr}) >= 2,
             "Training must vary both coarse and fine dimensions")
    _require(bool(set.intersection(*(set(i["supported"]) for i in tr + te))),
             "A common demonstrated reflection mechanism must apply to every example")
    return None


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed audit, count boundaries, rejection and equivariance checks."""
    signatures, records = set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_signature(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read_input(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                records.append((split, info, pair["output"].shape))
        if seed in (0, 17, 41):
            assert _signature(p) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len({i["grid"].shape for _, i, _ in records}) >= 25
    assert len({shape for _, _, shape in records}) >= 35
    assert len({i["mask"].shape for _, i, _ in records}) >= 10
    assert len({i["stamp"].shape for _, i, _ in records}) >= 12
    assert set(_MODELS) <= {m for _, i, _ in records for m in i["supported"]}
    assert {i["background"] for _, i, _ in records} == set(range(10))
    assert {i["line"] for _, i, _ in records} == set(range(10))
    assert {i["erased"] for _, i, _ in records} == set(range(10))
    patterns = {(_key(i["mask"]), i["stamp"].shape) for _, i, _ in records}
    assert len(patterns) >= 100
    assert any(i["stamp"].shape[0] != i["stamp"].shape[1] for _, i, _ in records)
    assert any(not np.any(i["stamp"] == i["background"]) for _, i, _ in records)
    assert any(np.all(i["stamp"][1:-1, 1:-1] == i["background"])
               and min(i["stamp"].shape) >= 4 for _, i, _ in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(173, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, True), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(19, 2, 1)
    # Coordinate systems are both transformed; the rule is color-equivariant.
    for turns in (1, 2, 3):
        for pair in p["train"] + p["test"]:
            assert np.array_equal(solve(np.rot90(pair["input"], turns)), np.rot90(pair["output"], turns))
    perm = np.random.default_rng(719).permutation(10)
    for pair in p["train"] + p["test"]:
        assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    damaged = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("Wrong output accepted")
    # No visible evidence for a completely erased coarse orbit must fail.
    bad = np.ones((4, 4), dtype=np.int8)
    bad[1:3, 1:3] = 0
    bad[0, 0] = bad[0, -1] = bad[-1, 0] = bad[-1, -1] = -1
    try:
        _infer_occupancy(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Unwitnessed occupancy accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds and all 30 legal count combinations)")
