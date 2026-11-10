"""Decode an inflated mosaic, resize it by unary counts, then tile a stencil.

Only the carrier is run-collapsed. Background is visibly encoded at (0, 0).
All sampling and diagnostics are bounded; this module writes no files.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 680478273, 74215422, 383700856, 870922496, 170611754,
              179956329, 263554312, 82056427, 586406604, 247837451,
              387184871, 518737567, 419169431, 97025024, 423475157,
              683207531, 632477853, 901951066, 812879944, 827651059,
              189899560, 231480814, 777237753, 220431873, 217411446,
              707457113, 562123275, 638709723, 994286007, 991307473,
              191362652, 210167415, 769895825, 529381962, 205123351,
              104377132, 765356812, 580672032, 528452028, 242900305, 17364070)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(train, test):
    if not _integer(train) or not 2 <= train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(test) or not 1 <= test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return int(train), int(test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid, sides 1..30, colors 0..9")
    return a


def _runs(a):
    """Maximal equality runs of complete vectors; separated repeats stay apart."""
    rs = np.r_[0, np.flatnonzero(np.any(a[1:] != a[:-1], axis=1)) + 1]
    cs = np.r_[0, np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0)) + 1]
    return (a[np.ix_(rs, cs)].copy(), np.diff(np.r_[rs, a.shape[0]]),
            np.diff(np.r_[cs, a.shape[1]]))


def _resize(a, height, width):
    return a[(np.arange(height) * a.shape[0] // height)[:, None],
             (np.arange(width) * a.shape[1] // width)[None, :]]


def _tile(a, height, width):
    return a[np.arange(height)[:, None] % a.shape[0],
             np.arange(width)[None, :] % a.shape[1]]


def _render(mosaic, stencil, height, width, background):
    base = _resize(mosaic, height, width)
    ink = _tile(stencil, height, width)
    return np.where(ink == background, base, ink).astype(np.int8)


def _decode(input_grid):
    g = _grid(input_grid)
    if g.shape[0] != g.shape[1] or g.shape[0] % 2:
        raise ValueError("Input must contain four equal square panels")
    n = g.shape[0] // 2
    if not 3 <= n <= 15:
        raise ValueError("Panel side must be 3..15")
    bg = int(g[0, 0])
    left, right, carrier, stencil = g[:n, :n], g[:n, n:], g[n:, :n], g[n:, n:]
    if right[0, 0] != bg or np.any(carrier == bg):
        raise ValueError("Counter corner markers must agree; carrier must be filled")
    h, w = int(np.count_nonzero(left != bg)), int(np.count_nonzero(right != bg))
    if not 1 <= h <= 30 or not 1 <= w <= 30:
        raise ValueError("Counter dimensions must be in 1..30")
    mosaic, rows, cols = _runs(carrier)
    return n, bg, h, w, left, right, carrier, stencil, mosaic, rows, cols


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only rule; also defines floor resampling outside sampled scale ratios."""
    _, bg, h, w, _, _, _, stencil, mosaic, _, _ = _decode(input_grid)
    return _render(mosaic, stencil, h, w, bg)


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _normalized(a):
    roles = {}
    return a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _hidden_seam(a):
    return (any(np.any(a[:, i] == a[:, i-1]) and np.any(a[:, i] != a[:, i-1])
                for i in range(1, a.shape[1]))
            or any(np.any(a[i] == a[i-1]) and np.any(a[i] != a[i-1])
                   for i in range(1, a.shape[0])))


REQUIRED_EVIDENCE = frozenset((
    "collapse_rows", "collapse_columns", "skip_collapse", "uneven", "hidden_seam",
    "counter_swap", "counter_bbox", "counter_one_ink", "counter_color_count",
    "mosaic_tile_rows", "mosaic_tile_columns", "stencil_resize", "collapse_stencil",
    "overlay_first", "opaque_background", "ignore_stencil", "phase_rows", "phase_columns",
    "visible_carrier", "visible_overwrite", "repeat_stencil",
))


def _evidence(pair):
    n, bg, h, w, left, right, carrier, stencil, m, rows, cols = _decode(pair["input"])
    out = np.asarray(pair["output"])
    base, tiled = _resize(m, h, w), _tile(stencil, h, w)
    cover = set()

    def compare(name, alternative):
        if not np.array_equal(out, alternative):
            cover.add(name)

    def overlay(a, s=tiled):
        return np.where(s == bg, a, s)

    compare("skip_collapse", overlay(_resize(carrier, h, w)))
    compare("collapse_rows", overlay(_resize(m.repeat(rows, axis=0), h, w)))
    compare("collapse_columns", overlay(_resize(m.repeat(cols, axis=1), h, w)))
    if len(set(map(int, rows))) > 1 or len(set(map(int, cols))) > 1:
        cover.add("uneven")
    if _hidden_seam(m):
        cover.add("hidden_seam")
    if h != w:
        cover.add("counter_swap")
    stats = [[], [], []]
    for panel in (left, right):
        rr, cc = np.nonzero(panel != bg)
        _, counts = np.unique(panel[panel != bg], return_counts=True)
        stats[0].append(int((rr.max()-rr.min()+1)*(cc.max()-cc.min()+1)))
        stats[1].append(int(counts.max()))
        stats[2].append(len(counts))
    for name, dims in zip(("counter_bbox", "counter_one_ink", "counter_color_count"), stats):
        if tuple(dims) != (h, w):
            cover.add(name)
    rr, cc = np.arange(h)[:, None], np.arange(w)[None, :]
    compare("mosaic_tile_rows", overlay(m[rr % m.shape[0], cc*m.shape[1]//w]))
    compare("mosaic_tile_columns", overlay(m[rr*m.shape[0]//h, cc % m.shape[1]]))
    compare("stencil_resize", overlay(base, _resize(stencil, h, w)))
    compare("collapse_stencil", overlay(base, _tile(_runs(stencil)[0], h, w)))
    fused = np.where(stencil == bg, carrier, stencil)
    compare("overlay_first", _resize(_runs(fused)[0], h, w))
    compare("opaque_background", tiled)
    compare("ignore_stencil", base)
    compare("phase_rows", overlay(base, stencil[(rr+1) % n, cc % n]))
    compare("phase_columns", overlay(base, stencil[rr % n, (cc+1) % n]))
    if np.any(tiled == bg):
        cover.add("visible_carrier")
    if np.any((tiled != bg) & (tiled != base)):
        cover.add("visible_overwrite")
    if h > n or w > n:
        cover.add("repeat_stencil")
    return cover


def _partition(rng, total, parts):
    if parts == 1:
        return np.array([total])
    # Mixture of nearly equal, random, and one dominant expansion band.
    mode = int(rng.integers(3))
    if mode == 0:
        result = np.full(parts, total // parts)
        result[rng.permutation(parts)[:total % parts]] += 1
    elif mode == 1:
        cuts = np.r_[0, np.sort(rng.choice(np.arange(1, total), parts-1, replace=False)), total]
        result = np.diff(cuts)
    else:
        result = np.ones(parts, dtype=int)
        result[int(rng.integers(parts))] += total-parts
    return result


def _mosaic(rng, shape, palette):
    cells = shape[0] * shape[1]
    if cells == 1:
        return np.full(shape, rng.choice(palette), dtype=np.int8)
    k = int(rng.integers(2, min(6, len(palette), cells)+1))
    inks = rng.choice(palette, k, replace=False)
    kind = int(rng.integers(5))
    for _ in range(128):
        rr, cc = np.indices(shape)
        if kind == 0:
            labels = (rr + cc + int(rng.integers(k))) % k
        elif kind == 1:
            labels = rng.integers(k, size=(min(2, shape[0]), min(3, shape[1])))
            labels = labels[rr % labels.shape[0], cc % labels.shape[1]]
        elif kind == 2:
            labels = np.zeros(shape, dtype=int)
            labels[rng.random(shape) < .4] = 1
            # Sparse marks and repeated base-color neighbors expose global seams.
            if k > 2:
                labels[rng.random(shape) < .2] = k-1
        else:
            labels = rng.integers(k, size=shape)
        m = inks[labels].astype(np.int8)
        if _runs(m)[0].shape == shape:
            return m
    # Fully distinct neighboring labels guarantee boundaries for any shape <=5.
    rr, cc = np.indices(shape)
    return inks[(rr+cc) % k].astype(np.int8)


def _counter(rng, n, count, bg, palette):
    panel = np.full((n, n), bg, dtype=np.int8)
    coords = np.arange(1, n*n)  # (0,0) is the explicit background marker.
    style = int(rng.integers(3))
    if style == 0:
        coords = rng.permutation(coords)
    elif style == 1:
        coords = np.array(sorted(coords, key=lambda x: (int(x) % n, int(x) // n)))
    else:
        r, c = int(rng.integers(n)), int(rng.integers(n))
        coords = np.array(sorted(coords, key=lambda x: (abs(int(x)//n-r)+abs(int(x)%n-c), int(x))))
    positions = coords[:count]
    ink_count = 1 if rng.random() < .4 else int(rng.integers(1, min(4, count, len(palette))+1))
    inks = rng.choice(palette, ink_count, replace=False)
    values = np.r_[inks, rng.choice(inks, count-ink_count)]
    rng.shuffle(values)
    panel.flat[positions] = values
    return panel


def _stencil(rng, n, bg, palette):
    s = np.full((n, n), bg, dtype=np.int8)
    kind = int(rng.integers(8))
    if kind == 0:
        return s
    rr, cc = np.indices((n, n))
    if kind == 1:
        mask = rng.random((n, n)) < float(rng.uniform(.1, .8))
    elif kind == 2:
        period = int(rng.integers(2, n+1))
        mask = (rr + int(rng.integers(period))) % period == 0
    elif kind == 3:
        period = int(rng.integers(2, n+1))
        mask = (cc + int(rng.integers(period))) % period == 0
    elif kind == 4:
        thickness = int(rng.integers(1, max(2, n//2)))
        mask = np.minimum.reduce([rr, cc, n-1-rr, n-1-cc]) < thickness
    elif kind == 5:
        mask = (rr//int(rng.integers(1, 4)) + cc//int(rng.integers(1, 4))) % 2 == int(rng.integers(2))
    elif kind == 6:
        r, c = int(rng.integers(n)), int(rng.integers(n))
        mask = abs(rr-r) + abs(cc-c) <= int(rng.integers(1, max(2, n//2)))
    else:
        mask = np.zeros((n, n), dtype=bool)
        mask.flat[rng.choice(n*n, int(rng.integers(1, n+1)), replace=False)] = True
    if mask.all():
        mask.flat[int(rng.integers(n*n))] = False
    if not mask.any():
        mask.flat[int(rng.integers(n*n))] = True
    k = 1 if rng.random() < .5 else int(rng.integers(1, min(4, len(palette), int(mask.sum()))+1))
    inks = rng.choice(palette, k, replace=False)
    s[mask] = rng.choice(inks, int(mask.sum()))
    return s


def _dimension(rng, logical, n):
    cap = min(30, n*n-1)
    choices = np.arange(logical, cap+1, logical)
    if logical == 1:
        choices = np.arange(1, cap+1)
    return int(rng.choice(choices))


def _assemble(rng, bg, palette, n, m, rows, cols, h, w, stencil):
    carrier = np.repeat(np.repeat(m, rows, axis=0), cols, axis=1)
    left, right = _counter(rng, n, h, bg, palette), _counter(rng, n, w, bg, palette)
    return {"input": np.block([[left, right], [carrier, stencil]]).astype(np.int8),
            "output": _render(m, stencil, h, w, bg)}


def _sample_pair(rng, bg, palette):
    n = int(rng.integers(3, 16))
    choices = np.arange(1, min(5, n-1)+1)
    weights = np.where(choices == 1, .12, 1.)
    a, b = map(int, rng.choice(choices, size=2, p=weights/weights.sum()))
    m = _mosaic(rng, (a, b), palette)
    rows, cols = _partition(rng, n, a), _partition(rng, n, b)
    h, w = _dimension(rng, a, n), _dimension(rng, b, n)
    return _assemble(rng, bg, palette, n, m, rows, cols, h, w,
                     _stencil(rng, n, bg, palette))


def _training_diverse(pairs):
    d = [_decode(p["input"]) for p in pairs]
    return (len({x[0] for x in d}) >= 2 and len({(x[2], x[3]) for x in d}) >= 2
            and len({_normalized(x[8]) for x in d}) >= 2
            and len({_normalized(x[7]) for x in d}) >= 2)


def _fallback(rng, bg, palette):
    """Two varied constructive teaching witnesses if the candidate pool is unlucky."""
    a, b, c, d = map(int, palette[:4])
    specs = [(6, np.array([[a, a, b], [c, b, a]], dtype=np.int8),
              np.array([1, 5]), np.array([1, 2, 3]), 10, 15),
             (7, np.array([[a, b], [c, a], [a, b]], dtype=np.int8),
              np.array([1, 4, 2]), np.array([5, 2]), 12, 18)]
    pairs = []
    for n, m, rows, cols, h, w in specs:
        stencil = np.full((n, n), bg, dtype=np.int8)
        stencil[1:3, 1:3] = d
        stencil[-1, -2] = b
        pair = _assemble(rng, bg, palette, n, m, rows, cols, h, w, stencil)
        # Guarantee multicolor occupancy and a nonrectangular mask, independently
        # of the ordinary counter sampler's random monochrome branch.
        for offset, count in ((0, h), (n, w)):
            panel = np.full((n, n), bg, dtype=np.int8)
            positions = np.r_[np.arange(1, count), n*n-1]
            panel.flat[positions] = np.asarray(palette)[:3][np.arange(count) % 3]
            pair["input"][:n, offset:offset+n] = panel
        pairs.append(pair)
    return pairs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train_count, test_count = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    bg = int(rng.integers(10))
    palette = rng.permutation([c for c in range(10) if c != bg])[:int(rng.integers(4, 10))]
    pool, covers, training = [], [], None
    for _ in range(64):
        pair = _sample_pair(rng, bg, palette)
        cover = _evidence(pair)
        for j in rng.permutation(len(pool)):
            proposed = [pool[int(j)], pair]
            if REQUIRED_EVIDENCE <= covers[int(j)] | cover and _training_diverse(proposed):
                training = proposed
                break
        if training is not None:
            break
        pool.append(pair)
        covers.append(cover)
    if training is None:
        training = _fallback(rng, bg, palette)
    puzzle = {"train": training, "test": []}
    seen = {_key(p["input"]) for p in training}
    train_shapes = {p["output"].shape for p in training}
    for split, target in (("train", train_count), ("test", test_count)):
        while len(puzzle[split]) < target:
            for _ in range(128):
                pair = _sample_pair(rng, bg, palette)
                if _key(pair["input"]) in seen:
                    continue
                if split == "test":
                    if pair["output"].shape in train_shapes or "skip_collapse" not in _evidence(pair):
                        continue
                break
            else:
                raise RuntimeError("Bounded sampler could not find a distinct informative pair")
            puzzle[split].append(pair)
            seen.add(_key(pair["input"]))
            if split == "train":
                train_shapes.add(pair["output"].shape)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, backgrounds, evidence = set(), set(), set()
    shapes = {np.asarray(p["output"]).shape for p in puzzle["train"]}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(g)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            n, bg, h, w, _, _, carrier, stencil, m, rows, cols = _decode(g)
            backgrounds.add(bg)
            assert out.shape == (h, w)
            assert np.array_equal(np.repeat(np.repeat(m, rows, axis=0), cols, axis=1), carrier)
            assert _runs(m)[0].shape == m.shape, "Invisible logical boundaries"
            assert m.shape[0] == 1 or h % m.shape[0] == 0
            assert m.shape[1] == 1 or w % m.shape[1] == 0
            assert np.array_equal(solve(g), out), "Output disagrees with input-only rule"
            assert not np.any(out == bg), "Transparent holes must expose filled carrier"
            tiled, base = _tile(stencil, h, w), _resize(m, h, w)
            assert np.array_equal(out[tiled == bg], base[tiled == bg])
            assert np.array_equal(out[tiled != bg], tiled[tiled != bg])
            cover = _evidence(pair)
            if split == "train":
                evidence |= cover
            else:
                assert out.shape not in shapes, "Missing output-geometry shift"
                assert "skip_collapse" in cover, "Test hides the contribution of A"
    assert len(backgrounds) == 1
    assert REQUIRED_EVIDENCE <= evidence, "Missing training evidence: " + str(REQUIRED_EVIDENCE-evidence)
    assert _training_diverse(puzzle["train"])


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.uint8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 host seeds, exhaustive legal count support, mutants, and bounded metrics."""
    shapes, panels, mosaics, profiles, masks, backgrounds, outputs = (set() for _ in range(7))
    stats = {"puzzles": 42, "pairs": 0, "empty_stencils": 0, "partial_tiles": 0,
             "hidden_seams": 0, "separated_repeats": 0, "black_payload": 0}
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                n, bg, h, w, _, _, _, s, m, rows, cols = _decode(pair["input"])
                stats["pairs"] += 1
                panels.add(n)
                shapes.add((h, w))
                mosaics.add(_normalized(m))
                profiles.add((tuple(map(int, rows)), tuple(map(int, cols))))
                masks.add(_key(s != bg))
                backgrounds.add(bg)
                outputs.add(_normalized(pair["output"]))
                stats["empty_stencils"] += bool(np.all(s == bg))
                stats["partial_tiles"] += h % n != 0 or w % n != 0
                stats["hidden_seams"] += _hidden_seam(m)
                stats["black_payload"] += bg != 0 and bool(np.any(m == 0) or np.any(s == 0))
                stats["separated_repeats"] += any(
                    np.array_equal(m[i], m[j]) for i in range(m.shape[0]) for j in range(i+2, m.shape[0])) or any(
                    np.array_equal(m[:, i], m[:, j]) for i in range(m.shape[1]) for j in range(i+2, m.shape[1]))
    assert len(signatures) == 42 and len(panels) >= 10 and len(backgrounds) >= 8
    assert len(shapes) >= 60 and len(mosaics) >= 50 and len(profiles) >= 80
    assert len(masks) >= 70 and len(outputs) >= 100
    assert all(value > 0 for value in stats.values())
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            validate_generated_puzzle(generate_puzzle(0, train, test))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4., 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Explicit fallback checks and a corrupted-label rejection.
    for seed in range(4):
        rng = np.random.default_rng(seed)
        pairs = _fallback(rng, 0, np.arange(1, 10))
        assert REQUIRED_EVIDENCE <= set().union(*(_evidence(p) for p in pairs))
    damaged = generate_puzzle(0)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # Run collapse is local: the third row/column must survive a separated repeat.
    repeated = np.array([[1, 2, 1], [3, 1, 3], [1, 2, 1]])
    assert np.array_equal(_runs(repeated)[0], repeated)
    assert np.array_equal(_resize(np.array([[1, 2], [3, 4]]), 3, 3),
                          np.array([[1, 1, 2], [1, 1, 2], [3, 3, 4]]))
    stats.update(panel_sizes=sorted(panels), output_shapes=len(shapes),
                 normalized_mosaics=len(mosaics), expansion_profiles=len(profiles),
                 stencil_masks=len(masks), backgrounds=sorted(backgrounds),
                 normalized_outputs=len(outputs), legal_count_combinations=30)
    return stats


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
