"""Repair a mosaic: mirror tile colors, repeat unreflected local motifs.

Zero denotes missing pixels. The color at the outer corner denotes the intact
one-cell lattice. Everything else, including the carrier, is inferred from the
input. No parent generator, examples, seed, or hidden convention is used by solve.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np


_TILE_SHAPES = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3),
                (3, 4), (4, 3), (4, 4))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= n <= 30 for n in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _fingerprint(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


@lru_cache(maxsize=512)
def _orbits(nr, nc, a, b):
    seen, result = set(), []
    for r in range(nr):
        for c in range(nc):
            if (r, c) in seen:
                continue
            rows = {r, a - r} if 0 <= a - r < nr else {r}
            cols = {c, b - c} if 0 <= b - c < nc else {c}
            orbit = tuple(sorted((y, x) for y in rows for x in cols))
            seen.update(orbit)
            result.append(orbit)
    return tuple(result)


def _folds(n):
    # At least four tile addresses participate on each axis. Through-cell
    # folds and fringes of either sign are allowed, without a best-score tie.
    return [a for a in range(1, 2 * n - 2)
            if sum(0 <= a - r < n for r in range(n)) >= 4]


def _restore_labels(labels):
    answers = []
    nr, nc = labels.shape
    for a in _folds(nr):
        for b in _folds(nc):
            restored = labels.copy()
            for orbit in _orbits(nr, nc, a, b):
                colors = {int(labels[p]) for p in orbit if labels[p] >= 0}
                if len(colors) != 1:
                    break
                color = next(iter(colors))
                for p in orbit:
                    restored[p] = color
            else:
                answers.append(((a, b), restored))
    return answers


def _split_tiles(grid):
    sep = int(grid[0, 0])
    if sep == 0:
        raise ValueError("the outer lattice is intact and nonzero")
    rows = np.flatnonzero(np.all(grid == sep, axis=1))
    cols = np.flatnonzero(np.all(grid == sep, axis=0))
    if (len(rows) < 5 or len(cols) < 5 or rows[0] != 0 or cols[0] != 0
            or rows[-1] != grid.shape[0] - 1 or cols[-1] != grid.shape[1] - 1):
        raise ValueError("missing outer lattice or too few tiles")
    dr, dc = np.diff(rows), np.diff(cols)
    if len(set(dr)) != 1 or len(set(dc)) != 1:
        raise ValueError("nonuniform tile dimensions")
    th, tw = int(dr[0] - 1), int(dc[0] - 1)
    if (th, tw) not in _TILE_SHAPES:
        raise ValueError("unsupported tile geometry")
    nr, nc = len(rows) - 1, len(cols) - 1
    tiles = np.empty((nr, nc, th, tw), dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            tiles[r, c] = grid[rows[r] + 1:rows[r + 1],
                               cols[c] + 1:cols[c + 1]]
    if np.any(tiles == sep):
        raise ValueError("lattice color occurs inside a tile")
    return sep, tiles


def _full_box(motif):
    return (2 <= int(motif.sum()) <= motif.size - 2
            and motif[0].any() and motif[-1].any()
            and motif[:, 0].any() and motif[:, -1].any())


def _analyze(grid):
    """Infer carrier, local occupancy, tile labels, and unique address folds."""
    sep, tiles = _split_tiles(grid)
    nr, nc, th, tw = tiles.shape
    evidence = [[{int(v) for v in tiles[:, :, y, x].flat if v != 0}
                 for x in range(tw)] for y in range(th)]
    if any(not values for row in evidence for values in row):
        raise ValueError("a local motif coordinate has no surviving witness")
    candidates = []
    for carrier in sorted(set(int(v) for v in tiles.flat) - {0}):
        motif = np.zeros((th, tw), dtype=bool)
        valid = True
        for y in range(th):
            for x in range(tw):
                values = evidence[y][x]
                if carrier in values and values != {carrier}:
                    valid = False
                motif[y, x] = carrier not in values
        if not valid or not _full_box(motif):
            continue
        labels = np.full((nr, nc), -1, dtype=np.int8)
        for r in range(nr):
            for c in range(nc):
                values = set(int(v) for v in tiles[r, c].flat) - {0, carrier}
                if len(values) > 1:
                    valid = False
                    break
                if values:
                    labels[r, c] = next(iter(values))
            if not valid:
                break
        if not valid or len(set(int(v) for v in labels.flat if v >= 0)) < 2:
            continue
        for axes, restored in _restore_labels(labels):
            candidates.append(dict(separator=sep, carrier=carrier, motif=motif,
                                   labels=labels, restored=restored, axes=axes,
                                   tiles=tiles))
    if len(candidates) != 1:
        raise ValueError("input does not determine unique carrier and tile folds")
    return candidates[0]


def _assemble(tiles, separator):
    nr, nc, th, tw = tiles.shape
    grid = np.full((nr * (th + 1) + 1, nc * (tw + 1) + 1),
                   separator, dtype=np.int8)
    for r in range(nr):
        for c in range(nc):
            y, x = 1 + r * (th + 1), 1 + c * (tw + 1)
            grid[y:y + th, x:x + tw] = tiles[r, c]
    return grid


def _render(motif, labels, carrier, separator):
    tiles = np.where(motif[None, None], labels[:, :, None, None], carrier)
    return _assemble(tiles.astype(np.int8), separator)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover solely from this grid, preserving every nonzero input pixel."""
    grid = _grid(input_grid)
    info = _analyze(grid)
    out = _render(info['motif'], info['restored'], info['carrier'], info['separator'])
    if not np.array_equal(out[grid != 0], grid[grid != 0]):
        raise ValueError("visible pixels contradict the inferred mosaic")
    return out


def _nearest_labels(labels):
    result = labels.copy()
    known = [tuple(int(v) for v in p) for p in np.argwhere(labels >= 0)]
    for r, c in np.argwhere(labels < 0):
        q = min(known, key=lambda p: (abs(p[0] - r) + abs(p[1] - c), p))
        result[r, c] = labels[q]
    return result


def _pixel_mirror_shortcut(grid, info):
    """A concrete competing rule: copy raw reflected pixels, without tiling."""
    th, tw = info['motif'].shape
    a, b = info['axes']
    sy, sx = (a + 1) * (th + 1), (b + 1) * (tw + 1)
    result = grid.copy()
    for r, c in np.argwhere(grid == 0):
        for y, x in ((sy - r, c), (r, sx - c), (sy - r, sx - c)):
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] != 0:
                result[r, c] = grid[y, x]
                break
    return result


def _draw_motif(rng, shape):
    for _ in range(512):
        count = int(rng.integers(2, shape[0] * shape[1] - 1))
        m = np.zeros(shape[0] * shape[1], dtype=bool)
        m[rng.choice(m.size, count, replace=False)] = True
        m = m.reshape(shape)
        if (_full_box(m) and not np.array_equal(m, m[::-1])
                and not np.array_equal(m, m[:, ::-1])
                and (shape[0] != shape[1] or not np.array_equal(m, m.T))):
            return m
    raise RuntimeError("bounded motif sampling exhausted")


def _geometry(rng, shape, kind):
    maxima = [(29 // (s + 1)) for s in shape]
    lower = 5 if kind == 'offset' else 4
    nr, nc = (int(rng.integers(lower, n + 1)) for n in maxima)
    offsets = []
    for n in (nr, nc):
        limit = min(2, n - 4)
        offsets.append(0 if kind == 'center' or limit == 0 else
                       int(rng.integers(-limit, limit + 1)))
    if kind == 'offset' and not any(offsets):
        d = int(rng.integers(2))
        offsets[d] = int(rng.choice((-1, 1)))
    return nr, nc, nr - 1 + offsets[0], nc - 1 + offsets[1]


def _lost_labels(rng, geometry, count):
    orbits = _orbits(*geometry)
    four = [o for o in orbits if len(o) == 4]
    orbit = four[int(rng.integers(len(four)))]
    survivor = orbit[int(rng.integers(4))]
    lost = set(p for p in orbit if p != survivor)
    remaining = [p for o in orbits for p in o if p not in lost]
    for i in rng.permutation(len(remaining)):
        if len(lost) == count:
            break
        p = remaining[int(i)]
        orbit = next(o for o in orbits if p in o)
        if sum(q not in lost for q in orbit) > 1:
            lost.add(p)
    if len(lost) != count:
        raise RuntimeError("too many hidden labels requested")
    return lost


def _damage(rng, motif, labels, lost, carrier):
    nr, nc = labels.shape
    th, tw = motif.shape
    tiles = np.where(motif[None, None], labels[:, :, None, None], carrier).astype(np.int8)
    mask = rng.random(tiles.shape) < float(rng.uniform(0.12, 0.38))
    style = int(rng.integers(3))
    if style == 1:  # Short row/column cuts, chosen independently per tile.
        for r in range(nr):
            for c in range(nc):
                if rng.random() < 0.65:
                    if rng.random() < 0.5:
                        mask[r, c, int(rng.integers(th)), :] = True
                    else:
                        mask[r, c, :, int(rng.integers(tw))] = True
    elif style == 2:  # Rectangular bites crossing tile boundaries.
        for _ in range(int(rng.integers(1, 5))):
            r0, c0 = int(rng.integers(nr)), int(rng.integers(nc))
            rh, cw = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            y0, x0 = int(rng.integers(th)), int(rng.integers(tw))
            mask[r0:r0 + rh, c0:c0 + cw, y0:, x0:] = True
    accent_points = [tuple(p) for p in np.argwhere(motif)]
    carrier_points = [tuple(p) for p in np.argwhere(~motif)]
    protected = np.zeros_like(mask)
    # Every tile loses both an occupied and an empty motif coordinate. Thus
    # there is no complete input template, even in the simplest demonstration.
    for r in range(nr):
        for c in range(nc):
            for points in (accent_points, carrier_points):
                y, x = points[int(rng.integers(len(points)))]
                protected[r, c, y, x] = True
            if (r, c) in lost:
                protected[r, c][motif] = True
                if rng.random() < 0.55:
                    mask[r, c] = True
    mask |= protected
    # All local positions have some evidence, but it is distributed among
    # incomplete tiles. Revealing witnesses never restores protected pixels.
    for y in range(th):
        for x in range(tw):
            if mask[:, :, y, x].all():
                eligible = np.argwhere(~protected[:, :, y, x])
                if not len(eligible):
                    return None
                r, c = eligible[int(rng.integers(len(eligible)))]
                mask[r, c, y, x] = False
    # Each known-label tile shows both roles. Hidden-label tiles may still
    # show carrier pixels, or may be erased completely.
    for r in range(nr):
        for c in range(nc):
            for role in ((~motif,) if (r, c) in lost else (motif, ~motif)):
                if not np.any(role & ~mask[r, c]):
                    # Keep fully blank hidden-label tiles blank.
                    if (r, c) in lost and mask[r, c].all():
                        continue
                    eligible = np.argwhere(role & ~protected[r, c])
                    if not len(eligible):
                        return None
                    y, x = eligible[int(rng.integers(len(eligible)))]
                    mask[r, c, y, x] = False
    tiles[mask] = 0
    return tiles


def _make_pair(rng, shape, geometry, palette, separator, carrier, hidden_count, used, used_motifs):
    nr, nc, a, b = geometry
    orbits = _orbits(*geometry)
    for _ in range(256):
        motif = _draw_motif(rng, shape)
        if _key(motif) in used_motifs:
            continue
        k = int(rng.integers(3, min(5, len(orbits), len(palette)) + 1))
        colors = rng.choice(palette, k, replace=False)
        values = rng.integers(k, size=len(orbits))
        values[rng.permutation(len(orbits))[:k]] = np.arange(k)
        labels = np.empty((nr, nc), dtype=np.int8)
        for value, orbit in zip(values, orbits):
            for p in orbit:
                labels[p] = colors[value]
        lost = _lost_labels(rng, geometry, hidden_count)
        damaged = _damage(rng, motif, labels, lost, carrier)
        if damaged is None:
            continue
        inp = _assemble(damaged, separator)
        if _key(inp) in used:
            continue
        try:
            info = _analyze(inp)
        except ValueError:
            continue
        if info['axes'] != (a, b) or not np.array_equal(info['restored'], labels):
            continue
        if np.array_equal(_nearest_labels(info['labels']), labels):
            continue
        out = _render(motif, labels, carrier, separator)
        if np.array_equal(_pixel_mirror_shortcut(inp, info), out):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError("bounded mosaic sampling exhausted")


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(np.arange(1, 10))]
    separator, carrier = palette[:2]
    accents = palette[2:]
    shape = _TILE_SHAPES[int(rng.integers(len(_TILE_SHAPES)))]
    puzzle, used, used_motifs = {'train': [], 'test': []}, set(), set()
    kinds = ['center', 'offset'] + ['mixed'] * (nt - 2)
    rng.shuffle(kinds)
    for split, count in (('train', nt), ('test', ns)):
        for i in range(count):
            kind = kinds[i] if split == 'train' else 'mixed'
            geometry = _geometry(rng, shape, kind)
            hidden = 3 if split == 'train' else int(rng.integers(6, 9))
            pair = _make_pair(rng, shape, geometry, accents, separator, carrier, hidden, used, used_motifs)
            used.add(_key(pair['input']))
            used_motifs.add(_key(_analyze(pair['input'])['motif']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Validate returned arrays, not sampler metadata or latent intended labels."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, conventions, all_motifs = set(), set(), set()
    geometry_kinds, train_motifs = set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.any(out == 0)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            tiles, motif = info['tiles'], info['motif']
            nr, nc, th, tw = tiles.shape
            assert _key(motif) not in all_motifs, "example motifs must be distinct"
            all_motifs.add(_key(motif))
            conventions.add((info['separator'], info['carrier'], th, tw))
            assert np.array_equal(inp != out, inp == 0), "visible pixels must be preserved"
            assert np.array_equal(solve(inp), out)
            assert _full_box(motif)
            assert not np.array_equal(motif, motif[::-1])
            assert not np.array_equal(motif, motif[:, ::-1])
            assert th != tw or not np.array_equal(motif, motif.T)
            assert 3 <= len(np.unique(info['restored'])) <= 5
            # Check the exact motif/carrier footprint and labels tile by tile.
            _, clean_tiles = _split_tiles(out)
            for r in range(nr):
                for c in range(nc):
                    assert np.array_equal(clean_tiles[r, c] == info['carrier'], ~motif)
                    assert np.all(clean_tiles[r, c][motif] == info['restored'][r, c])
                    assert np.any((tiles[r, c] == 0) & motif)
                    assert np.any((tiles[r, c] == 0) & ~motif)
                    if info['labels'][r, c] >= 0:
                        assert np.any(tiles[r, c] == info['carrier'])
            unknown = info['labels'] < 0
            hidden_count = int(unknown.sum())
            assert (hidden_count == 3 if split == 'train' else 6 <= hidden_count <= 8)
            assert any(len(o) == 4 and sum(unknown[p] for p in o) == 3
                       for o in _orbits(nr, nc, *info['axes'])), "both folds need a diagonal witness"
            assert not np.array_equal(_nearest_labels(info['labels']), info['restored'])
            assert not np.array_equal(_pixel_mirror_shortcut(inp, info), out)
            if split == 'train':
                geometry_kinds.add(info['axes'] == (nr - 1, nc - 1))
                train_motifs.add(_key(motif))
    assert len(conventions) == 1, "shared palette roles and tile size"
    assert geometry_kinds == {False, True}, "centered and offset folds are demonstrated"
    assert len(train_motifs) >= 2, "motif must affect the answer"


def self_test() -> dict:
    """Bounded 42-seed diversity audit, API counts, and semantic negative tests."""
    shapes, tile_shapes, motifs, outputs, axes, palettes, densities = (set() for _ in range(7))
    fingerprints, blank_counts, fringe_repairs = set(), set(), 0
    for seed in range(42):
        p = generate_puzzle(seed)
        assert _fingerprint(p) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                info = _analyze(inp)
                tiles, motif = info['tiles'], info['motif']
                nr, nc, th, tw = tiles.shape
                shapes.add(inp.shape)
                tile_shapes.add((th, tw))
                motifs.add(_key(motif))
                outputs.add(_key(out))
                densities.add((motif.size, int(motif.sum())))
                palettes.add((info['separator'], info['carrier']))
                a, b = info['axes']
                axes.add((int(np.sign(a - nr + 1)), int(np.sign(b - nc + 1)), a % 2, b % 2))
                blank_counts.add(int(np.all(tiles == 0, axis=(2, 3)).sum()))
                for r, c in np.argwhere(info['labels'] < 0):
                    fringe_repairs += int(not 0 <= a - r < nr or not 0 <= b - c < nc)
                # Explicitly compare both parent ablations, including footprint.
                nearest = _render(motif, _nearest_labels(info['labels']), info['carrier'], info['separator'])
                assert not np.array_equal(nearest, out)
                assert not np.array_equal(_pixel_mirror_shortcut(inp, info), out)
                assert np.array_equal(solve(inp.tolist()), out)
    assert len(fingerprints) == 42
    assert tile_shapes == set(_TILE_SHAPES)
    assert len(shapes) >= 25 and len(motifs) >= 80 and len(outputs) >= 190
    assert len(palettes) >= 25 and len(densities) >= 12
    assert len({x[:2] for x in axes}) >= 7 and len({x[2:] for x in axes}) == 4
    assert fringe_repairs > 0 and max(blank_counts) >= 3
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(900 + 10 * nt + ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Negative tests cover a wrong restored accent, wrong restored carrier,
    # and modification of an originally visible pixel.
    for role in ('accent', 'carrier', 'visible'):
        p = generate_puzzle(73)
        pair = p['test'][0]
        info = _analyze(pair['input'])
        eligible = pair['input'] != 0 if role == 'visible' else pair['input'] == 0
        if role != 'visible':
            eligible &= (pair['output'] == info['carrier']) if role == 'carrier' else (pair['output'] != info['carrier'])
        point = tuple(np.argwhere(eligible)[0])
        pair['output'][point] = (int(pair['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"validator accepted corrupted {role}")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(fingerprints),
                grid_shapes=len(shapes), tile_shapes=sorted(tile_shapes),
                local_motifs=len(motifs), outputs=len(outputs),
                ordered_palette_roles=len(palettes),
                density_pairs=len(densities), signed_axis_parities=len(axes),
                fringe_color_repairs=fringe_repairs, blank_tile_counts=sorted(blank_counts))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
