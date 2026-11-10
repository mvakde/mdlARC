"""Restoration stencils: recover a mirrored textile, then print by material.

The key's earlier rows win only where their selected textile color occurs.
Corner guides and all other hidden cells must first be reconstructed from the
same cloth's horizontal/vertical reflection orbits. No file I/O or global state.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid bounds"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "color bounds"
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _corners(box, diagonal):
    r, s, c, d = box  # exclusive upper bounds
    if diagonal == 0:
        return {(r, c), (r, c+1), (r+1, c),
                (s-1, d-1), (s-1, d-2), (s-2, d-1)}
    return {(r, d-1), (r, d-2), (r+1, d-1),
            (s-1, c), (s-1, c+1), (s-2, c)}


def _orbits(shape):
    h, w = shape
    rr, cc = np.indices(shape)
    return np.minimum(rr, h-1-rr)*((w+1)//2) + np.minimum(cc, w-1-cc)


def _recover(cloth, unknown):
    """Recover cellwise using all surviving H/V/HV mates; reject conflicts."""
    ids = _orbits(cloth.shape)
    n = int(ids.max()) + 1
    lo = np.full(n, 10, dtype=np.int16)
    hi = np.full(n, -1, dtype=np.int16)
    np.minimum.at(lo, ids[~unknown], cloth[~unknown])
    np.maximum.at(hi, ids[~unknown], cloth[~unknown])
    assert np.all(hi >= 0), "a whole reflection orbit is hidden"
    assert np.array_equal(lo, hi), "visible mirror counterparts disagree"
    return hi[ids].astype(np.int8)


def _parse_canonical(a):
    """Canonical key: vertical cue/sample rows above a one-row blank seam."""
    candidates = []
    for n in range(2, 5):
        if a.shape[0] <= n+4 or a.shape[1] < 5:
            continue
        hole = int(a[n, 0])
        if not np.all(a[n] == hole):
            continue
        columns = [np.flatnonzero(row != hole) for row in a[:n]]
        if any(len(cols) != 2 for cols in columns):
            continue
        col = int(columns[0][0])
        if not all(np.array_equal(cols, [col, col+1]) for cols in columns):
            continue
        cues = list(map(int, a[:n, col]))
        targets = list(map(int, a[:n, col+1]))
        if len(set(cues)) != n or set(cues) & set(targets):
            continue
        cloth = a[n+1:]
        boxes, markers, diagonals = [], [], []
        valid = True
        for cue in cues:
            coords = {tuple(map(int, rc)) for rc in np.argwhere(cloth == cue)}
            if len(coords) != 6:
                valid = False
                break
            r, s = min(p[0] for p in coords), max(p[0] for p in coords)+1
            c, d = min(p[1] for p in coords), max(p[1] for p in coords)+1
            h, w = s-r, d-c
            if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
                valid = False
                break
            # Recognize local Ls, separately from the construction helper.
            local = {(x-r, y-c) for x, y in coords}
            pats = (
                {(0, 0), (0, 1), (1, 0), (h-1, w-1), (h-1, w-2), (h-2, w-1)},
                {(0, w-1), (0, w-2), (1, w-1), (h-1, 0), (h-1, 1), (h-2, 0)},
            )
            ds = [i for i, pat in enumerate(pats) if local == pat]
            if len(ds) != 1:
                valid = False
                break
            boxes.append((r, s, c, d))
            markers.append(coords)
            diagonals.append(ds[0])
        if not valid or len(set(boxes)) != n:
            continue
        unknown = np.zeros(cloth.shape, dtype=bool)
        cover = np.zeros((n, *cloth.shape), dtype=bool)
        for i, (r, s, c, d) in enumerate(boxes):
            cover[i, r:s, c:d] = True
        unknown = cover.any(axis=0)
        guide = np.isin(cloth, cues)
        if not np.array_equal(unknown, (cloth == hole) | guide):
            continue
        if not np.any(cloth == hole):
            continue
        visible_colors = set(map(int, np.unique(cloth[~unknown])))
        if not 2 <= len(visible_colors) <= 4 or not set(targets) <= visible_colors:
            continue
        if hole in visible_colors or set(cues) & visible_colors:
            continue
        try:
            restored = _recover(cloth, unknown)
        except AssertionError:
            continue
        candidates.append(dict(n=n, hole=hole, key_col=col, cloth=cloth,
                               cues=cues, targets=targets, boxes=boxes,
                               markers=markers, diagonals=diagonals, cover=cover,
                               unknown=unknown, restored=restored))
    assert len(candidates) == 1, "not a unique canonical textile/key"
    return candidates[0]


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for turn in range(4):
        canonical = np.rot90(a, turn)
        try:
            record = _parse_canonical(canonical)
        except AssertionError:
            continue
        candidates.append((turn, canonical, record))
    assert len(candidates) == 1, "key orientation is ambiguous or invalid"
    return candidates[0]


def _prediction(canonical, record, mode="normal"):
    """Visit each pixel: first eligible key row wins. No generation metadata."""
    full = record["restored"]
    painted = full.copy()
    indices = list(range(record["n"]))
    if mode == "reverse":
        indices.reverse()
    for r, c in np.argwhere(record["unknown"]):
        r, c = int(r), int(c)
        for i in indices:
            if not record["cover"][i, r, c]:
                continue
            if mode == "solid" or full[r, c] == record["targets"][i]:
                painted[r, c] = record["cues"][i]
                break
            if mode == "opaque":
                break
    output = canonical.copy()
    output[record["n"]+1:] = painted
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive geometry, reflected substrate, eligibility and priority from input."""
    turn, canonical, record = _parse(input_grid)
    return np.rot90(_prediction(canonical, record), -turn).copy()


def _features(canonical, rec):
    n, full, cover, hidden = rec["n"], rec["restored"], rec["cover"], rec["unknown"]
    normal = _prediction(canonical, rec)
    texture = canonical.copy()
    texture[n+1:] = full
    eligible = cover & (full[None] == np.asarray(rec["targets"])[:, None, None])
    winner = np.full(full.shape, -1, dtype=np.int8)
    for i in range(n-1, -1, -1):
        winner[eligible[i]] = i
    h, w = full.shape
    diagonal_only = hidden & hidden[::-1] & hidden[:, ::-1] & ~hidden[::-1, ::-1]
    sizes = [(s-r, d-c) for r, s, c, d in rec["boxes"]]
    return dict(
        n=n, shape=canonical.shape, cloth_shape=full.shape,
        palette=tuple(map(int, np.unique(full))), hole=rec["hole"],
        boxes=tuple(rec["boxes"]), sizes=tuple(sizes),
        diagonals=tuple(rec["diagonals"]),
        edge=any(r == 0 or s == h or c == 0 or d == w for r, s, c, d in rec["boxes"]),
        thin=any(min(size) == 2 for size in sizes),
        crossing_h=any(2*r < h-1 < 2*(s-1) for r, s, c, d in rec["boxes"]),
        crossing_v=any(2*c < w-1 < 2*(d-1) for r, s, c, d in rec["boxes"]),
        overlap=int(np.count_nonzero(cover.sum(axis=0) > 1)),
        diagonal_only=int(diagonal_only.sum()),
        printed=int(np.count_nonzero(winner >= 0)),
        unprinted_hidden=int(np.count_nonzero(hidden & (winner < 0))),
        removed_guides=sum(normal[n+1:][r, c] != rec["cues"][i]
                           for i, cells in enumerate(rec["markers"]) for r, c in cells),
        all_rows_visible=all(np.any(winner == i) for i in range(n)),
        priority=not np.array_equal(normal, _prediction(canonical, rec, "reverse")),
        transparency=not np.array_equal(normal, _prediction(canonical, rec, "opaque")),
        needs_A=not np.array_equal(normal, _prediction(canonical, rec, "solid")),
        needs_B=not np.array_equal(normal, texture),
        reflected_output=bool(np.array_equal(normal[n+1:], normal[n+1:][::-1])
                              and np.array_equal(normal[n+1:], normal[n+1:][:, ::-1])),
    )


def _pattern(rng, h, w, palette):
    bh, bw = (h+1)//2, (w+1)//2
    style = int(rng.integers(4))
    if style == 0:
        base = rng.choice(palette, (bh, bw))
    elif style == 1:
        small = rng.choice(palette, ((bh+1)//2, (bw+1)//2))
        base = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:bh, :bw].copy()
    elif style == 2:
        rr, cc = np.indices((bh, bw))
        phase = rng.integers(len(palette), size=bh)
        base = np.asarray(palette)[(cc+phase[:, None]) % len(palette)]
    else:
        weights = rng.uniform(.5, 3, len(palette))
        base = rng.choice(palette, (bh, bw), p=weights/weights.sum())
    # Ensure every material has an orbit; all orbits retain a visible witness.
    places = rng.choice(base.size, len(palette), replace=False)
    base.flat[places] = palette
    rr, cc = np.indices((h, w))
    return base[np.minimum(rr, h-1-rr), np.minimum(cc, w-1-cc)].astype(np.int8)


def _box(rng, h, w, interior):
    margin = 1 if interior else 0
    hh = int(rng.integers(2, max(4, min(h-2*margin, (h*3)//4))+1))
    ww = int(rng.integers(2, max(4, min(w-2*margin, (w*3)//4))+1))
    if min(hh, ww) == 2 and max(hh, ww) < 4:
        ww = 4
    r = int(rng.integers(margin, h-margin-hh+1))
    c = int(rng.integers(margin, w-margin-ww+1))
    return r, r+hh, c, c+ww


def _geometry(rng, h, w, n, interior, kind):
    boxes = []
    if kind == "half":
        # Keep a whole opposite half as a direct source. This permits compact,
        # nested, staggered and chain-like overlaps without a crossing template.
        if rng.random() < .5:
            boxes = [_box(rng, h, w//2, interior) for _ in range(n)]
            if rng.random() < .5:
                boxes = [(r, s, w-d, w-c) for r, s, c, d in boxes]
        else:
            boxes = [_box(rng, h//2, w, interior) for _ in range(n)]
            if rng.random() < .5:
                boxes = [(h-s, h-r, c, d) for r, s, c, d in boxes]
    elif kind == "cross":
        # A vertical window on one side and a horizontal window on the other
        # axis conceal three members of some orbits, leaving the diagonal mate.
        a = int(rng.integers(0 if not interior else 1, max(2, h//2-2)))
        b = int(rng.integers(a+2, min(h//2, a+5)+1))
        c = int(rng.integers(0 if not interior else 1, max(2, w//2-2)))
        d = int(rng.integers(c+2, min(w//2, c+5)+1))
        # Slightly offset extents avoid colliding registration marks.
        boxes = [(max(int(interior), a-1), min(h-int(interior), h-a), c, d),
                 (a, b, max(int(interior), c-1), min(w-int(interior), w-c))]
        if rng.random() < .5:
            boxes = [(h-s, h-r, c0, d0) for r, s, c0, d0 in boxes]
        if rng.random() < .5:
            boxes = [(r, s, w-d0, w-c0) for r, s, c0, d0 in boxes]
    boxes.extend(_box(rng, h, w, interior) for _ in range(n-len(boxes)))
    if len(set(boxes)) != n:
        return None
    markers, used = [], set()
    for box in boxes:
        choices = list(map(int, rng.permutation(2)))
        selected = None
        for diagonal in choices:
            cells = _corners(box, diagonal)
            if len(cells) == 6 and not cells & used:
                selected = cells
                break
        if selected is None:
            return None
        used.update(selected)
        markers.append(selected)
    hidden = np.zeros((h, w), dtype=bool)
    for r, s, c, d in boxes:
        hidden[r:s, c:d] = True
    ids = _orbits((h, w))
    visible_ids = np.unique(ids[~hidden])
    if len(visible_ids) != int(ids.max())+1:
        return None
    if not .07 <= hidden.mean() <= .63:
        return None
    return boxes, markers, hidden


def _make_pair(rng, n, turn, interior, edge_required, rich, easy, seen):
    for attempt in range(650):
        h, w = int(rng.integers(12, 26)), int(rng.integers(12, 31))
        kind = "half" if easy else ("cross" if rich else str(rng.choice(["half", "half", "cross", "free"])))
        geometry = _geometry(rng, h, w, n, interior, kind)
        if geometry is None:
            continue
        boxes, markers, hidden = geometry
        if edge_required and not any(r == 0 or s == h or c == 0 or d == w for r, s, c, d in boxes):
            continue
        k = int(rng.integers(2, 5))
        colors = list(map(int, rng.permutation(10)))
        hole, palette, cues = colors[0], colors[1:k+1], colors[k+1:k+1+n]
        full = _pattern(rng, h, w, palette)
        targets = [palette[0], palette[0]]
        targets.extend(palette[(i-1) % k] for i in range(2, n))
        # Randomize table rank jointly with each complete stencil identity.
        order = list(map(int, rng.permutation(n)))
        boxes = [boxes[i] for i in order]
        markers = [markers[i] for i in order]
        targets = [targets[i] for i in order]
        col = int(rng.integers(w-1))
        inp = np.full((n+1+h, w), hole, dtype=np.int8)
        inp[:n, col] = cues
        inp[:n, col+1] = targets
        cloth = inp[n+1:]
        cloth[:] = full
        cloth[hidden] = hole
        for cue, cells in zip(cues, markers):
            for r, c in cells:
                cloth[r, c] = cue
        # Label generation uses the known completed textile and vector masks;
        # the public solver instead reconstructs it and chooses each pixel.
        output = inp.copy()
        painted = full.copy()
        for i in range(n-1, -1, -1):
            r, s, c, d = boxes[i]
            patch = painted[r:s, c:d]
            patch[full[r:s, c:d] == targets[i]] = cues[i]
        output[n+1:] = painted
        try:
            rec = _parse_canonical(inp)
        except AssertionError:
            continue
        feat = _features(inp, rec)
        if not (feat["all_rows_visible"] and feat["priority"] and feat["needs_A"]
                and feat["needs_B"] and feat["unprinted_hidden"] >= 3
                and feat["removed_guides"] > 0):
            continue
        if rich and not (feat["transparency"] and feat["diagonal_only"] >= 2):
            continue
        assert np.array_equal(_prediction(inp, rec), output)
        rotated_in = np.rot90(inp, turn).copy()
        if _key(rotated_in) in seen:
            continue
        return dict(input=rotated_in, output=np.rot90(output, turn).copy())
    raise RuntimeError("bounded construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one reproducible puzzle. All conventions are visible or fixed."""
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    if not _integer(num_train) or not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    seed = int(seed)
    rng = np.random.default_rng(seed if seed >= 0 else seed % (1 << 64))
    regime = str(rng.choice(["layers", "edge", "key_side"]))
    first_side = int(rng.integers(4))
    train_sides = [first_side, (first_side+1) % 4]
    test_sides = [(first_side+2) % 4, (first_side+3) % 4]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            training = split == "train"
            n = (2 if index == 0 else 3) if training else (4 if regime == "layers" else int(rng.integers(2, 5)))
            if training and index >= 2:
                n = int(rng.integers(2, 4 if regime == "layers" else 5))
            sides = train_sides if training else test_sides
            turn = int(rng.choice(sides)) if regime == "key_side" else int(rng.integers(4))
            rich = training and index == 1
            pair = _make_pair(rng, n, turn,
                              interior=regime == "edge" and training,
                              edge_required=regime == "edge" and not training,
                              rich=rich, easy=training and index == 0, seen=seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            turn, canonical, rec = _parse(a)
            prediction = _prediction(canonical, rec)
            assert np.array_equal(np.rot90(prediction, -turn), b), "incorrect label"
            out = np.rot90(b, turn)
            n, full, hidden = rec["n"], rec["restored"], rec["unknown"]
            assert np.array_equal(out[:n+1], canonical[:n+1]), "key changed"
            assert np.array_equal(out[n+1:][~hidden], canonical[n+1:][~hidden]), "visible cloth changed"
            assert not np.any(out[n+1:] == rec["hole"]), "unrestored hole"
            assert np.array_equal(full, full[::-1]) and np.array_equal(full, full[:, ::-1])
            assert not np.shares_memory(solve(a), a)
            feature = _features(canonical, rec)
            assert feature["all_rows_visible"] and feature["priority"]
            assert feature["needs_A"] and feature["needs_B"] and feature["unprinted_hidden"] >= 3
            assert feature["removed_guides"] > 0, "guide-removal behavior is not visible"
            feature["side"] = turn
            records[split].append(feature)
    tr, te = records["train"], records["test"]
    assert any(f["n"] == 2 for f in tr), "missing readable two-stencil demonstration"
    assert any(f["diagonal_only"] == 0 for f in tr), "missing direct-source demonstration"
    assert any(f["transparency"] and f["diagonal_only"] >= 2 for f in tr), "missing connecting-concept demonstration"
    # Check an actual visible shift, not a latent sampler regime flag.
    assert (max(f["n"] for f in te) > max(f["n"] for f in tr)
            or (not any(f["edge"] for f in tr) and all(f["edge"] for f in te))
            or not ({f["side"] for f in tr} & {f["side"] for f in te})), "no intended train/test shift"
    return True


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 streamed seeds, legal counts, mutations and equivariance."""
    signatures, shapes, layouts, sides, palettes, holes, counts = set(), set(), set(), set(), set(), set(), set()
    totals = dict(pairs=0, thin=0, edge=0, diagonal_only=0, transparency=0,
                  crossing_h=0, crossing_v=0, odd_cloth=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed in (0, 17, 41):
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"].copy()
                predicted = solve(a)
                assert np.array_equal(a, pair["input"]), "solver mutated input"
                assert np.array_equal(predicted, pair["output"])
                turn, canonical, rec = _parse(a)
                f = _features(canonical, rec)
                totals["pairs"] += 1
                for name in ("thin", "edge", "diagonal_only", "transparency", "crossing_h", "crossing_v"):
                    totals[name] += bool(f[name])
                totals["odd_cloth"] += any(x % 2 for x in f["cloth_shape"])
                shapes.add(a.shape)
                layouts.add((f["cloth_shape"], f["boxes"]))
                sides.add(turn)
                palettes.add(f["palette"])
                holes.add(f["hole"])
                counts.add(f["n"])
    assert len(signatures) == 42 and len(layouts) >= 150 and len(shapes) >= 70
    assert sides == {0, 1, 2, 3} and counts == {2, 3, 4} and holes == set(range(10))
    assert len(palettes) >= 60
    for name in ("thin", "edge", "diagonal_only", "transparency", "crossing_h", "crossing_v", "odd_cloth"):
        assert totals[name] >= 8, (name, totals)
    for nt in range(2, 11):
        for nq in range(1, min(nt-1, 4)+1):
            generate_puzzle(42000+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(613, 2, 1)
    pair = p["test"][0]
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
    permutation = np.random.default_rng(72).permutation(10).astype(np.int8)
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong label accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                layouts=len(layouts), palettes=len(palettes), **totals)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
