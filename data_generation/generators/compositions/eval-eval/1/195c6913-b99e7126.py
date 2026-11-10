"""Nested staircase engraving: a terrain map whose visited cells contain walks.

Only NumPy and the standard library are used. No file access or hidden generation
parameters are involved in inference. All sampling loops have finite limits.
"""
from __future__ import annotations

import hashlib
import itertools
import numpy as np


_SHAPES = ((2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4))
_REQUIRED = {"multiplicity", "bend_phase", "marker_phase", "tile_phase",
             "unit_phase", "outer_marker_phase", "edge_stop", "missing_path", "missing_offpath",
             "whole_tile", "cycling"}


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.uint8).tobytes()


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "noninteger cells"
    assert np.all((0 <= a) & (a <= 9)), "invalid colors"
    return a.astype(np.int8, copy=True)


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _walk(mask, start):
    """Cell-step inference walker. Directions are right, up, right, up, ... ."""
    p, vertical = start, False
    path, hits, legs = [], [], [0]
    while True:
        assert mask[p] and p not in path
        path.append(p)
        legs[-1] += 1
        q = (p[0]-1, p[1]) if vertical else (p[0], p[1]+1)
        if not _inside(mask.shape, q):
            end = "edge"
            break
        if mask[q]:
            p = q
            continue
        hits.append(q)
        q = (p[0], p[1]+1) if vertical else (p[0]-1, p[1])
        if not _inside(mask.shape, q) or not mask[q]:
            end = "blocked"
            break
        vertical = not vertical
        legs.append(0)
        p = q
    return {"path": tuple(path), "hits": tuple(hits), "legs": tuple(legs), "end": end}


def _runs_walk(mask, start):
    """Construction cross-check: maximal slices rather than single-cell turns."""
    current, axis = start, 0
    path, hits, legs = [], [], []
    for _ in range(sum(mask.shape)):
        dr, dc = ((0, 1), (-1, 0))[axis]
        leg = []
        r, c = current
        while _inside(mask.shape, (r, c)) and mask[r, c]:
            leg.append((r, c))
            r, c = r+dr, c+dc
        assert leg
        path.extend(leg)
        legs.append(len(leg))
        if not _inside(mask.shape, (r, c)):
            end = "edge"
            break
        hits.append((r, c))
        axis = 1-axis
        dr, dc = ((0, 1), (-1, 0))[axis]
        current = (leg[-1][0]+dr, leg[-1][1]+dc)
        if not _inside(mask.shape, current) or not mask[current]:
            end = "blocked"
            break
    else:
        raise AssertionError("walk did not terminate")
    return {"path": tuple(path), "hits": tuple(hits), "legs": tuple(legs), "end": end}


def _intervals(values):
    groups = []
    for x in map(int, values):
        if groups and x == groups[-1][-1]+1:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def _origins(mask, rows, cols, supplied):
    h, w = mask.shape
    return [(y, x) for y in range(rows-h+1) for x in range(cols-w+1)
            if all(0 <= r-y < h and 0 <= c-x < w and mask[r-y, c-x]
                   for r, c in supplied)]


def _tile_slice(info, p):
    h, w, t, _, _ = info["geometry"]
    return (slice(4+t+p[0]*(h+t), 4+t+p[0]*(h+t)+h),
            slice(t+p[1]*(w+t), t+p[1]*(w+t)+w))


def _decode_canonical(g):
    """Decode a top program strip and its lattice, with no split information."""
    assert g.shape[0] >= 11 and g.shape[1] >= 12
    divider = int(g[0, 0])
    assert np.all(g[0] == divider) and np.all(g[3:5] == divider)
    assert np.array_equal(g[1], g[2])
    groups = _intervals(np.flatnonzero(g[1] != divider))
    assert 3 <= len(groups) <= 6 and all(len(s) == 2 for s in groups)
    assert all(g[1, s[0]] == g[1, s[1]] for s in groups)
    assert all(b[0]-a[-1] == 2 for a, b in zip(groups[:-2], groups[1:-1]))
    assert groups[-1][0]-groups[-2][-1] >= 4
    assert groups[0][0] >= 1 and groups[-1][-1] < g.shape[1]-1
    word = tuple(int(g[1, s[0]]) for s in groups[:-1])
    marker = int(g[1, groups[-1][0]])
    assert 2 <= len(set(word)) <= 4 and marker not in word
    body = g[4:]
    rg = _intervals(np.flatnonzero(np.all(body == divider, axis=1)))
    cg = _intervals(np.flatnonzero(np.all(body == divider, axis=0)))
    assert len(rg) >= 3 and len(cg) >= 3
    assert rg[0][0] == cg[0][0] == 0
    assert rg[-1][-1] == body.shape[0]-1 and cg[-1][-1] == body.shape[1]-1
    thicknesses = {len(s) for s in rg+cg}
    assert len(thicknesses) == 1
    t = thicknesses.pop()
    assert t in (1, 2)
    heights = {b[0]-a[-1]-1 for a, b in zip(rg, rg[1:])}
    widths = {b[0]-a[-1]-1 for a, b in zip(cg, cg[1:])}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert (h, w) in _SHAPES
    rows, cols = len(rg)-1, len(cg)-1
    assert rows >= h and cols >= w
    geometry = (h, w, t, rows, cols)
    info = {"geometry": geometry, "divider": divider, "word": word,
            "marker": marker, "canonical": g}
    types = {}
    for r in range(rows):
        for c in range(cols):
            tile = g[_tile_slice(info, (r, c))]
            assert divider not in tile and marker not in tile
            types.setdefault(_key(tile), [tile.copy(), []])[1].append((r, c))
    assert len(types) == 2
    ordinary_record, glyph_record = sorted(types.values(), key=lambda v: -len(v[1]))
    ordinary, ordinary_pos = ordinary_record
    glyph, supplied = glyph_record
    assert len(ordinary_pos) > len(supplied) >= 2
    oc = set(map(int, np.unique(ordinary)))
    assert len(oc) == 2 and not (oc & (set(word) | {marker, divider}))
    extra = set(map(int, np.unique(glyph)))-oc
    assert word[0] in extra and len(extra) == 2
    terrain = (extra-{word[0]}).pop()
    assert terrain not in set(word) | {divider, marker}
    mask = (glyph == terrain) | (glyph == word[0])
    starts = np.argwhere(glyph == word[0])
    assert len(starts) == 1 and 4 <= int(mask.sum()) < h*w
    assert len(np.unique(glyph[~mask])) == 1 and int(glyph[~mask][0]) in oc
    start = tuple(map(int, starts[0]))
    r, c = start
    assert c == 0 or not mask[r, c-1]
    origins = _origins(mask, rows, cols, supplied)
    assert len(origins) == 1, "ambiguous stencil translation"
    origin = origins[0]
    local_supplied = {(r-origin[0], c-origin[1]) for r, c in supplied}
    active = set(map(tuple, np.argwhere(mask)))
    assert local_supplied < active
    record = _walk(mask, start)
    assert len(record["path"]) >= 3 and active-set(record["path"])
    info.update(ordinary=ordinary, glyph=glyph, terrain=terrain, mask=mask,
                start=start, origin=origin, supplied=local_supplied,
                active=active, record=record)
    return info


def _infer(grid):
    candidates = []
    for k in range(4):
        canonical = np.rot90(grid, k).copy()
        try:
            info = _decode_canonical(canonical)
        except (AssertionError, IndexError, ValueError):
            continue
        info["rotation"] = k
        candidates.append(info)
    assert len(candidates) == 1, "program side must identify exactly one orientation"
    return candidates[0]


def _render(info, *, model="correct"):
    """Construct the two-scale engraving from decoded semantic objects."""
    out = info["canonical"].copy()
    out[:4] = info["divider"]
    y, x = info["origin"]
    path, hits = info["record"]["path"], info["record"]["hits"]
    word, glyph = info["word"], info["glyph"]
    n = len(path)
    if model == "deduplicate":
        word = tuple(dict.fromkeys(word))
    active = info["active"]
    if model == "visible_only":
        active = info["supplied"]
    for r, c in active:
        target = _tile_slice(info, (y+r, x+c))
        if model == "ink_only" and (r, c) not in info["supplied"]:
            tile = out[target]
            tile[info["mask"]] = glyph[info["mask"]]
        else:
            out[target] = glyph
    if model == "completion_only":
        return out
    leg_id = []
    leg_offset = []
    for j, length in enumerate(info["record"]["legs"]):
        leg_id.extend([j]*length)
        leg_offset.extend(range(length))
    for j, (r, c) in enumerate(path):
        if model == "visible_only" and (r, c) not in active:
            continue
        target = _tile_slice(info, (y+r, x+c))
        tile = out[target]
        for i, p in enumerate(path):
            phase = j*n+i
            if model == "tile_reset":
                phase = i
            elif model == "unit_phase":
                phase = j+i
            elif model == "bend_reset":
                phase = j*n+leg_offset[i]
            elif model == "outer_marker_consumes":
                phase = j*n+i+leg_id[j]
            elif model == "marker_consumes":
                phase = j*(n+len(hits))+i+leg_id[i]
            tile[p] = word[phase % len(word)]
        for p in hits:
            tile[p] = info["marker"]
    for r, c in hits:
        out[_tile_slice(info, (y+r, x+c))] = info["marker"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover orientation, roles, map registration and both walks from input."""
    info = _infer(_grid(input_grid))
    return np.rot90(_render(info), -info["rotation"]).copy()


def _evidence(info):
    out = _render(info)
    models = {"deduplicate": "multiplicity", "tile_reset": "tile_phase",
              "unit_phase": "unit_phase", "bend_reset": "bend_phase",
              "marker_consumes": "marker_phase", "ink_only": "whole_tile",
              "outer_marker_consumes": "outer_marker_phase"}
    evidence = {label for model, label in models.items()
                if not np.array_equal(out, _render(info, model=model))}
    missing = info["active"]-info["supplied"]
    path = set(info["record"]["path"])
    if missing & path:
        evidence.add("missing_path")
    if missing-path:
        evidence.add("missing_offpath")
    n = len(path)
    if n*n >= 2*len(info["word"]):
        evidence.add("cycling")
    if info["record"]["end"] == "edge":
        r, c = info["record"]["path"][-1]
        # At an edge exit, a route must not take an available next turn.
        q = (r-1, c) if len(info["record"]["legs"]) % 2 else (r, c+1)
        if _inside(info["mask"].shape, q) and info["mask"][q]:
            evidence.add("edge_stop")
    return evidence


def _word(rng, maximum, witness):
    length = int(rng.integers(3 if witness else 2, maximum+1))
    distinct = int(rng.integers(2, min(4, length-(1 if witness else 0))+1))
    labels = list(range(distinct))+list(map(int, rng.integers(distinct, size=length-distinct)))
    rng.shuffle(labels)
    return tuple(labels)


def _geometry(rng, shape=None):
    if shape is None:
        shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
    h, w = shape
    t = int(rng.integers(1, 3))
    # Four header rows leave at most 26 body rows. Width has the full budget.
    nr = int(rng.integers(h, (26-t)//(h+t)+1))
    minimum_cols = max(w, (15-t+w+t-1)//(w+t))
    nc = int(rng.integers(minimum_cols, (30-t)//(w+t)+1))
    return h, w, t, nr, nc


def _fragment(rng, mask, origin, rows, cols, path, sparse):
    active = [tuple(map(int, p)) for p in np.argwhere(mask)]
    n = len(active)
    low = 2 if sparse else max(2, (n+1)//2)
    high = min((n-1)//2 if sparse else n-2, (rows*cols-1)//2)
    if high < low:
        return None
    # At most C(12,6)=924 subsets at one count, never a 2^canvas search. Try a bounded
    # random sample first, then exhaust the finite admissible subsets if needed.
    for _ in range(40):
        count = int(rng.integers(low, high+1))
        supplied = {active[int(i)] for i in rng.choice(n, count, replace=False)}
        if not ((set(active)-supplied) & set(path)) or not (set(active)-supplied-set(path)):
            continue
        absolute = [(r+origin[0], c+origin[1]) for r, c in supplied]
        if _origins(mask, rows, cols, absolute) == [origin]:
            return supplied
    # Avoid retaining a large subset array; enumerate in constant auxiliary space.
    sizes = list(range(low, high+1))
    rng.shuffle(sizes)
    for count in sizes:
        for chosen in itertools.combinations(active, count):
            supplied = set(chosen)
            if not ((set(active)-supplied) & set(path)) or not (set(active)-supplied-set(path)):
                continue
            absolute = [(r+origin[0], c+origin[1]) for r, c in supplied]
            if _origins(mask, rows, cols, absolute) == [origin]:
                return supplied
    return None


def _assemble(geometry, palette, labels, ordinary_labels, mask, start, origin, supplied):
    h, w, t, rows, cols = geometry
    divider, base, secondary, terrain, marker = map(int, palette[:5])
    word = tuple(int(palette[5+v]) for v in labels)
    glyph = np.where(mask, terrain, base).astype(np.int8)
    glyph[start] = word[0]
    ordinary = np.where(ordinary_labels, secondary, base).astype(np.int8)
    g = np.full((4+rows*(h+t)+t, cols*(w+t)+t), divider, np.int8)
    for j, color in enumerate(word):
        g[1:3, 1+3*j:3+3*j] = color
    g[1:3, -3:-1] = marker
    info = {"geometry": geometry}
    for r in range(rows):
        for c in range(cols):
            p = (r-origin[0], c-origin[1])
            g[_tile_slice(info, (r, c))] = glyph if p in supplied else ordinary
    # The construction oracle uses maximal-run tracing and known parameters.
    info.update(canonical=g, divider=divider, ordinary=ordinary, glyph=glyph,
                marker=marker, terrain=terrain, word=word, mask=mask, start=start,
                origin=origin, supplied=supplied,
                active=set(map(tuple, np.argwhere(mask))), record=_runs_walk(mask, start))
    return info


def _make_pair(rng, sparse, witness=False):
    for attempt in range(240):
        # A finite fallback keeps the informative 3-leg geometry available. The
        # random branch samples masks, rather than a letter or shape vocabulary.
        fallback = attempt >= 200
        geometry = _geometry(rng, (3, 4) if fallback else None)
        h, w, t, rows, cols = geometry
        width = cols*(w+t)+t
        maximum = min(5, (width-6)//3)
        labels = _word(rng, maximum, witness)
        palette = list(map(int, rng.permutation(10)))
        if fallback:
            mask = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0]], bool)
            starts = [(2, 0)]
            labels = (0, 0, 1)
        else:
            mask = np.zeros((h, w), bool)
            count = int(rng.integers(5, min(h*w, 13)))
            mask.flat[rng.choice(h*w, count, replace=False)] = True
            starts = [(r, c) for r in range(h) for c in range(w-1)
                      if mask[r, c] and (c == 0 or not mask[r, c-1])]
            rng.shuffle(starts)
        for start in starts:
            record = _runs_walk(mask, start)
            path = record["path"]
            if len(path) < 3 or not set(map(tuple, np.argwhere(mask)))-set(path):
                continue
            if witness and (len(record["legs"]) < 3 or len(path) % len(labels) == 0):
                continue
            origin = (int(rng.integers(rows-h+1)), int(rng.integers(cols-w+1)))
            supplied = _fragment(rng, mask, origin, rows, cols, path, sparse)
            if supplied is None:
                continue
            ordinary = rng.integers(0, 2, size=(h, w), dtype=np.int8).astype(bool)
            # Keep both ordinary colors, including a non-ink replacement witness.
            ordinary[0, 0] = False
            ordinary[-1, -1] = True
            candidates = np.argwhere(~mask)
            ordinary[tuple(candidates[0])] = True
            if ordinary.all():
                ordinary[tuple(np.argwhere(mask)[0])] = False
            info = _assemble(geometry, palette, labels, ordinary, mask, start, origin, supplied)
            evidence = _evidence(info)
            if witness and not _REQUIRED <= evidence:
                continue
            k = int(rng.integers(4))
            inp, out = np.rot90(info["canonical"], k).copy(), np.rot90(_render(info), k).copy()
            try:
                decoded = _infer(inp)
            except AssertionError:
                continue
            assert decoded["record"] == record
            assert np.array_equal(solve(inp), out)
            return {"input": inp, "output": out}
    raise RuntimeError("bounded map/fragment construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed):
        raise ValueError("seed must be an integer")
    if (not _integer(num_train) or not _integer(num_test) or
            not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("require integer counts: 2..10 train, 1..4 test, train > test")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle, used, train_masks = {"train": [], "test": []}, set(), set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            for _ in range(24):
                pair = _make_pair(rng, sparse=split == "test", witness=split == "train" and i == 0)
                info = _infer(pair["input"])
                mask_key = _key(info["mask"])
                if _key(pair["input"]) in used:
                    continue
                if split == "train" and i == 1 and mask_key in train_masks:
                    continue
                if split == "test" and mask_key in train_masks:
                    continue
                break
            else:
                raise RuntimeError("bounded distinct-map sampling exhausted")
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
            if split == "train":
                train_masks.add(mask_key)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence, masks = set(), set(), {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _infer(inp)
            assert info["record"] == _runs_walk(info["mask"], info["start"])
            expected = np.rot90(_render(info), -info["rotation"])
            assert np.array_equal(expected, out), "incorrect nested engraving"
            canonical_out = np.rot90(out, info["rotation"])
            assert np.all(canonical_out[:4] == info["divider"])
            y, x = info["origin"]
            active = {(y+r, x+c) for r, c in info["active"]}
            hits = {(y+r, x+c) for r, c in info["record"]["hits"]}
            assert not active & hits
            h, w, t, rows, cols = info["geometry"]
            # Check whole untouched tiles explicitly, including inactive holes.
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in active | hits:
                        assert np.array_equal(canonical_out[_tile_slice(info, (r, c))], info["ordinary"])
            coverage = _evidence(info)
            assert {"missing_path", "missing_offpath", "whole_tile"} <= coverage
            assert not np.array_equal(_render(info), _render(info, model="visible_only"))
            assert not np.array_equal(_render(info), _render(info, model="completion_only"))
            n = len(info["supplied"])
            a = len(info["active"])
            if split == "train":
                assert 2*n >= a
                evidence |= coverage
            else:
                assert 2*n < a
            masks[split].add(_key(info["mask"]))
    assert _REQUIRED <= evidence, _REQUIRED-evidence
    assert len(masks["train"]) >= 2
    assert not masks["train"] & masks["test"]


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role])
                digest.update(bytes(g.shape))
                digest.update(g.astype(np.uint8).tobytes())
    return digest.hexdigest()


def self_test() -> dict:
    """42 puzzles streamed through bounded aggregate sets; nothing is saved."""
    signatures, shapes, masks, geometry, words, rotations, origins, fragments = (set() for _ in range(8))
    ends, legs, residues, palettes = set(), set(), set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _infer(pair["input"])
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                pairs += 1
                shapes.add(pair["input"].shape)
                masks.add(_key(info["mask"]))
                geometry.add(info["geometry"])
                labels = {}
                words.add(tuple(labels.setdefault(c, len(labels)) for c in info["word"]))
                rotations.add(info["rotation"])
                origins.add(info["origin"])
                fragments.add((info["mask"].shape, tuple(sorted(info["supplied"]))))
                ends.add(info["record"]["end"])
                legs.add(len(info["record"]["legs"]))
                residues.add(len(info["record"]["path"]) % len(info["word"]))
                palettes.add(tuple(map(int, np.unique(pair["input"]))))
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 30 and len(masks) >= 60 and len(geometry) >= 35
    assert {value[:2] for value in geometry} == set(_SHAPES)
    assert len(words) >= 10 and rotations == {0, 1, 2, 3}
    assert len(origins) >= 8 and len(fragments) >= 70 and len(palettes) >= 20
    assert ends == {"edge", "blocked"} and {1, 2, 3} <= legs and {0, 1, 2} <= residues
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(9173, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (3.0, 1), (True, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Pin both horizontal and vertical exits with terrain around the corner.
    for rows, start, endpoint in ((["001", "011", "110"], (2, 0), (1, 2)),
                                  (["011", "110", "110"], (2, 0), (0, 1))):
        mask = np.array([[c == "1" for c in row] for row in rows])
        record = _walk(mask, start)
        assert record == _runs_walk(mask, start)
        assert record["end"] == "edge" and record["path"][-1] == endpoint
    # Output mutations must fail the complete semantic validator.
    puzzle = generate_puzzle(0)
    for model in ("deduplicate", "tile_reset", "unit_phase", "bend_reset",
                  "marker_consumes", "outer_marker_consumes", "ink_only", "visible_only", "completion_only"):
        pair = puzzle["train"][0]
        info = _infer(pair["input"])
        bad = np.rot90(_render(info, model=model), -info["rotation"]).copy()
        assert not np.array_equal(bad, pair["output"]), model
        altered = {s: list(puzzle[s]) for s in ("train", "test")}
        altered["train"][0] = {"input": pair["input"], "output": bad}
        try:
            validate_generated_puzzle(altered)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {model}")
    # Exercise the disclosed fallback at dense and sparse clue levels directly.
    mask = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [1, 1, 0, 0]], bool)
    ordinary = np.zeros((3, 4), bool)
    ordinary[0, 1] = True
    for supplied in ({(0, 3), (2, 0), (1, 1), (1, 2)}, {(0, 3), (2, 0)}):
        f = _assemble((3, 4, 1, 6, 5), tuple(range(10)), (0, 0, 1),
                      ordinary, mask, (2, 0), (1, 1), supplied)
        assert _REQUIRED <= _evidence(f)
        assert np.array_equal(solve(f["canonical"]), _render(f))
        path = f["record"]["path"]
        # First two nested walks spell AABAA and BAABA, respectively.
        for j, expected in ((0, (5, 5, 6, 5, 5)), (1, (6, 5, 5, 6, 5))):
            r, c = path[j]
            tile = _render(f)[_tile_slice(f, (1+r, 1+c))]
            assert tuple(int(tile[p]) for p in path) == expected
    # Smallest canonical height and a strongly rectangular coarse lattice.
    mask = np.array([[1, 1, 1], [1, 1, 0]], bool)
    f = _assemble((2, 3, 1, 2, 5), tuple(range(10)), (0, 0, 1),
                  np.array([[0, 0, 0], [0, 0, 1]], bool), mask, (1, 0),
                  (0, 1), {(0, 2), (1, 0)})
    assert f["canonical"].shape == (11, 21)
    assert np.array_equal(solve(f["canonical"]), _render(f))
    # Translation ambiguity and a damaged divider must not decode.
    ambiguous = _assemble((2, 3, 1, 4, 6), tuple(range(10)), (0, 0, 1),
                          np.array([[0, 0, 0], [0, 0, 1]], bool), mask,
                          (1, 0), (1, 1), {(0, 0), (0, 1)})["canonical"]
    damaged = puzzle["train"][0]["input"].copy()
    info = _infer(damaged)
    damaged = info["canonical"].copy()
    damaged[4, 0] = (info["divider"]+1) % 10
    for bad_input in (ambiguous, damaged):
        try:
            solve(bad_input)
        except AssertionError:
            pass
        else:
            raise AssertionError("invalid input decoded")
    # The complete scene, including its program, is equivariant to quarter turns
    # and bijective palette changes. Neither check has access to a sampling seed.
    palette = np.array([8, 5, 9, 0, 7, 1, 4, 6, 2, 3], dtype=np.int8)
    for pair in puzzle["train"]:
        for k in range(4):
            inp = np.rot90(pair["input"], k)
            expected = np.rot90(pair["output"], k)
            assert np.array_equal(solve(inp), expected)
            assert np.array_equal(solve(palette[inp]), palette[expected])
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "masks": len(masks), "geometries": len(geometry),
            "word_patterns": len(words), "origins": len(origins),
            "fragments": len(fragments), "palettes": len(palettes)}
