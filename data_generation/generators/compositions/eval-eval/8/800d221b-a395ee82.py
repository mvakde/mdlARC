"""Docking parliament: ballot panels become glyphs whose contacts cast votes.

Only NumPy and the standard library are used. No evidence files are needed at
runtime. All decoding is from the supplied input; generation records are used
only to construct the expected output, then checked by pixel reconstruction.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PANEL_SHAPES = ((1, 3), (3, 1), (3, 3))
_SELF_TEST_SEEDS = (
    0, 224762774, 939281383, 619907143, 681138404, 195876686,
    501975959, 95710337, 915881094, 705458712, 929142819, 864429865,
    120243289, 557543589, 268326921, 511579238, 950926777, 831179161,
    756546418, 981913583, 356392847, 318774863, 169471086, 828498692,
    687177067, 890833487, 818304866, 385772109, 175797212, 668438903,
    502292880, 269828384, 750190956, 683534540, 112925862, 480957738,
    304740862, 523495912, 650683011, 468010957, 945841000, 201550291,
)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(cells)
    return result


def _contacts(mask):
    """Number of opposing mask pixels that dock across a macro-cell side."""
    vertical = int(np.count_nonzero(mask[0] & mask[-1]))
    horizontal = int(np.count_nonzero(mask[:, 0] & mask[:, -1]))
    return vertical, horizontal


def _tight_connected(mask):
    return (all((mask[0].any(), mask[-1].any(), mask[:, 0].any(),
                 mask[:, -1].any())) and len(_components(mask)) == 1)


def _asymmetric(mask):
    alternatives = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        alternatives += [mask.T, np.flipud(mask).T, np.fliplr(mask).T,
                         np.rot90(mask, 2).T]
    return all(not np.array_equal(mask, other) for other in alternatives)


def _glyph(rng, high_axis=None, previous=None, informative=False, unequal=False):
    """Random erosion, retaining connectivity, tightness and visible ports.

    The two teaching glyphs have a narrow end and a broad body. Their exact
    interior, dimensions, narrow-end position and reflections still vary.
    This directly constructs a contact-ratio witness without an unbounded
    rejection search or a rendered motif bank.
    """
    if high_axis is not None:
        options = []
        for body_h in (4, 5):
            for body_w in (3, 4, 5):
                dims = (body_h, body_w) if high_axis == 1 else (body_w, body_h)
                if previous is None or all(a != b for a, b in zip(dims, previous)):
                    if not informative or dims[0] != dims[1]:
                        options.append((body_h, body_w))
        h, w = _pick(rng, options)
        mask = np.ones((h, w), dtype=bool)
        end = int(rng.choice((0, h - 1)))
        # A noncentral narrow end makes the initial rectangle asymmetric.
        column = int(_pick(rng, [c for c in range(w) if 2*c != w-1]))
        mask[end] = False
        mask[end, column] = True
        if high_axis == 0:
            mask = mask.T.copy()
        attempts = int(rng.integers(0, max(2, mask.size // 2)))
        for _ in range(attempts):
            p = tuple(map(int, _pick(rng, np.argwhere(mask))))
            candidate = mask.copy()
            candidate[p] = False
            v, h = _contacts(candidate)
            counts = (v, h)
            if (min(counts) > 0 and counts[high_axis] > 2*counts[1-high_axis]
                    and _tight_connected(candidate)
                    and (not informative or _asymmetric(candidate))):
                mask = candidate
        assert _asymmetric(mask) if informative else True
        return mask
    # Ordinary glyphs include solid blocks, holes, thin paths and notched masks.
    for _ in range(64):
        h, w = map(int, rng.integers(2, 6, size=2))
        mask = np.ones((h, w), dtype=bool)
        target = int(rng.integers(max(h, w) + 1, h*w + 1))
        for index in rng.permutation(h*w):
            if int(mask.sum()) <= target:
                break
            p = divmod(int(index), w)
            candidate = mask.copy()
            candidate[p] = False
            v, horizontal = _contacts(candidate)
            if v and horizontal and _tight_connected(candidate):
                mask = candidate
        v, horizontal = _contacts(mask)
        if not unequal or v != horizontal:
            return mask
    # A finite, geometrically valid unequal-contact fallback.
    return np.ones((2, 3), dtype=bool)


def _transformed_point(point, dims, transform):
    r, c = point
    rows, cols = dims
    for _ in range(transform % 4):
        r, c = cols-1-c, r
        rows, cols = cols, rows
    if transform >= 4:
        c = cols-1-c
    return (r, c), (rows, cols)


def _layout(rng, dims, degree, high_axis=None, previous_anchor=None,
            noncentral=False, transform=0):
    rows, cols = dims
    choices = []
    for r in range(rows):
        for c in range(cols):
            actual, actual_dims = _transformed_point((r, c), dims, transform)
            if previous_anchor is not None and (actual[0] == previous_anchor[0] or
                                                actual[1] == previous_anchor[1]):
                continue
            if noncentral and actual == ((actual_dims[0]-1)//2, (actual_dims[1]-1)//2):
                continue
            available = [(r+dr, c+dc) for dr, dc in _DIRS
                         if 0 <= r+dr < rows and 0 <= c+dc < cols]
            for neighbors in itertools.combinations(available, degree):
                axes = [0 if a != r else 1 for a, b in neighbors]
                if degree == 2 and set(axes) != {0, 1}:
                    continue
                if high_axis is not None and axes.count(high_axis) != 1:
                    continue
                choices.append(((r, c), neighbors))
    if not choices:
        return None
    anchor, neighbors = _pick(rng, choices)
    occupied = {anchor, *neighbors}
    forbidden = {(anchor[0]+dr, anchor[1]+dc) for dr, dc in _DIRS} - occupied
    allowed = {(r, c) for r in range(rows) for c in range(cols)} - forbidden
    # Keep a tight map while allowing both connected and scattered spectators.
    for axis, boundary in ((0, 0), (0, rows-1), (1, 0), (1, cols-1)):
        if not any(p[axis] == boundary for p in occupied):
            candidates = sorted(p for p in allowed if p[axis] == boundary)
            if not candidates:
                return None
            occupied.add(_pick(rng, candidates))
    target = int(rng.integers(max(len(occupied), degree+2), len(allowed)+1))
    while len(occupied) < target:
        remaining = sorted(allowed - occupied)
        frontier = [p for p in remaining if any(
            (p[0]+dr, p[1]+dc) in occupied for dr, dc in _DIRS)]
        occupied.add(_pick(rng, frontier if frontier and rng.random() < .7 else remaining))
    return occupied, anchor, neighbors


def _ballot(rng, majority, minority, dims=None, strong=False):
    h, w = dims if dims is not None else _pick(rng, _PANEL_SHAPES)
    size = h*w
    nminor = 1 if strong else int(rng.integers(1, (size-1)//2+1))
    values = np.full(size, majority, dtype=np.int8)
    # Scattered, banded and clustered minorities; counts alone define the vote.
    order = (rng.permutation(size) if rng.random() < .6 else
             np.argsort(np.abs(np.arange(size)-int(rng.integers(size)))))
    values[order[:nminor]] = minority
    return values.reshape(h, w)


def _place(rng, mask, dims, occupied, anchor, ballots, labels, roles):
    """Fit the input map and anchored output on one plane before padding."""
    bg, ink, s0, s1 = roles
    h, w = mask.shape
    mh, mw = dims
    oh, ow = mh*h, mw*w
    ph, pw = 4*(mh-1)+3, 4*(mw-1)+3
    tr, tc = anchor[0]*h, anchor[1]*w
    rr, cc = np.meshgrid(np.arange(-ph-1, oh+2), np.arange(-pw-1, ow+2), indexing="ij")
    minr, minc = np.minimum(0, rr), np.minimum(0, cc)
    maxr, maxc = np.maximum(oh, rr+ph), np.maximum(ow, cc+pw)
    separation = ((rr+ph < tr) | (tr+h < rr) | (cc+pw < tc) | (tc+w < cc))
    valid = separation & (maxr-minr <= 30) & (maxc-minc <= 30)
    candidates = np.argwhere(valid)
    if not len(candidates):
        return None
    i, j = map(int, _pick(rng, candidates))
    pr, pc = int(rr[i, j]), int(cc[i, j])
    bh, bw = int(maxr[i, j]-minr[i, j]), int(maxc[i, j]-minc[i, j])
    padding = int(rng.integers(5))
    gh = 30 if padding == 0 else int(rng.integers(bh, min(30, bh+6)+1))
    gw = 30 if padding == 0 else int(rng.integers(bw, min(30, bw+6)+1))
    top = int(rng.integers(gh-bh+1))-int(minr[i, j])
    left = int(rng.integers(gw-bw+1))-int(minc[i, j])
    inp = np.full((gh, gw), bg, dtype=np.int8)
    mr, mc = np.nonzero(mask)
    inp[top+tr+mr, left+tc+mc] = ink
    for r, c in sorted(occupied):
        cr, cc = top+pr+1+4*r, left+pc+1+4*c
        if (r, c) == anchor:
            inp[cr, cc] = ink
        else:
            panel = ballots[(r, c)]
            a, b = panel.shape
            inp[cr-a//2:cr+a//2+1, cc-b//2:cc+b//2+1] = panel
    values, counts = np.unique(inp, return_counts=True)
    if (len(values) != 4 or int(values[np.argmax(counts)]) != bg or
            np.count_nonzero(counts == counts.max()) != 1):
        return None
    vertical, horizontal = _contacts(mask)
    votes = {s0: 0, s1: 0}
    for dr, dc in _DIRS:
        p = anchor[0]+dr, anchor[1]+dc
        if p in labels:
            votes[labels[p]] += vertical if dr else horizontal
    if min(votes.values()) == 0 or votes[s0] == votes[s1]:
        return None
    winner = max(votes, key=votes.get)
    # Expected output is built directly from the sampled construction records.
    out = np.full_like(inp, bg)
    for r, c in sorted(occupied):
        color = winner if (r, c) == anchor else labels[(r, c)]
        out[top+r*h+mr, left+c*w+mc] = color
    return {"input": inp, "output": out}


def _decode(input_grid):
    inp = np.asarray(input_grid)
    assert inp.ndim == 2 and all(1 <= n <= 30 for n in inp.shape)
    assert np.issubdtype(inp.dtype, np.integer) and np.all((0 <= inp) & (inp <= 9))
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 4 and np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    candidates = []
    for color in values:
        if color != bg:
            components = _components(inp == color)
            if len(components) == 2 and sorted(map(len, components))[0] == 1:
                candidates.append((int(color), components))
    assert len(candidates) == 1, "need a glyph and its singleton anchor"
    ink, components = candidates[0]
    glyph = next(c for c in components if len(c) > 1)
    marker = next(c[0] for c in components if len(c) == 1)
    tr, tc = np.min(glyph, axis=0).tolist()
    br, bc = np.max(glyph, axis=0).tolist()
    mask = inp[tr:br+1, tc:bc+1] == ink
    h, w = mask.shape
    assert 2 <= h <= 5 and 2 <= w <= 5 and _tight_connected(mask)
    vertical, horizontal = _contacts(mask)
    assert vertical > 0 and horizontal > 0
    signals = tuple(int(c) for c in values if c not in (bg, ink))
    panels = _components(np.isin(inp, signals))
    assert len(panels) >= 3
    center_records = {}
    for cells in panels:
        r0, c0 = np.min(cells, axis=0).tolist()
        r1, c1 = np.max(cells, axis=0).tolist()
        a, b = r1-r0+1, c1-c0+1
        assert (a, b) in _PANEL_SHAPES and len(cells) == a*b
        block = inp[r0:r1+1, c0:c1+1]
        n0, n1 = (int(np.count_nonzero(block == s)) for s in signals)
        assert n0 > 0 and n1 > 0 and n0 != n1
        center = (r0+r1)//2, (c0+c1)//2
        center_records[center] = (signals[int(n1 > n0)], (r0, c0, r1, c1), (n0, n1))
    centers = [marker, *center_records]
    rbase = min(p[0] for p in centers)
    cbase = min(p[1] for p in centers)
    assert all((r-rbase) % 4 == 0 and (c-cbase) % 4 == 0 for r, c in centers)
    def coarse(p):
        return (p[0]-rbase)//4, (p[1]-cbase)//4
    anchor = coarse(marker)
    records = {coarse(p): record for p, record in center_records.items()}
    occupied = {anchor, *records}
    mh, mw = max(r for r, c in occupied)+1, max(c for r, c in occupied)+1
    assert 2 <= mh <= 5 and 2 <= mw <= 5
    # The padded map envelope and template have at least one blank band between.
    mapbox = rbase-1, cbase-1, rbase+4*(mh-1)+1, cbase+4*(mw-1)+1
    assert (br+1 < mapbox[0] or mapbox[2]+1 < tr or
            bc+1 < mapbox[1] or mapbox[3]+1 < tc)
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    assert 0 <= top and top+mh*h <= inp.shape[0]
    assert 0 <= left and left+mw*w <= inp.shape[1]
    votes = {s: 0 for s in signals}
    weights = {}
    for dr, dc in _DIRS:
        p = anchor[0]+dr, anchor[1]+dc
        if p in records:
            weight = vertical if dr else horizontal
            weights[p] = weight
            votes[records[p][0]] += weight
    assert min(votes.values()) > 0 and votes[signals[0]] != votes[signals[1]]
    winner = max(votes, key=votes.get)
    return {"background": bg, "ink": ink, "signals": signals, "mask": mask,
            "template": (tr, tc), "map_shape": (mh, mw), "anchor": anchor,
            "records": records, "occupied": occupied, "weights": weights,
            "votes": votes, "winner": winner, "top": top, "left": left,
            "contacts": (vertical, horizontal), "shape": inp.shape}


def _render(info):
    bg, mask = info["background"], info["mask"]
    h, w = mask.shape
    mh, mw = info["map_shape"]
    # Kronecker masks reconstruct the whole mosaic, independently of _place's
    # construction loop over glyph pixels.
    coarse = np.full((mh, mw), bg, dtype=np.int8)
    for p, record in info["records"].items():
        coarse[p] = record[0]
    coarse[info["anchor"]] = info["winner"]
    occupancy = np.zeros((mh, mw), dtype=bool)
    for p in info["occupied"]:
        occupancy[p] = True
    enlarged = np.repeat(np.repeat(coarse, h, axis=0), w, axis=1)
    stamp = np.full_like(enlarged, bg)
    foreground = np.kron(occupancy, mask)
    stamp[foreground] = enlarged[foreground]
    out = np.full(info["shape"], bg, dtype=np.int8)
    top, left = info["top"], info["left"]
    out[top:top+mh*h, left:left+mw*w] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive majority labels, docking weights and anchored copies from pixels."""
    return _render(_decode(input_grid))


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _make_pair(rng, roles, degree, witness=None, high_axis=None,
               previous_dims=None, previous_anchor=None, transform=0,
               winner_index=None):
    for _ in range(96):
        mask = _glyph(rng, high_axis, previous_dims,
                      informative=witness == 0, unequal=degree == 2)
        maxdim = 4 if witness is not None else 5
        dims = (int(rng.integers(3, maxdim+1)), int(rng.integers(3, maxdim+1)))
        layout = _layout(rng, dims, degree, high_axis,
                         previous_anchor=previous_anchor,
                         noncentral=witness == 0, transform=transform)
        if layout is None:
            continue
        occupied, anchor, neighbors = layout
        signals = sorted(roles[2:])
        desired = int(rng.integers(2)) if winner_index is None else winner_index
        winner, loser = signals[desired], signals[1-desired]
        labels = {}
        if witness is not None:
            for p in neighbors:
                axis = 0 if p[0] != anchor[0] else 1
                labels[p] = winner if axis == high_axis else loser
        else:
            v, horizontal = _contacts(mask)
            assignments = []
            for bits in range(1, (1 << degree)-1):
                tally = sum((1 if bits >> i & 1 else -1)*
                            (v if p[0] != anchor[0] else horizontal)
                            for i, p in enumerate(neighbors))
                if tally > 0:
                    assignments.append(bits)
            if not assignments:
                continue
            bits = int(_pick(rng, assignments))
            labels = {p: winner if bits >> i & 1 else loser
                      for i, p in enumerate(neighbors)}
        for p in sorted(occupied - {anchor} - set(neighbors)):
            labels[p] = loser if witness == 0 else int(_pick(rng, signals))
        ballots = {}
        for p, majority in sorted(labels.items()):
            minority = next(s for s in signals if s != majority)
            pdims, strong = None, False
            if witness == 0:
                pdims = _pick(rng, ((1, 3), (3, 1))) if majority == winner else (3, 3)
                strong = majority == loser
            elif witness == 1:
                pdims = (3, 3) if p in neighbors and majority == winner else _pick(rng, ((1, 3), (3, 1)))
                strong = True
            ballots[p] = _ballot(rng, majority, minority, pdims, strong)
        pair = _place(rng, mask, dims, occupied, anchor, ballots, labels, roles)
        if pair is None:
            continue
        # Centered odd ballot rectangles make these whole-scene isometries exact.
        for name in pair:
            a = np.rot90(pair[name], transform % 4)
            pair[name] = (np.fliplr(a) if transform >= 4 else a).copy()
        info = _decode(pair["input"])
        assert np.array_equal(_render(info), pair["output"])
        return pair, mask.shape, anchor
    raise RuntimeError("bounded construction could not fit a valid parliament")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    stable_palette = rng.random() < .25
    transform = int(rng.integers(8))
    first_axis, first_winner = int(rng.integers(2)), int(rng.integers(2))
    puzzle = {"train": [], "test": []}
    used = set()
    previous_dims = previous_anchor = first_signals = None
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            witness = index if split == "train" and index < 2 else None
            roles = palette if stable_palette else tuple(map(int, rng.choice(10, 4, replace=False)))
            if witness == 1 and not stable_palette and set(roles[2:]) == first_signals:
                pool = [c for c in range(10) if c not in roles[:2] and c not in first_signals]
                roles = (*roles[:2], *map(int, rng.choice(pool, 2, replace=False)))
            if witness == 0:
                first_signals = set(roles[2:])
            degree = 3 if split == "train" else int(rng.choice((2, 4)))
            high_axis = (first_axis if witness == 0 else 1-first_axis) if witness is not None else None
            desired = (first_winner if witness == 0 else 1-first_winner) if witness is not None else None
            for _ in range(24):
                pair, dims, anchor = _make_pair(
                    rng, roles, degree, witness, high_axis,
                    previous_dims if witness == 1 else None,
                    previous_anchor if witness == 1 else None,
                    transform if witness is not None else int(rng.integers(8)), desired)
                key = _key(pair["input"])
                if key in used:
                    continue
                if witness == 1:
                    first = _decode(puzzle["train"][0]["input"])
                    second = _decode(pair["input"])
                    if any(a == b for a, b in zip(first["anchor"], second["anchor"])):
                        continue
                used.add(key)
                puzzle[split].append(pair)
                if witness == 0:
                    previous_dims = dims
                    previous_anchor = _decode(pair["input"])["anchor"]
                break
            else:
                raise RuntimeError("bounded input uniqueness/anchor search exhausted")
    # The teaching witnesses have no fixed position in the returned demonstrations.
    order = rng.permutation(ntrain)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(info, inp):
    winner, signals = info["winner"], info["signals"]
    labels = {p: record[0] for p, record in info["records"].items()}
    neighbor_margin = sum(1 if labels[p] == winner else -1 for p in info["weights"])
    all_margin = sum(1 if s == winner else -1 for s in labels.values())
    other = next(s for s in signals if s != winner)
    raw_margin = int(np.count_nonzero(inp == winner)-np.count_nonzero(inp == other))
    return {"degree": len(info["weights"]), "neighbor_margin": neighbor_margin,
            "all_margin": all_margin, "raw_margin": raw_margin,
            "winner_index": signals.index(winner),
            "spectators": len(labels)-len(info["weights"])}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(_render(info), out)
            assert not np.array_equal(inp, out)
            assert not np.any(out == info["ink"]), "construction ink must disappear"
            assert np.count_nonzero(out != info["background"]) == len(info["occupied"])*int(info["mask"].sum())
            # Count actual output pixel edges at the collector. This does not
            # use the opposing-boundary intersection formula used by the solver.
            mr, mc = np.nonzero(info["mask"])
            tr, tc = info["template"]
            collector = set(zip((tr+mr).tolist(), (tc+mc).tolist()))
            contact_votes = {s: 0 for s in info["signals"]}
            for r, c in collector:
                for dr, dc in _DIRS:
                    q = r+dr, c+dc
                    if (q not in collector and 0 <= q[0] < out.shape[0]
                            and 0 <= q[1] < out.shape[1] and int(out[q]) in contact_votes):
                        contact_votes[int(out[q])] += 1
            assert contact_votes == info["votes"]
            metrics = _measure(info, inp)
            assert metrics["spectators"] >= 1
            (training if split == "train" else testing).append((info, metrics))
    assert all(m["degree"] == 3 for _, m in training)
    assert all(m["degree"] in (2, 4) for _, m in testing)
    assert {m["winner_index"] for _, m in training} == {0, 1}
    assert any(m["neighbor_margin"] < 0 and m["all_margin"] < 0 and m["raw_margin"] < 0
               for _, m in training), "need a port/area/unweighted counterexample"
    witnesses = [(i, m) for i, m in training if m["neighbor_margin"] < 0]
    assert any(i["contacts"][0] > 2*i["contacts"][1] for i, m in witnesses)
    assert any(i["contacts"][1] > 2*i["contacts"][0] for i, m in witnesses)
    assert len({i["anchor"][0] for i, _ in training}) >= 2
    assert len({i["anchor"][1] for i, _ in training}) >= 2
    assert len({i["mask"].shape[0] for i, _ in training}) >= 2
    assert len({i["mask"].shape[1] for i, _ in training}) >= 2
    assert any(_asymmetric(i["mask"]) and i["mask"].shape[0] != i["mask"].shape[1]
               for i, _ in training)
    assert any(i["anchor"] != ((i["map_shape"][0]-1)//2, (i["map_shape"][1]-1)//2)
               for i, _ in training)


def self_test():
    """Streaming checks over the 42 requested seeds; retain metrics, not grids."""
    shapes, masks, layouts, anchors, contacts, palettes, populations = (set() for _ in range(7))
    signatures, test_degrees, winner_indices = set(), set(), set()
    pairs = 0
    for seed in _SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _decode(inp)
                assert np.array_equal(solve(inp.copy()), out)
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                shapes.add(inp.shape)
                masks.add(_key(info["mask"]))
                layouts.add((info["map_shape"], tuple(sorted(info["occupied"]))))
                anchors.add(info["anchor"])
                contacts.add(info["contacts"])
                palettes.add((info["background"], info["ink"], *info["signals"]))
                populations.add(len(info["occupied"]))
                winner_indices.add(info["signals"].index(info["winner"]))
                if split == "test":
                    test_degrees.add(len(info["weights"]))
                pairs += 1
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42 and pairs == 210
    assert len(shapes) >= 50 and len(masks) >= 60 and len(layouts) >= 100
    assert len(anchors) >= 10 and len(contacts) >= 10 and len(palettes) >= 50
    assert len(populations) >= 10 and test_degrees == {2, 4} and winner_indices == {0, 1}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(713, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, 1.5)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    q = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
               for k in ("input", "output"))
    # Input-only metamorphic checks: every palette permutation and scene
    # isometry must preserve the same local-majority/contact rule.
    pair = p["train"][0]
    perm = np.array([7, 3, 0, 8, 2, 9, 1, 5, 6, 4], dtype=np.int8)
    assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
    for k in range(4):
        for reflected in (False, True):
            a, b = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
            if reflected:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    # Dropping local majority flips ordinary copies, even if the collector
    # accidentally keeps the same color in a particular alternative rule.
    info = _decode(pair["input"])
    wrong = {**info, "records": {pos: (next(s for s in info["signals"] if s != rec[0]), *rec[1:])
                                for pos, rec in info["records"].items()}}
    assert not np.array_equal(_render(wrong), pair["output"])
    # Mutate one required copy pixel and ensure validation catches it.
    bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    i = _decode(bad["test"][0]["input"])
    r, c = np.argwhere(i["mask"])[0]
    pos = i["template"][0]+r, i["template"][1]+c
    bad["test"][0]["output"][pos] = i["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted collector accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "canvas_shapes": len(shapes), "glyph_masks": len(masks),
            "map_layouts": len(layouts), "anchor_positions": len(anchors),
            "contact_profiles": len(contacts), "palettes": len(palettes),
            "occupied_counts": sorted(populations), "test_degrees": sorted(test_degrees),
            "independent_solver": False}
