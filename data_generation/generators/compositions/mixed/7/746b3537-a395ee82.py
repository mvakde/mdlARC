"""Anchored substitution, then maximal equal-vector run compression.

Only NumPy and the standard library are used. No files are read or written by
generation, solving, validation, or self_test; all experiments are memory-only.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _keeps(a):
    """Keep the first representative of each adjacent full-vector run."""
    rows = np.r_[True, np.any(a[1:] != a[:-1], axis=1)]
    cols = np.r_[True, np.any(a[:, 1:] != a[:, :-1], axis=0)]
    return rows, cols


def _compress(a):
    rows, cols = _keeps(a)
    return a[np.ix_(rows, cols)].copy()


def _components(a, color):
    remaining = set(map(tuple, np.argwhere(a == color)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], []
        while stack:
            r, c = stack.pop()
            group.append((int(r), int(c)))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        groups.append(group)
    return groups


def _decode(input_grid):
    """Recover geometry and semantic roles entirely from the input pixels."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 3 or counts.max() * 2 <= a.size:
        raise ValueError("three roles and a strict-majority background required")
    bg = int(colors[counts.argmax()])
    foreground = [int(c) for c in colors if c != bg]
    groups = {c: _components(a, c) for c in foreground}
    choices = [c for c in foreground if len(groups[c]) == 2
               and sorted(map(len, groups[c]))[0] == 1
               and sorted(map(len, groups[c]))[1] >= 3]
    if len(choices) != 1:
        raise ValueError("template and exceptional point must be unique")
    ink = choices[0]
    marker = next(c for c in foreground if c != ink)
    glyph = np.array(next(g for g in groups[ink] if len(g) > 1))
    anchor_cell = next(g[0] for g in groups[ink] if len(g) == 1)
    if not all(len(g) == 1 for g in groups[marker]):
        raise ValueError("ordinary sites must be isolated pixels")
    tr, tc = map(int, glyph.min(axis=0))
    br, bc = map(int, glyph.max(axis=0))
    mask = a[tr:br+1, tc:bc+1] == ink
    h, w = mask.shape
    if min(h, w) < 2:
        raise ValueError("template must span both axes")
    points = [anchor_cell] + [g[0] for g in groups[marker]]
    pr, pc = map(int, np.min(points, axis=0))
    er, ec = map(int, np.max(points, axis=0))
    if not (br+1 < pr or er+1 < tr or bc+1 < pc or ec+1 < tc):
        raise ValueError("template and map boxes require a blank separating band")
    if any((r-pr) % 2 or (c-pc) % 2 for r, c in points):
        raise ValueError("constellation must use pitch two")
    layout = np.zeros(((er-pr)//2+1, (ec-pc)//2+1), dtype=bool)
    for r, c in points:
        layout[(r-pr)//2, (c-pc)//2] = True
    anchor = ((anchor_cell[0]-pr)//2, (anchor_cell[1]-pc)//2)
    top, left = tr-anchor[0]*h, tc-anchor[1]*w
    if not (0 <= top and 0 <= left and top+h*layout.shape[0] <= a.shape[0]
            and left+w*layout.shape[1] <= a.shape[1]):
        raise ValueError("anchored mosaic would leave the canvas")
    return {"grid": a, "roles": (bg, ink, marker), "mask": mask,
            "layout": layout, "anchor": anchor, "top": top, "left": left,
            "template_top": tr, "template_left": tc}


def _stamp(meta):
    """Build B's intermediate on the original canvas, before A's collapse."""
    bg, ink, marker = meta["roles"]
    mask, layout = meta["mask"], meta["layout"]
    h, w = mask.shape
    body = np.full((h*layout.shape[0], w*layout.shape[1]), bg, dtype=np.int8)
    body[np.kron(layout, mask)] = ink
    special = np.zeros_like(layout)
    special[meta["anchor"]] = True
    body[np.kron(special, mask)] = marker
    canvas = np.full_like(meta["grid"], bg)
    r, c = meta["top"], meta["left"]
    canvas[r:r+body.shape[0], c:c+body.shape[1]] = body
    return canvas


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Unrotated, anchor-relative color inversion; then compress the canvas."""
    return _compress(_stamp(_decode(input_grid)))


def _asymmetric(a):
    return all(not np.array_equal(a, b) for b in (
        a[::-1], a[:, ::-1], a[::-1, ::-1], a.T,
        a.T[::-1], a.T[:, ::-1], a.T[::-1, ::-1]))


def _basis(rng, kind):
    if kind == "solid":
        return np.ones((1, 1), dtype=bool)
    if kind == "ring":
        h, w = map(int, rng.integers(3, 6, size=2))
        a = np.ones((h, w), dtype=bool)
        a[1:-1, 1:-1] = rng.random((h-2, w-2)) < 0.4
        a[int(rng.integers(1, h-1)), int(rng.integers(1, w-1))] = False
        # Delete isolated interior ink, retaining the connected perimeter.
        components = _components(a.astype(np.int8), 1)
        keep = next(g for g in components if (0, 0) in g)
        a[:] = False
        for p in keep:
            a[p] = True
        a = _compress(a)
        # Four logical rows/columns leave room for unequal thickness and a map.
        if max(a.shape) > 4:
            return np.array(
                [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
        return a
    for _ in range(64):
        h, w = map(int, rng.integers(2, 5, size=2))
        a = np.zeros((h, w), dtype=bool)
        steps = [0]*(h-1) + [1]*(w-1)
        rng.shuffle(steps)
        r = c = 0
        a[r, c] = True
        for step in steps:
            r += step == 0
            c += step == 1
            a[r, c] = True
        for _ in range(int(rng.integers(h*w//2+1))):
            candidates = [(r, c) for r in range(h) for c in range(w) if not a[r, c]
                          and any(a[x, y] for x, y in ((r-1, c), (r+1, c),
                                  (r, c-1), (r, c+1)) if 0 <= x < h and 0 <= y < w)]
            if candidates:
                a[_pick(rng, candidates)] = True
        a = _compress(a)
        if min(a.shape) >= 2 and _asymmetric(a):
            if rng.integers(2):
                a = a[::-1]
            if rng.integers(2):
                a = a[:, ::-1]
            return a.copy()
    # A rectangle with distinct profiles stays asymmetric after compression.
    # The former square fallback had an anti-diagonal reflection symmetry.
    return np.array([[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool)


def _weights(rng, n, target=False, force_mixed=False):
    if n == 1:
        return np.array([int(rng.integers(3, 5)) if target else 2])
    values = rng.integers(1, 3, size=n)
    if force_mixed:
        values[:] = 1
        values[int(rng.integers(n))] = 2
        for i in range(n):
            if rng.random() < 0.35 and values.sum() < 7:
                values[i] = 2
        if np.all(values == 2):
            values[int(rng.integers(n))] = 1
    if target:
        values[:] = 1
        values[int(rng.integers(n))] = int(rng.integers(3, 5))
        for i in rng.permutation(n):
            if values[i] == 1 and values.sum() < 8 and rng.random() < 0.4:
                values[i] = 2
    return values


def _make_mask(rng, kind, teaching=False, target=False):
    base = _basis(rng, kind)
    rw = _weights(rng, base.shape[0], target=target and bool(rng.integers(2)),
                  force_mixed=teaching)
    cw = _weights(rng, base.shape[1], target=target and rw.max() < 3,
                  force_mixed=teaching)
    return np.repeat(np.repeat(base, rw, axis=0), cw, axis=1)


def _layout(rng, mask, mode, previous_anchor=None):
    h, w = mask.shape
    if mode == "seams":
        mh = int(rng.integers(3, min(4, 28//h)+1))
        mw = int(rng.integers(3, min(4, 28//w)+1))
        anchor = (_pick(rng, (0, mh-1)), _pick(rng, (0, mw-1)))
        a = np.ones((mh, mw), dtype=bool)
        if rng.random() < 0.7:
            a[:] = False
            a[anchor[0], :] = True
            a[:, anchor[1]] = True
        return a, anchor
    if mode == "gaps":
        mh = int(rng.integers(4, min(5, 28//h)+1))
        mw = int(rng.integers(4, min(5, 28//w)+1))
        # Omit one internal band on each axis, while retaining endpoints.
        rows = list(range(mh))
        cols = list(range(mw))
        rows.remove(int(rng.integers(1, mh-1)))
        cols.remove(int(rng.integers(1, mw-1)))
        a = np.zeros((mh, mw), dtype=bool)
        a[np.ix_(rows, cols)] = rng.random((len(rows), len(cols))) < rng.uniform(0.3, 0.9)
        a[rows[0], cols] = True
        a[rows, cols[0]] = True
        choices = [(r, c) for r in rows for c in cols
                   if previous_anchor is None or (r != previous_anchor[0] and c != previous_anchor[1])]
        anchor = _pick(rng, choices)
        a[anchor] = True
        return a, anchor
    mh = int(rng.integers(1, min(6, 28//h)+1))
    mw = int(rng.integers(1, min(6, 28//w)+1))
    if mh*mw == 1:
        mh = 2
    a = rng.random((mh, mw)) < rng.uniform(0.2, 0.9)
    for r in (0, mh-1):
        a[r, int(rng.integers(mw))] = True
    for c in (0, mw-1):
        a[int(rng.integers(mh)), c] = True
    if a.sum() < 2:
        a[-1, -1] = a[0, 0] = True
    return a, tuple(map(int, _pick(rng, np.argwhere(a))))


def _place(rng, mask, layout, anchor, roles):
    """Place the input around its valid latent canvas, with no clipping."""
    h, w = mask.shape
    mh, mw = layout.shape
    oh, ow = h*mh, w*mw
    ph, pw = 2*mh-1, 2*mw-1
    tr, tc = anchor[0]*h, anchor[1]*w
    positions = []
    for pr in range(-ph-1, oh+2):
        for pc in range(-pw-1, ow+2):
            if not (pr+ph < tr or tr+h < pr or pc+pw < tc or tc+w < pc):
                continue
            r0, c0 = min(0, pr), min(0, pc)
            gh, gw = max(oh, pr+ph)-r0, max(ow, pc+pw)-c0
            if gh <= 30 and gw <= 30:
                positions.append((pr, pc, r0, c0, gh, gw))
    if not positions:
        raise ValueError("no placement fits")
    pr, pc, r0, c0, gh, gw = _pick(rng, positions)
    gh += int(rng.integers(min(3, 30-gh)+1))
    gw += int(rng.integers(min(3, 30-gw)+1))
    foreground = int(mask.sum()+layout.sum())
    while gh*gw <= 2*foreground:
        if gh < 30 and (gh <= gw or gw == 30):
            gh += 1
        elif gw < 30:
            gw += 1
        else:
            raise ValueError("background cannot be made a strict majority")
    union_h, union_w = max(oh, pr+ph)-r0, max(ow, pc+pw)-c0
    top = -r0 + int(rng.integers(gh-union_h+1))
    left = -c0 + int(rng.integers(gw-union_w+1))
    bg, ink, marker = roles
    inp = np.full((gh, gw), bg, dtype=np.int8)
    rr, cc = np.nonzero(mask)
    inp[top+tr+rr, left+tc+cc] = ink
    for r, c in np.argwhere(layout):
        inp[top+pr+2*r, left+pc+2*c] = ink if (r, c) == anchor else marker
    # Generation uses an explicit stamp loop; solve uses Kronecker expansion.
    raw = np.full_like(inp, bg)
    for r, c in np.argwhere(layout):
        raw[top+r*h+rr, left+c*w+cc] = marker if (r, c) == anchor else ink
    return {"input": inp, "output": _compress(raw)}


def _blank_only(a, bg):
    rows, cols = np.all(a == bg, axis=1), np.all(a == bg, axis=0)
    kr = np.r_[True, ~(rows[1:] & rows[:-1])]
    kc = np.r_[True, ~(cols[1:] & cols[:-1])]
    return a[np.ix_(kr, kc)]


def _unique_vectors(a):
    def keep(vectors):
        seen, result = set(), []
        for i, v in enumerate(vectors):
            key = v.tobytes()
            if key not in seen:
                result.append(i)
                seen.add(key)
        return result
    return a[np.ix_(keep(a), keep(a.T))]


def _histogram_collapse(a):
    rh = np.array([np.bincount(row, minlength=10) for row in a])
    ch = np.array([np.bincount(col, minlength=10) for col in a.T])
    kr = np.r_[True, np.any(rh[1:] != rh[:-1], axis=1)]
    kc = np.r_[True, np.any(ch[1:] != ch[:-1], axis=1)]
    return a[np.ix_(kr, kc)]


def _features(meta):
    raw = _stamp(meta)
    out = _compress(raw)
    bg = meta["roles"][0]
    h, w = meta["mask"].shape
    mh, mw = meta["layout"].shape
    seam_rows = [meta["top"]+r*h for r in range(1, mh)]
    seam_cols = [meta["left"]+c*w for c in range(1, mw)]
    return {"raw": raw, "output": out,
            "row_seam": any(np.array_equal(raw[r-1], raw[r]) and np.any(raw[r] != bg)
                            for r in seam_rows),
            "col_seam": any(np.array_equal(raw[:, c-1], raw[:, c]) and np.any(raw[:, c] != bg)
                            for c in seam_cols),
            "recurrence": not np.array_equal(_unique_vectors(raw), out),
            "histogram_witness": not np.array_equal(_histogram_collapse(raw), out),
            "blank_witness": not np.array_equal(_blank_only(raw, bg), out),
            "gaps": (bool(np.any(~meta["layout"].any(axis=1))),
                     bool(np.any(~meta["layout"].any(axis=0))))}


def _counts(num_train, num_test):
    for name, v, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    consistent_palette = bool(rng.random() < 0.25)
    base_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    first = None
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            mode = "seams" if split == "train" and i == 0 else "gaps" if split == "train" and i == 1 else "free"
            for _ in range(96):
                kind = "ring" if mode == "seams" else "open" if mode == "gaps" else _pick(rng, ("ring", "open", "open", "solid"))
                mask = _make_mask(rng, kind, teaching=mode == "seams", target=split == "test")
                if mode == "gaps" and (max(mask.shape) > 7
                                        or mask.shape[0] == first["mask"].shape[0]
                                        or mask.shape[1] == first["mask"].shape[1]):
                    continue
                layout, anchor = _layout(rng, mask, mode, first["anchor"] if mode == "gaps" else None)
                roles = base_roles if consistent_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _place(rng, mask, layout, anchor, roles)
                meta = _decode(pair["input"])
                f = _features(meta)
                if mode == "seams" and not (f["row_seam"] and f["col_seam"] and f["recurrence"]):
                    continue
                if mode == "gaps" and not f["histogram_witness"]:
                    continue
                if _key(pair["input"]) in used:
                    continue
                used.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if mode == "seams":
                    first = meta
                break
            else:
                raise RuntimeError("bounded construction attempts exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in used
            used.add(_key(inp))
            meta = _decode(inp)
            f = _features(meta)
            raw = f["raw"]
            assert np.array_equal(out, f["output"])
            assert np.count_nonzero(raw != meta["roles"][0]) == int(meta["mask"].sum()*meta["layout"].sum())
            output_colors = set(map(int, np.unique(out)))
            assert set(meta["roles"][1:]) <= output_colors <= set(meta["roles"])
            # Each omitted parent makes an observably different answer.
            assert not np.array_equal(raw, out)
            assert not np.array_equal(_compress(inp), out)
            assert not np.any(np.all(out[1:] == out[:-1], axis=1))
            assert not np.any(np.all(out[:, 1:] == out[:, :-1], axis=0))
            assert np.array_equal(_compress(raw.T), out.T)
            assert np.array_equal(_compress(out), out)
            if split == "train":
                training.append((meta, f))
    assert any(f["row_seam"] and f["col_seam"] and f["recurrence"] and f["blank_witness"] for _, f in training)
    assert any(f["gaps"] == (True, True) and f["histogram_witness"] for _, f in training)
    assert len({m["anchor"][0] for m, _ in training}) >= 2
    assert len({m["anchor"][1] for m, _ in training}) >= 2
    assert len({m["mask"].shape[0] for m, _ in training}) >= 2
    assert len({m["mask"].shape[1] for m, _ in training}) >= 2
    assert any(_asymmetric(_compress(m["mask"])) for m, _ in training)


def self_test():
    """42 sequential seeds, all count combinations, no saved grids or logs."""
    signatures, shapes, masks, maps, anchors, palettes, max_runs = (set() for _ in range(7))
    seam_pairs = gap_pairs = 0
    relabel = np.array([7, 4, 9, 0, 8, 3, 1, 6, 2, 5], dtype=np.int8)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                meta = _decode(pair["input"])
                f = _features(meta)
                for role in ("input", "output"):
                    a = pair[role]
                    digest.update(bytes(a.shape))
                    digest.update(a.tobytes())
                shapes.add(pair["input"].shape)
                masks.add(_key(meta["mask"]))
                maps.add(_key(meta["layout"]))
                anchors.add(meta["anchor"])
                palettes.add(meta["roles"])
                seam_pairs += f["row_seam"] and f["col_seam"]
                gap_pairs += f["gaps"] == (True, True)
                assert np.array_equal(solve(pair["input"]), pair["output"])
                kr, kc = _keeps(meta["mask"])
                runs = [np.diff(np.r_[np.flatnonzero(keep), len(keep)]).max() for keep in (kr, kc)]
                max_runs.add(int(max(runs)))
                if split == "test":
                    assert max(runs) >= 3
        signatures.add(digest.digest())
        # Input-only decoding must commute with every rectangle isometry and
        # arbitrary color relabeling, including changes of the background digit.
        pair = puzzle["train"][seed % len(puzzle["train"])]
        for turns in range(4):
            for reflect in (False, True):
                inp = np.rot90(pair["input"], turns)
                out = np.rot90(pair["output"], turns)
                if reflect:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                assert np.array_equal(solve(relabel[inp]), relabel[out])
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(masks) >= 50 and len(maps) >= 60
    assert len(anchors) >= 15 and len(palettes) >= 40
    assert {2, 3, 4} <= max_runs and seam_pairs >= 42 and gap_pairs >= 42
    # Force all random attempts to produce symmetric L masks. The bounded
    # fallback must still satisfy the asymmetric logical-template contract.
    class SymmetricAttempts:
        def integers(self, low, high=None, size=None):
            return np.full(size, 2, dtype=int) if size is not None else 0

        def shuffle(self, values):
            pass

    fallback = _basis(SymmetricAttempts(), "open")
    assert _asymmetric(_compress(fallback))
    assert len(_components(fallback.astype(np.int8), 1)) == 1
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            a, b = generate_puzzle(901, nt, nq), generate_puzzle(901, nt, nq)
            assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact vector equality, not histogram equality or global deduplication.
    a = np.array([[1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1]], dtype=np.int8)
    assert np.array_equal(_compress(a), a[[0, 2, 3]])
    assert not np.array_equal(_histogram_collapse(a), _compress(a))
    assert not np.array_equal(_unique_vectors(a), _compress(a))
    # An edge-filling reconstruction legitimately removes the background role.
    filled = np.zeros((9, 8), dtype=np.int8)
    filled[:3, :2] = 4
    for r in range(3):
        for c in range(4):
            filled[4+2*r, 2*c] = 4 if (r, c) == (0, 0) else 7
    assert np.array_equal(solve(filled), np.array([[7, 4], [4, 4]], dtype=np.int8))
    bad = generate_puzzle(7, 2, 1)
    bad["train"][0]["output"][0, 0] = (int(bad["train"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted answer accepted")
