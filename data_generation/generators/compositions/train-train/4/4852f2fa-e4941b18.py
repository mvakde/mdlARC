"""Crates off a ledge: count-controlled tile pile after a push around a block.

Composition of parent A (4852f2fa: extract a payload tile, count isolated
counter markers, concatenate that many exact copies) and parent B (e4941b18:
a pusher sends a token along a block face, around the far end and down the
exterior side lane to the boundary the block touches).

Canonical frame (block attached to the bottom boundary; any of the four
rotations may be used):

* one solid block touches the bottom boundary; its top face is exposed;
* a payload tile (tight bounding box, one colour, background holes allowed)
  rests on the top face, with its bottom row in the strip row just above it;
* one singleton pusher sits in that strip row, on one side of the tile;
* N isolated singleton counters of a fifth colour lie elsewhere.

Output: the pusher advances to the tile's near-edge cell in the strip row,
the tile is pushed away from the pusher, around that end of the block, and
falls down the exterior lane beside it.  The counters say how many identical
crates fall: N exact copies of the tile pile up from the boundary in that
lane, contiguous along the fall axis, the first copy touching the boundary.
The original tile, pusher position and counters are erased; the block and
background are unchanged.
"""
from __future__ import annotations

import numpy as np

TILE_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (3, 4), (4, 3), (4, 2)]


# --------------------------------------------------------------------------
# Input-only solver
# --------------------------------------------------------------------------

def _analyze_canonical(g):
    """Return role information if ``g`` is a valid bottom-attached frame."""
    H, W = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    order = np.argsort(-cnts)
    if len(vals) < 5:
        return None
    if len(vals) > 1 and cnts[order[0]] == cnts[order[1]]:
        return None
    bg = int(vals[order[0]])
    count = {int(v): int(c) for v, c in zip(vals, cnts)}
    best = None
    for c in count:
        if c == bg:
            continue
        rr, cc = np.nonzero(g == c)
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(rr) or area < 4 or r1 != H - 1 or r0 == 0:
            continue
        if c0 == 0 and c1 == W - 1:
            continue
        if best is None or area > best[0]:
            best = (area, c, int(r0), int(c0), int(c1))
    if best is None:
        return None
    _, block, top, L, R = best
    strip = g[top - 1, L:R + 1]
    present = sorted({int(v) for v in strip} - {bg, block})
    if len(present) != 2:
        return None
    singles = [c for c in present if count[c] == 1]
    multis = [c for c in present if count[c] >= 2]
    if len(singles) != 1 or len(multis) != 1:
        return None
    pusher, tile = singles[0], multis[0]
    rr, cc = np.nonzero(g == tile)
    tr0, tr1, tc0, tc1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    if tr1 != top - 1 or tc0 < L or tc1 > R:
        return None
    others = [c for c in count if c not in (bg, block, tile, pusher)]
    if len(others) != 1:
        return None
    counter = others[0]
    pr, pc = (int(v[0]) for v in np.nonzero(g == pusher))
    if pr != top - 1:
        return None
    if pc < tc0:
        direction = "right"
    elif pc > tc1:
        direction = "left"
    else:
        return None
    pattern = g[tr0:tr1 + 1, tc0:tc1 + 1] == tile
    return {"bg": bg, "block": block, "tile": tile, "pusher": pusher, "counter": counter,
            "top": top, "L": L, "R": R, "tr0": tr0, "tc0": tc0, "h": tr1 - tr0 + 1,
            "w": tc1 - tc0 + 1, "pattern": pattern, "pusher_col": pc,
            "direction": direction, "N": count[counter], "H": H, "W": W}


def _render_canonical(g, info, N=None, direction=None, anchor="floor"):
    """Apply the rule (or a shortcut variant) in the canonical frame.

    Returns None when the variant would leave the canvas."""
    N = info["N"] if N is None else N
    direction = info["direction"] if direction is None else direction
    H, W = g.shape
    h, w, bg = info["h"], info["w"], info["bg"]
    out = g.copy()
    for c in (info["tile"], info["pusher"], info["counter"]):
        out[g == c] = bg
    near = info["tc0"] if info["direction"] == "right" else info["tc0"] + w - 1
    out[info["top"] - 1, near] = info["pusher"]
    if direction == "right":
        lc = info["R"] + 1
    else:
        lc = info["L"] - w
    if lc < 0 or lc + w > W:
        return None
    for k in range(N):
        if anchor == "floor":
            r = H - (k + 1) * h
        else:  # shortcut: copies hang downward from the tile's own level
            r = info["top"] - h + k * h
        if r < 0 or r + h > H:
            return None
        region = out[r:r + h, lc:lc + w]
        region[info["pattern"]] = info["tile"]
    return out


def _frames(g):
    found = []
    for k in range(4):
        cand = np.rot90(g, -k)
        info = _analyze_canonical(cand)
        if info is not None:
            found.append((k, cand, info))
    return found


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    frames = _frames(g)
    if len(frames) != 1:
        raise ValueError(f"expected exactly one canonical frame, found {len(frames)}")
    k, cand, info = frames[0]
    out = _render_canonical(cand, info)
    if out is None:
        raise ValueError("pile does not fit on the canvas")
    return np.rot90(out, k).astype(int)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _random_tile(rng, h, w):
    for _ in range(200):
        density = rng.uniform(0.35, 0.9)
        m = rng.random((h, w)) < density
        if m[0].any() and m[-1].any() and m[:, 0].any() and m[:, -1].any() and m.sum() >= 2:
            return m
    m = np.ones((h, w), dtype=bool)
    return m


def _dilate(mask):
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src = mask[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)]
            out[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)] |= src
    return out


def _layout(rng, spec):
    """Build one canonical right-exit layout, then mirror for left exits."""
    h, w, N = spec["h"], spec["w"], spec["N"]
    col = spec["colors"]
    bg, block, tile, pusher, counter = (col[k] for k in ("bg", "block", "tile", "pusher", "counter"))
    big = spec.get("big", False)
    sky_extra = int(rng.integers(1 if big else 0, (11 if big else 7)))
    top = h + 1 + sky_extra
    d_min = 2
    if spec.get("gap"):
        d_min = max(d_min, N * h + 1)
    d_min = max(d_min, N * h - top)
    d_max = 30 - top
    if d_min > d_max:
        return None
    D = int(rng.integers(d_min, min(d_max, d_min + (12 if big else 6)) + 1))
    H = top + D
    sw_min = w + (3 if spec.get("far") else 2)
    Sw = sw_min + int(rng.integers(4 if big else 0, (17 if big else 8)))
    mE = w + int(rng.integers(0, 4))
    mO = 0 if rng.random() < 0.25 else int(rng.integers(1, 7))
    W = Sw + mE + mO
    if W > 30:
        return None
    L, R = mO, mO + Sw - 1
    g = np.full((H, W), bg, dtype=int)
    g[top:, L:R + 1] = block
    lo, hi = L + 1, R - w + 1
    choices = list(range(lo, hi + 1))
    if spec.get("far"):
        choices = [c for c in choices if R - (c + w - 1) > c - L]
    elif spec.get("near"):
        choices = [c for c in choices if R - (c + w - 1) < c - L]
    if not choices:
        return None
    c0 = int(rng.choice(choices))
    pattern = spec["pattern"] if spec.get("pattern") is not None else _random_tile(rng, h, w)
    tr0 = top - h
    g[tr0:top, c0:c0 + w][pattern] = tile
    p = c0 - 1 if rng.random() < 0.3 else int(rng.integers(L, c0))
    g[top - 1, p] = pusher
    # counters: isolated singletons away from the path band and exit lane
    allowed = g == bg
    allowed[:, R + 1:R + 1 + w] = False
    allowed[max(0, tr0 - 1):top, :] = False
    blocked = _dilate(g != bg)
    allowed &= ~blocked
    cells = []
    for _ in range(N):
        rr, cc = np.nonzero(allowed)
        if len(rr) == 0:
            return None
        i = int(rng.integers(len(rr)))
        r, c = int(rr[i]), int(cc[i])
        g[r, c] = counter
        cells.append((r, c))
        one = np.zeros_like(allowed)
        one[r, c] = True
        allowed &= ~_dilate(one)
    vals, cnts = np.unique(g, return_counts=True)
    cm = dict(zip(vals.tolist(), cnts.tolist()))
    if cm[bg] <= max(v for k, v in cm.items() if k != bg):
        return None
    # independent construction of the expected output
    out = g.copy()
    out[tr0:top, c0:c0 + w] = bg
    out[top - 1, p] = bg
    for r, c in cells:
        out[r, c] = bg
    out[top - 1, c0] = pusher
    for k in range(N):
        r = H - (k + 1) * h
        out[r:r + h, R + 1:R + 1 + w][pattern] = tile
    if spec["direction"] == "left":
        g, out = np.fliplr(g), np.fliplr(out)
    k = spec["rot"]
    return np.rot90(g, k).copy(), np.rot90(out, k).copy()


def _palette(rng, mode, base):
    if mode == "fixed":
        return dict(base)
    if mode == "vary_scene":
        rest = [c for c in range(10) if c not in (base["tile"], base["pusher"], base["counter"])]
        bg, block = (int(v) for v in rng.choice(rest, 2, replace=False))
        return {**base, "bg": bg, "block": block}
    cols = [int(v) for v in rng.choice(10, 5, replace=False)]
    return dict(zip(("bg", "block", "tile", "pusher", "counter"), cols))


def _check_counts(num_train, num_test):
    for v in (num_train, num_test):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2..10 train, 1..4 test and num_train > num_test")


def _try_puzzle(rng, num_train, num_test):
    total = num_train + num_test
    # orientations
    if rng.random() < 0.45:
        rots = [int(rng.integers(4))] * num_train
        test_rots = [rots[0]] * num_test
    else:
        kinds = [int(v) for v in rng.choice(4, int(rng.integers(2, min(4, num_train) + 1)), replace=False)]
        rots = kinds + [int(rng.choice(kinds)) for _ in range(num_train - len(kinds))]
        rng.shuffle(rots)
        test_rots = [int(rng.choice(kinds)) for _ in range(num_test)]
    # palette
    pmode = str(rng.choice(["fixed", "vary_scene", "vary_all"], p=[0.45, 0.35, 0.2]))
    base = dict(zip(("bg", "block", "tile", "pusher", "counter"),
                    (int(v) for v in rng.choice(10, 5, replace=False))))
    # tile shapes / patterns
    smode = str(rng.choice(["3x3", "fixed", "vary"], p=[0.35, 0.25, 0.4]))
    fixed_shape = TILE_SHAPES[int(rng.integers(len(TILE_SHAPES)))]
    same_pattern = rng.random() < 0.25
    shared = None
    if same_pattern:
        sh = (3, 3) if smode == "3x3" else fixed_shape
        shared = _random_tile(rng, *sh)

    def shape():
        if shared is not None:
            return shared.shape
        if smode == "3x3":
            return (3, 3)
        if smode == "fixed":
            return fixed_shape
        return TILE_SHAPES[int(rng.integers(len(TILE_SHAPES)))]

    # train specs
    dirs = ["right", "left"] + [str(rng.choice(["right", "left"])) for _ in range(num_train - 2)]
    rng.shuffle(dirs)
    Ns = [int(v) for v in rng.integers(1, 6, num_train)]
    if len(set(Ns)) < 2 or max(Ns) < 2:
        Ns[0], Ns[-1] = 1 + int(rng.integers(0, 2)), 3 + int(rng.integers(0, 3))
        rng.shuffle(Ns)
    idx = rng.permutation(num_train)
    gap_i, far_i = int(idx[0]), int(idx[1])
    specs = []
    for i in range(num_train):
        h, w = shape()
        far = i == far_i or rng.random() < 0.3
        specs.append({"h": h, "w": w, "N": Ns[i], "direction": dirs[i], "rot": rots[i],
                      "gap": i == gap_i or rng.random() < 0.4, "far": far,
                      "near": (not far) and rng.random() < 0.3, "pattern": shared,
                      "colors": _palette(rng, pmode, base)})
    # test specs: a train/test shift chosen per puzzle
    tmode = str(rng.choice(["extrapolate", "larger", "familiar", "mixed"], p=[0.3, 0.3, 0.15, 0.25]))
    for j in range(num_test):
        h, w = shape()
        mode = tmode if tmode != "mixed" else str(rng.choice(["extrapolate", "larger", "familiar"]))
        if mode == "extrapolate":
            N = max(Ns) + 1 + int(rng.integers(0, 2))
        elif mode == "familiar":
            N = int(rng.choice([n for n in Ns if n >= 2]))
        else:
            N = int(rng.integers(2, max(Ns) + 2))
        far = rng.random() < 0.5
        specs.append({"h": h, "w": w, "N": N, "direction": str(rng.choice(["right", "left"])),
                      "rot": test_rots[j], "gap": rng.random() < 0.4, "far": far,
                      "near": (not far) and rng.random() < 0.3, "big": mode == "larger",
                      "pattern": shared, "colors": _palette(rng, pmode, base)})
    pairs, seen = [], set()
    for spec in specs:
        made = None
        for _ in range(40):
            made = _layout(rng, spec)
            if made is not None and made[0].tobytes() + bytes(made[0].shape) not in seen:
                break
            made = None
        if made is None:
            return None
        seen.add(made[0].tobytes() + bytes(made[0].shape))
        pairs.append({"input": made[0], "output": made[1]})
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    try:
        validate_generated_puzzle(puzzle)
    except (AssertionError, ValueError):
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(300):
        puzzle = _try_puzzle(rng, int(num_train), int(num_test))
        if puzzle is not None:
            return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _pair_info(inp):
    frames = _frames(np.asarray(inp, dtype=int))
    assert len(frames) == 1, "input must have exactly one canonical frame"
    return frames[0]


def validate_generated_puzzle(puzzle):
    assert set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert len(train) >= 2 and len(test) >= 1
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            k, cand, info = _pair_info(inp)
            assert np.array_equal(solve(inp), out), "solver disagrees with constructed output"
            # family invariants
            assert info["N"] >= 1
            assert info["pattern"][0].any() and info["pattern"][-1].any()
            assert info["pattern"][:, 0].any() and info["pattern"][:, -1].any()
            assert info["N"] * info["h"] <= info["H"]
            assert info["tr0"] >= 1, "tile must not touch the canvas edge"
            canon_out = np.rot90(out, -k)
            # block and background unchanged, counters gone, exactly one pusher
            assert np.array_equal(canon_out == info["block"], cand == info["block"])
            assert not (canon_out == info["counter"]).any()
            assert (canon_out == info["pusher"]).sum() == 1
            # counters isolated from every other non-background cell
            cr, cc = np.nonzero(cand == info["counter"])
            for r, c in zip(cr, cc):
                win = cand[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                assert (win != info["bg"]).sum() == 1, "counter not isolated"
            # tile copies: N exact copies in the pile
            assert (canon_out == info["tile"]).sum() == info["N"] * int(info["pattern"].sum())
            info = dict(info, rot=k, cand=cand, out=canon_out)
            infos[split].append(info)
    tr = infos["train"]
    # demonstrations expose both concepts
    assert {i["direction"] for i in tr} == {"left", "right"}, "train must show both push directions"
    assert len({i["N"] for i in tr}) >= 2 and max(i["N"] for i in tr) >= 2, "train must vary counts"
    assert all(i["N"] >= 2 for i in infos["test"]), "test piles have at least two crates"
    assert {i["rot"] for i in infos["test"]} <= {i["rot"] for i in tr}, "unseen orientation at test"

    def differs(info, **kw):
        alt = _render_canonical(info["cand"], info, **kw)
        return alt is None or not np.array_equal(alt, info["out"])

    flip = {"left": "right", "right": "left"}
    # shortcuts that drop parent A (count) or parent B (push direction / fall) fail somewhere
    assert any(differs(i, N=1) for i in tr), "ignoring the count must fail in train"
    assert any(differs(i, N=i["N"] + 1) for i in tr), "counting the pusher too must fail"
    assert any(differs(i, anchor="face") for i in tr), "hanging the pile from the ledge must fail"
    assert any(differs(i, direction=flip[i["direction"]]) for i in tr)

    def nearest_end(i):
        dl = i["tc0"] - i["L"]
        dr = i["R"] - (i["tc0"] + i["w"] - 1)
        return "right" if dr < dl else "left" if dl < dr else None

    assert any(nearest_end(i) not in (None, i["direction"]) for i in tr), \
        "train must refute the nearest-end shortcut"
    for i in infos["test"]:
        assert differs(i, N=1) and differs(i, direction=flip[i["direction"]])
    return True


def self_test():
    seeds = range(42)
    rots, Ns, shapes, dirs, sizes, colors = set(), set(), set(), set(), set(), set()
    for s in seeds:
        p = generate_puzzle(s)
        validate_generated_puzzle(p)
        again = generate_puzzle(s)
        for a, b in zip(p["train"] + p["test"], again["train"] + again["test"]):
            assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for pair in p["train"] + p["test"]:
            k, cand, info = _pair_info(pair["input"])
            rots.add(k)
            Ns.add(info["N"])
            shapes.add((info["h"], info["w"]))
            dirs.add(info["direction"])
            sizes.add(np.asarray(pair["input"]).shape)
            colors.add((info["bg"], info["block"], info["tile"]))
    assert rots == {0, 1, 2, 3}
    assert len(Ns) >= 6 and len(shapes) >= 6 and dirs == {"left", "right"}
    assert len(sizes) >= 40 and len(colors) >= 30
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (2, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    generate_puzzle(5, 2, 1)
    generate_puzzle(6, 10, 4)
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
