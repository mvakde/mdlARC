"""Overlap-wave packing: stable edge packing controls local state exchange.

All semantic information is visible in each input. A frame's single side mark
selects its edge; two cavity templates are supplied by the tiles themselves.
"""
from __future__ import annotations

from math import gcd
from functools import reduce
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _band(a, b):
    return max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"])


def _overlap(a, b, gap=0):
    return (max(a["r"], b["r"]) < min(a["r"] + a["h"], b["r"] + b["h"]) + gap
            and max(a["c"], b["c"]) < min(a["c"] + a["w"], b["c"] + b["w"]) + gap)


def _order(objects, d):
    return sorted((i for i, o in enumerate(objects) if o["d"] == d),
                  key=lambda i: (objects[i]["c"] if d == 0 else
                                 -objects[i]["c"] - objects[i]["w"], objects[i]["r"]))


def _pack(objects, width):
    """Boundary recurrence and longest predecessor-chain depth."""
    out = [dict(o) for o in objects]
    for d in (0, 1):
        prior = []
        for i in _order(objects, d):
            blockers = [j for j in prior if _band(objects[i], objects[j])]
            out[i]["depth"] = 1 + max((out[j]["depth"] for j in blockers), default=0)
            out[i]["c"] = (max((out[j]["c"] + out[j]["w"] for j in blockers), default=0)
                            if d == 0 else min((out[j]["c"] for j in blockers), default=width)
                            - out[i]["w"])
            if not 0 <= out[i]["c"] <= width - out[i]["w"]:
                raise ValueError("packing chain exceeds canvas")
            prior.append(i)
    if any(_overlap(a, b) for i, a in enumerate(out) for b in out[i + 1:]):
        raise ValueError("opposite final groups collide")
    return out


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            edges = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            lengths.extend(b - a for a, b in zip(edges, edges[1:]))
    s = reduce(gcd, lengths)
    small = crop[::s, ::s]
    if not np.array_equal(crop, np.repeat(np.repeat(small, s, 0), s, 1)):
        raise ValueError("nonuniform scale blocks")
    return s, small


def _read_frames(grid):
    bg = int(np.argmax(np.bincount(grid.ravel(), minlength=10)))
    if np.count_nonzero(grid == bg) * 2 <= grid.size:
        raise ValueError("background must occupy more than half the input")
    covered = grid == bg
    occupied = np.zeros(grid.shape, dtype=bool)
    objects = []
    while np.any(~covered):
        r, c = map(int, np.argwhere(~covered)[0])
        right, bottom = c, r
        while right < grid.shape[1] and grid[r, right] != bg:
            right += 1
        while bottom < grid.shape[0] and grid[bottom, c] != bg:
            bottom += 1
        crop = grid[r:bottom, c:right]
        if np.any(occupied[r:bottom, c:right]):
            raise ValueError("interleaved tile enclosures")
        s, small = _primitive(crop)
        h, w = small.shape
        if min(h, w) < 5:
            raise ValueError("logical tiles need at least a 3 by 3 cavity")
        frame = int(small[0, 0])
        perimeter = np.zeros((h, w), dtype=bool)
        perimeter[0] = perimeter[-1] = True
        perimeter[:, 0] = perimeter[:, -1] = True
        marks = np.argwhere(perimeter & (small != frame))
        if len(marks) != 1:
            raise ValueError("each frame needs exactly one side mark")
        y, x = map(int, marks[0])
        sides = {(h // 2, 0): (1, 0), (h // 2, w - 1): (1, 1),
                 (0, w // 2): (0, 0), (h - 1, w // 2): (0, 1)}
        if (y, x) not in sides:
            raise ValueError("mark must be centered on a side")
        marker = int(small[y, x])
        cavity = small[1:-1, 1:-1].copy()
        if marker == bg or frame == bg or np.any((cavity == frame) | (cavity == marker)):
            raise ValueError("frame and marker colors are reserved")
        axis, d = sides[y, x]
        objects.append(dict(r=r, c=c, h=bottom-r, w=right-c, s=s,
                            logical=(h, w), frame=frame, marker=marker,
                            axis=axis, d=d, cavity=cavity))
        occupied[r:bottom, c:right] = True
        covered[r:bottom, c:right] = True
    if len(objects) < 2 or len({o["axis"] for o in objects}) != 1:
        raise ValueError("tiles must share one packing axis")
    if any(_overlap(a, b, 1) for i, a in enumerate(objects) for b in objects[i + 1:]):
        raise ValueError("input tile rectangles need a background gap")
    return bg, objects


def _read_input(value):
    grid = _grid(value)
    bg, objects = _read_frames(grid)
    axis = objects[0]["axis"]
    if axis == 0:
        grid = grid.T.copy()
        bg, objects = _read_frames(grid)
    for field in ("logical", "frame", "marker"):
        if len({o[field] for o in objects}) != 1:
            raise ValueError("pair-local frame conventions must agree")
    signatures = sorted({_key(o["cavity"]) for o in objects})
    if len(signatures) != 2:
        raise ValueError("input must contain exactly two cavity states")
    templates = [next(o["cavity"] for o in objects if _key(o["cavity"]) == k).copy()
                 for k in signatures]
    minimum = min(o["s"] for o in objects)
    if {_key(o["cavity"]) for o in objects if o["s"] == minimum} != set(signatures):
        raise ValueError("both states need a common minimum reference scale")
    common = (templates[0] == templates[1]) & (templates[0] != bg)
    if np.count_nonzero(common) < 3:
        raise ValueError("states need a shared visible interior scaffold")
    for o in objects:
        o["state"] = signatures.index(_key(o["cavity"]))
    return dict(grid=grid, background=bg, axis=axis, objects=objects, templates=templates)


def _render(shape, bg, objects, templates, exchange=False):
    grid = np.full(shape, bg, dtype=np.int8)
    occupied = np.zeros(shape, dtype=bool)
    for o in objects:
        r, c, h, w, s = (o[k] for k in ("r", "c", "h", "w", "s"))
        if not (0 <= r <= shape[0]-h and 0 <= c <= shape[1]-w):
            raise ValueError("tile clipped")
        if np.any(occupied[r:r+h, c:c+w]):
            raise ValueError("tile collision")
        lh, lw = h // s, w // s
        small = np.full((lh, lw), o["frame"], dtype=np.int8)
        small[lh // 2, 0 if o["d"] == 0 else lw - 1] = o["marker"]
        state = o["state"] ^ (int(o["depth"] % 2 == 1) if exchange else 0)
        small[1:-1, 1:-1] = templates[state]
        grid[r:r+h, c:c+w] = np.repeat(np.repeat(small, s, 0), s, 1)
        occupied[r:r+h, c:c+w] = True
    return grid


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read marks, local templates, scales and overlap chains from this input."""
    info = _read_input(input_grid)
    settled = _pack(info["objects"], info["grid"].shape[1])
    out = _render(info["grid"].shape, info["background"], settled, info["templates"], True)
    return out if info["axis"] == 1 else out.T.copy()


def _design(rng, h, w, bg, colors, kind):
    """Pair-local sparse or dense states, registered by their enclosure."""
    shape = h - 2, w - 2
    points = list(np.ndindex(shape))
    rng.shuffle(points)
    n = int(rng.integers(3, len(points) - 2))
    stable, available = points[:n], points[n:]
    a, b = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
    for j, p in enumerate(stable):
        color = colors[0] if j < 3 or rng.random() < .7 else colors[1]
        a[p] = b[p] = color
    if kind == "recolor":
        changing = available[:int(rng.integers(1, len(available) + 1))]
        for j, p in enumerate(changing):
            a[p], b[p] = colors[2 + j % 2], colors[4]
    elif kind == "relocate":
        cut = int(rng.integers(1, len(available)))
        for p in available[:cut]:
            a[p] = colors[2]
        for p in available[cut:]:
            b[p] = colors[2]
    else:
        # One common-support recoloring and independently sized relocated patches.
        a[available[0]], b[available[0]] = colors[2], colors[4]
        cut = int(rng.integers(2, len(available)))
        for j, p in enumerate(available[1:cut]):
            a[p] = colors[2 + j % 2]
        for p in available[cut:]:
            b[p] = colors[4]
    return [a, b]


def _object(r, c, h, w, s, d, state, frame, marker):
    return dict(r=int(r), c=int(c), h=h*s, w=w*s, s=s, d=d,
                state=state, frame=frame, marker=marker)


def _chain_layout(rng, h, w, length, enlarged):
    # Adjacent links overlap; nonadjacent endpoints can be disjoint. The empty
    # band also supplies independent tiles that reuse the same destination edge.
    d = int(rng.integers(2))
    if enlarged == 3:
        steps = [3] * (length - 1)
    else:
        steps = [int(rng.integers((h + 1)//2, h)) for _ in range(length - 1)]
    rows = [0]
    for step in steps:
        rows.append(rows[-1] + step)
    if rng.integers(2):
        rows = [rows[-1] - r for r in rows]
    chain_h = max(rows) + h
    minimum_w = length*w + length-1
    width = int(rng.integers(minimum_w, 31))
    chain_free = width - minimum_w
    gaps = rng.multinomial(chain_free, np.full(length+1, 1/(length+1)))
    cols, c = [], int(gaps[0])
    for i in range(length):
        cols.append(c)
        c += w + 1 + int(gaps[i+1])
    lower = chain_h + 1
    height = int(rng.integers(lower + h*enlarged, 31))
    dy = int(rng.integers(height-(lower+h*enlarged)+1))
    records = [(r+dy, c, 1, d, 0 if i < 2 else 1)
               for i, (r, c) in enumerate(zip(rows, cols))]
    # Both reference states occur at scale 1, regardless of the enlarged copy.
    records += [(lower+dy, int(rng.integers(width-w*enlarged-w)), enlarged, d, 1),
                (lower+dy, width-w, 1, 1-d, 1)]
    # Keep the independent tiles separated even in their source placement.
    if d == 1:
        records = [(r, width-c-w*s, s, direction, state)
                   for r, c, s, direction, state in records]
    return (height, width), records


def _mixed_layout(rng, h, w):
    # Two disjoint small branches both follow a large root. A fourth tile
    # bridges them: three ancestors, but longest-chain depth three, not four.
    d = int(rng.integers(2))
    width = int(rng.integers(4*w+3, 31))
    height = int(rng.integers(3*h+2, 31))
    dy = int(rng.integers(height-(3*h+2)+1))
    dx = int(rng.integers(width-(4*w+2)+1))
    records = [(dy, dx, 2, d, 0), (dy, dx+2*w+1, 1, d, 0),
               (dy+h+1, dx+2*w+1, 1, d, 1),
               (dy+h-1, dx+3*w+2, 1, d, 1),
               (dy+2*h+2, int(rng.integers(width-w+1)), 1, 1-d, 1)]
    if d == 1:
        records = [(r, width-c-w*s, s, direction, state)
                   for r, c, s, direction, state in records]
    return (height, width), records


def _free_layout(rng, h, w, frame, marker):
    shape = int(rng.integers(max(14, 2*h+1), 31)), int(rng.integers(max(16, 2*w+1), 31))
    objects = []
    target = int(rng.integers(2, 9))
    for i in range(target):
        scale = 1 if i < 2 else int(_pick(rng, [1, 1, 1, 2]))
        for _ in range(100):
            if h*scale > shape[0] or w*scale > shape[1]:
                break
            o = _object(rng.integers(shape[0]-h*scale+1), rng.integers(shape[1]-w*scale+1),
                        h, w, scale, int(rng.integers(2)), i if i < 2 else int(rng.integers(2)), frame, marker)
            if any(_overlap(o, p, 1) for p in objects):
                continue
            try:
                _pack(objects+[o], shape[1])
            except ValueError:
                continue
            objects.append(o)
            break
        else:
            continue
    if len(objects) < 2 or {o["state"] for o in objects if o["s"] == 1} != {0, 1}:
        raise ValueError("reference placement failed")
    return shape, objects


def _make_pair(rng, axis, purpose):
    for _ in range(160):
        h = 5 if purpose == "deep" else int(rng.integers(5, 8))
        w = int(rng.integers(5, 7 if purpose in ("mixed", "deep") else 8))
        palette = [int(c) for c in rng.permutation(10)]
        bg, frame, marker = palette[:3]
        kind = "combined" if purpose == "teach" else _pick(rng, ["recolor", "relocate", "combined"])
        templates = _design(rng, h, w, bg, palette[3:8], kind)
        try:
            if purpose == "mixed":
                shape, records = _mixed_layout(rng, h, w)
            elif purpose in ("teach", "deep"):
                enlarged = int(_pick(rng, [2, 2, 3])) if purpose == "deep" else 1
                shape, records = _chain_layout(rng, h, w, 4 if purpose == "deep" else 3, enlarged)
            else:
                shape, objects = _free_layout(rng, h, w, frame, marker)
            if purpose != "free":
                objects = [_object(r, c, h, w, s, d, state, frame, marker)
                           for r, c, s, d, state in records]
            if any(_overlap(a, b, 1) for i, a in enumerate(objects) for b in objects[i+1:]):
                continue
            if rng.integers(2):
                for o in objects:
                    o["state"] ^= 1
            settled = _pack(objects, shape[1])
            inp = _render(shape, bg, objects, templates)
            out = _render(shape, bg, settled, templates, True)
            if min(np.count_nonzero(inp == bg), np.count_nonzero(out == bg))*2 <= inp.size:
                continue
            if np.array_equal(inp, out):
                continue
            if axis == 0:
                inp, out = inp.T.copy(), out.T.copy()
            # Decoder agreement is an author-owned check, not independent proof.
            if not np.array_equal(solve(inp), out):
                raise AssertionError("rendered labels and input reader disagree")
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    axis = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            purpose = ("teach" if split == "train" and i == 0 else "mixed"
                       if split == "train" and i == 1 else "deep"
                       if split == "test" and i == 0 else "free")
            for _ in range(40):
                pair = _make_pair(rng, axis, purpose)
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _audit_settle(objects, shape):
    """Supplemental check: unit-cell physical packing and recursive graph depth.

    This does not call the generator's boundary recurrence.
    """
    out = [dict(o) for o in objects]
    for d in (0, 1):
        group = [o for o in out if o["d"] == d]
        for _ in range(shape[1]*max(1, len(group))+1):
            moved = False
            for o in group:
                candidate = dict(o, c=o["c"] + (-1 if d == 0 else 1))
                if (0 <= candidate["c"] <= shape[1]-o["w"]
                        and not any(p is not o and _overlap(candidate, p) for p in group)):
                    o["c"] = candidate["c"]
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError("physical packing failed to converge")
    memo = {}
    def depth(i):
        if i not in memo:
            a = objects[i]
            predecessors = [j for j, b in enumerate(objects) if a["d"] == b["d"] and _band(a, b)
                            and (b["c"]+b["w"] <= a["c"] if a["d"] == 0 else
                                 b["c"] >= a["c"]+a["w"])]
            memo[i] = 1 + max((depth(j) for j in predecessors), default=0)
        return memo[i]
    for i, o in enumerate(out):
        o["depth"] = depth(i)
    return out


def _features(info, settled):
    a, b = info["templates"]
    bg = info["background"]
    objects = info["objects"]
    return dict(axis=info["axis"], shape=info["grid"].shape, count=len(objects),
                logical=objects[0]["logical"], scales=tuple(sorted(o["s"] for o in objects)),
                depths=tuple(sorted(o["depth"] for o in settled)),
                moving=sum(a["c"] != b["c"] for a, b in zip(objects, settled)),
                recolor=bool(np.any((a != bg) & (b != bg) & (a != b))),
                relocate=bool(np.any((a == bg) != (b == bg))),
                vocabulary=tuple(tuple((x != bg).ravel()) for x in (a, b)),
                background=bg,
                palette=tuple(int(x) for x in np.unique(info["grid"])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, axes, train = set(), set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            axes.add(info["axis"])
            settled = _audit_settle(info["objects"], info["grid"].shape)
            expected = _render(info["grid"].shape, info["background"], settled, info["templates"], True)
            actual = out if info["axis"] == 1 else out.T
            assert np.array_equal(expected, actual), "incorrect packing, depth or template exchange"
            assert np.array_equal(solve(inp), out)
            assert np.count_nonzero(out == info["background"])*2 > out.size
            if split == "train":
                train.append((info, settled))
    assert len(axes) == 1, "generated puzzles use one demonstrated axis"
    # These are demonstration-coverage checks, not assumptions made by solve.
    assert any({1, 2, 3} <= {o["depth"] for o in settled} for _, settled in train)
    assert any(any(a["state"] == b["state"] and a["s"] == b["s"]
                       and settled[i]["depth"] % 2 != settled[j]["depth"] % 2
                       for i, a in enumerate(info["objects"]) for j, b in enumerate(info["objects"]))
               for info, settled in train), "missing state-versus-depth witness"
    assert {o["state"] for info, settled in train for o in settled if o["depth"] % 2} == {0, 1}
    assert any(len({o["s"] for o in info["objects"]}) > 1 for info, _ in train)
    assert any(_features(info, settled)["recolor"] and _features(info, settled)["relocate"]
               for info, settled in train)
    assert any(any(a["c"] != b["c"] for a, b in zip(info["objects"], settled)) for info, settled in train)


def self_test():
    """Bounded 42-seed variation, count boundaries, ablations and mutations."""
    corpus = [generate_puzzle(seed) for seed in range(42)]
    fingerprints, rows, shapes, vocabularies = set(), [], set(), set()
    for seed, p in enumerate(corpus):
        fingerprint = tuple(_key(pair[k]) for s in ("train", "test") for pair in p[s] for k in ("input", "output"))
        fingerprints.add(fingerprint)
        again = generate_puzzle(seed)
        assert fingerprint == tuple(_key(pair[k]) for s in ("train", "test") for pair in again[s] for k in ("input", "output"))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _read_input(pair["input"])
                settled = _audit_settle(info["objects"], info["grid"].shape)
                f = _features(info, settled)
                rows.append(f)
                shapes.add(pair["input"].shape)
                vocabularies.add((f["logical"], f["vocabulary"]))
                # Omitting A: exchange every tile at its original anchor.
                fixed = [dict(o, depth=1) for o in info["objects"]]
                no_a = _render(info["grid"].shape, info["background"], fixed, info["templates"], True)
                # Omitting B: compact the carriers and copy their original states.
                no_b = _render(info["grid"].shape, info["background"], settled, info["templates"])
                canonical_out = pair["output"] if info["axis"] == 1 else pair["output"].T
                if split == "train" and pair is p["train"][0]:
                    assert not np.array_equal(no_a, canonical_out)
                    assert not np.array_equal(no_b, canonical_out)
                    # Unconditional exchange also fails on the even chain link.
                    all_swap = _render(info["grid"].shape, info["background"],
                                       [dict(o, depth=1) for o in settled], info["templates"], True)
                    assert not np.array_equal(all_swap, canonical_out)
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(vocabularies) >= 100
    assert {r["axis"] for r in rows} == {0, 1}
    assert {1, 2, 3} <= {s for r in rows for s in r["scales"]}
    assert {1, 2, 3, 4} <= {d for r in rows for d in r["depths"]}
    assert len({r["count"] for r in rows}) >= 5
    assert len({r["logical"] for r in rows}) >= 6
    assert len({r["background"] for r in rows}) == 10
    assert {(r["recolor"], r["relocate"]) for r in rows} == {(True, False), (False, True), (True, True)}
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            generate_puzzle(2026, nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = corpus[0]
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = broken["test"][0]["output"]
    point = tuple(np.argwhere(out != broken["test"][0]["input"])[0])
    out[point] = (int(out[point])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, explicit counts, ablations and mutations)")
