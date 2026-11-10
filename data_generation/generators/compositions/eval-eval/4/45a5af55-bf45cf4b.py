"""Fused stamp contours: hierarchical placement followed by shared inward shells.

No files, network, parent modules, random state, or training labels enter solve.
Generation uses block writes and erosion; solve uses indexed support and BFS.
"""
from collections import deque
import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 941761746, 238123947, 784005380, 576620928, 24568996,
              975280365, 419808386, 336472573, 876638105, 208864929,
              697668594, 627284891, 228702237, 804396569, 842387075,
              848593800, 354950329, 330474881, 201730768, 353164325,
              203680140, 992513103, 206853259, 710965245, 871870084,
              401525301, 891608373, 868931742, 298276640, 380845327,
              141536530, 505423451, 830293922, 648081231, 305650632,
              817453136, 514949493, 218220162, 921638813, 639349127,
              618407739)
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a


def _key(a):
    return a.shape, a.tobytes()


def _components(binary):
    remaining = set(map(tuple, np.argwhere(binary)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in OFFSETS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _bbox(binary):
    r, c = np.nonzero(binary)
    assert len(r), "an encoding must have visible support"
    return int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1


def _crop(a, box):
    r0, r1, c0, c1 = box
    return a[r0:r1, c0:c1].copy()


def _separated(a, b):
    # Half-open rectangles with at least one blank row or column between them.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    assert all(np.all(side == bg) for side in (a[0], a[-1], a[:, 0], a[:, -1]))
    multicolor = [cells for cells in _components(a != bg)
                  if len({int(a[r, c]) for r, c in cells}) > 1]
    assert len(multicolor) == 1, "one multicolor stripe legend is required"
    legend_cells = multicolor[0]
    marker = np.zeros(a.shape, dtype=bool)
    for cell in legend_cells:
        marker[cell] = True
    lb = _bbox(marker)
    legend = _crop(a, lb)
    assert np.all(legend != bg) and len(legend_cells) == legend.size
    assert 2 <= legend.shape[1] <= 4 and 3 <= legend.shape[0] <= 16
    assert np.all(legend == legend[:, :1]), "legend rows must be uniform"
    stamp_ink = int(legend[0, 0])
    code = legend[1:, 0].astype(int)
    assert len(set(map(int, code))) >= 2
    assert stamp_ink not in code and bg not in code
    remaining = a.copy()
    remaining[marker] = bg
    colors = set(map(int, np.unique(remaining))) - {bg}
    assert stamp_ink in colors and len(colors) == 2
    mask_ink = (colors - {stamp_ink}).pop()
    assert mask_ink not in code
    mb, sb = _bbox(remaining == mask_ink), _bbox(remaining == stamp_ink)
    assert all(_separated(x, y) for x, y in ((mb, sb), (mb, lb), (sb, lb)))
    mask_crop, stamp_crop = _crop(remaining, mb), _crop(remaining, sb)
    assert set(map(int, np.unique(mask_crop))) <= {bg, mask_ink}
    assert set(map(int, np.unique(stamp_crop))) <= {bg, stamp_ink}
    mask, stamp = mask_crop == mask_ink, stamp_crop == stamp_ink
    assert mask.shape[0] * stamp.shape[0] <= 30
    assert mask.shape[1] * stamp.shape[1] <= 30
    return {"background": bg, "mask_ink": mask_ink, "stamp_ink": stamp_ink,
            "mask": mask, "stamp": stamp, "code": code,
            "boxes": (mb, sb, lb), "legend_width": legend.shape[1]}


def _place(mask, stamp):
    h, w = stamp.shape
    support = np.zeros((mask.shape[0] * h, mask.shape[1] * w), dtype=bool)
    for r, c in np.argwhere(mask):
        support[r*h:(r+1)*h, c*w:(c+1)*w] = stamp
    return support


def _erode_depth(support, diagonal=True):
    """Depth zero is the occupied boundary; all exterior cells have depth -1."""
    current = support.copy()
    depth = np.full(support.shape, -1, dtype=int)
    level = 0
    offsets = ((r, c) for r in range(3) for c in range(3)
               if diagonal or r == 1 or c == 1)
    offsets = tuple(offsets)
    while np.any(current):
        depth[current] = level
        pad = np.pad(current, 1, constant_values=False)
        current = np.logical_and.reduce([pad[r:r+support.shape[0], c:c+support.shape[1]]
                                         for r, c in offsets])
        level += 1
    return depth


def _bfs_depth(support):
    """Chessboard distance to any blank cell, including an exterior blank rim."""
    pad = np.pad(support, 1, constant_values=False)
    distance = np.full(pad.shape, -1, dtype=int)
    queue = deque()
    for r, c in np.argwhere(~pad):
        distance[r, c] = 0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for dr, dc in OFFSETS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < pad.shape[0] and 0 <= nc < pad.shape[1]
                    and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance[1:-1, 1:-1] - 1


def _paint(support, depth, code, bg):
    output = np.full(support.shape, int(bg), dtype=np.int8)
    output[support] = np.asarray(code)[np.minimum(depth[support], len(code)-1)]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read(input_grid)
    mask, stamp = info["mask"], info["stamp"]
    h, w = stamp.shape
    rr, cc = np.indices((mask.shape[0]*h, mask.shape[1]*w))
    support = mask[rr // h, cc // w] & stamp[rr % h, cc % w]
    depth = _bfs_depth(support)
    assert depth.max() >= len(info["code"])-1, "every legend level must be reachable"
    return _paint(support, depth, info["code"], info["background"])


def _has_cavity(binary):
    return any(all(r not in (0, binary.shape[0]-1) and c not in (0, binary.shape[1]-1)
                   for r, c in comp) for comp in _components(~binary))


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_mask(rng, shape, family):
    h, w = shape
    rr, cc = np.indices(shape)
    mask = np.ones(shape, dtype=bool)
    if family == "ring":
        mask = (rr == 0) | (rr == h-1) | (cc == 0) | (cc == w-1)
        # A variable rectangular cavity, with a possibly thicker outer rim.
        if h > 3 and w > 3 and rng.random() < .5:
            mask[:] = True
            r0, c0 = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
            mask[r0:int(rng.integers(r0+1, h)), c0:int(rng.integers(c0+1, w))] = False
    elif family == "notch":
        dh = int(rng.integers(1, max(2, h)))
        dw = int(rng.integers(1, max(2, w)))
        mask[:min(dh, h-1), :min(dw, w-1)] = False
    elif family == "diagonal":
        mask[:] = False
        for r in range(h):
            mask[r, round(r*(w-1)/max(1, h-1))] = True
        for c in range(w):
            mask[round(c*(h-1)/max(1, w-1)), c] = True
    elif family == "islands":
        mask = (rr % 2 == 0) & (cc % 2 == 0)
        mask[0, 0] = mask[-1, -1] = True
    elif family == "stairs":
        mask = cc <= np.floor((rr+1)*w/h).astype(int)
    elif family != "full":
        raise ValueError(f"unknown mask family {family}")
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    # Crop instead of inventing invisible padding for thin masks.
    return _crop(mask, _bbox(mask))


def _make_stamp(rng, shape, holes=False, asymmetric=False):
    h, w = shape
    stamp = np.ones(shape, dtype=bool)
    if holes:
        # Two cells of material separate internal cavities from exterior notches.
        r0, c0 = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        if rng.random() < .45:
            r1, c1 = int(rng.integers(r0+1, h-1)), int(rng.integers(c0+1, w-1))
            stamp[r0:r1, c0:c1] = False
        else:
            stamp[r0, c0] = False
            for _ in range(int(rng.integers(3))):
                stamp[int(rng.integers(2, h-2)), int(rng.integers(2, w-2))] = False
    if asymmetric:
        stamp[0, 1] = False
    elif rng.random() < .3:
        side = int(rng.integers(4))
        if side == 0:
            stamp[0, :int(rng.integers(1, w))] = False
        elif side == 1:
            stamp[-1, :int(rng.integers(1, w))] = False
        elif side == 2:
            stamp[:int(rng.integers(1, h)), 0] = False
        else:
            stamp[:int(rng.integers(1, h)), -1] = False
    return stamp


def _layout(rng, arrays, bg):
    # Pack independent, upright encodings, then vary offsets and blank padding.
    order = [int(i) for i in rng.permutation(3)]
    horizontal = bool(rng.integers(2))
    if not horizontal:
        arrays = [a.T for a in arrays]
    gap = int(rng.integers(1, 4))
    width = sum(a.shape[1] for a in arrays) + 2*gap + 2
    if width > 30:
        gap = 1
        width = sum(a.shape[1] for a in arrays) + 4
    assert width <= 30
    height = max(a.shape[0] for a in arrays) + 2
    height += int(rng.integers(min(6, 30-height)+1))
    width += int(rng.integers(min(5, 30-width)+1))
    canvas = np.full((height, width), bg, dtype=np.int8)
    unused = width - (sum(a.shape[1] for a in arrays) + 2*gap + 2)
    left = 1 + int(rng.integers(unused+1))
    for i in order:
        a = arrays[i]
        top = int(rng.integers(1, height-a.shape[0]))
        canvas[top:top+a.shape[0], left:left+a.shape[1]] = a
        left += a.shape[1] + gap
    return canvas if horizontal else canvas.T.copy()


def _sample_code(rng, depth, available, witness):
    maximum = min(6, int(depth.max())+1)
    if witness == "merge":
        assert maximum >= 3
        return np.array([available[0], available[0], available[1]], dtype=int)
    length = int(rng.integers(2, maximum+1))
    if witness == "cavity":
        length = min(length, 3)
        code = np.array([available[i % 2] for i in range(length)], dtype=int)
    else:
        k = int(rng.integers(2, min(len(available), length)+1))
        style = int(rng.integers(4))
        labels = np.arange(length) % k if style == 0 else np.sort(np.arange(length) % k) if style == 1 else rng.integers(k, size=length)
        labels[0], labels[-1] = 0, k-1
        if style == 3 and length >= 3:
            labels[-1], labels[1] = 0, 1
        code = np.asarray(available)[labels]
    return code


def _make_pair(rng, palette, regime, split, index):
    bg, mi, si = map(int, palette[:3])
    witness = "merge" if split == "train" and index == 0 else "cavity" if split == "train" and index == 1 else None
    if witness == "merge":
        shape = (2, 3) if rng.integers(2) else (3, 2)
        mask = np.ones(shape, dtype=bool)
        mask[0, -1] = False
        if rng.integers(2):
            mask = np.flipud(mask)
        # A completely solid stamp exposes deep contours crossing tile seams.
        stamp = np.ones((int(rng.integers(4, 7)), int(rng.integers(4, 7))), dtype=bool)
    else:
        coarse_hole = ((regime == 1) if split == "train" else (regime == 0))
        fine_hole = not coarse_hole
        if split == "test" and rng.random() < .5:
            coarse_hole = fine_hole = True
        if split == "train" and witness is None:
            coarse_hole = coarse_hole and bool(rng.integers(2))
            fine_hole = (regime == 0) and bool(rng.integers(2))
        sh = int(rng.integers(5 if fine_hole else 4, 9))
        sw = int(rng.integers(5 if fine_hole else 4, 9))
        lo = 3 if coarse_hole else 2
        mh = int(rng.integers(lo, min(6, 30//sh)+1))
        mw = int(rng.integers(lo, min(6, 30//sw)+1))
        family = "ring" if coarse_hole else _choose(rng, ["full", "notch", "diagonal", "islands", "stairs"])
        mask = _make_mask(rng, (mh, mw), family)
        stamp = _make_stamp(rng, (sh, sw), fine_hole, asymmetric=witness == "cavity")
    support = _place(mask, stamp)
    depth = _erode_depth(support)
    if depth.max() < 1:
        return None
    code = _sample_code(rng, depth, palette[3:], witness)
    legend_width = int(rng.integers(2, 5))
    legend = np.repeat(np.r_[si, code][:, None], legend_width, axis=1)
    arrays = [np.where(mask, mi, bg), np.where(stamp, si, bg), legend]
    inp = _layout(rng, arrays, bg)
    return {"input": inp, "output": _paint(support, depth, code, bg)}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(2))
    stable = rng.random() < .2
    palette = rng.permutation(10)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(48):
                colors = palette if stable else rng.permutation(10)
                pair = _make_pair(rng, colors, regime, split, index)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _local_coloring(info):
    stamp, mask = info["stamp"], info["mask"]
    tile = _paint(stamp, _erode_depth(stamp), info["code"], info["background"])
    h, w = stamp.shape
    result = np.full((mask.shape[0]*h, mask.shape[1]*w), info["background"], dtype=np.int8)
    for r, c in np.argwhere(mask):
        result[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    return result


def _alternative(info, name):
    mask, stamp, code, bg = info["mask"], info["stamp"], info["code"], info["background"]
    support = _place(mask, stamp)
    depth = _erode_depth(support, diagonal=name != "manhattan")
    if name == "local":
        return _local_coloring(info)
    if name == "no_a":
        return np.where(support, code[0], bg)
    if name == "no_b_mask":
        support = _place(np.ones_like(mask), stamp)
        depth = _erode_depth(support)
    if name == "no_b_stamp":
        support = _place(mask, np.ones_like(stamp))
        depth = _erode_depth(support)
    if name == "reverse":
        code = code[::-1]
    if name == "collapse":
        code = code[np.r_[True, code[1:] != code[:-1]]]
    if name == "cycle":
        depth = depth % len(code)
    return _paint(support, depth, code, bg)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            # Normalize dtype for duplicate detection on list/array inputs.
            key = _key(inp.astype(np.int8))
            assert key not in seen
            seen.add(key)
            info = _read(inp)
            infos[split].append(info)
            support = _place(info["mask"], info["stamp"])
            depth = _erode_depth(support)
            assert depth.max() >= len(info["code"])-1
            assert out.shape == support.shape
            assert np.array_equal(out != info["background"], support)
            assert np.array_equal(out, _paint(support, depth, info["code"], info["background"]))
            assert np.array_equal(solve(inp), out)
            assert set(map(int, np.unique(out))) - {info["background"]} == set(map(int, info["code"]))
            assert info["mask_ink"] not in out and info["stamp_ink"] not in out
    # All these simpler hypotheses must be contradicted by training alone.
    alternatives = ("no_a", "no_b_mask", "no_b_stamp", "local", "manhattan", "reverse", "collapse", "cycle")
    for name in alternatives:
        assert any(not np.array_equal(_alternative(info, name), pair["output"])
                   for info, pair in zip(infos["train"], puzzle["train"])), name
    # The visible geometry must establish the orientation of BOTH sources.
    for role in ("mask", "stamp"):
        for transform in range(1, 8):
            if all(np.array_equal(_place(_transforms(i["mask"])[transform] if role == "mask" else i["mask"],
                                         _transforms(i["stamp"])[transform] if role == "stamp" else i["stamp"]),
                                  _place(i["mask"], i["stamp"])) for i in infos["train"]):
                assert all(np.array_equal(_transforms(i[role])[transform], i[role]) for i in infos["test"])
    train_coarse = any(_has_cavity(i["mask"]) for i in infos["train"])
    train_fine = any(_has_cavity(i["stamp"]) for i in infos["train"])
    assert train_coarse != train_fine, "one cavity source scale is held out"
    assert all(_has_cavity(i["stamp"] if train_coarse else i["mask"]) for i in infos["test"])


def _digest(puzzle):
    payload = {s: [{k: np.asarray(v).tolist() for k, v in p.items()} for p in puzzle[s]]
               for s in ("train", "test")}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def self_test():
    """Stream 42 seeds; retain signatures and counters, never a corpus of grids."""
    signatures, shapes, masks, stamps, codes, colors, regimes = (set() for _ in range(7))
    counts = {"pairs": 0, "coarse_cavities": 0, "fine_cavities": 0,
              "disconnected_masks": 0, "rectangular_stamps": 0, "repeated_codes": 0}
    first = None
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(_digest(p))
        regimes.add(any(_has_cavity(_read(x["input"])["mask"]) for x in p["train"]))
        for pair in p["train"] + p["test"]:
            i = _read(pair["input"])
            counts["pairs"] += 1
            shapes.add((pair["input"].shape, pair["output"].shape))
            masks.add(_key(i["mask"]))
            stamps.add(_key(i["stamp"]))
            rolemap = {}
            codes.add(tuple(rolemap.setdefault(int(c), len(rolemap)) for c in i["code"]))
            colors.update(map(int, np.unique(pair["input"])))
            counts["coarse_cavities"] += int(_has_cavity(i["mask"]))
            counts["fine_cavities"] += int(_has_cavity(i["stamp"]))
            counts["disconnected_masks"] += int(len(_components(i["mask"])) > 1)
            counts["rectangular_stamps"] += int(i["stamp"].shape[0] != i["stamp"].shape[1])
            counts["repeated_codes"] += int(np.any(i["code"][1:] == i["code"][:-1]))
        if first is None:
            first = p
    assert len(signatures) == 42 and len(regimes) == 2 and colors == set(range(10))
    assert len(shapes) >= 70 and len(masks) >= 25 and len(stamps) >= 45 and len(codes) >= 5
    assert all(counts[name] >= 5 for name in counts if name != "pairs")
    assert _digest(generate_puzzle(0)) == _digest(first)
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000 + 10*a + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact semantic fixtures: a solid rectangle reproduces A's square frames.
    square = np.ones((8, 8), dtype=bool)
    expected = np.array([2, 2, 5, 7])[np.minimum.outer(np.minimum(np.arange(8), np.arange(8)[::-1]),
                                                                     np.minimum(np.arange(8), np.arange(8)[::-1]))]
    assert np.array_equal(_paint(square, _bfs_depth(square), [2, 2, 5, 7], 0), expected)
    square[3, 3] = False
    assert _bfs_depth(square)[2, 2] == 0  # diagonal contact with a hole is boundary
    assert np.array_equal(_bfs_depth(square), _erode_depth(square))
    # Input-only fixtures beyond the sampler: maximum depth/code, a one-cell
    # primitive, and a disconnected stamp. Build the visible sources directly.
    separated = np.ones((5, 7), dtype=bool)
    separated[:, 3] = False
    fixtures = [(np.ones((5, 5), dtype=bool), np.ones((6, 6), dtype=bool),
                 np.array([1 + j % 7 for j in range(15)])),
                (np.ones((8, 9), dtype=bool), np.ones((1, 1), dtype=bool),
                 np.array([1, 2, 3, 4])),
                (np.array([[1, 1, 0], [1, 1, 1]], dtype=bool), separated,
                 np.array([1, 2]))]
    for mask, stamp, code in fixtures:
        legend = np.repeat(np.r_[9, code][:, None], 2, axis=1)
        arrays = [np.where(mask, 8, 0), np.where(stamp, 9, 0), legend]
        canvas = np.zeros((max(a.shape[0] for a in arrays)+2,
                           sum(a.shape[1] for a in arrays)+4), dtype=int)
        left = 1
        for a in arrays:
            canvas[1:1+a.shape[0], left:left+a.shape[1]] = a
            left += a.shape[1]+1
        support = _place(mask, stamp)
        expected = _paint(support, _erode_depth(support), code, 0)
        assert np.array_equal(solve(canvas), expected)
    assert expected.shape == (10, 21)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "mask_patterns": len(masks), "stamp_patterns": len(stamps),
            "normalized_codes": len(codes), "cavity_transfer_directions": len(regimes), **counts}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
