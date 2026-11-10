"""Stencil welding: a rotated shared stencil gains local support in two contexts.

Only the standard library and NumPy are used. There is no puzzle-wide solver state.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=False)


def _support(mask):
    result = np.zeros(mask.shape, dtype=bool)
    blocks = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
    result[:-1, :-1] |= blocks
    result[1:, :-1] |= blocks
    result[:-1, 1:] |= blocks
    result[1:, 1:] |= blocks
    return result


def _corners(box):
    t, l, b, r = box
    return ((t, l), (t, r), (b, l), (b, r))


def _inside(box):
    t, l, b, r = box
    return np.s_[t + 1:b, l + 1:r]


def _separated(a, b):
    return (a[2] + 1 < b[0] or b[2] + 1 < a[0]
            or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _frame_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l + 1:r] = mask[b, l + 1:r] = True
        mask[t + 1:b, l] = mask[t + 1:b, r] = True
    return mask


def _find_boxes(mask):
    # Match maximal horizontal bars with their two continuous vertical sides.
    runs = {}
    for row in range(mask.shape[0]):
        padded = np.r_[False, mask[row], False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        for left, end in zip(starts, ends):
            if end - left >= 2 and left > 0 and end < mask.shape[1]:
                runs.setdefault((int(left), int(end)), []).append(row)
    boxes = []
    for (left, end), rows in runs.items():
        for top, bottom in combinations(rows, 2):
            box = (top, left - 1, bottom, end)
            if (bottom - top >= 3
                    and mask[top + 1:bottom, left - 1].all()
                    and mask[top + 1:bottom, end].all()
                    and not mask[_inside(box)].any()
                    and not any(mask[p] for p in _corners(box))):
                boxes.append(box)
    return sorted(boxes)


def _parse(value):
    a = _grid(value)
    bg = int(a[0, 0])
    if not (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)):
        raise ValueError("the outer margin must be background")
    colors = list(map(int, np.unique(a)))
    if len(colors) != 3:
        raise ValueError("expected background, frame and ink colors")
    interpretations = []
    for frame in colors:
        if frame == bg:
            continue
        ink = next(c for c in colors if c not in (bg, frame))
        boxes = _find_boxes(a == frame)
        if len(boxes) != 3 or not all(_separated(x, y) for x, y in combinations(boxes, 2)):
            continue
        if not np.array_equal(_frame_mask(a.shape, boxes), a == frame):
            continue
        covered = np.zeros(a.shape, dtype=bool)
        tagged = []
        legal = True
        for i, box in enumerate(boxes):
            t, l, b, r = box
            covered[t:b + 1, l:r + 1] = True
            marks = [p for p in _corners(box) if a[p] == ink]
            if marks:
                if len(marks) != 1:
                    legal = False
                tagged.append(i)
        if not legal or len(tagged) != 1 or np.any(a[~covered] != bg):
            continue
        source = tagged[0]
        sh, sw = a[_inside(boxes[source])].shape
        if any(a[_inside(box)].shape != (sw, sh)
               for i, box in enumerate(boxes) if i != source):
            continue
        interpretations.append((a, bg, frame, ink, boxes, source))
    if len(interpretations) != 1:
        raise ValueError("expected one unambiguous three-frame scene")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, bg, frame, ink, boxes, source = _parse(input_grid)
    rotated = np.rot90(a[_inside(boxes[source])] == ink)
    out = a.copy()
    for i, box in enumerate(boxes):
        if i != source:
            union = rotated | (a[_inside(box)] == ink)
            out[_inside(box)] = np.where(_support(union), ink, bg)
    return out


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return (a.shape, a.tobytes())


def _nonrectangle(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        stack = [remaining.pop()]
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        rr, cc = zip(*cells)
        if len(cells) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1):
            return True
    return False


def _stencil(rng, shape, rich):
    h, w = shape
    r = np.zeros(shape, dtype=bool)
    # A supported nucleus, sometimes a stepped union of overlapping 2x2 blocks.
    top = int(rng.integers(h - (2 if not rich else 3) + 1))
    left = int(rng.integers(w - (2 if not rich else 3) + 1))
    r[top:top + 2, left:left + 2] = True
    if rich:
        r[top + 1:top + 3, left + 1:left + 3] = True
    for _ in range(int(rng.integers(2, 5))):
        t, l = int(rng.integers(h - 1)), int(rng.integers(w - 1))
        patch = np.ones((2, 2), dtype=bool)
        patch.flat[int(rng.integers(4))] = False
        r[t:t + 2, l:l + 2] |= patch
    for _ in range(int(rng.integers(1, 4))):
        t, l = int(rng.integers(h)), int(rng.integers(w))
        length = int(rng.integers(1, 5))
        if rng.random() < .5:
            r[t, l:min(w, l + length)] = True
        else:
            r[t:min(h, t + length), l] = True
    return r


def _receivers(rng, r):
    h, w = r.shape
    windows = []
    supported = _support(r)
    for t in range(h - 1):
        for l in range(w - 1):
            patch = r[t:t + 2, l:l + 2]
            if 1 <= patch.sum() <= 3 and (patch & ~supported[t:t + 2, l:l + 2]).any():
                windows.append((t, l))
    if len(windows) < 2 or (r & ~supported).sum() < 3:
        return None
    result = []
    for _ in range(2):
        u = rng.random(r.shape) < float(rng.uniform(.025, .09))
        # Complementary fragments complete selected partial 2x2 blocks.
        n = min(len(windows), int(rng.integers(1, 4)))
        for index in rng.choice(len(windows), n, replace=False):
            t, l = windows[int(index)]
            u[t:t + 2, l:l + 2] |= ~r[t:t + 2, l:l + 2]
        if rng.random() < .55:
            t, l = int(rng.integers(h - 1)), int(rng.integers(w - 1))
            u[t:t + 2, l:l + 2] = True
        # A thin receiver stroke challenges component-size filtering.
        t, l = int(rng.integers(h)), int(rng.integers(w - 2))
        u[t, l:l + 3] = True
        union = r | u
        welded = _support(union)
        if not ((union & ~welded).any()
                and (welded & r & ~u).any() and (welded & u & ~r).any()
                and not np.array_equal(welded, _support(supported | u))
                and not np.array_equal(welded, _support(r | _support(u)))):
            return None
        result.append(u)
    a, b = (_support(r | u) for u in result)
    if np.array_equal(a & r, b & r):
        return None
    return result


def _layout(rng, sizes):
    # Free rectangle packing, with a guaranteed legal 30x30 search stratum.
    order = sorted(range(3), key=lambda i: -sizes[i][0] * sizes[i][1])
    for attempt in range(100):
        if attempt < 70:
            h, w = int(rng.integers(17, 31)), int(rng.integers(17, 31))
        else:
            h = w = 30
        placed = {}
        for i in order:
            bh, bw = sizes[i]
            if h - bh < 2 or w - bw < 2:
                break
            positions = [(r, c) for r in range(1, h - bh)
                         for c in range(1, w - bw)]
            rng.shuffle(positions)
            for t, l in positions:
                box = (t, l, t + bh - 1, l + bw - 1)
                if all(_separated(box, old) for old in placed.values()):
                    placed[i] = box
                    break
            else:
                break
        if len(placed) == 3:
            return (h, w), [placed[i] for i in range(3)]
    raise RuntimeError("packing failed")


def _paint(shape, boxes, masks, colors, corner):
    bg, frame, ink = colors
    a = np.full(shape, bg, dtype=np.int8)
    a[_frame_mask(shape, boxes)] = frame
    for box, mask in zip(boxes, masks):
        a[_inside(box)] = np.where(mask, ink, bg)
    a[_corners(boxes[0])[corner]] = ink
    return a


def _alternative_fits(inp, out, transform):
    a, bg, frame, ink, boxes, source = _parse(inp)
    stencil = a[_inside(boxes[source])] == ink
    rotated = transform(stencil)
    for i, box in enumerate(boxes):
        if i != source:
            u = a[_inside(box)] == ink
            if u.shape != rotated.shape:
                return False
            if not np.array_equal(out[_inside(box)] == ink, _support(u | rotated)):
                return False
    return True


def _transforms():
    return [lambda a, k=k: np.rot90(a, k) for k in (0, 2, 3)] + [
        lambda a, k=k: np.rot90(np.fliplr(a), k) for k in range(4)]


def _make_pair(rng, source_shape, colors, rich, corner):
    h, w = source_shape
    for _ in range(500):
        r = _stencil(rng, (w, h), rich)
        receivers = _receivers(rng, r)
        if receivers is None:
            continue
        final = [_support(r | u) for u in receivers]
        if rich and not any(_nonrectangle(mask) for mask in final):
            continue
        shape, boxes = _layout(rng, [(h + 2, w + 2), (w + 2, h + 2), (w + 2, h + 2)])
        s = np.rot90(r, -1)
        inp = _paint(shape, boxes, [s] + receivers, colors, corner)
        out = _paint(shape, boxes, [s] + final, colors, corner)
        # Every demonstration exposes rotation even after support filtering.
        if any(_alternative_fits(inp, out, f) for f in _transforms()):
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("could not sample an informative stencil")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 3, replace=False)))
    palette_mode = int(rng.integers(3))
    tag_offset = int(rng.integers(4))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            for _ in range(50):
                if split == "train":
                    h, w = int(rng.integers(5, 9)), int(rng.integers(5, 9))
                else:
                    h, w = int(rng.integers(9, 12)), int(rng.integers(4, 9))
                    if rng.random() < .5:
                        h, w = w, h
                if palette_mode == 0:
                    colors = palette
                elif palette_mode == 1:
                    others = [c for c in range(10) if c != palette[1]]
                    bg, ink = map(int, rng.choice(others, 2, replace=False))
                    colors = (bg, palette[1], ink)
                else:
                    colors = tuple(map(int, rng.choice(10, 3, replace=False)))
                pair = _make_pair(rng, (h, w), colors,
                                  rich=(index == 1 or (split == "test" and rng.random() < .7)),
                                  corner=(tag_offset + index) % 4)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _slow_support(mask, size=2):
    # Deliberately separate coordinate-loop check of the local semantic rule.
    supported = np.zeros(mask.shape, dtype=bool)
    for row in range(mask.shape[0] - size + 1):
        for col in range(mask.shape[1] - size + 1):
            if all(mask[row + dr, col + dc] for dr in range(size) for dc in range(size)):
                for dr in range(size):
                    for dc in range(size):
                        supported[row + dr, col + dc] = True
    return supported


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen = set()
    training_nonrectangle = False
    training_rejects_three = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, bg, frame, ink, boxes, source = _parse(pair["input"])
            out = _grid(pair["output"])
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            s = a[_inside(boxes[source])] == ink
            # Explicit coordinate mapping, independent of np.rot90 in solve.
            h, w = s.shape
            r = np.array([[s[col, w - 1 - row] for col in range(h)]
                          for row in range(w)], dtype=bool)
            changed_region = np.zeros(a.shape, dtype=bool)
            outputs = []
            for i, box in enumerate(boxes):
                if i == source:
                    continue
                sl = _inside(box)
                changed_region[sl] = True
                u = a[sl] == ink
                union = r | u
                expected = _slow_support(union)
                assert np.array_equal(out[sl], np.where(expected, ink, bg))
                assert (union & ~expected).any(), "must demonstrate deletion"
                assert (expected & r & ~u).any(), "copied ink must survive"
                assert (expected & u & ~r).any(), "receiver ink must survive"
                assert not np.array_equal(expected, _slow_support(u)), "A must matter"
                assert not np.array_equal(expected, _slow_support(r)), "receiver must matter"
                assert not np.array_equal(expected, _slow_support(_slow_support(r) | u)), "precleaning stencil must fail"
                assert not np.array_equal(expected, _slow_support(r | _slow_support(u))), "precleaning receiver must fail"
                outputs.append(expected)
                if split == "train":
                    training_nonrectangle |= _nonrectangle(expected)
                    training_rejects_three |= not np.array_equal(expected, _slow_support(union, 3))
            assert np.array_equal(out[~changed_region], a[~changed_region])
            assert not np.array_equal(outputs[0] & r, outputs[1] & r), "same stencil must behave differently in the two contexts"
            assert np.array_equal(solve(a), out)
    assert training_nonrectangle, "demonstrate support beyond solid rectangles"
    assert training_rejects_three, "distinguish 2x2 from 3x3 support"
    for alternative in _transforms():
        assert not all(_alternative_fits(p["input"], np.asarray(p["output"]), alternative)
                       for p in puzzle["train"]), "rotation must be identifiable"


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    shapes, source_shapes, structures, palettes, tags, signatures = (set() for _ in range(6))
    deleted, jointly_supported, nonrectangles = [], [], 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, bg, frame, ink, boxes, source = _parse(pair["input"])
                shapes.add(a.shape)
                s = a[_inside(boxes[source])] == ink
                source_shapes.add(s.shape)
                r = np.rot90(s)
                palettes.add((bg, frame, ink))
                tags.add(next(j for j, p in enumerate(_corners(boxes[source])) if a[p] == ink))
                normalized = np.where(a == bg, 0, np.where(a == frame, 1, 2)).astype(np.int8)
                structures.add(_key(normalized))
                for i, box in enumerate(boxes):
                    if i == source:
                        continue
                    u = a[_inside(box)] == ink
                    weld = _support(r | u)
                    deleted.append(int(((r | u) & ~weld).sum()))
                    jointly_supported.append(int((weld & ~(_support(r) | _support(u))).sum()))
                    nonrectangles += _nonrectangle(weld)
        if seed in (0, 21, 41):
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42 and len(shapes) >= 50
    assert len(source_shapes) >= 25 and len(structures) == 210
    assert len(palettes) >= 30 and tags == {0, 1, 2, 3}
    assert min(deleted) > 0 and min(jointly_supported) > 0 and nonrectangles >= 42
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # The input-only solver also handles clean and tiny members not sampled above.
    boxes = [(1, 1, 4, 5), (1, 8, 5, 11), (8, 3, 12, 6)]
    s = np.array([[1, 0, 1], [1, 1, 0]], dtype=bool)
    masks = [s, np.zeros((3, 2), bool), np.ones((3, 2), bool)]
    a = _paint((15, 15), boxes, masks, (0, 5, 8), 0)
    out = solve(a)
    assert np.array_equal(out[_inside(boxes[1])] == 8, _slow_support(np.rot90(s)))
    assert np.all(out[_inside(boxes[2])] == 8)
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes),
            "source_shapes": len(source_shapes), "normalized_structures": len(structures),
            "palettes": len(palettes), "tag_corners": len(tags),
            "deleted_cells_range": [min(deleted), max(deleted)],
            "joint_support_cells_range": [min(jointly_supported), max(jointly_supported)],
            "nonrectangular_receivers": nonrectangles}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
