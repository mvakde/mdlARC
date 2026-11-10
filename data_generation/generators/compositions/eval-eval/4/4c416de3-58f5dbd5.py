"""Color-matched corner stencils perforate plates by overlap parity.

Only standard Python and NumPy are used. The solver parses its input afresh;
reference masks, geometry and palette roles are all visible in that input.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(a, color):
    visited = np.zeros(a.shape, dtype=bool)
    result = []
    for r, c in np.argwhere(a == color):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        stack, points = [(r, c)], []
        visited[r, c] = True
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < a.shape[0] and 0 <= xx < a.shape[1]
                        and not visited[yy, xx] and a[yy, xx] == color):
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        result.append(points)
    return result


def _corner_points(rect):
    t, l, b, r = rect
    return ((t, l), (t, r), (b, l), (b, r))


def _badge_points(rect):
    return tuple((y - dy, x - dx)
                 for (y, x), (dy, dx) in zip(_corner_points(rect), _DIRS))


def _perimeter(rect):
    t, l, b, r = rect
    return ({(t, x) for x in range(l, r + 1)}
            | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t, b + 1)}
            | {(y, r) for y in range(t, b + 1)})


def _placed(mask, q, h, w, orientation="reflect"):
    """Return mask and its inset top-left position for an inward corner."""
    dy, dx = _DIRS[q]
    if orientation == "reflect":
        m = mask[::-1 if dy < 0 else 1, ::-1 if dx < 0 else 1]
    elif orientation == "fixed":
        m = mask
    elif orientation == "rotate":
        m = np.rot90(mask, (0, -1, 1, 2)[q])
    else:
        raise ValueError("unknown orientation")
    kh, kw = m.shape
    assert kh <= h and kw <= w, "stencil does not fit the plate"
    return m, (0 if dy > 0 else h - kh), (0 if dx > 0 else w - kw)


def _read_with_frame(a, bg, frame):
    rects = []
    for component in _components(a, frame):
        t = min(y for y, _ in component)
        l = min(x for _, x in component)
        b = max(y for y, _ in component)
        r = max(x for _, x in component)
        rect = (t, l, b, r)
        assert b - t >= 3 and r - l >= 3
        assert set(component) == _perimeter(rect), "not an intact frame"
        assert t >= 1 and l >= 1 and b < a.shape[0] - 1 and r < a.shape[1] - 1
        rects.append(rect)
    assert len(rects) >= 3
    rects.sort()
    for i, (t, l, b, r) in enumerate(rects):
        for tt, ll, bb, rr in rects[:i]:
            assert b + 2 < tt - 1 or bb + 2 < t - 1 or r + 2 < ll - 1 or rr + 2 < l - 1

    accounted = np.zeros(a.shape, dtype=bool)
    references, plates, templates = [], [], {}
    for rect in rects:
        t, l, b, r = rect
        accounted[t:b + 1, l:r + 1] = True
        badges = {}
        for q, point in enumerate(_badge_points(rect)):
            color = int(a[point])
            if color != bg:
                assert color != frame
                badges[q] = color
                accounted[point] = True
        interior = a[t + 1:b, l + 1:r]
        is_reference = (set(badges) == {0}
                        and np.all(np.isin(interior, (bg, badges[0])))
                        and np.any(interior == bg) and np.any(interior == badges[0]))
        if is_reference:
            color = badges[0]
            assert color not in templates, "duplicate reference color"
            templates[color] = interior == color
            references.append((rect, color))
        else:
            plates.append({"rect": rect, "badges": badges})
    assert np.all(accounted | (a == bg)), "unregistered content"
    assert len(templates) >= 2 and plates
    assert len({m.shape for m in templates.values()}) == 1
    payloads = set(templates)
    material_colors = set()
    for p in plates:
        t, l, b, r = p["rect"]
        inside = a[t + 1:b, l + 1:r]
        assert not np.any(inside == bg)
        material_colors |= set(map(int, np.unique(inside))) - payloads
        assert set(p["badges"].values()) <= payloads
    assert len(material_colors) == 1
    material = material_colors.pop()
    assert material not in {bg, frame} | payloads
    kh, kw = next(iter(templates.values())).shape
    for p in plates:
        t, l, b, r = p["rect"]
        interior = a[t + 1:b, l + 1:r]
        h, w = interior.shape
        assert h >= kh and w >= kw and np.any(interior == material)
        assert set(map(int, np.unique(interior))) <= {material} | set(p["badges"].values())
        membership = {color: np.zeros((h, w), dtype=bool)
                      for color in set(p["badges"].values())}
        for q, color in p["badges"].items():
            mask, y, x = _placed(templates[color], q, h, w)
            footprint = np.zeros((h, w), dtype=bool)
            footprint[y:y + kh, x:x + kw] = mask
            assert np.any(footprint & (interior == color)), "marked corner lacks an interior clue"
            membership[color] |= footprint
        for color, permitted in membership.items():
            assert np.all((interior != color) | permitted), "clue outside selected stencil"
    return {"background": bg, "frame": frame, "material": material,
            "templates": templates, "references": references, "plates": plates,
            "rectangles": rects, "mask_shape": (kh, kw)}


def _decode(a):
    bg = int(a[0, 0])
    assert (np.all(a[0] == bg) and np.all(a[-1] == bg)
            and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)), "missing background border"
    candidates = []
    for frame in map(int, np.unique(a)):
        if frame == bg:
            continue
        try:
            candidates.append(_read_with_frame(a, bg, frame))
        except AssertionError:
            continue
    assert len(candidates) == 1, "missing or ambiguous scene interpretation"
    return candidates[0]


def _render(a, info, combine="xor", orientation="reflect", selection="color", positive=False):
    """Production rendering plus explicitly named countermodels for validation."""
    out = a.copy()
    bg, mat = info["background"], info["material"]
    colors = sorted(info["templates"])
    profiles = []
    for p in info["plates"]:
        t, l, b, r = p["rect"]
        h, w = b - t - 1, r - l - 1
        coverage = np.zeros((h, w), dtype=np.int8)
        result = np.full((h, w), mat, dtype=np.int8)
        for q, color in p["badges"].items():
            selected = color
            if selection == "swapped" and color in colors[:2]:
                selected = colors[1] if color == colors[0] else colors[0]
            elif selection == "one_mask":
                selected = colors[0]
            m, y, x = _placed(info["templates"][selected], q, h, w, orientation)
            mh, mw = m.shape
            coverage[y:y + mh, x:x + mw] += m
            if positive:
                result[y:y + mh, x:x + mw][m] = color
        if not positive:
            holes = coverage % 2 == 1 if combine == "xor" else coverage > 0
            result[holes] = bg
        out[t + 1:b, l + 1:r] = result
        profiles.append(coverage)
    return out, profiles


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer palette, frames, reference masks and stamps from this input only."""
    a = _grid(input_grid)
    try:
        return _render(a, _decode(a))[0]
    except AssertionError as exc:
        raise ValueError("input is outside the registered corner-stamp family") from exc


def _clue_only(a, info):
    out = a.copy()
    for p in info["plates"]:
        t, l, b, r = p["rect"]
        interior = out[t + 1:b, l + 1:r]
        interior[interior != info["material"]] = info["background"]
    return out


def _witnesses(a, info, expected):
    alternatives = {
        "union": {"combine": "union"},
        "fixed_orientation": {"orientation": "fixed"},
        "quarter_turns": {"orientation": "rotate"},
        "wrong_color": {"selection": "swapped"},
        "shared_mask": {"selection": "one_mask"},
        "positive_completion": {"positive": True},
    }
    result = {"clue_only": not np.array_equal(_clue_only(a, info), expected)}
    for name, kwargs in alternatives.items():
        try:
            alternative = _render(a, info, **kwargs)[0]
            result[name] = not np.array_equal(alternative, expected)
        except AssertionError:
            result[name] = True
    return result


def _sample_mask(rng, h, w):
    """Registered masks may be simple, irregular, symmetric or disconnected."""
    for _ in range(60):
        mode = int(rng.integers(8))
        m = np.zeros((h, w), dtype=bool)
        if mode == 0:
            for y in range(h):
                m[y, int(round(y * (w - 1) / (h - 1)))] = True
        elif mode == 1:
            m[int(rng.integers(h)), :] = True
            m[:, int(rng.integers(w))] = True
        elif mode == 2:
            m[0, :] = m[-1, :] = True
        elif mode == 3:
            m[:] = True
            m.flat[int(rng.integers(m.size))] = False
        else:
            m = rng.random((h, w)) < float(rng.uniform(.18, .8))
            if mode == 4:
                m |= m[:, ::-1]
            elif mode == 5:
                m |= m[::-1, :]
        if rng.random() < .5:
            m = m[:, ::-1]
        if rng.random() < .5:
            m = m[::-1, :]
        if 2 <= int(m.sum()) < m.size:
            return m.copy()
    m = np.zeros((h, w), dtype=bool)
    m[0, 0] = m[-1, -1] = True
    return m


def _pack(rng, sizes):
    """Place padded objects freely, with one blank cell between envelopes."""
    for _ in range(24):
        h, w = map(int, rng.integers(20, 31, size=2))
        occupied = np.zeros((h, w), dtype=bool)
        origins = [None] * len(sizes)
        # Larger objects are attempted first; positions remain random.
        order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
        for i in order:
            sh, sw = sizes[i]
            choices = [(y, x) for y in range(1, h - sh)
                       for x in range(1, w - sw)
                       if not occupied[y:y + sh, x:x + sw].any()]
            if not choices:
                break
            y, x = choices[int(rng.integers(len(choices)))]
            origins[i] = (y, x)
            occupied[y - 1:y + sh + 1, x - 1:x + sw + 1] = True
        else:
            return (h, w), origins
    raise AssertionError("packing budget exhausted")


def _draw_frame(a, rect, frame):
    t, l, b, r = rect
    a[t, l:r + 1] = a[b, l:r + 1] = frame
    a[t:b + 1, l] = a[t:b + 1, r] = frame


def _scene(rng, split, index, base_q):
    witness = split == "train" and index == 0
    if witness:
        kh = kw = int(rng.integers(3, 5))
        nref, nplate = 2, 1
    else:
        kh, kw = map(int, rng.integers(2, 5, size=2))
        if rng.random() < .14:
            kh = int(rng.integers(2, 6))
            kw = int(rng.integers(2, 6))
        nref = int(rng.integers(2, 5))
        nplate = int(rng.choice((1, 1, 2, 2, 3))) if max(kh, kw) <= 3 else int(rng.choice((1, 1, 2)))
    roles = list(map(int, rng.choice(10, 3 + nref, replace=False)))
    bg, frame, mat = roles[:3]
    colors = sorted(roles[3:])
    masks = {color: _sample_mask(rng, kh, kw) for color in colors}
    assert len({m.tobytes() for m in masks.values()}) > 1
    plans = []
    for i in range(nplate):
        if witness:
            ph = int(rng.integers(kh, 2 * kh + 1))
            pw = int(rng.integers(kw, 2 * kw))
            badges = {base_q: colors[0], base_q ^ 1: colors[1]}
        else:
            ph = int(rng.integers(kh, 2 * kh + 3))
            pw = int(rng.integers(kw, 2 * kw + 3))
            if split == "test" and i == 0:
                # Four corners are a held-out combination; encourage overlap.
                ph = int(rng.integers(kh, 2 * kh))
                pw = int(rng.integers(kw, 2 * kw))
                qs = list(range(4))
            elif split == "train" and index == 1 and i == 0:
                qs = [base_q ^ 2, base_q ^ 3]
                if rng.random() < .5:
                    qs.append(base_q)
            else:
                count = int(rng.integers(0 if i else 1, 4))
                qs = list(map(int, rng.choice(4, count, replace=False)))
            badges = {q: int(rng.choice(colors)) for q in qs}
        plans.append((ph, pw, badges))
    sizes = [(kh + 4, kw + 4)] * nref + [(ph + 4, pw + 4) for ph, pw, _ in plans]
    shape, origins = _pack(rng, sizes)
    a = np.full(shape, bg, dtype=np.int8)
    for color, (y, x) in zip(colors, origins[:nref]):
        rect = (y + 1, x + 1, y + kh + 2, x + kw + 2)
        _draw_frame(a, rect, frame)
        a[y, x] = color
        a[y + 2:y + kh + 2, x + 2:x + kw + 2][masks[color]] = color
    out = a.copy()
    for (ph, pw, badges), (y, x) in zip(plans, origins[nref:]):
        rect = (y + 1, x + 1, y + ph + 2, x + pw + 2)
        _draw_frame(a, rect, frame)
        _draw_frame(out, rect, frame)
        interior = a[y + 2:y + ph + 2, x + 2:x + pw + 2]
        interior[:] = mat
        coverage = np.zeros((ph, pw), dtype=np.int8)
        used = set()
        # Reserve one distinct interior fragment for every marked corner first.
        footprints = {}
        for q, color in badges.items():
            dy, dx = _DIRS[q]
            points = [(int(u) if dy > 0 else ph - 1 - int(u),
                       int(v) if dx > 0 else pw - 1 - int(v))
                      for u, v in np.argwhere(masks[color])]
            footprints[q] = points
            for point in points:
                coverage[point] += 1
            choices = [p for p in points if p not in used]
            assert choices, "corner clues collide"
            near = [p for p in choices
                    if (p[0] < ph / 2 if dy > 0 else p[0] >= ph / 2)
                    and (p[1] < pw / 2 if dx > 0 else p[1] >= pw / 2)]
            choices = near or choices
            chosen = choices[int(rng.integers(len(choices)))]
            interior[chosen] = color
            used.add(chosen)
            point = _badge_points(rect)[q]
            a[point] = out[point] = color
        for q, color in badges.items():
            choices = [p for p in footprints[q] if p not in used]
            # The input fragments are usually much sparser than the full mask.
            if choices and rng.random() < .45:
                chosen = choices[int(rng.integers(len(choices)))]
                interior[chosen] = color
                used.add(chosen)
        out[y + 2:y + ph + 2, x + 2:x + pw + 2] = np.where(coverage % 2, bg, mat)
        assert np.count_nonzero(interior == mat) >= max(1, ph * pw // 3)
    # Trim unused space, preserve a blank border, and occasionally add padding.
    yy, xx = np.nonzero(a != bg)
    top, left, bottom, right = int(yy.min()) - 1, int(xx.min()) - 1, int(yy.max()) + 2, int(xx.max()) + 2
    a, out = a[top:bottom, left:right], out[top:bottom, left:right]
    padding = []
    for dim in a.shape:
        extra = int(rng.integers(min(4, 30 - dim) + 1))
        if rng.random() < .12:
            extra = 30 - dim
        before = int(rng.integers(extra + 1))
        padding.append((before, extra - before))
    a = np.pad(a, padding, constant_values=bg)
    out = np.pad(out, padding, constant_values=bg)
    info = _decode(a)
    assert np.array_equal(_render(a, info)[0], out)
    assert np.any(a != out)
    if witness:
        assert all(_witnesses(a, info, out).values()), "uninformative initial demonstration"
    return {"input": a, "output": out}


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_q = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            for _ in range(180):
                try:
                    pair = _scene(rng, split, i, base_q)
                    key = _key(pair["input"])
                    assert key not in seen
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, orientations = set(), set()
    taught = {name: False for name in ("union", "fixed_orientation", "quarter_turns", "wrong_color",
                                      "shared_mask", "positive_completion", "clue_only")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = _key(a)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _decode(a)
            expected, profiles = _render(a, info)
            assert np.array_equal(expected, out), "incorrect parity, matching, reflection or preserved region"
            assert np.any(a != out)
            mutable = np.zeros(a.shape, dtype=bool)
            for p, coverage in zip(info["plates"], profiles):
                t, l, b, r = p["rect"]
                mutable[t + 1:b, l + 1:r] = True
                interior_out = out[t + 1:b, l + 1:r]
                assert np.all(interior_out[coverage % 2 == 0] == info["material"])
                assert np.all(interior_out[coverage % 2 == 1] == info["background"])
                assert np.all(out[t, l:r + 1] == info["frame"])
                assert np.all(out[b, l:r + 1] == info["frame"])
                if split == "train":
                    assert len(p["badges"]) <= 3
                    orientations |= set(p["badges"])
            assert np.array_equal(a[~mutable], out[~mutable]), "changed a reference, badge, frame or background"
            if split == "train":
                for name, witnessed in _witnesses(a, info, out).items():
                    taught[name] |= witnessed
            else:
                assert any(len(p["badges"]) == 4 for p in info["plates"])
    assert orientations == set(range(4)), "training omits a corner orientation"
    assert all(taught.values()), f"training permits a countermodel: {taught}"


def _signature(puzzle):
    return hashlib.sha256(b"".join(bytes(a.shape) + a.tobytes()
                                  for split in ("train", "test") for pair in puzzle[split]
                                  for a in (_grid(pair["input"]), _grid(pair["output"])))).hexdigest()


def _normalized(a):
    mapping = {}
    return bytes(mapping.setdefault(int(c), len(mapping)) for c in a.ravel())


def self_test():
    """Bounded 42-seed coverage audit, determinism, count and mutation checks."""
    signatures, structures, shapes, mask_shapes, plate_shapes = set(), set(), set(), set(), set()
    counts, material_colors, bg_colors, frame_colors, mask_counts = set(), set(), set(), set(), set()
    coverage_levels, marks, reference_counts, plate_counts = set(), set(), set(), set()
    mask_inks, distractors, repeated, unmarked = set(), set(), 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        if seed < 2:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _grid(pair["input"])
                prediction = solve(a.copy())
                assert np.array_equal(prediction, pair["output"])
                info = _decode(a)
                structures.add((a.shape, _normalized(a)))
                shapes.add(a.shape)
                mask_shapes.add(info["mask_shape"])
                bg_colors.add(info["background"])
                frame_colors.add(info["frame"])
                material_colors.add(info["material"])
                reference_counts.add(len(info["references"]))
                plate_counts.add(len(info["plates"]))
                used_colors = []
                _, profiles = _render(a, info)
                for p, coverage in zip(info["plates"], profiles):
                    t, l, b, r = p["rect"]
                    plate_shapes.add((b - t - 1, r - l - 1))
                    marks.add(len(p["badges"]))
                    used_colors.extend(p["badges"].values())
                    unmarked += not p["badges"]
                    coverage_levels.update(map(int, np.unique(coverage)))
                repeated += len(used_colors) > len(set(used_colors))
                distractors.add(len(set(info["templates"]) - set(used_colors)))
                for m in info["templates"].values():
                    mask_counts.add(int(m.sum()))
                    mask_inks.add((m.shape, m.tobytes()))
                counts.add((len(info["references"]), len(info["plates"])))
    assert len(signatures) == 42 and len(structures) >= 200
    assert len(shapes) >= 60 and len(mask_shapes) >= 10 and len(plate_shapes) >= 30
    assert {2, 3, 4} <= reference_counts and {1, 2, 3} <= plate_counts
    assert {0, 1, 2, 3, 4} <= marks and {0, 1, 2, 3, 4} <= coverage_levels
    assert {0, 1, 2} <= distractors and repeated > 30 and unmarked > 0
    assert min(len(bg_colors), len(frame_colors), len(material_colors)) == 10
    assert len(mask_inks) >= 120 and len(mask_counts) >= 12
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A correct schema does not excuse an unrelated change or wrong parity.
    p = generate_puzzle(17, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("unrelated output mutation accepted")
    p = generate_puzzle(18, 2, 1)
    a = p["train"][0]["input"]
    p["train"][0]["output"] = _render(a, _decode(a), combine="union")[0]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("union countermodel accepted")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "color_normalized_structures": len(structures), "input_shapes": len(shapes),
            "mask_shapes": sorted(mask_shapes), "plate_shapes": len(plate_shapes),
            "reference_counts": sorted(reference_counts), "plate_counts": sorted(plate_counts),
            "coverage_levels": sorted(coverage_levels), "distractor_counts": sorted(distractors),
            "distinct_masks": len(mask_inks), "repeated_color_pairs": repeated,
            "unmarked_plates": unmarked}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
