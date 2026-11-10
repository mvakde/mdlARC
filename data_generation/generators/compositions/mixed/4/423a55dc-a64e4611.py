"""Bottom-anchored shear followed by clipped 3x3 negative-space highlighting.

No files are read or written by this module. Highlight 3 is a family constant;
the strict majority input color is background. Only NumPy and the stdlib are used.
"""
from __future__ import annotations

import hashlib

import numpy as np


HIGHLIGHT = 3


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a


def _roles(a):
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 2 or counts.max() * 2 <= a.size or HIGHLIGHT in colors:
        raise ValueError("input needs two colors excluding 3 and a strict background majority")
    background = int(colors[np.argmax(counts)])
    foreground = int(colors[np.argmin(counts)])
    return background, foreground


def _shear(mask, model="bottom"):
    """A mask transform; other models are used only for diagnostic checks."""
    rows, cols = np.nonzero(mask)
    bottom = int(rows.max())
    if model == "none":
        return mask.copy()
    distance = bottom - rows
    if model == "canvas":
        distance = mask.shape[0] - 1 - rows
    elif model == "rigid":
        distance = np.full_like(rows, bottom - int(rows.min()))
    elif model == "top":
        distance = rows - int(rows.min())
    elif model == "double":
        distance = 2 * distance
    elif model == "right":
        distance = -distance
    elif model != "bottom":
        raise ValueError(model)
    target = cols - distance
    keep = (target >= 0) & (target < mask.shape[1])
    out = np.zeros_like(mask, dtype=bool)
    out[rows[keep], target[keep]] = True
    return out


def _clear(mask, radius=1, diagonal=True):
    """True exactly at foreground-free clipped neighborhood centers."""
    h, w = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    near = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if diagonal or abs(dy) + abs(dx) <= radius:
                near |= padded[radius + dy:radius + dy + h,
                               radius + dx:radius + dx + w]
    return ~near


def _paint(mask, background, foreground, clear=None):
    out = np.full(mask.shape, background, dtype=np.int8)
    out[mask] = foreground
    if clear is None:
        clear = _clear(mask)
    out[clear & ~mask] = HIGHLIGHT
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive both stages solely from the given input, without mutation."""
    a = _grid(input_grid)
    background, foreground = _roles(a)
    return _paint(_shear(a == foreground), background, foreground)


def _components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, group = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(group)
    return groups


def _internal_cores(mask):
    h, w = mask.shape
    return [g for g in _components(_clear(mask))
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in g)]


def _shape(rng, h, w, kind, cavities=1):
    """Connected masks with variable runs, holes, notches, tips and branches."""
    m = np.zeros((h, w), dtype=bool)
    if kind == "solid":
        m[:] = True
    elif kind == "frame":
        m[0] = m[-1] = True
        m[:, 0] = m[:, -1] = True
    elif kind == "cavities":
        m[:] = True
        # Every cavity has a foreground roof, floor and separating wall.
        widths = [int(v) for v in rng.integers(5, 8, size=cavities)]
        w = sum(widths) + cavities + 1
        m = np.ones((h, w), dtype=bool)
        left = 1
        for breadth in widths:
            m[1:-1, left:left + breadth] = False
            left += breadth + 1
    elif kind == "middle":
        m[:, w - 2] = True
        m[0, 2:] = True
        m[1] = True
        for r in range(2, h):
            lo = int(rng.integers(0, w - 1))
            hi = int(rng.integers(w - 1, w + 1))
            m[r, lo:hi] = True
    elif kind == "comb":
        spine = int(rng.integers(0, w))
        m[:, spine] = True
        bars = set(int(v) for v in rng.choice(h, size=int(rng.integers(2, h + 1)), replace=False))
        bars.add(h // 2)
        for r in bars:
            lo = int(rng.integers(0, spine + 1))
            hi = int(rng.integers(spine + 1, w + 1))
            m[r, lo:hi] = True
        m[h // 2] = True
    elif kind == "notched":
        m[:] = True
        spine = w // 2
        for _ in range(int(rng.integers(1, h + 1))):
            r = int(rng.integers(0, h))
            if rng.random() < .5:
                m[r, :int(rng.integers(0, spine + 1))] = False
            else:
                m[r, int(rng.integers(spine + 1, w + 1)):] = False
        m[h // 2] = True
    elif kind == "meander":
        center = int(rng.integers(0, w))
        for r in range(h):
            old = center
            center = int(np.clip(center + rng.integers(-2, 3), 0, w - 1))
            lo = max(0, min(old, center) - int(rng.integers(0, 4)))
            hi = min(w, max(old, center) + int(rng.integers(1, 4)))
            m[r, lo:hi] = True
        m[h // 2] = True
    else:
        raise ValueError(kind)
    if kind != "middle" and rng.random() < .5:
        m = m[:, ::-1].copy()
    return m


def _place(rng, m, clipping, background, foreground, diagnostic=False):
    h, w = m.shape
    minimum = max(0, max(h - 1 - r - int(np.flatnonzero(m[r])[0]) for r in range(h)))
    if clipping == "middle":
        left = h - 3
    elif clipping == "yes":
        if minimum == 0:
            return None
        left = int(rng.integers(0, min(minimum - 1, 30 - w) + 1))
    else:
        if minimum > 30 - w:
            return None
        left = int(rng.integers(minimum, min(minimum + 6, 30 - w) + 1))
    top = int(rng.integers(0, 6))
    bottom_margin = int(rng.integers(1 if diagnostic else 0, 7))
    height = min(30, h + top + bottom_margin)
    width = min(30, left + w + int(rng.integers(0, 8)))
    aspect = int(rng.integers(0, 4))
    if aspect == 0:
        height = width = max(height, width)
    elif aspect == 1:
        height = min(30, max(height, width + int(rng.integers(1, 6))))
    elif aspect == 2:
        width = min(30, max(width, height + int(rng.integers(1, 6))))
    # This visibly disambiguates color roles, independently of corner contact.
    mass = int(m.sum())
    if height * width <= 2 * mass:
        width = min(30, max(width, 2 * mass // height + 1))
    if height * width <= 2 * mass:
        height = min(30, max(height, 2 * mass // width + 1))
    a = np.full((height, width), background, dtype=np.int8)
    a[top:top + h, left:left + w][m] = foreground
    return a


def _key(a):
    return a.shape, a.tobytes()


def _draw(rng, role, background, foreground, seen, shift=None, train_max=None):
    for _ in range(96):
        cavities = 1
        if role == "pocket":
            h, w, kind, clipping = int(rng.integers(5, 7)), int(rng.integers(7, 9)), "frame", "no"
        elif role == "clip":
            h, w = int(rng.integers(5, 9)), int(rng.integers(5, 11))
            kind = str(rng.choice(("middle", "notched", "comb", "meander")))
            clipping = "middle" if kind == "middle" else "yes"
        else:
            h, w = int(rng.integers(3, 10)), int(rng.integers(3, 12))
            kind = str(rng.choice(("solid", "frame", "comb", "notched", "meander")))
            clipping = str(rng.choice(("no", "yes")))
            if role == "test":
                if shift == "height":
                    h = int(rng.integers(train_max[0] + 1, min(15, train_max[0] + 4) + 1))
                    w = int(rng.integers(4, 14))
                elif shift == "width":
                    w = int(rng.integers(train_max[1] + 2, min(20, train_max[1] + 6) + 1))
                    h = int(rng.integers(3, train_max[0] + 1))
                else:
                    cavities = int(rng.integers(2, 4))
                    h = int(rng.integers(5, 7 if cavities == 3 else 10))
                    kind = "cavities"
            if kind == "frame" and (h < 5 or w < 7):
                kind = "comb"
        m = _shape(rng, h, w, kind, cavities)
        a = _place(rng, m, clipping, background, foreground, role in ("pocket", "clip"))
        if a is None or _key(a) in seen:
            continue
        mask = a == foreground
        s = _shear(mask)
        clear = _clear(s)
        causal = (~mask & ~s) & (_clear(mask) != clear)
        if (np.array_equal(mask, s) or not causal.any() or clear.sum() < 2
                or np.count_nonzero(np.any(s, axis=1)) < 2
                or not np.any(_clear(s, diagonal=False) & ~clear)):
            continue
        if role == "pocket" and not any(1 <= len(g) <= 4 for g in _internal_cores(s)):
            continue
        if role == "clip" and (s.sum() == mask.sum() or np.array_equal(m, m[:, ::-1])):
            continue
        if role == "test" and shift == "cavities" and len(_internal_cores(s)) < 2:
            continue
        seen.add(_key(a))
        return {"input": a, "output": solve(a)}
    raise RuntimeError("bounded sampler could not construct a diagnostic distinct figure")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one puzzle; defaults are exactly four training and one test pair."""
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c != HIGHLIGHT])]
    backgrounds = palette[:2] if rng.random() < .5 else palette[:1]
    shift = str(rng.choice(("height", "width", "cavities")))
    roles = ["pocket", "clip"] + ["extra"] * (num_train - 2)
    rng.shuffle(roles)
    puzzle, seen, prior_foregrounds = {"train": [], "test": []}, set(), []
    for index, role in enumerate(roles + ["test"] * num_test):
        background = int(rng.choice(backgrounds))
        available = [c for c in palette if c != background]
        if index == 1:
            available = [c for c in available if c != prior_foregrounds[0]]
        elif prior_foregrounds and rng.random() < .5:
            available = [c for c in prior_foregrounds if c != background] or available
        foreground = int(rng.choice(available))
        if role == "test":
            facts = [_facts(p) for p in puzzle["train"]]
            maximum = max(f["height"] for f in facts), max(f["width"] for f in facts)
        else:
            maximum = None
        pair = _draw(rng, role, background, foreground, seen, shift, maximum)
        puzzle["test" if role == "test" else "train"].append(pair)
        if role != "test":
            prior_foregrounds.append(foreground)
    return puzzle


def _facts(pair):
    a, out = np.asarray(pair["input"]), np.asarray(pair["output"])
    background, foreground = _roles(a)
    mask = a == foreground
    rows, cols = np.nonzero(mask)
    s = _shear(mask)
    clear = _clear(s)
    crop = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    return {"background": background, "foreground": foreground,
            "height": int(crop.shape[0]), "width": int(crop.shape[1]),
            "crop": _key(crop), "clipped": int(mask.sum() - s.sum()),
            "causal_cells": int(np.count_nonzero((~mask & ~s) & (_clear(mask) != clear))),
            "small_core": any(1 <= len(g) <= 4 for g in _internal_cores(s)),
            "cores": len(_internal_cores(s)), "mask": mask, "sheared": s,
            "clear": clear, "output": out}


def _alternatives(a):
    bg, fg = _roles(a)
    mask, s = a == fg, _shear(a == fg)
    candidates = {model: _paint(_shear(mask, model), bg, fg)
                  for model in ("none", "canvas", "rigid", "top", "double", "right")}
    candidates["no_highlight"] = _paint(s, bg, fg, np.zeros_like(s))
    candidates["scan_before"] = _paint(s, bg, fg, _clear(mask))
    candidates["all_background"] = _paint(s, bg, fg, ~s)
    candidates["orthogonal"] = _paint(s, bg, fg, _clear(s, diagonal=False))
    candidates["radius_two"] = _paint(s, bg, fg, _clear(s, radius=2))
    interior = _clear(s)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    candidates["complete_windows_only"] = _paint(s, bg, fg, interior)
    large = np.zeros_like(s)
    for group in _components(_clear(s)):
        if len(group) >= 5:
            for r, c in group:
                large[r, c] = True
    candidates["component_cutoff_five"] = _paint(s, bg, fg, large)
    # A union of full blank 3x3 windows is a different operation from centers.
    union = np.zeros_like(s)
    for r in range(1, s.shape[0] - 1):
        for c in range(1, s.shape[1] - 1):
            if not s[r - 1:r + 2, c - 1:c + 2].any():
                union[r - 1:r + 2, c - 1:c + 2] = True
    candidates["whole_window_union"] = _paint(s, bg, fg, union)
    return candidates


def validate_generated_puzzle(puzzle):
    """Check structure, literal coordinate/neighborhood invariants and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, facts, rejected = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape and _key(a) not in seen
            seen.add(_key(a))
            f = _facts(pair)
            facts[split].append(f)
            bg, fg, mask = f["background"], f["foreground"], f["mask"]
            assert len(_components(mask)) == 1
            assert f["height"] >= 3 and f["causal_cells"] > 0
            # Check every surviving source coordinate without invoking solve.
            literal = np.zeros_like(mask)
            bottom = int(np.nonzero(mask)[0].max())
            for r, c in np.argwhere(mask):
                target = int(c) - (bottom - int(r))
                if target >= 0:
                    literal[int(r), target] = True
            assert np.array_equal(out == fg, literal)
            assert np.array_equal(mask[bottom], literal[bottom])
            assert set(int(v) for v in np.unique(out)) == {bg, fg, HIGHLIGHT}
            for r in range(a.shape[0]):
                for c in range(a.shape[1]):
                    if literal[r, c]:
                        continue
                    neighbors = literal[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
                    assert out[r, c] == (bg if neighbors.any() else HIGHLIGHT)
            original = a.copy()
            assert np.array_equal(solve(a), out) and np.array_equal(a, original)
            if split == "train":
                rejected.update(name for name, prediction in _alternatives(a).items()
                                if not np.array_equal(prediction, out))
    train = facts["train"]
    assert len({f["foreground"] for f in train}) >= 2
    assert any(f["clipped"] > 0 for f in train) and any(f["clipped"] == 0 for f in train)
    assert any(f["small_core"] for f in train)
    assert rejected == set(_alternatives(np.asarray(puzzle["train"][0]["input"])))
    mh, mw = max(f["height"] for f in train), max(f["width"] for f in train)
    assert all(f["height"] > mh or f["width"] > mw or f["cores"] >= 2
               for f in facts["test"])


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded: 42 seeds plus all 30 legal count combinations; no disk writes."""
    # Thin canvases exercise clipped neighborhoods without any complete 3x3.
    assert np.array_equal(solve(np.array([[9, 0, 0, 0]])), [[9, 0, 3, 3]])
    assert np.array_equal(solve(np.array([[7], [0], [0]])), [[7], [0], [3]])
    # Upper rows may disappear completely; the original bottom stays fixed.
    clipping_case = np.zeros((6, 5), dtype=int)
    clipping_case[:5, 0] = 7
    clipping_case[4, 1] = 7
    assert np.array_equal(solve(clipping_case),
                          [[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3],
                           [0, 0, 0, 3, 3], [7, 7, 0, 3, 3], [0, 0, 0, 3, 3]])
    shapes, crops, backgrounds, foregrounds, signatures, aspects = set(), set(), set(), set(), set(), set()
    shifts, first_clipping, clip_styles = set(), set(), set()
    total = clipped = small = causal = edge_highlights = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        assert signature == _digest(generate_puzzle(seed))
        signatures.add(signature)
        train = [_facts(p) for p in puzzle["train"]]
        mh, mw = max(f["height"] for f in train), max(f["width"] for f in train)
        first_clipping.add(train[0]["clipped"] > 0)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _facts(pair)
                shapes.add(a.shape)
                crops.add(f["crop"])
                backgrounds.add(f["background"])
                foregrounds.add(f["foreground"])
                aspects.add("square" if a.shape[0] == a.shape[1] else "tall" if a.shape[0] > a.shape[1] else "wide")
                total += 1
                clipped += f["clipped"] > 0
                small += f["small_core"]
                causal += f["causal_cells"]
                clear = f["clear"]
                edge_highlights += bool(clear[0].any() or clear[-1].any() or clear[:, 0].any() or clear[:, -1].any())
                losses = f["mask"].sum(axis=1) - f["sheared"].sum(axis=1)
                top = int(np.nonzero(f["mask"])[0].min())
                clip_styles.add("none" if not losses.any() else "middle" if losses[top] == 0 else "top")
                if split == "test":
                    shifts.add("height" if f["height"] > mh else "width" if f["width"] > mw else "cores")
        # Validate the same puzzle in its JSON-compatible nested-list form.
        if seed == 0:
            validate_generated_puzzle({s: [{r: p[r].tolist() for r in ("input", "output")}
                                          for p in puzzle[s]] for s in ("train", "test")})
    assert len(signatures) == 42 and len(shapes) >= 50 and len(crops) >= 80
    assert len(backgrounds) >= 7 and len(foregrounds) >= 7
    assert aspects == {"square", "tall", "wide"}
    assert first_clipping == {False, True} and clip_styles == {"none", "middle", "top"}
    assert "height" in shifts and "width" in shifts
    assert 42 <= small and 42 <= clipped < total and causal > 100
    legal_calls = 0
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(901, nt, ne))
            legal_calls += 1
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {nt}/{ne}")
    return {"seeds": 42, "default_pairs": total, "legal_count_calls": legal_calls,
            "canvas_shapes": len(shapes), "cropped_masks": len(crops),
            "background_colors": len(backgrounds), "foreground_colors": len(foregrounds),
            "clipped_pairs": clipped, "small_core_pairs": small,
            "causal_witness_cells": causal, "edge_highlight_pairs": edge_highlights,
            "test_shift_features": sorted(shifts)}
