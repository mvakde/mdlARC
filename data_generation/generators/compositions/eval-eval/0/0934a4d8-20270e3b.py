"""Mirror mosaic weld: registration defines the frame for joint reconstruction.

Only NumPy and the standard library are used. No parent generator is imported.
All coordinates in the parser are relative to the observed registration bars.
"""
from __future__ import annotations

import numpy as np


def _check_grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells, diagonal=True):
    remaining = set(cells)
    result = []
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r or c) and (diagonal or not (r and c))]
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(cells):
    if not cells:
        return False
    t, b, l, r = _bbox(cells)
    return len(cells) >= 2 and len(cells) == (b - t + 1) * (r - l + 1)


def _crop(known, background, axis=0):
    known = {q: v for q, v in known.items() if v != background}
    if not known:
        raise ValueError("Empty mosaic")
    t, b, l, r = _bbox(known)
    if max(b - t + 1, r - l + 1) > 30:
        raise ValueError("Assembly exceeds ARC dimensions")
    a = np.full((b - t + 1, r - l + 1), background, dtype=np.int8)
    for (y, x), v in known.items():
        a[y - t, x - l] = v
    return a.T.copy() if axis else a


def _registered(p, seam=-1, strict=True):
    """Known pigment wins over missing observations; guides are transparent."""
    known = dict(p["anchor"]["known"])
    mover = {(r + seam, c): v for (r, c), v in p["mover"]["known"].items()}
    if strict and any(q in known and known[q] != v for q, v in mover.items()):
        raise ValueError("Conflicting overlapping pigments")
    known.update(mover)
    unknown = (set(p["anchor"]["unknown"])
               | {(r + seam, c) for r, c in p["mover"]["unknown"]})
    guides = {(r, c) for r in (0, seam) for c in range(p["length"])}
    return known, unknown, guides


def _complete(p):
    known, unknown, guides = _registered(p)
    # The axis lies halfway between adjacent guides, not at a fragment midpoint.
    mate = lambda q: (-1 - q[0], q[1])
    if any(mate(q) in known and known[mate(q)] != v for q, v in known.items()):
        raise ValueError("Visible registered reflection witnesses disagree")
    unresolved = unknown - known.keys()
    if any(mate(q) in unresolved for q in unresolved):
        raise ValueError("Both members of a reflection orbit were erased")
    answer = dict(known)
    for q in unresolved:
        answer[q] = known.get(mate(q), p["background"])
    # After fusion, lack of pigment means background, including uncovered guides.
    foreground = {q: v for q, v in answer.items() if v != p["background"]}
    if any(foreground.get(mate(q)) != v for q, v in foreground.items()):
        raise ValueError("Completed silhouette is not reflected about the guides")
    info = {
        "unresolved": len(unresolved),
        "recovered_pigment": sum(answer[q] != p["background"] for q in unresolved),
        "recovered_background": sum(answer[q] == p["background"] for q in unresolved),
        "borrowed": len(unknown & known.keys()),
        "overlap": len(p["anchor"]["known"].keys()
                       & {(r - 1, c) for r, c in p["mover"]["known"]}),
        "covered_guides": len(guides & foreground.keys()),
        "uncovered_guides": len(guides - foreground.keys()),
        "cross_fragment_witnesses": 0,
        "exclusive_cross_fragment_witnesses": 0,
        "missing": unresolved,
    }
    for name, other in (("anchor", "mover"), ("mover", "anchor")):
        shift = 0 if name == "anchor" else -1
        other_shift = 0 if other == "anchor" else -1
        own_missing = {(r + shift, c) for r, c in p[name]["unknown"]} & unresolved
        other_known = {(r + other_shift, c) for r, c in p[other]["known"]}
        own_known = {(r + shift, c) for r, c in p[name]["known"]}
        info["cross_fragment_witnesses"] += sum(mate(q) in other_known for q in own_missing)
        info["exclusive_cross_fragment_witnesses"] += sum(
            mate(q) in other_known and mate(q) not in own_known for q in own_missing)
    return _crop(answer, p["background"], p["axis"]), info


def _parse_candidates(a):
    """Find roles by geometry and registered evidence, never by output colors.

    Every non-background cell belongs to one of exactly two eight-connected
    fragments. Each has a congruent straight guide, length >= 2. Erasure color
    forms one rectangle in each affected fragment. Other colors are pigments.
    """
    colors = [int(c) for c in np.unique(a)]
    if not 5 <= len(colors) <= 8:
        return []
    by_color = {v: {tuple(map(int, q)) for q in np.argwhere(a == v)} for v in colors}
    bar_options = []
    for marker in colors:
        bars = _components(by_color[marker], False)
        if len(bars) != 2 or len(bars[0]) != len(bars[1]) or len(bars[0]) < 2:
            continue
        for axis in (0, 1):
            if all(len({q[axis] for q in b}) == 1
                   and max(q[1-axis] for q in b) - min(q[1-axis] for q in b) + 1 == len(b)
                   for b in bars):
                bar_options.append((marker, bars, axis))
    candidates = []
    all_cells = set().union(*by_color.values())
    comp_cache = {}
    for marker, bars, axis in bar_options:
        for background in colors:
            if background == marker:
                continue
            if background not in comp_cache:
                comp_cache[background] = _components(all_cells - by_color[background])
            parts = comp_cache[background]
            if len(parts) != 2:
                continue
            owners = []
            for part in parts:
                owned = [b for b in bars if b <= part]
                if len(owned) != 1:
                    break
                owners.append((part, owned[0]))
            if len(owners) != 2:
                continue
            for erasure in colors:
                if erasure in (background, marker):
                    continue
                rectangles = [part & by_color[erasure] for part, _ in owners]
                if not any(rectangles) or any(cells and not _rectangle(cells) for cells in rectangles):
                    continue
                records = []
                for (part, bar), holes in zip(owners, rectangles):
                    r0 = min(q[axis] for q in bar)
                    c0 = min(q[1-axis] for q in bar)
                    convert = lambda q: (q[axis] - r0, q[1-axis] - c0)
                    known = {convert(q): int(a[q]) for q in part
                             if int(a[q]) not in (marker, erasure)}
                    occupied = {convert(q) for q in part}
                    length = len(bar)
                    negative = any((-1, c) in known for c in range(length))
                    positive = any((1, c) in known for c in range(length))
                    if negative == positive:
                        break
                    facing = 1 if negative else -1
                    if any((facing, c) in occupied for c in range(length)):
                        break
                    records.append({"negative": negative, "known": known,
                                    "unknown": {convert(q) for q in holes},
                                    "part": part, "origin": (r0, c0)})
                if len(records) != 2 or records[0]["negative"] == records[1]["negative"]:
                    continue
                records.sort(key=lambda x: not x["negative"])
                p = {"background": background, "marker": marker, "erasure": erasure,
                     "axis": axis, "length": len(bars[0]),
                     "anchor": records[0], "mover": records[1]}
                try:
                    output, info = _complete(p)
                except ValueError:
                    continue
                p["output"], p["info"] = output, info
                candidates.append(p)
    return candidates


def _parse(a):
    candidates = _parse_candidates(a)
    if len(candidates) != 1:
        raise ValueError(f"Expected one geometric role assignment; found {len(candidates)}")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from input pixels. No training, seed or generator state."""
    return _parse(_check_grid(input_grid))["output"].copy()


def _alternative(p, model):
    """Concrete counter-rules used only to audit teaching examples."""
    if model in ("seam_zero", "seam_reverse"):
        seam = 0 if model == "seam_zero" else 1
        known, _, _ = _registered(p, seam, strict=False)
        return _crop(known, p["background"], p["axis"])
    known, unknown, guides = _registered(p)
    missing = unknown - known.keys()
    out = dict(known)
    if model == "repeat_half":
        h = max(-min(r for r, c in known), max(r for r, c in known) + 1)
        for r, c in missing:
            out[r, c] = known.get((r + h if r < 0 else r - h, c), p["background"])
    elif model == "unknown_paint":
        out.update({q: p["erasure"] for q in missing})
    elif model in ("guide_erase", "guide_paint"):
        for r, c in missing:
            out[r, c] = known.get((-1-r, c), p["background"])
        for q in guides:
            out[q] = p["background"] if model == "guide_erase" else p["marker"]
    elif model == "anchor_only":
        out = dict(p["anchor"]["known"])
    elif model == "mover_only":
        out = {(r-1, c): v for (r, c), v in p["mover"]["known"].items()}
    elif model != "no_reflection":
        raise ValueError(model)
    return _crop(out, p["background"], p["axis"])


_ALTERNATIVES = ("no_reflection", "repeat_half", "unknown_paint", "seam_zero",
                 "seam_reverse", "guide_erase", "guide_paint", "anchor_only", "mover_only")


def _pattern(rng, h, w, palette, style, silhouette):
    rr, cc = np.indices((h, w))
    if style == "random":
        top = rng.choice(palette, (h, w))
    elif style == "patches":
        small = rng.choice(palette, ((h+1)//2, (w+1)//2))
        top = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)[:h, :w]
    elif style == "bands":
        top = np.asarray(palette)[(rr + cc // int(rng.integers(1, 4))) % len(palette)]
    else:
        motif = rng.choice(palette, (2, 3))
        top = motif[rr % 2, cc % 3].copy()
    mask = np.ones((h, w), dtype=bool)
    if silhouette == "rings":
        mask[:] = False
        mask[0] = mask[-2:] = True
        thickness = int(rng.integers(1, 3))
        mask[:, :thickness] = mask[:, -thickness:] = True
    elif silhouette == "terraces":
        for r in range(h-2):
            lo = int(rng.integers(0, max(1, (h-r)//2 + 1)))
            hi = w - int(rng.integers(0, max(1, (h-r)//2 + 1)))
            mask[r, :lo] = mask[r, hi:] = False
    elif silhouette == "branches":
        mask[:] = False
        spine = int(rng.integers(1, w-1))
        mask[:, spine] = True
        mask[-2:] = True
        for r in range(h-2):
            lo, hi = int(rng.integers(spine+1)), int(rng.integers(spine, w))
            mask[r, lo:hi+1] = True
    elif silhouette == "notches":
        if h > 3:
            start = int(rng.integers(1, w-2))
            mask[1:h-2, start:start+int(rng.integers(1, min(4, w-start)))] = False
    # A second pigment must be visible away from the eventual guide line.
    top[h-2, 0], top[h-2, -1] = palette[0], palette[1]
    return top.astype(np.int8), mask


def _make_geometry(rng, h, w, length, silhouette, style, colors, partial, overlap):
    bg, erasure, marker, *palette = colors
    top, mask = _pattern(rng, h, w, palette, style, silhouette)
    offset = int(rng.integers(1, w-length))
    if partial:
        n = int(rng.integers(1, length))
        mask[-1, offset + rng.choice(length, n, replace=False)] = False
    top[~mask] = bg
    full = np.concatenate((top, top[::-1]), axis=0)
    cuts = rng.integers(-min(3, h-1), min(3, h-1), w)
    cuts[offset:offset+length] = -1
    extra = np.zeros(w, dtype=int)
    if overlap:
        outside = [c for c in range(w) if not offset <= c < offset+length]
        for c in rng.choice(outside, min(len(outside), int(rng.integers(1, 4))), replace=False):
            extra[int(c)] = int(rng.integers(1, 3))
    a, b = {}, {}
    for y in range(2*h):
        r = y-h
        for c in range(w):
            v = int(full[y, c])
            if v == bg:
                continue
            if r <= cuts[c] + extra[c]:
                a[r, c-offset] = v
            if r > cuts[c] - extra[c]:
                b[r+1, c-offset] = v
    for c in range(length):
        a[0, c] = marker
        b[0, c] = marker
    return a, b, full, offset


def _erase_options(rng, piece, h, w, offset, marker, erasure, direction):
    """Try bounded rectangular damage, possibly covering transparent pixels.

    Erasures may hide both pigment and the shape's background. They never
    obscure the attachment evidence needed to determine the mating direction.
    """
    candidates = []
    shift = 0 if direction == 0 else 1
    known = [q for q, v in piece.items() if v != marker]
    for _ in range(100):
        target = known[int(rng.integers(len(known)))]
        height = int(rng.integers(2, min(5, 2*h)+1))
        width = int(rng.integers(1, min(5, w)+1))
        r = int(np.clip(target[0] - int(rng.integers(height)), -h+shift, h+shift-height))
        c = int(np.clip(target[1] - int(rng.integers(width)), -offset, w-offset-width))
        holes = {(y, x) for y in range(r, r+height) for x in range(c, c+width)}
        protected = {(y, x) for y in (-1, 0, 1) for x in range(sum(v == marker for v in piece.values()))}
        if holes & protected or len(holes & set(known)) < 2:
            continue
        new = dict(piece)
        new.update({q: erasure for q in holes})
        if len(_components(new)) != 1:
            continue
        candidates.append(new)
        if len(candidates) >= 16:
            break
    return candidates


def _stage(rng, pieces, bg, stage):
    normalized = []
    for piece in pieces:
        t, b, l, r = _bbox(piece)
        normalized.append(({(y-t, x-l): v for (y, x), v in piece.items()}, b-t+1, r-l+1))
    if rng.integers(2):
        normalized.reverse()
    a, ah, aw = normalized[0]
    b, bh, bw = normalized[1]
    gap = int(rng.integers(1, 5))
    if stage == "horizontal":
        dr, dc = int(rng.integers(-min(3, bh-1), min(4, ah))), aw+gap
    elif stage == "vertical":
        dr, dc = ah+gap, int(rng.integers(-min(3, bw-1), min(4, aw)))
    else:
        dr, dc = ah+gap, aw+gap
    merged = dict(a)
    merged.update({(r+dr, c+dc): v for (r, c), v in b.items()})
    t, bottom, l, right = _bbox(merged)
    height, width = bottom-t+1, right-l+1
    if max(height, width) > 30:
        return None
    pads = [int(rng.integers(0, 4)) for _ in range(4)]
    pads[0] = min(pads[0], 30-height)
    pads[1] = min(pads[1], 30-height-pads[0])
    pads[2] = min(pads[2], 30-width)
    pads[3] = min(pads[3], 30-width-pads[2])
    canvas = np.full((height+pads[0]+pads[1], width+pads[2]+pads[3]), bg, dtype=np.int8)
    for (r, c), v in merged.items():
        canvas[r-t+pads[0], c-l+pads[2]] = v
    return canvas


def _palette(rng):
    return tuple(map(int, rng.choice(10, int(rng.integers(5, 9)), replace=False)))


def _make_pair(rng, profile, used):
    for attempt in range(280):
        h = int(rng.integers(*profile["heights"]))
        length = int(rng.choice(profile["lengths"]))
        w = int(rng.integers(max(6, length+2), 13))
        colors = profile["palette"]
        bg, erasure, marker = colors[:3]
        silhouette = profile["silhouette"] if attempt < 200 else "rings"
        a, b, full, offset = _make_geometry(rng, h, w, length, silhouette,
                                           profile["style"], colors,
                                           profile["partial"], profile["overlap"])
        if len(_components(a)) != 1 or len(_components(b)) != 1:
            continue
        pieces = [a, b]
        for owner in profile["damaged"]:
            options = _erase_options(rng, pieces[owner], h, w,
                                     offset, marker, erasure, owner)
            if not options:
                break
            pieces[owner] = options[int(rng.integers(len(options)))]
        else:
            inp = _stage(rng, pieces, bg, profile["stage"])
            if inp is None:
                continue
            rot = profile["rotation"]
            inp = np.rot90(inp, rot).copy()
            # Ground truth is the complete original mosaic, not solver output.
            content = {tuple(map(int, q)): int(full[tuple(q)]) for q in np.argwhere(full != bg)}
            out = np.rot90(_crop(content, bg), rot).copy()
            if _key(inp) in used:
                continue
            try:
                p = _parse(inp)
            except ValueError:
                continue
            info = p["info"]
            if (not np.array_equal(p["output"], out) or not info["recovered_pigment"]
                    or not info["exclusive_cross_fragment_witnesses"]
                    or len(np.unique(out[out != bg])) < 2):
                continue
            if profile["teach"]:
                # The two minimum demonstrations expose target-order reversal,
                # joint witnesses, and guide transparency, irrespective of seed.
                if np.array_equal(_alternative(p, "repeat_half"), out):
                    continue
                if profile["partial"] and (not info["uncovered_guides"] or not info["covered_guides"]):
                    continue
                if profile["fusion_teaching"] and (not info["recovered_background"] or not info["borrowed"]):
                    continue
            return {"input": inp, "output": out}
    raise RuntimeError("Bounded mosaic construction exhausted")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(("bar_interpolation", "larger_mosaic", "new_staging", "two_erased_fragments")))
    palette = _palette(rng)
    varying_palette = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    mixed_axes = bool(rng.integers(2))
    style = str(rng.choice(("random", "patches", "bands", "motifs")))
    silhouettes = ("solid", "rings", "terraces", "branches", "notches")
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            training = split == "train"
            heights = (4, 7)
            if regime == "larger_mosaic":
                heights = (4, 6) if training else (6, 8)
            lengths = (2, 4) if training and regime == "bar_interpolation" else ((3,) if regime == "bar_interpolation" else (2, 3, 4, 5))
            if training and i < 2 and regime == "bar_interpolation":
                lengths = (2,) if i == 0 else (4,)
            pair_rotation = (rotation + i % 2) % 4 if mixed_axes else rotation
            # A half-turn swaps the mating roles. Teach damage on each role,
            # even when the scene orientation changes between demonstrations.
            owner = (i % 2) ^ int(pair_rotation >= 2)
            damaged = (owner,) if training or regime != "two_erased_fragments" else (0, 1)
            if training and i >= 2 and regime != "two_erased_fragments" and rng.random() < .3:
                damaged = (0, 1)
            stage = "horizontal" if training and regime == "new_staging" else (str(rng.choice(("vertical", "diagonal"))) if regime == "new_staging" else str(rng.choice(("horizontal", "vertical", "diagonal"))))
            profile = {"heights": heights, "lengths": lengths,
                       "palette": _palette(rng) if varying_palette else palette,
                       "silhouette": str(rng.choice(silhouettes[1:] if training and i == 1 else silhouettes)),
                       "style": style if rng.random() < .6 else str(rng.choice(("random", "patches", "bands", "motifs"))),
                       "partial": training and i == 1 or (i > 1 and rng.random() < .4),
                       "overlap": True if training and i == 1 else bool(rng.integers(2)), "damaged": damaged,
                       "rotation": pair_rotation,
                       "stage": stage, "teach": training and i < 2,
                       "fusion_teaching": training and i == 1}
            pair = _make_pair(rng, profile, used)
            puzzle[split].append(pair)
            used.add(_key(pair["input"]))
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[sp], list) for sp in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert _key(a) not in seen, "Repeated input"
            seen.add(_key(a))
            p = _parse(a)
            assert np.array_equal(p["output"], b), "Wrong repaired mosaic or crop"
            assert p["marker"] not in b and p["erasure"] not in b
            assert p["info"]["recovered_pigment"] > 0, "Reflection must affect the answer"
            assert p["info"]["cross_fragment_witnesses"] > 0, "Registration must make a witness available"
            assert p["info"]["exclusive_cross_fragment_witnesses"] > 0, "Separate fragment repair must be insufficient"
            assert len(np.unique(b[b != p["background"]])) >= 2
            # Foreground border support establishes exact tight cropping.
            assert all(np.any(edge != p["background"]) for edge in (b[0], b[-1], b[:, 0], b[:, -1]))
            for model in ("no_reflection", "anchor_only", "mover_only"):
                assert not np.array_equal(_alternative(p, model), b), f"Unused concept: {model}"
            records[split].append(p)
    # Tests introduce combinations and scale/placement, not an unshown normal.
    assert {p["axis"] for p in records["test"]} <= {p["axis"] for p in records["train"]}
    for model in _ALTERNATIVES:
        trained_difference = any(not np.array_equal(_alternative(p, model), p["output"])
                                 for p in records["train"])
        if not trained_difference:
            assert all(np.array_equal(_alternative(p, model), p["output"])
                       for p in records["test"]), f"Untaught distinction: {model}"
    assert any(p["info"]["covered_guides"] and p["info"]["uncovered_guides"] for p in records["train"])
    assert any(p["info"]["recovered_background"] and p["info"]["borrowed"]
               for p in records["train"]), "Teach silhouette recovery and observed-over-unknown fusion"
    return True


def self_test():
    """Bounded 42-seed survey, count matrix and semantic mutation checks."""
    fingerprints, shapes, output_shapes, structures, roles = set(), set(), set(), set(), set()
    lengths, axes, features = set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[sp]) for sp in ("train", "test"))
        fingerprints.add(signature)
        if seed in (0, 21, 41):
            again = generate_puzzle(seed)
            assert signature == tuple(tuple((_key(p["input"]), _key(p["output"])) for p in again[sp]) for sp in ("train", "test"))
        for split in puzzle:
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                p = _parse(a)
                assert np.array_equal(solve(a.copy()), b)
                shapes.add(a.shape)
                output_shapes.add(b.shape)
                roles.add((p["background"], p["erasure"], p["marker"]))
                lengths.add(p["length"])
                axes.add(p["axis"])
                features.append(p["info"])
                mapping = {}
                structures.add((a.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in a.flat)))
        # Arbitrary pair-specific color bijections and whole-scene rotations.
        if seed == 0:
            transformed = {sp: [] for sp in puzzle}
            for sp in puzzle:
                for pair in puzzle[sp]:
                    perm = np.random.default_rng(71 + len(transformed[sp])).permutation(10)
                    transformed[sp].append({k: perm[np.rot90(v)].astype(np.int8) for k, v in pair.items()})
            validate_generated_puzzle(transformed)
    assert len(fingerprints) == 42
    assert len(shapes) >= 45 and len(output_shapes) >= 20 and len(structures) >= 180
    assert len(roles) >= 35 and lengths >= {2, 3, 4, 5} and axes == {0, 1}
    assert any(x["borrowed"] for x in features) and any(x["overlap"] for x in features)
    assert any(not x["overlap"] for x in features)
    assert any(x["recovered_background"] for x in features)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(817, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Every named counter-rule changes at least one demonstration in this puzzle.
    p = generate_puzzle(0, 2, 1)
    for model in _ALTERNATIVES:
        assert any(not np.array_equal(_alternative(_parse(pair["input"]), model), pair["output"])
                   for pair in p["train"])
    mutant = {sp: [{k: v.copy() for k, v in pair.items()} for pair in p[sp]] for sp in p}
    mutant["test"][0]["output"][0, 0] = (int(mutant["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(mutant)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("Wrong label accepted")
    # A conflicting visible reflection witness invalidates the input itself.
    original = p["train"][0]["input"]
    parsed = _parse(original)
    known, _, _ = _registered(parsed)
    mover_known = {(r-1, c) for r, c in parsed["mover"]["known"]}
    q = next(q for q in parsed["anchor"]["known"]
             if (-1-q[0], q[1]) in mover_known and q not in mover_known)
    r0, c0 = parsed["anchor"]["origin"]
    r, c = q[0]+r0, q[1]+c0
    address = (c, r) if parsed["axis"] else (r, c)
    corrupt = original.copy()
    pigments = set(known.values())
    corrupt[address] = next(v for v in sorted(pigments) if v != int(corrupt[address]))
    try:
        solve(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting visible witness accepted")
    # Losing both orbit members is rejected even if the color could be guessed.
    erased = {k: v for k, v in parsed.items()}
    erased["anchor"] = {k: (dict(v) if k == "known" else set(v) if k == "unknown" else v)
                        for k, v in parsed["anchor"].items()}
    erased["mover"] = {k: (dict(v) if k == "known" else set(v) if k == "unknown" else v)
                       for k, v in parsed["mover"].items()}
    mate = (-1-q[0], q[1])
    erased["anchor"]["known"].pop(q)
    erased["anchor"]["unknown"].add(q)
    erased["mover"]["known"].pop((mate[0]+1, mate[1]))
    erased["mover"]["unknown"].add((mate[0]+1, mate[1]))
    # Remove any redundant anchor witness so this orbit is wholly missing.
    erased["anchor"]["known"].pop(mate, None)
    try:
        _complete(erased)
    except ValueError:
        pass
    else:
        raise AssertionError("Fully erased reflection orbit accepted")
    # Solver support extends beyond the sampler: full-width length-six bars,
    # one-row damage, and a colored corner with no surrounding blank border.
    top = np.array([[1, 2, 3, 4, 1, 2], [3, 4, 1, 2, 3, 4],
                    [1, 3, 2, 4, 1, 3], [2, 4, 3, 1, 2, 4]], dtype=np.int8)
    full = np.concatenate((top, top[::-1]))
    a = {(r-4, c): int(full[r, c]) for r in range(4) for c in range(6)}
    b = {(r-3, c): int(full[r, c]) for r in range(4, 8) for c in range(6)}
    for cells in (a, b):
        cells.update({(0, c): 9 for c in range(6)})
    a.update({(-4, c): 8 for c in range(4)})
    scene = np.zeros((5, 13), dtype=np.int8)
    for (r, c), v in a.items():
        scene[r+4, c] = v
    for (r, c), v in b.items():
        scene[r, c+7] = v
    assert scene[0, 0] != 0
    assert np.array_equal(solve(scene), full)
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "normalized_inputs": len(structures),
            "role_triples": len(roles)}


if __name__ == "__main__":
    print(self_test())
