"""Reveal self-similar stamp fields behind an opaque rectangular pane.

The decoder uses pixels only. The sampler renders a complete scene, crops its
ground truth, then occludes it; it does not obtain labels by calling solve.
"""
from math import isqrt

import numpy as np


HOST_SEEDS = (0, 355380859, 480043125, 867505465, 478153360, 794517629,
              532615571, 702292189, 577115838, 919558150, 172929117,
              833099105, 58398229, 722400738, 564108725, 961551943,
              510956648, 249063231, 745519551, 308499287, 465045843,
              209330175, 468027437, 841207384, 477408760, 128582305,
              346075122, 743336736, 8370037, 795318335, 729100074,
              459059979, 947382850, 750256974, 443853375, 209569943,
              642728618, 870008431, 566693468, 323081038, 615606301,
              429268405)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "grid must contain integers"
    assert np.all((0 <= a) & (a <= 9)), "invalid ARC colors"
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} |
            {(b, x) for x in range(l, r + 1)} |
            {(y, l) for y in range(t + 1, b)} |
            {(y, r) for y in range(t + 1, b)})


def _area(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _draw_carrier(canvas, carrier, background, glyph_background):
    """Rendering from a complete mask, independent of the input decoder."""
    t, b, l, r = carrier["box"]
    mask, ink, frame = carrier["mask"], carrier["ink"], carrier["frame"]
    h, w = mask.shape
    glyph = np.where(mask, ink, glyph_background).astype(np.int8)
    canvas[t + 1:b, l + 1:r] = background
    for y, x in np.argwhere(mask):
        dy, dx = t + 1 + int(y) * h, l + 1 + int(x) * w
        canvas[dy:dy + h, dx:dx + w] = glyph
    for p in _perimeter(carrier["box"]):
        canvas[p] = frame


def _pane(grid, background):
    """A dedicated monochrome perimeter around a uniform, distinct fill."""
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == background:
            continue
        points = set(map(tuple, np.argwhere(grid == color)))
        box = _bbox(points)
        t, b, l, r = box
        if b - t < 2 or r - l < 2 or points != _perimeter(box):
            continue
        inside = grid[t + 1:b, l + 1:r]
        fill = int(inside[0, 0])
        if fill not in (color, background) and np.all(inside == fill):
            candidates.append((inside.size, box, color, fill))
    assert candidates, "no opaque pane"
    candidates.sort(reverse=True)
    assert len(candidates) == 1 or candidates[0][0] > candidates[1][0], "opaque pane size tie"
    _, box, frame, fill = candidates[0]
    # Both pane colors are dedicated roles, not colors borrowed from glyphs.
    assert set(map(tuple, np.argwhere(grid == fill))) == _area(
        (box[0] + 1, box[1] - 1, box[2] + 1, box[3] - 1))
    return box, frame, fill


def _decode_mask(grid, box, hidden, background):
    """Unify observations at coarse and fine scales; no reference tile needed.

    Inactive coarse sites are background. At an active site a background-of-
    glyph pixel means (coarse bit=1, fine bit=0), and ink means both bits=1.
    Each observation therefore constrains the SAME h by w Boolean array twice.
    """
    t, b, l, r = box
    h, w = isqrt(b - t - 1), isqrt(r - l - 1)
    if not (2 <= h <= 4 and 2 <= w <= 4 and
            h * h == b - t - 1 and w * w == r - l - 1):
        return []
    visible = [(y, x) for y in range(t + 1, b) for x in range(l + 1, r)
               if (y, x) not in hidden]
    colors = sorted({int(grid[p]) for p in visible} - {background})
    if len(colors) != 2:
        return []
    results = []
    for ink, glyph_background in (colors, colors[::-1]):
        bits = np.full((h, w), -1, dtype=np.int8)
        consistent = True

        def assign(y, x, value):
            nonlocal consistent
            if bits[y, x] != -1 and bits[y, x] != value:
                consistent = False
            bits[y, x] = value

        for y, x in visible:
            dy, dx = y - t - 1, x - l - 1
            coarse_y, coarse_x = dy // h, dx // w
            fine_y, fine_x = dy % h, dx % w
            color = int(grid[y, x])
            if color == background:
                assign(coarse_y, coarse_x, 0)
            elif color == glyph_background:
                assign(coarse_y, coarse_x, 1)
                assign(fine_y, fine_x, 0)
            elif color == ink:
                assign(coarse_y, coarse_x, 1)
                assign(fine_y, fine_x, 1)
            else:
                consistent = False
        if consistent and np.all(bits >= 0) and 3 <= bits.sum() < h * w:
            results.append({"box": box, "mask": bits.astype(bool), "ink": ink,
                            "glyph_background": glyph_background})
    return results


def _infer(input_grid):
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    assert np.all(grid[0] == background) and np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background) and np.all(grid[:, -1] == background)
    pane, pane_frame, fill = _pane(grid, background)
    hidden = _area(pane)
    strict = _area((pane[0] + 1, pane[1] - 1, pane[2] + 1, pane[3] - 1))
    carriers = []
    for frame in map(int, np.unique(grid)):
        if frame in (background, pane_frame, fill):
            continue
        points = set(map(tuple, np.argwhere(grid == frame)))
        box = _bbox(points)
        full = _perimeter(box)
        if points != full - hidden or not (full & strict):
            continue
        fits = _decode_mask(grid, box, hidden, background)
        if not fits:
            continue
        assert len(fits) == 1, "two ink-role explanations"
        carrier = fits[0]
        carrier["frame"] = frame
        carriers.append(carrier)
    assert 1 <= len(carriers) <= 3, "expected one to three recoverable carriers"
    assert len({c["glyph_background"] for c in carriers}) == 1
    glyph_background = carriers[0]["glyph_background"]
    roles = [background, pane_frame, fill, glyph_background]
    roles += [value for c in carriers for value in (c["frame"], c["ink"])]
    assert len(set(roles)) == len(roles), "color roles must be disjoint"
    occupied = set()
    under = np.full(grid.shape, background, dtype=np.int8)
    for carrier in carriers:
        cells = _area(carrier["box"])
        assert not (cells & occupied), "carrier boxes overlap"
        occupied.update(cells)
        _draw_carrier(under, carrier, background, glyph_background)
    visible = np.ones(grid.shape, dtype=bool)
    visible[pane[0]:pane[1] + 1, pane[2]:pane[3] + 1] = False
    assert np.array_equal(under[visible], grid[visible]), "unexplained visible pixels"
    t, b, l, r = pane
    output = under[t + 1:b, l + 1:r].copy()
    output[output == background] = fill
    return output, {"background": background, "pane": pane, "pane_frame": pane_frame,
                    "fill": fill, "glyph_background": glyph_background,
                    "carriers": carriers, "under": under}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the pane's restored strict-interior crop from the input alone."""
    return _infer(input_grid)[0]


def _asymmetric(mask):
    # Compare the whole h by w blueprint, including its inactive margins.
    return all(not (a.shape == mask.shape and np.array_equal(a, mask))
               for a in [np.rot90(mask, k) for k in (1, 2, 3)] +
               [np.fliplr(np.rot90(mask, k)) for k in range(4)])


def _mask(rng, h, w, require_asymmetry=False):
    # All macro rows/columns have activity. This supplies complementary clipped
    # stamps in the band mode; topology and density remain broadly variable.
    for _ in range(256):
        mask = np.zeros((h, w), dtype=bool)
        count = int(rng.integers(max(3, h, w), h * w))
        mask.flat[rng.choice(h * w, count, replace=False)] = True
        if (np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)) and
                (not require_asymmetry or _asymmetric(mask))):
            return mask
    # Finite constructive fallback, tested for every supported sampling shape.
    for code in range(1, 1 << (h * w)):
        mask = np.array([(code >> i) & 1 for i in range(h * w)], dtype=bool).reshape(h, w)
        if (3 <= mask.sum() < h * w and np.all(mask.any(axis=0)) and
                np.all(mask.any(axis=1)) and (not require_asymmetry or _asymmetric(mask))):
            return mask
    raise AssertionError("no asymmetric spanning mask for this shape")


def _render(shape, boxes, masks, pane, palette):
    background, pane_frame, fill, glyph_background = map(int, palette[:4])
    carriers = [{"box": box, "mask": mask, "frame": int(palette[4 + 2 * i]),
                 "ink": int(palette[5 + 2 * i])}
                for i, (box, mask) in enumerate(zip(boxes, masks))]
    under = np.full(shape, background, dtype=np.int8)
    for carrier in carriers:
        _draw_carrier(under, carrier, background, glyph_background)
    t, b, l, r = pane
    output = under[t + 1:b, l + 1:r].copy()
    output[output == background] = fill
    grid = under.copy()
    grid[t + 1:b, l + 1:r] = fill
    for p in _perimeter(pane):
        grid[p] = pane_frame
    return {"input": grid, "output": output}


def _orient(pair, turns, flip):
    return {k: (np.fliplr(np.rot90(v, turns)) if flip else np.rot90(v, turns)).copy()
            for k, v in pair.items()}


def _corner(rng, palette, require_asymmetry=False):
    h, w = 3, int(rng.choice((2, 3, 4)))
    ch, cw = h * h + 2, w * w + 2
    height = int(rng.integers(ch + 4, min(30, ch + 11) + 1))
    width = int(rng.integers(cw + 4, min(30, cw + 11) + 1))
    t, l = int(rng.integers(2, height - ch - 1)), int(rng.integers(2, width - cw - 1))
    box = (t, t + ch - 1, l, l + cw - 1)
    # A corner cut leaves a complete bottom row of macro tiles visible.
    dy = int(rng.integers(h + 1, h * h - h + 1))
    dx = int(rng.integers(w, max(w + 1, w * w - w + 1)))
    pane = (t - 1, t + 1 + dy, l - 1, l + 1 + dx)
    return _render((height, width), [box], [_mask(rng, h, w, require_asymmetry)], pane, palette)


def _band(rng, palette, count, asymmetric_cut, require_asymmetry=False):
    h = int(rng.choice((3, 4)))
    # Random rectangular tile widths subject only to the canvas packing budget.
    options = [(2,), (3,), (4,)] if count == 1 else None
    if count == 2:
        options = [(a, b) for a in (2, 3, 4) for b in (2, 3, 4)
                   if a * a + b * b + 5 <= 26]
    if count == 3:
        options = [(a, b, c) for a in (2, 3) for b in (2, 3) for c in (2, 3)
                   if a * a + b * b + c * c + 8 <= 26]
    widths = options[int(rng.integers(len(options)))]
    gaps = [int(rng.integers(1, 3)) for _ in range(count - 1)]
    packed = sum(w * w + 2 for w in widths) + sum(gaps)
    if packed > 26:
        gaps = [1] * (count - 1)
        packed = sum(w * w + 2 for w in widths) + sum(gaps)
    cuts = [(a, b) for a in range(1, h) for b in range(1, h)
            if a + b >= h and (not asymmetric_cut or a != b)]
    a, b = cuts[int(rng.integers(len(cuts)))] if asymmetric_cut else (h - 1, h - 1)
    # A shared pane may meet carriers at different fine-pixel phases.
    shifts = [s for s in range(-h, h + 1) if 1 <= a - s < h and 1 <= b + s < h]
    offsets = [int(rng.choice(shifts)) for _ in widths]
    top = 2 - min(offsets)
    last = top + max(offsets) + h * h + 1
    height = int(rng.integers(last + 3, min(30, last + 8) + 1))
    width = int(rng.integers(packed + 4, min(30, packed + 8) + 1))
    left = int(rng.integers(2, width - packed - 1))
    boxes, masks, x = [], [], left
    for i, w in enumerate(widths):
        y = top + offsets[i]
        boxes.append((y, y + h * h + 1, x, x + w * w + 1))
        masks.append(_mask(rng, h, w, require_asymmetry))
        x += w * w + 2 + (gaps[i] if i < count - 1 else 0)
    pane = (top + 1 + a, top + h * h - b, left - 1, x)
    return _render((height, width), boxes, masks, pane, palette)


def _features(grid):
    output, info = _infer(grid)
    pane, bg, fill = info["pane"], info["background"], info["fill"]
    closed, strict = _area(pane), _area((pane[0] + 1, pane[1] - 1, pane[2] + 1, pane[3] - 1))
    full_tiles, partial_tiles, erased_tiles, a_cells, b_cells, background_cells = 0, 0, 0, 0, 0, 0
    inactive_cells, ink_cells = 0, 0
    shapes, masks, phases = set(), set(), set()
    for carrier in info["carriers"]:
        t, bottom, l, right = carrier["box"]
        mask = carrier["mask"]
        h, w = mask.shape
        shapes.add((h, w))
        masks.add(_key(mask))
        phases.add(((pane[0] - t - 1) % h, (pane[2] - l - 1) % w))
        a_cells += len(_perimeter(carrier["box"]) & strict)
        for y, x in np.argwhere(~mask):
            dy, dx = t + 1 + int(y) * h, l + 1 + int(x) * w
            inactive_cells += len(_area((dy, dy + h - 1, dx, dx + w - 1)) & strict)
        for y, x in np.argwhere(mask):
            start_y, start_x = t + 1 + int(y) * h, l + 1 + int(x) * w
            cells = _area((start_y, start_y + h - 1, start_x, start_x + w - 1))
            exposed = cells - closed
            full_tiles += len(exposed) == len(cells)
            partial_tiles += 0 < len(exposed) < len(cells)
            erased_tiles += not exposed
            b_cells += len(cells & strict)
            background_cells += sum(info["under"][p] == info["glyph_background"] for p in cells & strict)
            ink_cells += sum(info["under"][p] == carrier["ink"] for p in cells & strict)
    return {"count": len(info["carriers"]), "tile_shapes": shapes, "masks": masks,
            "phases": phases, "full_tiles": full_tiles, "partial_tiles": partial_tiles,
            "erased_tiles": erased_tiles, "a_cells": a_cells, "b_cells": b_cells,
            "glyph_background_cells": background_cells, "inactive_cells": inactive_cells,
            "ink_cells": ink_cells, "output_shape": output.shape,
            "fill": fill, "background": bg, "info": info}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    a, b = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    puzzle, seen, first_masks = {"train": [], "test": []}, set(), set()
    for split, count in (("train", a), ("test", b)):
        for i in range(count):
            corner = split == "train" and (i == 0 or (i > 1 and rng.random() < .25))
            n = 1 if corner or (split == "train" and i == 1) else int(rng.integers(1 if split == "train" else 2, 4))
            asymmetric_cut = split == "test" or (split == "train" and i > 1)
            for _ in range(96):
                palette = rng.permutation(10)
                asymmetric_mask = split == "train" and i < 2
                pair = (_corner(rng, palette, asymmetric_mask) if corner else
                        _band(rng, palette, n, asymmetric_cut, asymmetric_mask))
                pair = _orient(pair, int(rng.integers(4)), bool(rng.integers(2)))
                try:
                    f = _features(pair["input"])
                    assert np.array_equal(solve(pair["input"]), pair["output"])
                    assert f["a_cells"] and f["b_cells"] and f["glyph_background_cells"]
                    assert f["ink_cells"]
                    if not corner:
                        assert f["full_tiles"] == 0 and f["erased_tiles"] > 0
                    if split == "train" and i == 1:
                        assert f["masks"] != first_masks
                    if split == "train" and i < 2:
                        assert f["inactive_cells"], "demonstrate inactive macro sites"
                    if _key(pair["input"]) in seen:
                        continue
                except AssertionError:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                if split == "train" and i == 0:
                    first_masks = f["masks"]
                break
            else:
                raise AssertionError("bounded scene construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate inputs"
            seen.add(_key(inp))
            expected, info = _infer(inp)
            assert np.array_equal(out, expected), "incorrect reconstructed pane interior"
            assert out.shape != inp.shape
            f = _features(inp)
            assert f["a_cells"] > 0, "omitting geometric completion must change the answer"
            assert f["b_cells"] > 0, "omitting scale echo must change the answer"
            assert f["glyph_background_cells"] > 0, "full glyph copying must matter"
            assert f["ink_cells"] > 0, "fine-scale stencil must affect hidden ink"
            # Every marker has at least one strict-interior hidden boundary cell.
            pane = info["pane"]
            strict = _area((pane[0] + 1, pane[1] - 1, pane[2] + 1, pane[3] - 1))
            for c in info["carriers"]:
                assert _perimeter(c["box"]) & strict
                mask, ink = c["mask"], c["ink"]
                h, w = mask.shape
                # Algebraic check of the rendered field, without tile painting:
                # coarse activity is M x all-ones, and fine ink is M x M.
                macro = np.kron(mask, np.ones((h, w), dtype=bool))
                micro = np.kron(mask, mask)
                field = np.where(micro, ink, np.where(macro, info["glyph_background"], info["background"]))
                t, bottom, l, right = c["box"]
                assert np.array_equal(field, info["under"][t + 1:bottom, l + 1:right])
            records[split].append(f)
    assert records["train"][0]["full_tiles"] > 0
    assert records["train"][1]["full_tiles"] == 0
    assert records["train"][1]["partial_tiles"] >= 2
    assert records["train"][0]["masks"] != records["train"][1]["masks"]
    for f in records["train"][:2]:
        assert f["inactive_cells"] > 0
        assert all(_asymmetric(c["mask"]) for c in f["info"]["carriers"])
    assert all(f["count"] >= 2 and f["full_tiles"] == 0 and f["erased_tiles"] > 0
               for f in records["test"])


def _puzzle_key(p):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in p[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded author-owned semantics, ablations, variation, and count checks."""
    signatures, shapes, masks, phases, layouts, normalized, backgrounds, counts = (set() for _ in range(8))
    partial_only, multi, disconnected, holes = 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                f = _features(inp)
                info = f["info"]
                shapes.add((inp.shape, out.shape))
                masks.update(f["masks"])
                phases.update(f["phases"])
                counts.add(f["count"])
                backgrounds.add(f["background"])
                layouts.add((inp.shape, info["pane"], tuple(c["box"] for c in info["carriers"])))
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(c), len(roles)) for c in inp.flat)))
                partial_only += f["full_tiles"] == 0
                multi += f["count"] > 1
                # Concrete ablations: reveal only loop perimeters, or only fields.
                t, b, l, r = info["pane"]
                just_a = np.full(out.shape, info["fill"], dtype=np.int8)
                without_a = out.copy()
                for c in info["carriers"]:
                    for y, x in _perimeter(c["box"]):
                        if t < y < b and l < x < r:
                            just_a[y - t - 1, x - l - 1] = c["frame"]
                            without_a[y - t - 1, x - l - 1] = info["fill"]
                    active = set(map(tuple, np.argwhere(c["mask"])))
                    todo, visited = [next(iter(active))], set()
                    while todo:
                        p = todo.pop()
                        if p in visited:
                            continue
                        visited.add(p)
                        y, x = p
                        todo.extend(q for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                                    if q in active and q not in visited)
                    disconnected += visited != active
                    m = c["mask"]
                    holes += any(not m[y, x] and m[y - 1, x] and m[y + 1, x] and
                                 m[y, x - 1] and m[y, x + 1]
                                 for y in range(1, m.shape[0] - 1) for x in range(1, m.shape[1] - 1))
                assert not np.array_equal(just_a, out) and not np.array_equal(without_a, out)
                assert np.array_equal(solve(inp.tolist()), out)
                before = inp.copy()
                solve(inp)
                assert np.array_equal(inp, before), "solver mutates its argument"
    assert len(signatures) == 42 and len(normalized) >= 180
    assert len(shapes) >= 70 and len(masks) >= 90 and len(layouts) >= 180
    assert counts == {1, 2, 3} and backgrounds == set(range(10))
    assert len(phases) >= 8 and partial_only >= 120 and multi >= 70
    assert disconnected >= 20 and holes >= 5
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    # The reader supports all nine 2..4 rectangular blueprint geometries,
    # including the symmetric 2x2 case outside the asymmetric demonstration
    # sampler, rather than depending on the sampled band layouts.
    rng = np.random.default_rng(2026)
    for h in range(2, 5):
        for w in range(2, 5):
            mask = _mask(rng, h, w)
            p = _render((h * h + 6, w * w + 6), [(2, h * h + 3, 2, w * w + 3)],
                        [mask], (1, 4, 1, 4), (0, 1, 2, 3, 4, 5))
            assert np.array_equal(solve(p["input"]), p["output"])
    # A diagonal arrangement and a blueprint with an empty row/column verify
    # that neither the packed band layout nor the spanning-mask sampler is a
    # requirement of the input-only rule.
    empty_margin = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=bool)
    other = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool)
    p = _render((30, 30), [(2, 12, 2, 12), (16, 26, 16, 26)],
                [empty_margin, other], (8, 20, 8, 20), tuple(range(8)))
    assert np.array_equal(solve(p["input"]), p["output"])
    # Observing only one fine row does not determine an arbitrary blueprint.
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool)
    p = _render((17, 17), [(2, 12, 2, 12)], [mask], (4, 13, 1, 13), (0, 1, 2, 3, 4, 5))
    assert _decode_mask(p["input"], (2, 12, 2, 12), _area((4, 13, 1, 13)), 0) == []
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123456 + 10 * a + b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {args}")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    damaged = generate_puzzle(0, 2, 1)
    _, info = _infer(damaged["train"][0]["input"])
    c = info["carriers"][0]
    p = next(iter(_perimeter(c["box"]) - _area(info["pane"])))
    damaged["train"][0]["input"][p] = info["background"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("broken visible outline accepted")
