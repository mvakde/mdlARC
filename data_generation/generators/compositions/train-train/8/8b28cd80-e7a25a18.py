"""Swatch-fed spirals: doubled miniature coordinates feed a winding paintbrush.

Only NumPy and the standard library are used. No parent module or data is loaded.
The input parser, including every semantic color role, is independent of sampling.
"""

from __future__ import annotations

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its legal range")
    return value


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any(grid < 0) or np.any(grid > 9)):
        raise ValueError("a grid must be an integer rectangle of colors 0..9, size 1..30")
    return grid.astype(np.int8, copy=False)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, tuple(int(x) for x in grid.flat)


def _border(grid, color):
    grid[0, :] = grid[-1, :] = color
    grid[:, 0] = grid[:, -1] = color


def _parse(input_grid):
    """Find a unique frame, background, dense miniature, paper and needle."""
    inp = _grid(input_grid)
    candidates = []
    colors = [int(x) for x in np.unique(inp)]
    for frame in colors:
        positions = np.argwhere(inp == frame)
        r0, c0 = (int(x) for x in positions.min(axis=0))
        r1, c1 = (int(x) for x in positions.max(axis=0))
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        if not (10 <= oh <= 30 and 10 <= ow <= 30 and oh % 2 == ow % 2 == 0):
            continue
        crop = inp[r0:r1 + 1, c0:c1 + 1]
        if (len(positions) != 2 * oh + 2 * ow - 4
                or not np.all(crop[0] == frame) or not np.all(crop[-1] == frame)
                or not np.all(crop[:, 0] == frame) or not np.all(crop[:, -1] == frame)):
            continue
        mh, mw = (oh - 2) // 2, (ow - 2) // 2
        for background in colors:
            if background == frame:
                continue
            content = np.argwhere((inp != background) & (inp != frame))
            if not len(content):
                continue
            mr0, mc0 = (int(x) for x in content.min(axis=0))
            mr1, mc1 = (int(x) for x in content.max(axis=0))
            if ((mr1 - mr0 + 1, mc1 - mc0 + 1) != (mh, mw)
                    or not (r0 < mr0 <= mr1 < r1 and c0 < mc0 <= mc1 < c1)):
                continue
            motif = inp[mr0:mr1 + 1, mc0:mc1 + 1]
            mc, counts = np.unique(motif, return_counts=True)
            if (background in mc or frame in mc or not 4 <= len(mc) <= 6
                    or np.count_nonzero(counts == 1) != 1):
                continue
            paper = int(motif[0, 0])
            if not (motif[0, -1] == motif[-1, 0] == motif[-1, -1] == paper):
                continue
            needle = int(mc[counts == 1][0])
            if paper == needle:
                continue
            expected = np.full(inp.shape, background, dtype=np.int8)
            framed = expected[r0:r1 + 1, c0:c1 + 1]
            _border(framed, frame)
            expected[mr0:mr1 + 1, mc0:mc1 + 1] = motif
            if not np.array_equal(inp, expected):
                continue
            needle_pos = tuple(int(x) for x in np.argwhere(motif == needle)[0])
            candidates.append({
                "frame": frame, "background": background, "paper": paper,
                "needle": needle, "needle_pos": needle_pos, "motif": motif.copy(),
                "ink": frozenset(int(x) for x in mc if x not in (paper, needle)),
                "outer": (oh, ow), "frame_pos": (r0, c0),
                "offset": (mr0 - r0 - 1, mc0 - c0 - 1),
            })
    if len(candidates) != 1:
        raise ValueError("input must have exactly one valid frame and miniature")
    return candidates[0]


def _spiral_path(shape, anchor):
    """Infinite N2,E2,S4,W4,N6,E6,... walk, retaining in-window visits.

    Off-window steps still happen; their sampler is neutral and they do not
    reset carried paint. The walk never revisits a cell. A generous finite
    radius covers every segment that could enter the finite window.
    """
    h, w = shape
    y, x = anchor
    path = [(y, x)]
    distance = max(y, x, h - 1 - y, w - 1 - x)
    directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
    for leg in range(2 * distance + 6):
        dy, dx = directions[leg % 4]
        for _ in range(2 * (leg // 2 + 1)):
            y, x = y + dy, x + dx
            if 0 <= y < h and 0 <= x < w:
                path.append((y, x))
    return path


def _shell_mask(shape, anchor):
    """An algebraic check on the walk, including its cuts and bridges."""
    yy, xx = np.indices(shape)
    ay, ax = anchor
    mask = np.maximum(abs(yy - ay), abs(xx - ax)) % 2 == 0
    for k in range(1, max(shape) + 1):
        y, x = ay - k, ax - k + 1
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            mask[y, x] = not bool(mask[y, x])
    return mask


def _render(found, *, magnify=True, transport=True):
    motif = found["motif"]
    h, w = motif.shape
    anchor = tuple(2 * x for x in found["needle_pos"])
    if magnify:
        sampler = np.repeat(np.repeat(motif, 2, axis=0), 2, axis=1)
    else:
        # A deliberately wrong comparison: same anchor and destination, but
        # source swatches are not magnified. Used only in validation/tests.
        sampler = np.full((2 * h, 2 * w), found["paper"], dtype=np.int8)
        sampler[:h, :w] = motif
    out = np.full(found["outer"], found["frame"], dtype=np.int8)
    interior = out[1:-1, 1:-1]
    interior[:] = found["background"]
    carried = found["needle"]
    for y, x in _spiral_path(interior.shape, anchor):
        sample = int(sampler[y, x])
        if sample != found["paper"]:
            carried = sample
        if transport or sample != found["paper"]:
            interior[y, x] = carried
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using this input alone; orientation and scale are family constants."""
    return _render(_parse(input_grid))


def _swatches(rng, shape, needle_pos, roles, inks, style):
    """Grow disjoint swatches; corners remain a visible neutral-paper key."""
    _, _, paper, needle = roles
    h, w = shape
    motif = np.full(shape, paper, dtype=np.int8)
    motif[needle_pos] = needle
    reserved = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), needle_pos}
    available = {(r, c) for r in range(h) for c in range(w)} - reserved
    for i, ink in enumerate(inks):
        max_take = min(8, len(available) - 2 * (len(inks) - i - 1))
        target = int(rng.integers(2, max(2, max_take) + 1))
        chosen = []
        for _ in range(target):
            options = sorted(available)
            if chosen and style == "patch":
                neighbors = [p for p in options if any(
                    abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in chosen)]
                if neighbors:
                    options = neighbors
            elif chosen and style == "bar":
                aligned = [p for p in options if p[0] == chosen[0][0]]
                if aligned:
                    options = aligned
            p = options[int(rng.integers(len(options)))]
            chosen.append(p)
            available.remove(p)
            motif[p] = ink
    return motif


def _dimension(rng, outer):
    slack = 30 - outer
    if rng.random() < .7:
        slack = min(slack, 5)
    return outer + int(rng.integers(slack + 1))


def _position(rng, slack):
    if rng.random() < .5:
        return int(rng.choice((0, slack)))
    return int(rng.integers(slack + 1))


def _make_pair(rng, shape, needle_pos, roles, ink_count, style, first=False):
    background, frame, paper, needle = roles
    ink_pool = [x for x in range(10) if x not in roles]
    inks = [int(x) for x in rng.choice(ink_pool, size=ink_count, replace=False)]
    h, w = shape
    outer = (2 * h + 2, 2 * w + 2)
    for _ in range(100):
        motif = _swatches(rng, shape, needle_pos, roles, inks, style)
        found = {"motif": motif, "needle_pos": needle_pos, "needle": needle,
                 "paper": paper, "background": background, "frame": frame,
                 "outer": outer}
        out = _render(found)
        visible = set(int(x) for x in np.unique(out[1:-1, 1:-1]))
        if not set(inks).issubset(visible):
            continue
        sampler = np.repeat(np.repeat(motif, 2, axis=0), 2, axis=1)
        path = _spiral_path(sampler.shape, tuple(2 * x for x in needle_pos))
        if not any(sampler[y, x] == paper and int(out[y + 1, x + 1]) in inks
                   for y, x in path):
            continue
        if np.array_equal(out, _render(found, transport=False)):
            continue
        if np.array_equal(out, _render(found, magnify=False)):
            continue
        break
    else:
        raise RuntimeError("could not construct informative swatches")
    height, width = (_dimension(rng, d) for d in outer)
    if first and (height, width) == outer:
        height += 1  # the first miniature is at most 8x8
    r0, c0 = _position(rng, height - outer[0]), _position(rng, width - outer[1])
    if rng.random() < .2:
        dr, dc = h // 2, w // 2
    else:
        dr, dc = int(rng.integers(h + 1)), int(rng.integers(w + 1))
    if first and (dr, dc) == (h // 2, w // 2):
        dr = 0
    inp = np.full((height, width), background, dtype=np.int8)
    _border(inp[r0:r0 + outer[0], c0:c0 + outer[1]], frame)
    inp[r0 + 1 + dr:r0 + 1 + dr + h, c0 + 1 + dc:c0 + 1 + dc + w] = motif
    return {"input": inp, "output": out}


def _random_shape(rng):
    sizes = np.arange(4, 15)
    probabilities = np.array([5, 6, 6, 5, 4, 3, 2, 2, 1, 1, 1], dtype=float)
    probabilities /= probabilities.sum()
    h = int(rng.choice(sizes, p=probabilities))
    w = h if rng.random() < .6 else int(rng.choice(sizes, p=probabilities))
    return h, w


def _center(shape):
    return tuple((x - 1) // 2 for x in shape)


def _plan(rng, nt, ns):
    h = int(rng.integers(5, 9))
    w = h if rng.random() < .6 else int(rng.integers(5, 9))
    base = h, w
    cy, cx = _center(base)
    if nt == 2:
        second = _random_shape(rng)
        while second == base:
            second = _random_shape(rng)
        train = [(base, (cy, cx)), (second, _center(second))]
        # With two demonstrations, reuse demonstrated anchor/size combinations.
        tests = [train[int(rng.integers(2))] for _ in range(ns)]
    else:
        # An L of needle coordinates isolates both coordinate axes.
        train = [(base, (cy, cx)), (base, (0, cx)), (base, (cy, 0))]
        for extra in range(nt - 3):
            shape = _random_shape(rng)
            while extra == 0 and shape == base:
                shape = _random_shape(rng)
            sh, sw = shape
            positions = [_center(shape), (0, (sw - 1) // 2),
                         (sh - 1, (sw - 1) // 2), ((sh - 1) // 2, 0),
                         ((sh - 1) // 2, sw - 1)]
            train.append((shape, positions[int(rng.integers(len(positions)))]))
        tests = [(base, (h - 1, cx)), (base, (cy, w - 1)),
                 (base, (h - 1, w - 2)), (base, (h - 2, w - 1))][:ns]
        # A scale extrapolation is supported only after a second shape is shown.
        if nt >= 4 and ns >= 2 and rng.random() < .6:
            shape = _random_shape(rng)
            while shape in {sh for sh, _ in train}:
                shape = _random_shape(rng)
            tests[-1] = shape, _center(shape)
    return train, tests


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    train_plan, test_plan = _plan(rng, nt, ns)
    shared_roles = bool(rng.integers(2))
    roles = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    puzzle = {"train": [], "test": []}
    used = set()
    for i, (shape, pos) in enumerate(train_plan + test_plan):
        split = "train" if i < nt else "test"
        if not shared_roles:
            roles = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
        ink_count = 2 + i if i < min(nt, 3) else int(rng.integers(2, min(nt, 3) + 2))
        style = ("patch", "bar", "scattered")[int(rng.integers(3))]
        for _ in range(30):
            pair = _make_pair(rng, shape, pos, roles, ink_count, style, first=i == 0)
            key = _key(pair["input"])
            if key not in used:
                used.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = (_grid(pair[k]) for k in ("input", "output"))
    found = _parse(inp)
    assert out.shape == found["outer"]
    assert np.all(out[0] == found["frame"]) and np.all(out[-1] == found["frame"])
    assert np.all(out[:, 0] == found["frame"]) and np.all(out[:, -1] == found["frame"])
    h, w = found["motif"].shape
    anchor = tuple(2 * x for x in found["needle_pos"])
    path = _spiral_path((2 * h, 2 * w), anchor)
    assert len(path) == len(set(path))
    mask = np.zeros((2 * h, 2 * w), dtype=bool)
    for p in path:
        mask[p] = True
    assert np.array_equal(mask, _shell_mask(mask.shape, anchor)), "wrong spiral seam"
    inner = out[1:-1, 1:-1]
    assert np.array_equal(inner != found["background"], mask)
    assert inner[anchor] == found["needle"]
    assert found["paper"] not in out and found["frame"] not in inner
    # Independently check the finite-state paint behavior against output visits.
    sampler = np.repeat(np.repeat(found["motif"], 2, axis=0), 2, axis=1)
    carried = found["needle"]
    neutral_ink_carries = 0
    for p in path:
        sample = int(sampler[p])
        if sample != found["paper"]:
            carried = sample
        else:
            neutral_ink_carries += carried in found["ink"]
        assert inner[p] == carried
    assert neutral_ink_carries > 0, "no evidence for carrying paint"
    assert found["ink"].issubset(set(int(x) for x in np.unique(inner)))
    assert not np.array_equal(out, _render(found, magnify=False)), "scale is irrelevant"
    assert not np.array_equal(out, _render(found, transport=False)), "transport is irrelevant"
    assert np.array_equal(out, solve(inp.copy()))
    return found


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen = set()
    found = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _check_pair(pair)
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            found[split].append(analysis)
    first = found["train"][0]
    assert first["needle_pos"] == _center(first["motif"].shape)
    assert first["offset"] != tuple(x // 2 for x in first["motif"].shape)
    assert np.asarray(puzzle["train"][0]["input"]).shape != first["outer"]
    if nt == 2:
        demonstrated = {(f["motif"].shape, f["needle_pos"]) for f in found["train"]}
        assert all((f["motif"].shape, f["needle_pos"]) in demonstrated
                   for f in found["test"])
        assert all(len(f["ink"]) <= 3 for f in found["test"])
    else:
        a, b, c = found["train"][:3]
        assert a["motif"].shape == b["motif"].shape == c["motif"].shape
        assert a["needle_pos"][1] == b["needle_pos"][1]
        assert a["needle_pos"][0] != b["needle_pos"][0]
        assert a["needle_pos"][0] == c["needle_pos"][0]
        assert a["needle_pos"][1] != c["needle_pos"][1]
        assert found["test"][0]["needle_pos"][0] == first["motif"].shape[0] - 1
    assert {len(f["ink"]) for f in found["train"]}.issuperset({2, 3})


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test():
    """Bounded 42-seed sweep, all count combinations, ablations and diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    shapes, anchors, frame_positions, offsets, palettes, structures = (set() for _ in range(6))
    role_colors = [set() for _ in range(4)]
    shared, mixed, rectangles, edge_frames, inside_frames, centered = 0, 0, 0, 0, 0, 0
    for puzzle in puzzles:
        roles_in_puzzle = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _parse(pair["input"])
                motif = f["motif"]
                shapes.add(motif.shape)
                anchors.add((motif.shape, f["needle_pos"]))
                frame_positions.add((pair["input"].shape, f["outer"], f["frame_pos"]))
                offsets.add((motif.shape, f["offset"]))
                roles = tuple(f[k] for k in ("background", "frame", "paper", "needle"))
                roles_in_puzzle.add(roles)
                for seen, color in zip(role_colors, roles):
                    seen.add(color)
                palettes.add(f["ink"])
                rectangles += motif.shape[0] != motif.shape[1]
                centered += f["offset"] == tuple(x // 2 for x in motif.shape)
                r, c = f["frame_pos"]
                oh, ow = f["outer"]
                ih, iw = pair["input"].shape
                edge_frames += r == 0 or c == 0 or r + oh == ih or c + ow == iw
                inside_frames += 0 < r and 0 < c and r + oh < ih and c + ow < iw
                renaming = {}
                structures.add((motif.shape, tuple(renaming.setdefault(int(x), len(renaming))
                                                   for x in motif.flat)))
        shared += len(roles_in_puzzle) == 1
        mixed += len(roles_in_puzzle) > 1
    assert len(shapes) >= 20 and len(anchors) >= 40
    assert len(frame_positions) >= 60 and len(offsets) >= 60
    assert len(structures) >= 180 and len(palettes) >= 40
    assert all(colors == set(range(10)) for colors in role_colors)
    assert shared >= 5 and mixed >= 5
    assert rectangles >= 20 and edge_frames >= 30 and inside_frames >= 3 and centered >= 5
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {nt}/{ns}")
    # Hand-check a seam and initial paint carry without relying on the renderer.
    path = _spiral_path((10, 12), (4, 4))
    assert path[:9] == [(4, 4), (3, 4), (2, 4), (2, 5), (2, 6),
                       (3, 6), (4, 6), (5, 6), (6, 6)]
    assert (2, 3) not in path and (1, 2) in path
    miniature = np.array([[int(c) for c in row] for row in
                          ("111111", "113311", "112141", "114411", "111111")],
                         dtype=np.int8)
    witness = np.zeros((14, 18), dtype=np.int8)
    _border(witness[1:13, 2:16], 9)
    witness[5:10, 4:10] = miniature
    inner = solve(witness)[1:-1, 1:-1]
    prefix = [(4, 4), (3, 4), (2, 4), (2, 5), (2, 6),
              (3, 6), (4, 6), (5, 6), (6, 6)]
    assert [int(inner[p]) for p in prefix] == [2, 3, 3, 3, 3, 3, 3, 3, 4]
    assert inner[2, 3] == 0 and inner[1, 2] == 4
    # An interior, noncentral needle is legal even though the sampler
    # emphasizes center and boundary positions. Full-canvas frames are legal.
    miniature[2, 2], miniature[1, 1] = 1, 2
    full_canvas = np.zeros((12, 14), dtype=np.int8)
    _border(full_canvas, 9)
    full_canvas[1:6, 7:13] = miniature
    parsed = _parse(full_canvas)
    assert parsed["needle_pos"] == (1, 1)
    assert solve(full_canvas)[3, 3] == 2
    for sh in ((8, 8), (8, 28), (28, 8), (28, 28)):
        for ay, ax in ((0, 2), (2, 4), (sh[0] - 2, sh[1] - 4)):
            visits = _spiral_path(sh, (ay, ax))
            walked = np.zeros(sh, dtype=bool)
            for p in visits:
                walked[p] = True
            assert len(visits) == int(walked.sum())
            assert np.array_equal(walked, _shell_mask(sh, (ay, ax)))
    return {"seeds": 42, "distinct_puzzles": 42, "source_shapes": len(shapes),
            "anchor_configurations": len(anchors), "frame_layouts": len(frame_positions),
            "motif_offsets": len(offsets), "motif_structures": len(structures),
            "ink_palettes": len(palettes), "shared_role_puzzles": shared,
            "mixed_role_puzzles": mixed, "rectangular_pairs": rectangles,
            "edge_frame_pairs": edge_frames, "interior_frame_pairs": inside_frames,
            "centered_miniatures": centered}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
