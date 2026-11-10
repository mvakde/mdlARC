"""Recover a motif by folding clues from the marker-selected half of a mirror field.

The solver reads only the supplied grid. Generation builds the answers from
latent motifs, then checks them against the input-only reconstruction.
"""
from __future__ import annotations

import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _marker_candidates(a):
    """Return (marker, background, cut_axis, selected_index, marker_positions)."""
    height, width = a.shape
    candidates = []
    for color, count in zip(*np.unique(a, return_counts=True)):
        if count != 2:
            continue
        (r0, c0), (r1, c1) = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        if (c0 == c1 and c0 in (0, width - 1) and r1 - r0 == 2
                and height % 2 == 0 and r0 // (height // 2) == r1 // (height // 2)):
            bg = int(a[r0 + 1, c0])
            if bg != color:
                candidates.append((int(color), bg, 0, int(r0 >= height // 2),
                                   ((r0, c0), (r1, c1))))
        if (r0 == r1 and r0 in (0, height - 1) and c1 - c0 == 2
                and width % 2 == 0 and c0 // (width // 2) == c1 // (width // 2)):
            bg = int(a[r0, c0 + 1])
            if bg != color:
                candidates.append((int(color), bg, 1, int(c0 >= width // 2),
                                   ((r0, c0), (r1, c1))))
    return candidates


def _parse(value):
    a = _grid(value)
    if a.shape[0] % 3 or a.shape[1] % 3:
        raise ValueError("the field must have three tile rows and three tile columns")
    candidates = _marker_candidates(a)
    if len(candidates) != 1:
        raise ValueError("expected one unambiguous border marker pair")
    return a, candidates[0]


def _coordinates(shape, flip_rows=True, flip_cols=True):
    height, width = shape
    h, w = height // 3, width // 3
    rows, cols = np.indices(shape)
    rr, cc = rows % h, cols % w
    if flip_rows:
        rr = np.where(rows // h == 1, rr, h - 1 - rr)
    if flip_cols:
        cc = np.where(cols // w == 1, cc, w - 1 - cc)
    return rows, cols, rr, cc


def _fold(a, candidate, *, side=None, flip_rows=True, flip_cols=True, tile=None, strict=True):
    marker, bg, axis, chosen, _ = candidate
    if side is not None:
        chosen = side
    rows, cols, rr, cc = _coordinates(a.shape, flip_rows, flip_cols)
    selected = (rows if axis == 0 else cols) // (a.shape[axis] // 2) == chosen
    selected &= (a != bg) & (a != marker)
    h, w = a.shape[0] // 3, a.shape[1] // 3
    if tile is not None:
        selected &= (rows // h == tile[0]) & (cols // w == tile[1])
    result = np.full((h, w), bg, dtype=np.int8)
    for r, c in np.argwhere(selected):
        r0, c0, color = int(rr[r, c]), int(cc[r, c]), int(a[r, c])
        if strict and result[r0, c0] not in (bg, color):
            raise ValueError("selected clues disagree after mirror folding")
        result[r0, c0] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the instruction, ignore the other half, and fold/merge selected clues."""
    a, candidate = _parse(input_grid)
    return _fold(a, candidate)


def _different_without(a, candidate, expected, **kwargs):
    # A concrete same-size shortcut: omit the indicated mirrors and overwrite
    # collisions in raster order. The real rule never permits a conflict.
    return not np.array_equal(_fold(a, candidate, strict=False, **kwargs), expected)


def _symmetry(a):
    return (bool(np.array_equal(a, a[::-1])),
            bool(np.array_equal(a, a[:, ::-1])),
            bool(np.array_equal(a, a[::-1, ::-1])))


def _marker_line(rng, shape, axis, side):
    extent = shape[axis] // 2
    start = side * extent + int(rng.integers(extent - 2))
    border = 0 if rng.integers(2) == 0 else shape[1 - axis] - 1
    return ([(start + d, border) for d in range(3)] if axis == 0
            else [(border, start + d) for d in range(3)])


def _sample_shape(rng, axis, band):
    if band == "small":
        cut, other = int(rng.choice((4, 6))), int(rng.integers(3, 7))
    elif band == "large":
        cut, other = int(rng.choice((8, 10))), int(rng.integers(7, 11))
    else:
        cut, other = int(rng.choice((4, 6, 8, 10))), int(rng.integers(3, 11))
    return (cut, other) if axis == 0 else (other, cut)


def _sample_motif(rng, shape, bg, palette, style, witness):
    """Textures are generation choices, never extra rules for the solver."""
    h, w = shape
    yy, xx = np.indices(shape)
    for _ in range(96):
        labels = np.zeros(shape, dtype=int)
        if style == "path":
            r, c = int(rng.integers(h)), int(rng.integers(w))
            for step in range(int(rng.integers(max(h, w), h * w + 1))):
                labels[r, c] = 1 + (step // 3) % len(palette)
                dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
                r, c = min(h - 1, max(0, r + dr)), min(w - 1, max(0, c + dc))
        elif style == "patches":
            rectangles = int(rng.integers(4, 8)) if witness else len(palette)
            for index in range(rectangles):
                color = 1 + index % len(palette)
                r, c = int(rng.integers(h)), int(rng.integers(w))
                rh, cw = int(rng.integers(1, max(2, h // 2 + 1))), int(rng.integers(1, max(2, w // 2 + 1)))
                labels[r:min(h, r + rh), c:min(w, c + cw)] = color
        elif style == "bands":
            period = int(rng.integers(2, 5))
            mask = (yy if rng.integers(2) else xx) % period == int(rng.integers(period))
            labels[mask] = 1 + (yy + xx)[mask] % len(palette)
        elif style == "diagonal":
            period = int(rng.integers(2, 5))
            mask = (yy + int(rng.choice((-1, 1))) * xx) % period == 0
            labels[mask] = 1 + (yy // 2 + xx)[mask] % len(palette)
        elif style == "sparse":
            n = min(h * w, int(rng.integers(1, 5)))
            indices = rng.choice(h * w, size=n, replace=False)
            labels.flat[indices] = 1 + np.arange(n) % len(palette)
        elif style == "symmetric":
            labels = rng.integers(0, len(palette) + 1, size=shape)
            mode = int(rng.integers(4))
            for r in range(h):
                for c in range(w):
                    orbit = {(r, c)}
                    if mode in (0, 3):
                        orbit.add((h - 1 - r, c))
                    if mode in (1, 3):
                        orbit.add((r, w - 1 - c))
                    if mode in (2, 3):
                        orbit.add((h - 1 - r, w - 1 - c))
                    labels[r, c] = labels[min(orbit)]
        else:
            mask = rng.random(shape) < rng.uniform(.20, .55)
            labels[mask] = rng.integers(1, len(palette) + 1, size=int(mask.sum()))
        if witness:
            # The two essential demonstrations use every chosen payload color,
            # enough distinct clues for fragmentation, and no reflection symmetry.
            indices = np.flatnonzero(labels)
            if len(indices) < 6:
                continue
            if len(np.unique(labels[labels != 0])) < len(palette):
                chosen = rng.choice(indices, size=len(palette), replace=False)
                for i, index in enumerate(chosen):
                    labels.flat[index] = 1 + i
        if not np.any(labels):
            continue
        lookup = np.array([bg, *palette], dtype=np.int8)
        motif = lookup[labels]
        if witness and any(_symmetry(motif)):
            continue
        return motif
    raise RuntimeError("could not sample a motif")


def _scatter(rng, motif, shape, axis, side, bg, forbidden_cells, redundancy, witness):
    """Put sparse witnesses at positions in their exact reflection orbits."""
    rows, cols, rr, cc = _coordinates(shape)
    selected = (rows if axis == 0 else cols) // (shape[axis] // 2) == side
    for p in forbidden_cells:
        selected[p] = False
    active = [tuple(map(int, p)) for p in np.argwhere(motif != bg)]
    for _ in range(80):
        field = np.full(shape, bg, dtype=np.int8)
        for i, (r, c) in enumerate(active):
            locations = np.argwhere(selected & (rr == r) & (cc == c))
            if len(locations) == 0:
                raise RuntimeError("marker erased an entire reflection orbit")
            # One irreplaceable corner clue forces both mirror axes to matter.
            corners = [p for p in locations if p[0] // motif.shape[0] != 1
                       and p[1] // motif.shape[1] != 1]
            if witness and i == 0 and corners:
                chosen = np.array([corners[int(rng.integers(len(corners)))]])
            else:
                count = 1 + int(rng.binomial(len(locations) - 1, redundancy))
                chosen = locations[rng.choice(len(locations), size=count, replace=False)]
            for p in chosen:
                field[tuple(p)] = motif[r, c]
        if not witness:
            return field
        candidate = (-1, bg, axis, side, ())
        tiles = {(int(r // motif.shape[0]), int(c // motif.shape[1]))
                 for r, c in np.argwhere(field != bg)}
        if len(tiles) < 3:
            continue
        if any(np.array_equal(_fold(field, candidate, tile=(tr, tc)), motif)
               for tr in range(3) for tc in range(3)):
            continue
        if not all(_different_without(field, candidate, motif, **kw) for kw in
                   ({"flip_rows": False}, {"flip_cols": False},
                    {"flip_rows": False, "flip_cols": False})):
            continue
        return field
    raise RuntimeError("could not distribute a fragmented mirror witness")


def _make_pair(rng, axis, side, band, roles, pool, style, witness, redundancy):
    bg, marker = roles
    shape = _sample_shape(rng, axis, band)
    field_shape = (3 * shape[0], 3 * shape[1])
    line = _marker_line(rng, field_shape, axis, side)
    ncolors = 2 if witness else int(rng.integers(1, min(4, len(pool)) + 1))
    palette = [int(x) for x in rng.choice(pool, size=ncolors, replace=False)]
    motif = _sample_motif(rng, shape, bg, palette, style, witness)
    if rng.random() < .55:
        rival = motif.copy()
        edits = rng.choice(motif.size, size=max(2, motif.size // 5), replace=False)
        rival.flat[edits] = rng.choice([bg, *pool], size=len(edits))
    else:
        rival = _sample_motif(rng, shape, bg, palette, str(rng.choice(
            ("path", "patches", "bands", "diagonal", "field"))), False)
    # Competing authorities contain a literal incompatible clue in the same
    # mirror orbit; folding everything cannot quietly produce the right tile.
    anchor = tuple(np.argwhere(motif != bg)[int(rng.integers(np.count_nonzero(motif != bg)))])
    rival[anchor] = int(rng.choice([c for c in pool if c != motif[anchor]]))
    left = _scatter(rng, motif, field_shape, axis, side, bg, line, redundancy, witness)
    right = _scatter(rng, rival, field_shape, axis, 1 - side, bg, [],
                     float(rng.uniform(.05, .75)), False)
    field = left.copy()
    field[right != bg] = right[right != bg]
    field[line[0]], field[line[2]] = marker, marker
    candidates = _marker_candidates(field)
    if len(candidates) != 1 or candidates[0][:4] != (marker, bg, axis, side):
        raise ValueError("ambiguous accidental border marker")
    if np.count_nonzero(field == bg) <= field.size // 2:
        raise ValueError("background must remain visually clear")
    return {"input": field, "output": motif.copy()}


def _integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or int(value) < low or (high is not None and int(value) > high)):
        raise ValueError(f"illegal {name}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    ns = _integer(num_test, "num_test", 1, 4)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    colors = [int(x) for x in rng.permutation(10)]
    fixed_roles = bool(rng.integers(2))
    first_axis, first_side = int(rng.integers(2)), int(rng.integers(2))
    shift = str(rng.choice(("grow", "shrink", "aspect"), p=(.4, .4, .2)))
    train_band, test_band = (("small", "large") if shift == "grow" else
                             ("large", "small") if shift == "shrink" else ("mixed", "mixed"))
    initial = [(first_axis, first_side), (1 - first_axis, 1 - first_side),
               (first_axis, 1 - first_side), (1 - first_axis, first_side)]
    puzzle, used = {"train": [], "test": []}, set()
    train_shapes = set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            axis, side = initial[index] if split == "train" and index < 4 else (
                int(rng.integers(2)), int(rng.integers(2)))
            witness = split == "train" and index < 2
            style = ("path" if index == 0 else "patches") if witness else str(rng.choice(
                ("path", "patches", "bands", "diagonal", "sparse", "symmetric", "field")))
            roles = tuple(colors[:2]) if fixed_roles else tuple(int(x) for x in rng.choice(10, 2, replace=False))
            pool = [c for c in colors if c not in roles][:int(rng.integers(3, 9))]
            redundancy = .48 if witness and index == 0 else .08 if witness else float(rng.uniform(.02, .65))
            for _ in range(128):
                try:
                    pair = _make_pair(rng, axis, side,
                                      train_band if split == "train" else test_band,
                                      roles, pool, style, witness, redundancy)
                except (ValueError, RuntimeError):
                    continue
                key = _key(pair["input"])
                if key in used:
                    continue
                if split == "test" and pair["output"].shape in train_shapes:
                    continue
                break
            else:
                raise RuntimeError("could not construct a unique, informative pair")
            used.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_shapes.add(pair["output"].shape)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    a, info = _parse(pair["input"])
    out = _grid(pair["output"])
    marker, bg, axis, side, positions = info
    _require(np.array_equal(solve(a), out), "answer is not selected mirror fold")
    _require(out.shape == (a.shape[0] // 3, a.shape[1] // 3), "wrong tile size")
    _require(not np.any(out == marker), "instruction color copied")
    rival = _fold(a, info, side=1 - side)
    _require(not np.array_equal(rival, out), "marker choice is irrelevant")
    conflict = (out != bg) & (rival != bg) & (out != rival)
    _require(bool(np.any(conflict)), "discarded evidence lacks a competing color")
    _require(np.count_nonzero(a == bg) > a.size // 2, "background is not clear")
    coords = _coordinates(a.shape)
    chosen = coords[axis] // (a.shape[axis] // 2) == side
    clues = chosen & (a != bg) & (a != marker)
    tile_masks = []
    for tr in range(3):
        for tc in range(3):
            tile_masks.append(_fold(a, info, tile=(tr, tc)))
    h, w = out.shape
    tiles = {(int(r // h), int(c // w)) for r, c in np.argwhere(clues)}
    counts = np.zeros(out.shape, dtype=int)
    for r, c in np.argwhere(clues):
        counts[coords[2][r, c], coords[3][r, c]] += 1
    corner_singletons = sum(counts[coords[2][r, c], coords[3][r, c]] == 1
                            and r // h != 1 and c // w != 1
                            for r, c in np.argwhere(clues))
    return {"axis": axis, "side": side,
            "edge": int(positions[0][1 - axis] != 0),
            "roles": (bg, marker), "shape": out.shape,
            "payload_colors": len(np.unique(out[out != bg])),
            "active": int(np.count_nonzero(out != bg)),
            "clues": int(clues.sum()), "tiles": len(tiles),
            "corner_singletons": int(corner_singletons),
            "single_tile_suffices": any(np.array_equal(t, out) for t in tile_masks),
            "no_rows_wrong": _different_without(a, info, out, flip_rows=False),
            "no_cols_wrong": _different_without(a, info, out, flip_cols=False),
            "no_flips_wrong": _different_without(a, info, out, flip_rows=False, flip_cols=False),
            "symmetry": _symmetry(out)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            analyses[split].append(_analyze({"input": a, "output": out}))
    train, tests = analyses["train"], analyses["test"]
    _require({i["axis"] for i in train[:2]} == {0, 1}, "both cut axes need demonstration")
    _require({i["side"] for i in train[:2]} == {0, 1}, "both half indices need demonstration")
    for i in train[:2]:
        _require(i["payload_colors"] >= 2 and i["active"] >= 6, "weak color/fragment evidence")
        _require(not any(i["symmetry"]), "essential demonstration is symmetric")
        _require(i["tiles"] >= 3 and not i["single_tile_suffices"], "single-tile shortcut")
        _require(i["corner_singletons"] >= 1, "no indispensable doubly reflected clue")
        _require(i["no_rows_wrong"] and i["no_cols_wrong"] and i["no_flips_wrong"],
                 "a mirror-axis omission fits the demonstrations")
    _require(all(i["shape"] not in {t["shape"] for t in train} for i in tests),
             "missing held-out tile geometry")


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]))
                 for split in ("train", "test"))


def _normalized(a):
    colors = {}
    return a.shape, tuple(colors.setdefault(int(c), len(colors)) for c in a.flat)


def self_test():
    """Bounded semantic, metamorphic, count, determinism and 42-seed coverage checks."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "seed diversity")
    _require(_puzzle_key(generate_puzzle(98765, 6, 3)) ==
             _puzzle_key(generate_puzzle(98765, 6, 3)), "determinism")
    features = {"train": [], "test": []}
    shapes, patterns, role_sets = set(), set(), []
    up = down = payload_zero = 0
    for seed, puzzle in enumerate(puzzles):
        infos = {s: [_analyze(p) for p in puzzle[s]] for s in ("train", "test")}
        role_sets.append({i["roles"] for s in infos for i in infos[s]})
        train_areas = [int(np.prod(i["shape"])) for i in infos["train"]]
        test_area = int(np.prod(infos["test"][0]["shape"]))
        up += test_area > max(train_areas)
        down += test_area < min(train_areas)
        for split in ("train", "test"):
            features[split].extend(infos[split])
            for pair, info in zip(puzzle[split], infos[split]):
                a, out = pair["input"], pair["output"]
                shapes.add(out.shape)
                patterns.add(_normalized(a))
                payload_zero += info["roles"][0] != 0 and np.any(out == 0)
        # Whole-scene geometry and arbitrary palette names must commute with solving.
        a, out = puzzle["train"][0]["input"], puzzle["train"][0]["output"]
        for altered, answer in ((a.T, out.T), (a[::-1], out[::-1]),
                                (a[:, ::-1], out[:, ::-1])):
            _require(np.array_equal(solve(altered), answer), "geometric equivariance")
        palette_map = np.random.default_rng(seed + 500).permutation(10)
        _require(np.array_equal(solve(palette_map[a]), palette_map[out]), "palette equivariance")
        # Relocate one clue within its orbit: its meaning must be unchanged.
        _, candidate = _parse(a)
        marker, bg, axis, side, _ = candidate
        coords = _coordinates(a.shape)
        selected = coords[axis] // (a.shape[axis] // 2) == side
        relocated = False
        for r, c in np.argwhere(selected & (a != bg) & (a != marker)):
            targets = np.argwhere(selected & (a == bg) &
                                  (coords[2] == coords[2][r, c]) &
                                  (coords[3] == coords[3][r, c]))
            for target in targets:
                b = a.copy()
                b[tuple(target)], b[r, c] = a[r, c], bg
                if len(_marker_candidates(b)) == 1:
                    _require(np.array_equal(solve(b), out), "orbit relocation")
                    relocated = True
                    break
            if relocated:
                break
        _require(relocated, "no relocation test available")
    for split in ("train", "test"):
        f = features[split]
        _require({(i["axis"], i["side"]) for i in f} == {(0, 0), (0, 1), (1, 0), (1, 1)},
                 f"{split} retained-side coverage")
        _require({i["edge"] for i in f} == {0, 1}, f"{split} marker edges")
        _require(any(any(i["symmetry"]) for i in f), f"{split} symmetry coverage")
        _require(any(i["payload_colors"] == 1 for i in f) and
                 any(i["payload_colors"] >= 3 for i in f), f"{split} palette-size coverage")
        _require(len({i["shape"] for i in f}) >= 12, f"{split} shape coverage")
    _require(len(shapes) >= 35 and len(patterns) >= 180, "structural diversity")
    _require(up >= 8 and down >= 8, "size shifts in both directions")
    _require(sum(len(r) == 1 for r in role_sets) >= 8 and
             sum(len(r) > 1 for r in role_sets) >= 8, "role regimes")
    _require(payload_zero >= 5, "zero must sometimes be payload")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(70000 + 10 * nt + ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "explicit counts")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts {nt}/{ns}")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "tile_shapes": len(shapes), "normalized_input_structures": len(patterns),
            "size_increases": up, "size_decreases": down,
            "fixed_role_puzzles": sum(len(r) == 1 for r in role_sets),
            "relabeled_role_puzzles": sum(len(r) > 1 for r in role_sets),
            "zero_payload_pairs": int(payload_zero)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
