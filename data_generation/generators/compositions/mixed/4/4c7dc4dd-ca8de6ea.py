"""A locally calibrated, reversible clockwise ring-interleaving analogy.

Only NumPy and the standard library are needed. No files are read or written.
The complete pair supplies the phase and the expansion blueprint; panel order
has no meaning. See task_spec.md for the complete family contract.
"""
from collections import deque
import hashlib
import json
from numbers import Integral

import numpy as np


# Clockwise from upper left. Inner points follow their own outer quadrant.
_PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 2),
              (2, 2), (2, 1), (2, 0), (1, 0))
_QUADRANTS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must be 2D, with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and
             np.all((a >= 0) & (a <= 9)), "colors must be integers 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _outline(a, box, color):
    r, c, h, w = box[:4]
    a[r, c:c+w] = a[r+h-1, c:c+w] = color
    a[r:r+h, c] = a[r:r+h, c+w-1] = color


def _inside(a, b):
    r, c, h, w = a[:4]
    y, x, bh, bw = b[:4]
    return y < r and x < c and r+h < y+bh and c+w < x+bw


def _outlines(a):
    """Recover isolated hollow rectangular components, regardless of color."""
    h, w = a.shape
    visited = np.zeros(a.shape, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(a[r, c])
            todo = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while todo:
                y, x = todo.popleft()
                cells.append((y, x))
                for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                    if (0 <= yy < h and 0 <= xx < w and not visited[yy, xx]
                            and int(a[yy, xx]) == color):
                        visited[yy, xx] = True
                        todo.append((yy, xx))
            if len(cells) < 16:
                continue
            ys, xs = zip(*cells)
            top, left, bottom, right = min(ys), min(xs), max(ys), max(xs)
            bh, bw = bottom-top+1, right-left+1
            if (bh >= 5 and bw >= 5 and len(cells) == 2*(bh+bw)-4
                    and all(y in (top, bottom) or x in (left, right)
                            for y, x in cells)):
                result.append((top, left, bh, bw, color))
    return result


def _ring_data(a, neutral):
    """Find the nine points and their geometry from visible pixels only."""
    coords = set(map(tuple, np.argwhere(a != neutral)))
    _require(len(coords) == 9, "expanded panel must contain nine visible points")
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    _require(len(rows) == len(cols) == 5, "rings need five row/column levels")
    cy, cx = rows[2], cols[2]
    oy, ox = cy-rows[0], cx-cols[0]
    iy, ix = cy-rows[1], cx-cols[1]
    _require(0 < iy < oy and 0 < ix < ox, "strictly nested rings required")
    _require(rows == [cy-oy, cy-iy, cy, cy+iy, cy+oy] and
             cols == [cx-ox, cx-ix, cx, cx+ix, cx+ox],
             "both rings must have a common center")
    outer = tuple((cy+dy*oy, cx+dx*ox) for dy, dx in _QUADRANTS)
    inner = tuple((cy+dy*iy, cx+dx*ix) for dy, dx in _QUADRANTS)
    positions = outer + inner + ((cy, cx),)
    _require(coords == set(positions), "unexpected point geometry")
    values = tuple(int(a[p]) for p in positions)
    return positions, values, (oy, ox, iy, ix, cy-oy,
                              a.shape[0]-1-cy-oy, cx-ox, a.shape[1]-1-cx-ox)


def _pack(values, phase, block=False):
    a = np.empty((3, 3), dtype=np.int8)
    sequence = values[:8] if block else tuple(
        values[j] for i in range(4) for j in (i, 4+i))
    for p, color in zip(_PERIMETER, sequence):
        a[p] = color
    a[1, 1] = values[8]
    return np.rot90(a, -phase).copy()


def _expand(compact, phase, shape, positions, neutral, block=False):
    a = np.rot90(compact, phase)
    sequence = [int(a[p]) for p in _PERIMETER]
    values = (sequence if block else sequence[::2] + sequence[1::2]) + [int(a[1, 1])]
    out = np.full(shape, neutral, dtype=np.int8)
    for p, color in zip(positions, values):
        out[p] = color
    return out


def _scene(input_grid):
    a = _grid(input_grid)
    outlines = _outlines(a)
    candidates = []
    for color in sorted({b[4] for b in outlines}):
        panels = sorted(b for b in outlines if b[4] == color)
        if len(panels) != 4:
            continue
        groups = [b for b in outlines if b[4] != color and
                  sum(_inside(p, b) for p in panels) == 2]
        if len(groups) != 2:
            continue
        membership = [[p for p in panels if _inside(p, b)] for b in groups]
        if len(set(membership[0] + membership[1])) != 4:
            continue
        crops = [[a[r+1:r+h-1, c+1:c+w-1].copy() for r, c, h, w, _ in ps]
                 for ps in membership]
        if any(sorted(p.shape for p in group)[0] != (3, 3) for group in crops):
            continue
        blanks = [(g, i) for g in range(2) for i in range(2)
                  if np.all(crops[g][i] == crops[g][i][0, 0])]
        if len(blanks) != 1:
            continue
        target_group, blank_slot = blanks[0]
        blank = crops[target_group][blank_slot]
        neutral = int(blank[0, 0])
        reference = crops[1-target_group]
        rings = [p for p in reference if min(p.shape) >= 5]
        compact = [p for p in reference if p.shape == (3, 3)]
        if len(rings) != len(compact) or len(rings) != 1:
            continue
        occupied = crops[target_group][1-blank_slot]
        if sorted(p.shape for p in crops[target_group]) != sorted(p.shape for p in reference):
            continue
        if neutral == color:
            continue
        positions, values, geometry = _ring_data(rings[0], neutral)
        _require(np.all(compact[0] != neutral), "compact reference has empty cells")
        phases = [q for q in range(4) if np.array_equal(_pack(values, q), compact[0])]
        _require(len(phases) == 1, "reference must specify one clockwise phase")
        if occupied.shape == (3, 3):
            _require(np.all(occupied != neutral), "occupied compact must be full")
            forward = False
        else:
            target_positions, _, _ = _ring_data(occupied, neutral)
            _require(target_positions == positions,
                     "expanded panels must share the reference blueprint")
            forward = True
        candidates.append(dict(neutral=neutral, frame=color, panels=membership,
                               groups=groups, reference_group=1-target_group,
                               reference=reference, ring=rings[0], compact=compact[0],
                               occupied=occupied, blank=blank, blank_slot=blank_slot,
                               positions=positions, geometry=geometry,
                               phase=phases[0], forward=forward))
    _require(len(candidates) == 1, "expected one unambiguous two-pair analogy")
    return candidates[0]


def _answer(scene, phase=None, block=False):
    q = scene["phase"] if phase is None else phase
    if scene["forward"]:
        _, values, _ = _ring_data(scene["occupied"], scene["neutral"])
        return _pack(values, q, block)
    return _expand(scene["occupied"], q, scene["ring"].shape,
                   scene["positions"], scene["neutral"], block)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the missing companion using only this input's evidence."""
    return _answer(_scene(input_grid))


def _geometry(rng, stage):
    if stage == 0:
        return (2, 2, 1, 1, 0, 0, 0, 0)
    if stage == 1:
        # A small rectangle, offset center, and changed inter-ring spacing.
        return (3, 2, int(rng.integers(1, 3)), 1, 0, 1, 1, 0)
    high = 4 if stage == 2 else 5
    oy, ox = map(int, rng.integers(2 if stage == 2 else 3, high, size=2))
    iy, ix = int(rng.integers(1, oy)), int(rng.integers(1, ox))
    margins = list(map(int, rng.integers(0, 2, size=4)))
    # Two same-orientation enclosures must fit one axis in 30 cells.
    if 2*oy+1+sum(margins[:2]) > 9 and 2*ox+1+sum(margins[2:]) > 9:
        margins[:2] = [0, 0]
    return (oy, ox, iy, ix, *margins)


def _ring(geometry, values, neutral):
    oy, ox, iy, ix, top, bottom, left, right = geometry
    cy, cx = top+oy, left+ox
    a = np.full((2*oy+1+top+bottom, 2*ox+1+left+right), neutral, dtype=np.int8)
    positions = [(cy+dy*oy, cx+dx*ox) for dy, dx in _QUADRANTS]
    positions += [(cy+dy*iy, cx+dx*ix) for dy, dx in _QUADRANTS] + [(cy, cx)]
    for p, color in zip(positions, values):
        a[p] = color
    return a


def _target_values(rng, palette, mode):
    p = list(map(int, rng.permutation(palette)))
    if mode == 0:
        values = p + [p[int(rng.integers(8))]]
    elif mode == 1:
        a, b, c, d, z = p[:5]
        values = [a, b, a, b, c, d, c, d, z]
    elif mode == 2:
        values = p[:4] + [p[0], p[4], p[1], p[5], p[0]]
    else:
        # Distinct outer quadrants keep arbitrary repeats readable.
        values = p[:4] + list(map(int, rng.choice(p[:6], 5)))
    _require(not np.array_equal(_pack(values, 0), _pack(values, 0, True)),
             "target does not distinguish alternating and block order")
    return tuple(values)


def _layout(rng, ring_shape):
    """Enumerate fitting group orientations, then randomize gaps and offsets."""
    rh, rw = ring_shape
    options = []
    for orientations in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for arrangement in (0, 1):
            shapes = [(rh+4, rw+10) if o == 0 else (rh+10, rw+4)
                      for o in orientations]
            height = sum(s[0] for s in shapes)+1 if arrangement == 0 else max(s[0] for s in shapes)
            width = max(s[1] for s in shapes) if arrangement == 0 else sum(s[1] for s in shapes)+1
            if height <= 30 and width <= 30:
                options.append((orientations, arrangement))
    _require(bool(options), "ring panels cannot fit in 30x30")
    orientations, arrangement = options[int(rng.integers(len(options)))]
    gaps = list(map(int, rng.integers(1, 4, size=2)))
    extras = [list(map(int, rng.integers(0, 3, size=2))) for _ in range(2)]

    def sizes():
        return [(rh+4+ex[0], rw+9+gap+ex[1]) if o == 0 else
                (rh+9+gap+ex[0], rw+4+ex[1])
                for o, gap, ex in zip(orientations, gaps, extras)]

    def bounds(shapes):
        return ((sum(s[0] for s in shapes)+1, max(s[1] for s in shapes))
                if arrangement == 0 else
                (max(s[0] for s in shapes), sum(s[1] for s in shapes)+1))

    shapes = sizes()
    if max(bounds(shapes)) > 30:
        gaps, extras = [1, 1], [[0, 0], [0, 0]]
        shapes = sizes()
    bh, bw = bounds(shapes)
    h, w = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    boxes = []
    if arrangement == 0:
        spare = h-sum(s[0] for s in shapes)-1
        allocation = rng.multinomial(spare, [1/3]*3)
        top = int(allocation[0])
        for i, (gh, gw) in enumerate(shapes):
            boxes.append((top, int(rng.integers(w-gw+1)), gh, gw))
            top += gh+1+int(allocation[1])
    else:
        spare = w-sum(s[1] for s in shapes)-1
        allocation = rng.multinomial(spare, [1/3]*3)
        left = int(allocation[0])
        for i, (gh, gw) in enumerate(shapes):
            boxes.append((int(rng.integers(h-gh+1)), left, gh, gw))
            left += gw+1+int(allocation[1])
    return (h, w), boxes, orientations, gaps


def _render(rng, geometry, ref_values, target_values, phase, forward,
            neutral, frame, group_colors):
    ref_ring = _ring(geometry, ref_values, neutral)
    # Labels are built directly from the nine logical tokens, without solve.
    ref_compact = _pack(ref_values, phase)
    target_ring = _ring(geometry, target_values, neutral)
    target_compact = _pack(target_values, phase)
    output = target_compact if forward else target_ring
    target = ([target_ring, np.full((3, 3), neutral, dtype=np.int8)] if forward
              else [np.full(ref_ring.shape, neutral, dtype=np.int8), target_compact])
    reference = [ref_ring, ref_compact]
    shape, boxes, orientations, gaps = _layout(rng, ref_ring.shape)
    contents = [reference, target] if rng.integers(2) else [target, reference]
    a = np.full(shape, neutral, dtype=np.int8)
    for box, orientation, gap, group, group_color in zip(
            boxes, orientations, gaps, contents, group_colors):
        r, c, gh, gw = box
        _outline(a, box, group_color)
        if rng.integers(2):
            group = group[::-1]
        sizes = [(p.shape[0]+2, p.shape[1]+2) for p in group]
        along_size = sum(s[1] if orientation == 0 else s[0] for s in sizes)+gap
        span = gw if orientation == 0 else gh
        along = int(rng.integers(1, span-along_size))
        for interior, (ph, pw) in zip(group, sizes):
            if orientation == 0:
                y, x = r+int(rng.integers(1, gh-ph)), c+along
                along += pw+gap
            else:
                y, x = r+along, c+int(rng.integers(1, gw-pw))
                along += ph+gap
            _outline(a, (y, x, ph, pw), frame)
            a[y+1:y+ph-1, x+1:x+pw-1] = interior
    return {"input": a, "output": output.copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                    (num_test, 1, 4, "num_test")):
        _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, Integral) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_geometries = set()
    first_phase = int(rng.integers(4))
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            stage = min(index, 2) if split == "train" else 3
            # Two demonstrations suffice even at the smallest legal count.
            phase = (first_phase if index == 0 else (first_phase+1) % 4) if split == "train" and index < 2 else int(rng.integers(4))
            forward = (index == 0) if split == "train" and index < 2 else bool(rng.integers(2))
            for attempt in range(32):
                geometry = _geometry(rng, stage)
                if split == "test" and geometry in training_geometries:
                    continue
                colors = list(map(int, rng.permutation(10)))
                neutral, frame = colors[:2]
                palette = colors[2:]
                outer_inner = list(map(int, rng.permutation(palette)))
                # Eight distinct reference perimeter tokens identify every slot.
                ref_values = tuple(outer_inner + [int(rng.choice(palette))])
                mode = 0 if split == "train" and index == 0 else int(rng.integers(1, 4))
                target_values = _target_values(rng, palette, mode)
                pair = _render(rng, geometry, ref_values, target_values, phase,
                               forward, neutral, frame, palette[:2])
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-geometry sampling exhausted")
            seen.add(_key(pair["input"]))
            if split == "train":
                training_geometries.add(geometry)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "expected exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            scene = _scene(inp)
            # Reconstruct the carrier from the detected frames and interiors.
            # This rejects unrelated noise that a permissive cropper could skip.
            carrier = np.full(inp.shape, scene["neutral"], dtype=np.int8)
            first, second = scene["groups"]
            r, c, h, w = first[:4]
            y, x, bh, bw = second[:4]
            _require(r+h <= y or y+bh <= r or c+w <= x or x+bw <= c,
                     "group enclosures must not overlap")
            for group, panels in zip(scene["groups"], scene["panels"]):
                _outline(carrier, group, group[4])
                for panel in panels:
                    _outline(carrier, panel, scene["frame"])
                    r, c, h, w = panel[:4]
                    crop = inp[r+1:r+h-1, c+1:c+w-1]
                    _require(not np.any(crop == scene["frame"]),
                             "panel frame color cannot be a point or neutral")
                    carrier[r+1:r+h-1, c+1:c+w-1] = crop
            _require(np.array_equal(carrier, inp), "unexpected carrier noise")
            expected = _answer(scene)
            _require(np.array_equal(expected, out), "incorrect missing companion")
            _require(out.shape == scene["blank"].shape, "blank dimensions lost")
            _require(not np.array_equal(_answer(scene, block=True), out),
                     "target fails to expose clockwise alternation")
            _require(not np.array_equal(_answer(scene, phase=(scene["phase"]+1) % 4), out),
                     "target fails to expose local phase")
            # Preserve all nine individual tokens, including repeated colors.
            if scene["forward"]:
                _, values, _ = _ring_data(scene["occupied"], scene["neutral"])
                restored = _expand(out, scene["phase"], scene["ring"].shape,
                                   scene["positions"], scene["neutral"])
                _require(np.array_equal(restored, scene["occupied"]), "round trip lost a point")
            else:
                _, values, _ = _ring_data(out, scene["neutral"])
                _require(np.array_equal(_pack(values, scene["phase"]), scene["occupied"]),
                         "inverse round trip lost a point")
            _require(sorted(values) == sorted(int(c) for c in
                     (out.flat if scene["forward"] else scene["occupied"].flat)),
                     "token multiset changed")
            features[split].append(scene)
    train = features["train"]
    _require({s["forward"] for s in train} == {False, True}, "train must teach both directions")
    _require(len({s["phase"] for s in train}) >= 2, "train must teach reference dependence")
    _require(len(set(_ring_data(train[0]["ring"], train[0]["neutral"])[1][:8])) == 8,
             "first reference must expose all perimeter correspondences")
    geometries = {s["geometry"] for s in train}
    _require(all(s["geometry"] not in geometries for s in features["test"]),
             "tests must recombine expansion geometry")


def _counterfactual_reference(pair):
    """Change only the demonstrated phase; keep the occupied target fixed."""
    scene = _scene(pair["input"])
    changed = np.asarray(pair["input"]).copy()
    panel = next(p for p in scene["panels"][scene["reference_group"]] if p[2:4] == (5, 5))
    r, c = panel[:2]
    changed[r+1:r+4, c+1:c+4] = np.rot90(scene["compact"], -1)
    answer = solve(changed)
    _require(not np.array_equal(answer, pair["output"]),
             "changing the reference did not change the answer")
    _require(np.array_equal(_scene(changed)["occupied"], scene["occupied"]),
             "counterfactual altered the target")
    # Changing the target center must affect only the reconstructed center.
    center_changed = np.asarray(pair["input"]).copy()
    target_group = scene["panels"][1-scene["reference_group"]]
    occupied_panel = next(p for i, p in enumerate(target_group) if i != scene["blank_slot"])
    r, c = occupied_panel[:2]
    local_center = scene["positions"][-1] if scene["forward"] else (1, 1)
    old = int(scene["occupied"][local_center])
    new = next(v for v in range(10) if v not in (old, scene["neutral"], scene["frame"]))
    center_changed[r+1+local_center[0], c+1+local_center[1]] = new
    expected = np.asarray(pair["output"]).copy()
    dest = (1, 1) if scene["forward"] else scene["positions"][-1]
    expected[dest] = new
    _require(np.array_equal(solve(center_changed), expected), "center is not copied separately")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Stream 42 seeds and bounded diagnostics; retain no puzzle payloads."""
    shapes, geometry, phases, layouts, neutrals, token_modes, signatures = (set() for _ in range(7))
    directions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(hashlib.sha256(puzzle["train"][0]["input"].tobytes()).digest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene = _scene(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                shapes.add(pair["input"].shape)
                geometry.add(scene["geometry"])
                phases.add(scene["phase"])
                directions.add(scene["forward"])
                layouts.add(tuple(tuple(b[:4]) for b in scene["groups"]))
                neutrals.add(scene["neutral"])
                values = (_ring_data(scene["occupied"], scene["neutral"])[1]
                          if scene["forward"] else tuple(map(int, scene["occupied"].flat)))
                token_modes.add(len(set(values)))
        for pair in puzzle["train"][:2]:
            _counterfactual_reference(pair)
    _require(len(signatures) == 42 and len(shapes) >= 30 and len(geometry) >= 60,
             "insufficient structural diversity")
    _require(phases == set(range(4)) and directions == {False, True} and
             len(neutrals) == 10 and len(layouts) >= 150 and len(token_modes) >= 4,
             "insufficient rule, layout or palette coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(700+10*nt+nq, nt, nq))
    _require(_puzzle_key(generate_puzzle(123, 7, 3)) ==
             _puzzle_key(generate_puzzle(123, 7, 3)), "seed is not deterministic")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0, 2, 1)
    pair = puzzle["test"][0]
    pair["output"] = pair["output"].copy()
    pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged label accepted")
    return {"seeds": 42, "pairs": 210, "distinct_first_inputs": len(signatures),
            "input_shapes": len(shapes), "ring_geometries": len(geometry),
            "enclosure_layouts": len(layouts), "phases": sorted(phases),
            "background_colors": sorted(neutrals), "target_color_counts": sorted(token_modes),
            "reference_counterfactuals": 84, "center_counterfactuals": 84,
            "all_legal_count_combinations": 30, "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
