"""Selected swatches become a cyclic program of concentric band thicknesses.

The input alone supplies every color, length, ordering, and rectangle bound.
No parent generator, stored pair, or generation metadata is used by solve().
"""
from __future__ import annotations

import numpy as np


BACKGROUND = 0
SELECTOR = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "Invalid grid color")
    return a


def _key(a):
    return a.shape, tuple(int(c) for c in a.flat)


def _components(a):
    """Nonbackground 4-components, each required to be a solid uniform box."""
    seen = np.zeros(a.shape, dtype=bool)
    boxes = []
    height, width = a.shape
    for r, c in np.argwhere(a != BACKGROUND):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                xx, yy = x + dx, y + dy
                if (0 <= xx < height and 0 <= yy < width
                        and a[xx, yy] != BACKGROUND and not seen[xx, yy]):
                    seen[xx, yy] = True
                    stack.append((xx, yy))
        r0, r1 = min(x for x, _ in cells), max(x for x, _ in cells) + 1
        c0, c1 = min(y for _, y in cells), max(y for _, y in cells) + 1
        color = int(a[r, c])
        _require(len(cells) == (r1-r0)*(c1-c0), "Nonrectangular input component")
        _require(bool(np.all(a[r0:r1, c0:c1] == color)), "Mixed input component")
        boxes.append((r0, r1, c0, c1, color))
    return boxes


def _depth(box):
    r0, r1, c0, c1 = box[:4]
    return (min(r1-r0, c1-c0)-1) // 2


def _separate(a, b):
    # Half-open boxes have at least one complete background row or column between.
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _read_canonical(a):
    boxes = _components(a)
    targets = [b for b in boxes if b[1]-b[0] >= 2 and b[3]-b[2] >= 2]
    _require(1 <= len(targets) <= 5, "Expected one to five target rectangles")
    _require(all(b[4] != SELECTOR for b in targets), "Selector used as target fill")
    _require(any(_depth(b) >= 1 for b in targets), "No target interior")
    marks = np.argwhere(a == SELECTOR)
    _require(2 <= len(marks) <= 4, "Expected two to four selectors")
    col = int(marks[0, 1])
    _require(bool(np.all(marks[:, 1] == col)), "Selectors not in one column")
    mark_rows = {int(r) for r in marks[:, 0]}
    bars = []
    for box in boxes:
        if box in targets:
            continue
        r0, r1, c0, c1, color = box
        if color == SELECTOR:
            _require(r1-r0 == c1-c0 == 1, "Selector is not an isolated cell")
            continue
        _require(r1-r0 == 1 and 1 <= c1-c0 <= 3, "Malformed swatch")
        _require(c0 == col + 2, "Swatch is not to the right of the selector gap")
        bars.append((r0, c0, c1-c0, color, r0 in mark_rows))
    bars.sort()
    _require(2 <= len(bars) <= 6, "Expected two to six swatches")
    _require(all(b[0]-a_[0] >= 2 for a_, b in zip(bars, bars[1:])), "Swatches touch")
    _require(mark_rows <= {b[0] for b in bars}, "Selector without a swatch")
    _require(sum(b[4] for b in bars) == len(marks), "Bad selected swatches")
    _require(len(bars)-len(marks) <= 2, "Too many unselected swatches")
    selected = [(b[3], b[2]) for b in bars if b[4]]
    _require(len({c for c, _ in selected}) >= 2, "Selected cycle has only one color")
    bar_colors = {b[3] for b in bars}
    _require(not (bar_colors & {b[4] for b in targets}), "Fill and swatch roles overlap")
    tape_box = (bars[0][0], bars[-1][0]+1, col, max(b[1]+b[2] for b in bars))
    _require(all(_separate(tape_box, b) for b in targets), "Tape touches a target")
    _require(all(_separate(a_, b) for i, a_ in enumerate(targets) for b in targets[i+1:]),
             "Target rectangles are not separated")
    word = tuple(c for c, length in selected for _ in range(length))
    return {"bars": bars, "selected": selected, "word": word,
            "targets": sorted(targets), "tape_box": tape_box}


def _read(input_grid):
    a = _grid(input_grid)
    candidates = []
    for turns in range(4):
        try:
            info = _read_canonical(np.rot90(a, turns))
        except AssertionError:
            continue
        info["turns"] = turns
        candidates.append(info)
    _require(len(candidates) == 1, "Ambiguous or missing swatch reading direction")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Keep rectangle borders, repeat the selected length-weighted cycle inward."""
    source = _grid(input_grid)
    info = _read(source)
    a = np.rot90(source, info["turns"])
    out = a.copy()
    word = np.asarray(info["word"], dtype=int)
    for r0, r1, c0, c1, _ in info["targets"]:
        rows = np.arange(r1-r0)[:, None]
        cols = np.arange(c1-c0)[None, :]
        distances = np.minimum(np.minimum(rows, r1-r0-1-rows),
                               np.minimum(cols, c1-c0-1-cols))
        inside = distances > 0
        region = out[r0:r1, c0:c1]
        region[inside] = word[(distances[inside]-1) % len(word)]
    return np.rot90(out, -info["turns"]).copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _program(rng, profile, budget, max_length):
    """Return ordered (color, length, selected) swatches, including decoys."""
    palette = [int(c) for c in rng.permutation(COLORS)]
    a, b, decoy = palette[:3]
    if profile == "measure":
        lengths = (2, 1) if rng.integers(2) else (1, 2)
        selected = [(a, lengths[0]), (b, lengths[1])]
    elif profile == "repeat":
        selected = [(a, 1), (b, max_length), (a, 1)]
    else:
        count = int(rng.integers(2, min(4, budget)+1))
        lengths = [1] * count
        spare = int(rng.integers(0, budget-count+1))
        while spare:
            eligible = [i for i in range(count) if lengths[i] < max_length]
            if not eligible:
                break
            lengths[_pick(rng, eligible)] += 1
            spare -= 1
        colors = [a, b] + [_pick(rng, palette[:int(rng.integers(2, 4))])
                          for _ in range(count-2)]
        selected = list(zip(colors, lengths))
    tape = [(c, length, True) for c, length in selected]
    extras = int(rng.integers(1 if profile in ("measure", "repeat") else 0, 3))
    for index in range(extras):
        # The first distractor is visibly different, but later ones can be exact copies.
        if index == 0:
            extra = (decoy, int(rng.integers(1, max_length+1)), False)
        else:
            color, length = _pick(rng, selected)
            extra = (color, length, False)
        position = 1 if profile == "measure" and index == 0 else int(rng.integers(len(tape)+1))
        tape.insert(position, extra)
    return tape


def _place(rng, height, width, shapes):
    """Pack a swatch tape and free rectangles, without a fixed bank layout."""
    result = {}
    order = sorted(range(len(shapes)), key=lambda i: -shapes[i][0]*shapes[i][1])
    for index in order:
        h, w, contact = shapes[index]
        candidates = []
        for r in range(height-h+1):
            for c in range(width-w+1):
                edge_r, edge_c = r == 0 or r+h == height, c == 0 or c+w == width
                if contact == "internal" and (edge_r or edge_c):
                    continue
                if contact == "edge" and not (edge_r or edge_c):
                    continue
                if contact == "corner" and not (edge_r and edge_c):
                    continue
                box = (r, r+h, c, c+w)
                if all(_separate(box, other) for other in result.values()):
                    candidates.append(box)
        if not candidates:
            return None
        result[index] = _pick(rng, candidates)
    return result


def _make_pair(rng, tape, primary_depth, count, turns, reset_witness=False):
    used = {c for c, _, _ in tape}
    fills = [c for c in COLORS if c not in used]
    _require(bool(fills), "No rectangle fill available")
    row_offsets = [0]
    for _ in tape[1:]:
        row_offsets.append(row_offsets[-1]+int(rng.integers(2, 4)))
    tape_h, tape_w = row_offsets[-1]+1, 2+max(length for _, length, _ in tape)
    for _ in range(160):
        short = 2*primary_depth+int(rng.integers(1, 3))
        long = min(26, short+int(rng.integers(0, 6)))
        ph, pw = (short, long) if rng.integers(2) else (long, short)
        lower_h, lower_w = max(16, ph+1, tape_h), max(16, pw+1, tape_w)
        height = int(rng.integers(lower_h, 31))
        width = int(rng.integers(lower_w, 31))
        contact = _pick(rng, ("free", "free", "internal", "edge", "corner"))
        shapes = [(tape_h, tape_w, "free"), (ph, pw, contact)]
        for _ in range(count-1):
            small = (int(rng.integers(5, 7)) if reset_witness else
                     int(rng.integers(2, min(9, 2*primary_depth+2)+1)))
            big = int(rng.integers(small, min(12, small+5)+1))
            h, w = (small, big) if rng.integers(2) else (big, small)
            shapes.append((h, w, _pick(rng, ("free", "free", "internal", "edge"))))
        placed = _place(rng, height, width, shapes)
        if placed is None:
            continue
        inp = np.zeros((height, width), dtype=np.int8)
        out = inp.copy()
        tr0, _, tc0, _ = placed[0]
        for offset, (color, length, chosen) in zip(row_offsets, tape):
            inp[tr0+offset, tc0+2:tc0+2+length] = color
            if chosen:
                inp[tr0+offset, tc0] = SELECTOR
        out[:] = inp
        for index in range(1, count+1):
            r0, r1, c0, c1 = placed[index]
            fill = _pick(rng, fills)
            inp[r0:r1, c0:c1] = fill
            out[r0:r1, c0:c1] = fill
            # Construction paints whole nested bands. solve instead indexes side distances.
            depth = 1
            limit = (min(r1-r0, c1-c0)-1)//2
            while depth <= limit:
                for color, length, chosen in tape:
                    if not chosen:
                        continue
                    for _ in range(length):
                        if depth > limit:
                            break
                        out[r0+depth:r1-depth, c0+depth:c1-depth] = color
                        depth += 1
        return {"input": np.rot90(inp, -turns).copy(),
                "output": np.rot90(out, -turns).copy()}
    raise RuntimeError("Could not pack a swatch tape and rectangles")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, "num_train", 2, 10)
    m = _count(num_test, "num_test", 1, 4)
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    max_length = int(rng.integers(2, 4))
    depth_cap = max_length+3  # the repeat witness has period max_length+2, plus a restart
    orientation_count = 1 if rng.random() < 0.5 else int(rng.integers(2, min(4, n)+1))
    orientations = [int(t) for t in rng.choice(4, orientation_count, replace=False)]
    train_turns = orientations + [_pick(rng, orientations) for _ in range(n-orientation_count)]
    rng.shuffle(train_turns)
    witness_indices = [int(i) for i in rng.choice(n, 2, replace=False)]
    profiles = {witness_indices[0]: "measure", witness_indices[1]: "repeat"}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", n), ("test", m)):
        for i in range(amount):
            profile = profiles.get(i, "free") if split == "train" else "free"
            tape = _program(rng, profile, depth_cap, max_length)
            period = sum(length for _, length, chosen in tape if chosen)
            if split == "test":
                primary_depth = depth_cap+int(rng.integers(1, 3))
            elif profile == "measure":
                primary_depth = 4
            elif profile == "repeat":
                primary_depth = depth_cap
            else:
                primary_depth = int(rng.integers(max(2, period), depth_cap+1))
            count = 2 if profile == "measure" else int(rng.integers(1, 5 if split == "test" else 6))
            turns = train_turns[i] if split == "train" else _pick(rng, orientations)
            for _ in range(20):
                pair = _make_pair(rng, tape, primary_depth, count, turns, profile == "measure")
                signature = _key(pair["input"])
                if signature not in seen:
                    puzzle[split].append(pair)
                    seen.add(signature)
                    break
            else:
                raise RuntimeError("Could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference_output(inp, info, mode="correct"):
    """Explicit cell-wise check, also used for semantic ablations."""
    a = np.rot90(inp, info["turns"])
    out = a.copy()
    bars = info["bars"]
    if mode == "all_bars":
        entries = [(b[3], b[2]) for b in bars]
    else:
        entries = info["selected"]
    if mode == "deduplicate":
        entries = list(dict.fromkeys(entries))
    if mode == "unit_width":
        word = [c for c, _ in entries]
    else:
        word = [c for c, length in entries for _ in range(length)]
    cursor = 0
    for r0, r1, c0, c1, _ in info["targets"]:
        for r in range(r0, r1):
            for c in range(c0, c1):
                distance = min(r-r0, r1-1-r, c-c0, c1-1-c)
                if mode == "row_stripes":
                    distance = r-r0
                    if r in (r0, r1-1) or c in (c0, c1-1):
                        distance = 0
                if distance:
                    if mode == "hold_last":
                        position = min(distance-1, len(word)-1)
                    elif mode == "carry_cursor":
                        position = (cursor+distance-1) % len(word)
                    else:
                        position = (distance-1) % len(word)
                    out[r, c] = word[position]
        cursor += _depth((r0, r1, c0, c1))
    return np.rot90(out, -info["turns"]).copy()


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Bad split container")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "Bad example counts")
    seen, infos = set(), {"train": [], "test": []}
    witnesses = dict.fromkeys(("all_bars", "unit_width", "deduplicate", "row_stripes",
                              "hold_last", "carry_cursor"), False)
    training_lengths, training_turns = set(), set()
    max_train_depth = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Bad pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "Shape changed")
            signature = _key(inp)
            _require(signature not in seen, "Repeated input")
            seen.add(signature)
            info = _read(inp)
            infos[split].append(info)
            _require(np.array_equal(_reference_output(inp, info), out), "Incorrect band completion")
            _require(np.array_equal(solve(inp), out), "Input-only solver disagrees")
            _require(not np.array_equal(inp, out), "Identity example")
            _require(set(np.unique(inp)) == set(np.unique(out)), "Colors added or lost")
            a, b = np.rot90(inp, info["turns"]), np.rot90(out, info["turns"])
            interior = np.zeros(a.shape, dtype=bool)
            for r0, r1, c0, c1, _ in info["targets"]:
                interior[r0+1:r1-1, c0+1:c1-1] = True
            _require(np.array_equal(a[~interior], b[~interior]), "Border, swatches, or background changed")
            _require(any(_depth(box) >= len(info["word"]) for box in info["targets"]),
                     "Program not fully exposed in the output")
            if split == "train":
                training_turns.add(info["turns"])
                training_lengths.update(length for _, length in info["selected"])
                max_train_depth = max(max_train_depth, *(_depth(box) for box in info["targets"]))
                for mode in witnesses:
                    witnesses[mode] |= not np.array_equal(_reference_output(inp, info, mode), out)
            else:
                _require(info["turns"] in training_turns, "Undemonstrated reading orientation")
                _require({length for _, length in info["selected"]} <= training_lengths,
                         "Undemonstrated swatch length")
    _require(all(witnesses.values()), "Training does not distinguish a named shortcut")
    _require(any(max(_depth(b) for b in x["targets"]) > len(x["word"]) for x in infos["train"]),
             "No training wraparound")
    _require(any(len(x["targets"]) >= 2 for x in infos["train"]), "No independent rectangle reset evidence")
    for info in infos["test"]:
        _require(max(_depth(b) for b in info["targets"]) > max_train_depth,
                 "Missing test depth continuation shift")


def _puzzle_key(puzzle):
    return tuple((split, tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                              for p in puzzle[split])) for split in ("train", "test"))


def self_test():
    """Bounded checks: all count pairs, 42 seeds, ablations, and visible interventions."""
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(6000+10*n+m, n, m)
            _require((len(p["train"]), len(p["test"])) == (n, m), "Counts changed")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(1, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    _require(_puzzle_key(generate_puzzle(993136894)) == _puzzle_key(generate_puzzle(993136894)),
             "Seed is not deterministic")
    # A valid family member outside the sampler's full-program-exposure restriction:
    # four length-three swatches and a shallow rectangle truncate before most entries.
    fixture = np.zeros((16, 17), dtype=np.int8)
    for r, color in ((1, 1), (4, 2), (7, 3), (10, 4)):
        fixture[r, 0], fixture[r, 2:5] = 5, color
    fixture[2:8, 8:17] = 9
    expected = fixture.copy()
    expected[3:7, 9:16] = 1
    _require(np.array_equal(solve(fixture), expected), "Long program truncation failed")
    for turns_ in range(4):
        _require(np.array_equal(solve(np.rot90(fixture, turns_)), np.rot90(expected, turns_)),
                 "Quarter-turn equivariance failed")
    signatures, shapes, palettes, layouts = set(), set(), set(), set()
    counts, selected_counts, periods, lengths, turns, depths, centers = (set() for _ in range(7))
    contacts, decoy_counts, key_sides = set(), set(), set()
    mixed_fill = repeated = gapped = changed_marker = changed_width = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                inp, out = pair["input"], pair["output"]
                info = _read(inp)
                a = np.rot90(inp, info["turns"])
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(int(c) for c in np.unique(inp))))
                counts.add(len(info["targets"]))
                selected_counts.add(len(info["selected"]))
                periods.add(len(info["word"]))
                lengths.update(length for _, length in info["selected"])
                turns.add(info["turns"])
                decoy_counts.add(len(info["bars"])-len(info["selected"]))
                repeated += len(set(info["selected"])) < len(info["selected"])
                mixed_fill += len({b[4] for b in info["targets"]}) > 1
                gapped += any(not bar[4] for bar in info["bars"][1:-1])
                layouts.add((a.shape, tuple(info["targets"]), tuple(info["bars"])))
                tr0, tr1, tc0, tc1 = info["tape_box"]
                for label, yes in (("top", tr1 <= a.shape[0]//2),
                                   ("bottom", tr0 >= a.shape[0]//2),
                                   ("left", tc1 <= a.shape[1]//2),
                                   ("right", tc0 >= a.shape[1]//2)):
                    if yes:
                        key_sides.add(label)
                for box in info["targets"]:
                    r0, r1, c0, c1, _ = box
                    d = _depth(box)
                    depths.add(d)
                    centers.add((r1-r0-2*d, c1-c0-2*d))
                    contacts.add("edge" if r0 == 0 or c0 == 0 or r1 == a.shape[0] or c1 == a.shape[1]
                                 else "internal")
                # Remove one visible marker while retaining a well-formed multi-color cycle.
                if len(info["selected"]) == 3:
                    bar = next(b for b in reversed(info["bars"]) if b[4])
                    edited = a.copy()
                    edited[bar[0], bar[1]-2] = BACKGROUND
                    try:
                        cf = solve(np.rot90(edited, -info["turns"]))
                    except AssertionError:
                        continue
                    # Compare target interiors, avoiding the deliberately changed marker.
                    target_mask = np.zeros(a.shape, dtype=bool)
                    for r0, r1, c0, c1, _ in info["targets"]:
                        target_mask[r0+1:r1-1, c0+1:c1-1] = True
                    changed_marker += bool(np.any(np.rot90(cf, info["turns"])[target_mask]
                                                  != np.rot90(out, info["turns"])[target_mask]))
                # Lengthen a selected swatch into free reserved tape space (when possible).
                for bar in info["bars"]:
                    r, c, length, _, chosen = bar
                    if chosen and length < 3 and c+length < a.shape[1] and a[r, c+length] == 0:
                        edited = a.copy()
                        edited[r, c+length] = bar[3]
                        try:
                            cf = solve(np.rot90(edited, -info["turns"]))
                        except AssertionError:
                            continue
                        target_mask = np.zeros(a.shape, dtype=bool)
                        for r0, r1, c0, c1, _ in info["targets"]:
                            target_mask[r0+1:r1-1, c0+1:c1-1] = True
                        changed_width += bool(np.any(np.rot90(cf, info["turns"])[target_mask]
                                                     != np.rot90(out, info["turns"])[target_mask]))
                        break
    _require(len(signatures) == 42, "Repeated seed puzzles")
    _require(len(shapes) >= 60 and len(layouts) >= 200 and len(palettes) >= 20, "Insufficient variation")
    _require(counts == {1, 2, 3, 4, 5} and selected_counts == {2, 3, 4}, "Missing count variation")
    _require({2, 3, 4, 5, 6} <= periods and lengths == {1, 2, 3}, "Missing band programs")
    _require(turns == {0, 1, 2, 3} and contacts == {"edge", "internal"}, "Missing geometry")
    _require({0, 1, 2} <= decoy_counts and key_sides == {"top", "bottom", "left", "right"},
             "Fixed key placement or distractor count")
    _require({0, 1, 2, 4, 5, 6, 7, 8} <= depths, "Missing shallow or deep rectangles")
    _require((1, 1) in centers and (2, 2) in centers
             and any(min(h, w) <= 2 and max(h, w) >= 4 for h, w in centers), "Missing core shapes")
    _require(min(mixed_fill, repeated, gapped, changed_marker, changed_width) >= 10,
             "Missing informative structures or causal interventions")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal count combinations, semantic ablations)")
