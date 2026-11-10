"""Repair corner stencils, then replace every entire row by its equality label.

The family fixes positive=5 and negative=0. Scene colors remain arbitrary.
No parent generator, file, random state, or demonstration is used by solve.
"""

import hashlib
import numpy as np


HOST_SEEDS = (0, 194765757, 344380827, 234233162, 114579308, 923803448,
              8967443, 261562, 201863050, 401169840, 252244126, 41731526,
              790498040, 236155932, 61997261, 778402926, 956212739,
              549291972, 88604118, 277068439, 910840348, 49846656,
              134250170, 251500895, 629433184, 198649926, 627975074,
              952232689, 309048393, 887239879, 534620460, 262492741,
              901706552, 232768977, 737853045, 826728915, 429738811,
              290149958, 946707653, 363702935, 114132448, 287619620)
_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, lo, hi=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid shape")
    _require(a.dtype.kind in "iu" and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _uniform(a):
    return np.all(a == a[:, :1], axis=1)


def _label(a):
    return np.repeat(np.where(_uniform(a), 5, 0).astype(np.int8)[:, None], a.shape[1], axis=1)


def _has_run(mask, length=4):
    run = 0
    for yes in mask:
        run = run + 1 if yes else 0
        if run >= length:
            return True
    return False


def _geometry(a):
    """Visible rails identify structure even when ornaments cover the borders."""
    h, w = a.shape
    _require(w >= 6 and w % 2 == 0, "even canvas width of at least six required")
    background = int(a[0, 0])
    _require(np.all(a[0] == background), "missing blank top margin")
    candidates = []
    empty_inside = np.all(a[:, 1:-1] == background, axis=1)
    for color in np.unique(a):
        color = int(color)
        if color == background:
            continue
        rails = (a[:, 0] == color) & (a[:, -1] == color) & empty_inside
        if _has_run(rails):
            candidates.append(color)
    _require(len(candidates) == 1, "missing or ambiguous structural rails")
    frame = candidates[0]
    borders = np.flatnonzero(np.any(a[:, 1:-1] == frame, axis=1)).tolist()
    _require(len(borders) in (2, 4), "expected one or two complete frames")
    frames = list(zip(borders[::2], borders[1::2]))
    structure = np.zeros_like(a, dtype=bool)
    corners = []
    for t, b in frames:
        _require(b - t >= 7, "frame too short")
        structure[t, :] = structure[b, :] = True
        structure[t:b + 1, 0] = structure[t:b + 1, -1] = True
        _require(np.all(a[t] != background) and np.all(a[b] != background), "broken border")
        _require(_has_run((a[t + 1:b, 0] == frame)
                          & (a[t + 1:b, -1] == frame)
                          & empty_inside[t + 1:b]), "unrecognizable frame arms")
        corners.extend(((t, 0, 0), (t, w - 1, 1), (b, 0, 2), (b, w - 1, 3)))
    _require(np.all(structure[a == frame]), "structural pixels outside frames")
    _require(np.all(a[structure] != background), "frame has erased cells")
    return background, frame, frames, corners


def _nearest(y, x, corners):
    ds = [(y - r) ** 2 + (x - c) ** 2 for r, c, _ in corners]
    _require(ds.count(min(ds)) == 1, "ambiguous corner association")
    return ds.index(min(ds))


def _read(a):
    background, frame, frames, corners = _geometry(a)
    assigned = [[] for _ in corners]
    for y, x in np.argwhere((a != background) & (a != frame)):
        y, x = int(y), int(x)
        assigned[_nearest(y, x, corners)].append((y, x))
    occurrences = []
    for i, points in enumerate(assigned):
        if not points:
            continue
        colors = {int(a[p]) for p in points}
        _require(len(colors) == 1, "one corner has multiple accents")
        y, x, q = corners[i]
        dy, dx = _DIRS[q]
        pattern = frozenset(((r - y) * dy, (c - x) * dx) for r, c in points)
        _require(all(abs(u) <= 3 and 0 <= v < a.shape[1] // 2 for u, v in pattern),
                 "clue outside its corner window")
        occurrences.append((i, colors.pop(), pattern))
    _require(len(occurrences) >= 2, "need a reference and a marked recipient")
    return {"background": background, "frame": frame, "frames": frames,
            "corners": corners, "occurrences": occurrences}


def _infer(occurrences, rotation=False):
    patterns = []
    for i, _, p in occurrences:
        patterns.append(frozenset((v, u) for u, v in p)
                        if rotation and i % 4 in (1, 2) else p)
    maximal = {p for p in patterns if all(q <= p for q in patterns)}
    _require(len(maximal) == 1, "no intact common stencil contains every clue")
    return maximal.pop()


def _paint(a, info, pattern, rotation=False, all_corners=False):
    out = a.copy()
    occurrences = list(info["occurrences"])
    if all_corners:
        marked = {i: c for i, c, _ in occurrences}
        # Countermodel: invent a mate for each lone marked corner.
        for i in range(len(info["corners"])):
            if i not in marked and (i ^ 1) in marked:
                occurrences.append((i, marked[i ^ 1], frozenset()))
    occupied = set()
    for i, color, _ in occurrences:
        y, x, q = info["corners"][i]
        dy, dx = _DIRS[q]
        local = {(v, u) for u, v in pattern} if rotation and q in (1, 2) else pattern
        for u, v in local:
            r, c = y + dy * u, x + dx * v
            _require(0 <= r < a.shape[0] and 0 <= c < a.shape[1], "stencil would be clipped")
            _require((r, c) not in occupied, "completed copies overlap")
            _require(_nearest(r, c, info["corners"]) == i, "completion belongs to another corner")
            _require(int(a[r, c]) in (info["background"], info["frame"], color),
                     "completion would overwrite another accent")
            occupied.add((r, c))
            out[r, c] = color
    return out


def _complete(value, rotation=False):
    a = _grid(value)
    info = _read(a)
    pattern = _infer(info["occurrences"], rotation)
    completed = _paint(a, info, pattern, rotation)
    info["stencil"] = pattern
    return completed, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reflect and repair each marked corner, then classify every complete row."""
    return _label(_complete(input_grid)[0])


def _stencil(rng, radius, half_width, test, training_index):
    rows = [u for u in range(-radius, radius + 1) if u]
    count = int(rng.integers(2, min(4, len(rows)) + 1)) if test else 1
    full = set(int(x) for x in rng.choice(rows, count, replace=False))
    pattern = set()
    for u in rows:
        if u in full:
            vs = range(half_width)
        else:
            n = int(rng.integers(0, half_width))
            vs = rng.choice(half_width, n, replace=False)
        pattern.update((u, int(v)) for v in vs)
    # The first witness has equal endpoints but a genuine interior gap.
    if not test and training_index == 0:
        hole_row = int(rng.choice([u for u in rows if u not in full]))
        pattern = {p for p in pattern if p[0] != hole_row}
        gap = int(rng.integers(1, half_width - 1))
        pattern.update((hole_row, v) for v in range(half_width) if v != gap)
    # Border-center cells remain structural, making anchors visible. The
    # second demonstration has no border-row ornament, exposing uniform f.
    if not (not test and training_index == 1) and rng.random() < .7:
        n = int(rng.integers(1, half_width - 1))
        pattern.update((0, int(v)) for v in rng.choice(half_width - 2, n, replace=False))
    # Ensure the sampled vertical extent is observable, without forcing a
    # particular corner/anchor cell or a retained anchor in partial copies.
    if not any(abs(u) == radius for u, _ in pattern):
        pattern.add((int(rng.choice((-radius, radius))), int(rng.integers(half_width))))
    return frozenset(pattern), sorted(full)


def _layout(rng, radius, count):
    minimum_height = 2 * radius + 6
    gap = 2 * radius + 1
    minimum_canvas = count * minimum_height + (count - 1) * gap + 2 * (radius + 1)
    h = int(rng.integers(minimum_canvas, 31))
    # One allocation, no canvases or histories retained across retries.
    parts = rng.multinomial(h - minimum_canvas, np.full(2 * count + 1, 1 / (2 * count + 1)))
    top = radius + 1 + int(parts[0])
    frames = []
    for i in range(count):
        height = minimum_height + int(parts[1 + 2 * i])
        bottom = top + height - 1
        frames.append((top, bottom))
        top = bottom + 1 + gap + int(parts[2 + 2 * i])
    return h, frames


def _make_pair(rng, roles, test, index):
    radius = int(rng.choice((1, 2, 3), p=(.5, .32, .18)))
    frame_count = 2 if radius == 1 and rng.random() < .36 else 1
    half_width = int(rng.integers(3, 9)) if rng.random() < .86 else int(rng.integers(9, 16))
    stencil, full_rows = _stencil(rng, radius, half_width, test, index)
    h, frames = _layout(rng, radius, frame_count)
    w = 2 * half_width
    background, frame = roles
    base = np.full((h, w), background, dtype=np.int8)
    corners = []
    for t, b in frames:
        base[t, :] = base[b, :] = frame
        base[t:b + 1, 0] = base[t:b + 1, -1] = frame
        corners.extend(((t, 0, 0), (t, w - 1, 1), (b, 0, 2), (b, w - 1, 3)))
    available = [c for c in range(10) if c not in roles]
    size = int(rng.integers(2 if not test and index == 0 else 1, 5))
    palette = [int(x) for x in rng.choice(available, size, replace=False)]
    witness_frame = int(rng.integers(frame_count))
    border = 2 * int(rng.integers(2))
    left, right = witness_frame * 4 + border, witness_frame * 4 + border + 1
    color = int(rng.choice(palette))
    marked = {left: color, right: color}
    other = [witness_frame * 4 + (border ^ 2), witness_frame * 4 + (border ^ 2) + 1]
    if not test and index == 0:
        marked[other[0]] = color
        marked[other[1]] = int(rng.choice([x for x in palette if x != color]))
    elif not test and index == 1:
        marked[int(rng.choice(other))] = int(rng.choice(palette))
    else:
        for i in other:
            if rng.random() < .65:
                marked[i] = int(rng.choice(palette))
    for i in range(len(corners)):
        if i // 4 != witness_frame and rng.random() < .6:
            marked[i] = int(rng.choice(palette))
    reference = int(rng.choice((left, right)))
    partner = right if reference == left else left
    lost_row = int(rng.choice(full_rows))
    inp, completed = base.copy(), base.copy()
    for i, accent in sorted(marked.items()):
        y, x, q = corners[i]
        dy, dx = _DIRS[q]
        ordered = sorted(stencil)
        if i == reference or (i != partner and rng.random() < .23):
            kept = stencil
        elif i == partner and not test and index == 0:
            # A visible one-cell repair, with the missing pixel away from ends.
            kept = stencil - {(lost_row, int(rng.integers(1, half_width - 1)))}
        elif i == partner and (test or index == 1):
            # The marked corner can lack the entire relevant row. A clue on
            # another row still requests the same full stencil.
            candidates = [p for p in ordered if p[0] != lost_row]
            rng.shuffle(candidates)
            limit = max(1, len(candidates) // (3 if test else 2))
            kept = frozenset(candidates[:int(rng.integers(1, limit + 1))])
        else:
            rng.shuffle(ordered)
            kept = frozenset(ordered[:int(rng.integers(1, len(ordered)))])
        for u, v in stencil:
            completed[y + dy * u, x + dx * v] = accent
        for u, v in kept:
            inp[y + dy * u, x + dx * v] = accent
    pair = {"input": inp, "output": _label(completed)}
    # Constructor truth and input-only reconstruction are separate paths.
    recovered, _ = _complete(inp)
    _require(np.array_equal(recovered, completed), "construction/reference disagreement")
    _require(np.any(_uniform(inp) != _uniform(completed)), "repair did not affect row decisions")
    if not test and index == 0:
        try:
            alternative = _label(_complete(inp, rotation=True)[0])
        except AssertionError:
            alternative = None
        _require(alternative is None or not np.array_equal(alternative, pair["output"]),
                 "first demonstration failed to expose the reflection convention")
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    puzzle = {"train": [], "test": []}
    seen, test_masks = set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for _ in range(64):
                try:
                    pair = _make_pair(rng, roles, split == "test", index)
                    key = _key(pair["input"])
                    mask_key = tuple(bool(x) for x in (pair["output"][:, 0] == 5))
                    _require(key not in seen, "duplicate input")
                    _require(split != "test" or mask_key not in test_masks, "repeated test mask")
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded construction attempts exhausted")
            seen.add(key)
            if split == "test":
                test_masks.add(mask_key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _full_rows(pattern, half_width):
    return {u for u, _ in pattern
            if all((u, v) in pattern for v in range(half_width))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "invalid split type")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid example counts")
    seen, roles, test_masks = set(), set(), set()
    endpoint_witness = color_witness = unmarked_witness = False
    frame_positive = background_positive = False
    rotation_fits_training = True
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, output = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == output.shape, "output changed shape")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            completed, info = _complete(a)
            pattern = info["stencil"]
            roles.add((info["background"], info["frame"]))
            _require(np.array_equal(output, _label(completed)), "incorrect complete-row labels")
            _require(np.any(completed != a), "no actual completion")
            _require(np.any(_uniform(completed) != _uniform(a)), "parent B is unnecessary")
            _require(not np.array_equal(output, completed), "parent A is unnecessary")
            _require(np.any(output == 0) and np.any(output == 5), "missing a row class")
            k = a.shape[1] // 2
            _require(all(v < k - 2 for u, v in pattern if u == 0), "border-center guard violated")
            full = _full_rows(pattern, k)
            correct_profile = (len(full) == 1 if split == "train" else 2 <= len(full) <= 4)
            _require(0 not in full and correct_profile, "train/test profile shift absent")
            radius = max(abs(u) for u, _ in pattern)
            _require(1 <= radius <= 3, "invalid stencil radius")
            for t, b in info["frames"]:
                _require(b - t >= 2 * radius + 5, "corner windows approach too closely")
            borders = [x for frame in info["frames"] for x in frame]
            _require(borders[0] > radius and a.shape[0] - 1 - borders[-1] > radius,
                     "clipped ornament or missing blank margin")
            for (_, b), (t, _) in zip(info["frames"], info["frames"][1:]):
                _require(t - b > 2 * radius, "neighboring corner windows overlap")
            if split == "train":
                mask = _uniform(completed)
                endpoint_witness |= bool(np.any((completed[:, 0] == completed[:, -1]) & ~mask))
                frame_positive |= bool(np.any(mask & (completed[:, 0] == info["frame"])))
                background_positive |= bool(np.any(mask & (completed[:, 0] == info["background"])))
                marked = {i: c for i, c, _ in info["occurrences"]}
                for i, c in marked.items():
                    if i % 2 == 0 and i + 1 in marked and c != marked[i + 1]:
                        color_witness = True
                filled = _paint(a, info, pattern, all_corners=True)
                unmarked_witness |= not np.array_equal(_label(filled), output)
                if rotation_fits_training:
                    try:
                        alternative = _label(_complete(a, rotation=True)[0])
                        rotation_fits_training = np.array_equal(alternative, output)
                    except AssertionError:
                        rotation_fits_training = False
            else:
                mask_key = tuple(bool(x) for x in (output[:, 0] == 5))
                _require(mask_key not in test_masks, "duplicate test masks")
                test_masks.add(mask_key)
                if rotation_fits_training:
                    try:
                        alternative = _label(_complete(a, rotation=True)[0])
                    except AssertionError:
                        alternative = output
                    _require(np.array_equal(alternative, output), "untaught corner convention")
    _require(len(roles) == 1, "scene conventions changed within puzzle")
    _require(endpoint_witness and color_witness and unmarked_witness,
             "demonstrations lack a whole-row or marked-only contrast")
    _require(frame_positive and background_positive, "literal structural colors not abstracted")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                a = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming checks: 42 host seeds, count contracts, countermodels."""
    shapes, stencils, masks, roles, frame_counts, radii, marks, signatures = (set() for _ in range(8))
    retained_sizes, full_counts = set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                before = a.copy()
                completed, info = _complete(a)
                prediction = solve(a)
                _require(np.array_equal(prediction, pair["output"]), "solve disagrees")
                _require(np.array_equal(a, before), "solve mutated its input")
                shapes.add(a.shape)
                p = info["stencil"]
                stencils.add(tuple(sorted(p)))
                masks.add(tuple(bool(x) for x in _uniform(completed)))
                roles.add((info["background"], info["frame"]))
                frame_counts.add(len(info["frames"]))
                radii.add(max(abs(u) for u, _ in p))
                marks.add(len(info["occurrences"]))
                retained_sizes.update(len(q) for _, _, q in info["occurrences"] if q != p)
                full_counts.add(sum(q == p for _, _, q in info["occurrences"]))
    _require(len(signatures) == 42 and len(stencils) >= 100, "insufficient structural variation")
    _require(len(shapes) >= 50 and len(masks) >= 100 and len(roles) >= 20, "insufficient scene variation")
    _require(frame_counts == {1, 2} and radii == {1, 2, 3}, "geometry ranges not exercised")
    _require(len(marks) >= 4 and len(retained_sizes) >= 10 and max(full_counts) >= 2,
             "marking/damage/reference variation missing")
    _require(max(w for _, w in shapes) == 30 and max(h for h, _ in shapes) == 30,
             "large legal grids not exercised")
    _require(_fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0)), "not reproducible")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(97 + nt * 10 + nq, nt, nq)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "wrong explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Regression checks that the validator reads labels rather than trusting
    # generator provenance, and accepts the contract's nested-list grids.
    puzzle = generate_puzzle(0, 2, 1)
    lists = {s: [{k: v.tolist() for k, v in pair.items()} for pair in puzzle[s]]
             for s in ("train", "test")}
    validate_generated_puzzle(lists)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]]
           for s in ("train", "test")}
    bad["test"][0]["output"][0, 0] = 7
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, streaming; no puzzle files written)")
