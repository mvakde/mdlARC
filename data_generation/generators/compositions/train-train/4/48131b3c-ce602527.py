"""Negative wallpaper seen through a magnifying aperture.

The reference swatches, the aperture bitmap, and all color roles are read from
the input. No generation metadata is used by solve().
"""

import copy
import json
import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer 1..30 grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _box(a, color):
    rr, cc = np.where(a == color)
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def _gap(a, b):
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _parse(value):
    """Identify the dominant background, unique frame, aperture, and swatches."""
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    if not 5 <= len(colors) <= 7 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("need 2..4 swatches and a unique dominant background")
    bg = int(colors[np.argmax(counts)])
    frames = []
    for color in colors:
        color = int(color)
        if color == bg:
            continue
        r0, r1, c0, c1 = _box(a, color)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        if not (np.all(sub[0] == color) and np.all(sub[-1] == color)
                and np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)):
            continue
        inner = sub[1:-1, 1:-1]
        ic = set(map(int, np.unique(inner)))
        if color in ic or bg not in ic or len(ic) != 2:
            continue
        signal = next(iter(ic - {bg}))
        signal_box = _box(a, signal)
        if not (r0 < signal_box[0] <= signal_box[1] < r1
                and c0 < signal_box[2] <= signal_box[3] < c1):
            continue
        frames.append((color, signal, (r0, r1, c0, c1), inner.copy()))
    if len(frames) != 1:
        raise ValueError("need one complete rectangular aperture frame")
    frame, signal, fb, aperture = frames[0]
    swatches = []
    for color in colors:
        color = int(color)
        if color in (bg, frame, signal):
            continue
        box = _box(a, color)
        r0, r1, c0, c1 = box
        crop = a[r0:r1 + 1, c0:c1 + 1]
        mask = crop == color
        if (not _gap(box, fb) or not all(d <= 15 for d in crop.shape)
                or not np.all((crop == bg) | mask) or mask.all()):
            raise ValueError("invalid or overlapping native swatch")
        if any(not _gap(box, s["box"]) for s in swatches):
            raise ValueError("swatches need a background gap")
        swatches.append({"color": color, "box": box, "mask": mask.copy()})
    if not 2 <= len(swatches) <= 4:
        raise ValueError("need 2..4 swatches")
    return {"background": bg, "frame": frame, "signal": signal,
            "frame_box": fb, "aperture": aperture == signal, "swatches": swatches}


def _matches(scene, invert=True):
    """Jointly match candidate, cyclic phase, inversion, scaling and clipping."""
    observed = scene["aperture"]
    vh, vw = observed.shape
    matches = []
    for swatch in scene["swatches"]:
        mask = swatch["mask"]
        h, w = mask.shape
        # The window is a prefix of the enlarged, four-tile output rectangle.
        if vh > 4 * h or vw > 4 * w:
            continue
        ri, ci = np.arange(vh) // 2, np.arange(vw) // 2
        for dr in range(h):
            for dc in range(w):
                prediction = mask[np.ix_((ri + dr) % h, (ci + dc) % w)]
                if invert:
                    prediction = ~prediction
                if not np.array_equal(prediction, observed):
                    continue
                native = mask[np.ix_((np.arange(2 * h) + dr) % h,
                                     (np.arange(2 * w) + dc) % w)]
                if invert:
                    native = ~native
                output = np.where(native, swatch["color"], scene["background"]).astype(np.int8)
                matches.append({"swatch": swatch, "shift": (dr, dc), "output": output})
    return matches


def _interpret(value):
    scene = _parse(value)
    matches = _matches(scene)
    # Periodic/symmetric swatches may have several equivalent phases. It is the
    # full colored output, including dimensions, that must be uniquely implied.
    answers = {_key(m["output"]) for m in matches}
    if len(answers) != 1:
        raise ValueError("aperture does not imply one complete colored wallpaper")
    return scene, matches[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the uniquely identified aperture-aligned native wallpaper."""
    return _interpret(input_grid)[1]["output"].copy()


def _tight(mask):
    return (mask.any() and not mask.all() and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _cyclic_equal(a, b):
    return a.shape == b.shape and any(
        np.array_equal(a, np.roll(b, (r, c), (0, 1)))
        for r in range(a.shape[0]) for c in range(a.shape[1]))


def _orientation_rich(mask):
    variants = [mask[::-1], mask[:, ::-1], mask[::-1, ::-1]]
    if mask.shape[0] == mask.shape[1]:
        variants += [np.rot90(mask), np.rot90(mask, 3), mask.T, mask.T[::-1, ::-1]]
    return all(not _cyclic_equal(mask, other) for other in variants)


def _mask(rng, shape, style):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(100):
        if style == "cross":
            a = (rr == int(rng.integers(h))) | (cc == int(rng.integers(w)))
        elif style == "sparse":
            a = rng.random(shape) < .22
        elif style == "dense":
            a = rng.random(shape) < .8
        elif style == "diagonal":
            period = int(rng.integers(2, 5))
            a = (rr + int(rng.choice((-1, 1))) * cc) % period == 0
        elif style == "repeat":
            a = np.tile(rng.random(w) < .5, (h, 1))
            a[int(rng.integers(h))] = rng.random(w) < .5
        elif style == "symmetric":
            a = rng.random(shape) < .35
            a |= a[:, ::-1]
        else:
            a = rng.random(shape) < float(rng.uniform(.3, .7))
        if _tight(a):
            return a
    # Corner anchors retain sparse and disconnected vocabulary without cropping.
    a = rng.random(shape) < .4
    a[0, 0] = a[-1, -1] = True
    a[h // 2, w // 2] = False
    return a


def _layout(rng, masks, aperture, colors, selected_rank):
    """Place a frame and a swatch band on any of four sides, with free offsets."""
    bg, frame, signal, *inks = colors
    # The list is a physical reading order within a band, independent of ink.
    if selected_rank:
        masks = masks[1:] + masks[:1]
        inks = inks[1:] + inks[:1]
    axis = int(rng.integers(2))
    frame_first = bool(rng.integers(2))
    sizes = [m.shape for m in masks]
    fshape = (aperture.shape[0] + 2, aperture.shape[1] + 2)
    cross = 1 - axis
    small_gap = 1 if sum(s[cross] for s in sizes) + 2 * (len(sizes) - 1) > 28 else int(rng.integers(1, 3))
    band_cross = sum(s[cross] for s in sizes) + small_gap * (len(sizes) - 1)
    band_axis = max(s[axis] for s in sizes)
    separation = int(rng.integers(1, 3))
    base_axis = fshape[axis] + band_axis + separation + 2
    base_cross = max(fshape[cross], band_cross) + 2
    if base_axis > 30 or base_cross > 30:
        return None
    extent = [0, 0]
    extent[axis] = int(rng.integers(base_axis, min(30, base_axis + 4) + 1))
    extent[cross] = int(rng.integers(base_cross, min(30, base_cross + 4) + 1))
    start = int(rng.integers(1, extent[axis] - (base_axis - 2)))
    if frame_first:
        fs, bs = start, start + fshape[axis] + separation
    else:
        bs, fs = start, start + band_axis + separation
    fp = [0, 0]
    fp[axis] = fs
    fp[cross] = int(rng.integers(1, extent[cross] - fshape[cross]))
    grid = np.full(tuple(extent), bg, dtype=np.int8)
    r, c = fp
    grid[r:r + fshape[0], c:c + fshape[1]] = frame
    grid[r + 1:r + fshape[0] - 1, c + 1:c + fshape[1] - 1] = np.where(aperture, signal, bg)
    cursor = int(rng.integers(1, extent[cross] - band_cross))
    for mask, color in zip(masks, inks):
        p = [0, 0]
        p[cross] = cursor
        p[axis] = bs + int(rng.integers(0, band_axis - mask.shape[axis] + 1))
        rr, cc = p
        view = grid[rr:rr + mask.shape[0], cc:cc + mask.shape[1]]
        view[mask] = color
        cursor += mask.shape[cross] + small_gap
    return grid


def _hidden_cells(scene, match):
    h, w = match["swatch"]["mask"].shape
    vh, vw = scene["aperture"].shape
    dr, dc = match["shift"]
    seen_r = set(((np.arange(vh) // 2 + dr) % h).tolist())
    seen_c = set(((np.arange(vw) // 2 + dc) % w).tolist())
    return [(r, c) for r in range(h) for c in range(w)
            if r not in seen_r or c not in seen_c]


def _make_pair(rng, kind, selected_rank, parity):
    styles = ("random", "cross", "sparse", "dense", "diagonal", "repeat", "symmetric")
    for _ in range(600):
        if kind == "seam":
            shape = (int(rng.integers(3, 6)), int(rng.integers(3, 6)))
            style = "random"
        else:
            shape = (int(rng.integers(4, 7)), int(rng.integers(4, 7)))
            style = styles[int(rng.integers(len(styles)))]
        h, w = shape
        selected = _mask(rng, shape, style)
        if kind == "seam":
            if not (_tight(~selected) and _orientation_rich(selected)
                    and not _cyclic_equal(selected, ~selected)):
                continue
            other = ~selected
        else:
            if rng.random() < .6:
                other = selected.copy()
                for p in rng.choice(h * w, size=int(rng.integers(1, 4)), replace=False):
                    other.flat[int(p)] ^= True
                if not _tight(other):
                    continue
            else:
                other = _mask(rng, shape, styles[int(rng.integers(len(styles)))])
        count = int(rng.integers(2, 5))
        masks = [selected, other]
        for _ in range(count - 2):
            s = shape if rng.random() < .55 else (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
            masks.append(_mask(rng, s, styles[int(rng.integers(len(styles)))]))
        dr, dc = int(rng.integers(h)), int(rng.integers(w))
        shifted = np.roll(~selected, (-dr, -dc), axis=(0, 1))
        if kind == "seam" and np.array_equal(shifted, ~selected):
            continue
        if kind == "seam":
            nh, nw = h + int(rng.integers(1, 3)), w + int(rng.integers(1, 3))
        elif kind == "clip":
            nh, nw = h - 1, w - 1
        else:
            if rng.integers(2):
                nh, nw = h - 2, w - 1
            else:
                nh, nw = h - 1, w - 2
        vh, vw = 2 * nh - parity[0], 2 * nw - parity[1]
        wallpaper = np.tile(shifted, (2, 2))
        magnified = np.repeat(np.repeat(wallpaper, 2, axis=0), 2, axis=1)
        aperture = magnified[:vh, :vw]
        if aperture.all() or not aperture.any():
            continue
        colors = tuple(int(v) for v in rng.choice(10, count + 3, replace=False))
        grid = _layout(rng, masks, aperture, colors, selected_rank)
        if grid is None:
            continue
        try:
            scene, match = _interpret(grid)
        except ValueError:
            continue
        bg, _, _, selected_color = colors[:4]
        expected = np.where(wallpaper, selected_color, bg).astype(np.int8)
        if not np.array_equal(expected, match["output"]):
            continue
        if kind != "seam":
            # A hidden interior bit is a direct causal witness for completion
            # from a reference, rather than downsampling the aperture alone.
            if not any(0 < r < h - 1 and 0 < c < w - 1 for r, c in _hidden_cells(scene, match)):
                continue
        return {"input": grid, "output": expected}
    raise RuntimeError(f"could not construct {kind} example in bounded attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle, with explicit teaching witnesses."""
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    # The two required lessons occur in random order and select opposite ends
    # of the swatch band. Extra demonstrations mix both aperture regimes.
    first_rank = int(rng.integers(2))
    axis = int(rng.integers(2))
    parity = (1, 0) if axis == 0 else (0, 1)
    plans = [("seam", first_rank, parity), ("clip", 1 - first_rank, parity[::-1])]
    for _ in range(num_train - 2):
        plans.append(("seam" if rng.random() < .4 else "clip", int(rng.integers(2)),
                      (int(rng.integers(2)), int(rng.integers(2)))))
    rng.shuffle(plans)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_roles = None
    for split, local_plans in (("train", plans),
                               ("test", [("test", int(rng.integers(2)), (1, 1)) for _ in range(num_test)])):
        for index, (kind, rank, cuts) in enumerate(local_plans):
            for _ in range(60):
                pair = _make_pair(rng, kind, rank, cuts)
                scene, match = _interpret(pair["input"])
                roles = (scene["background"], scene["frame"], scene["signal"], match["swatch"]["color"])
                if split == "train" and index == 1 and any(a == b for a, b in zip(roles, first_roles)):
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        first_roles = roles
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the rule, uniqueness, periodic completion, and teaching evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal counts")
    seen, teaching = set(), {"seam": False, "hidden": False, "phase": False,
                             "negative_trap": False, "orientation": False}
    ranks = set()
    training_roles = [set() for _ in range(4)]
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            k = _key(inp)
            _require(k not in seen, "duplicate input")
            seen.add(k)
            scene, match = _interpret(inp)
            _require(np.array_equal(out, match["output"]), "incorrect solution")
            swatch, shift = match["swatch"], match["shift"]
            mask, bg, ink = swatch["mask"], scene["background"], swatch["color"]
            h, w = mask.shape
            _require(out.shape == (2 * h, 2 * w), "must reconstruct four native tiles")
            _require(set(map(int, np.unique(out))) == {bg, ink}, "wrong output color roles")
            for r in range(2 * h):
                for c in range(2 * w):
                    expected = bg if mask[(r + shift[0]) % h, (c + shift[1]) % w] else ink
                    _require(out[r, c] == expected, "cellwise complement/phase invariant failed")
            for r, c in ((0, w), (h, 0), (h, w)):
                _require(np.array_equal(out[r:r+h, c:c+w], out[:h, :w]), "tiles differ")
            observed = scene["aperture"]
            enlarged = np.repeat(np.repeat(out == ink, 2, 0), 2, 1)
            _require(np.array_equal(enlarged[:observed.shape[0], :observed.shape[1]], observed),
                     "reconstructed output does not explain the aperture")
            if split == "train":
                for values, color in zip(training_roles, (bg, scene["frame"], scene["signal"], ink)):
                    values.add(color)
                vh, vw = observed.shape
                teaching["seam"] |= vh > 2*h and vw > 2*w
                teaching["hidden"] |= any(0 < r < h-1 and 0 < c < w-1
                                          for r, c in _hidden_cells(scene, match))
                teaching["phase"] |= not np.array_equal(out[:h, :w] == ink, ~mask)
                teaching["orientation"] |= vh > 2*h and vw > 2*w and _orientation_rich(mask)
                wrong = _matches(scene, invert=False)
                teaching["negative_trap"] |= bool(wrong) and all(
                    not np.array_equal(m["output"], out) for m in wrong)
                boxes = [s["box"] for s in scene["swatches"]]
                selected_box = swatch["box"]
                # Either row or column order describes the generated swatch band.
                for axis in (0, 2):
                    order = sorted(boxes, key=lambda b: (b[axis], b[2-axis]))
                    if all(order[i][axis+1] < order[i+1][axis] for i in range(len(order)-1)):
                        ranks.add("first" if order[0] == selected_box else
                                  "last" if order[-1] == selected_box else "middle")
    _require(all(teaching.values()), f"missing teaching evidence: {teaching}")
    _require({"first", "last"} <= ranks, "training must vary selected physical rank")
    _require(all(len(values) >= 2 for values in training_roles), "training must vary literal color roles")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) for s in ("train", "test")) and all(
        np.array_equal(x[k], y[k]) for s in ("train", "test")
        for x, y in zip(a[s], b[s]) for k in ("input", "output"))


def self_test():
    """Bounded 42-seed variation audit plus causal and corruption regressions."""
    stats = {k: set() for k in ("puzzles", "input_shapes", "output_shapes", "masks",
                                "palettes", "counts", "layouts", "phases", "densities")}
    hidden_test = seam_train = no_black = different_shapes = equivalent_phases = 0
    samples = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        samples.append(puzzle)
        stats["puzzles"].add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                scene, match = _interpret(pair["input"])
                m = match["swatch"]["mask"]
                stats["input_shapes"].add(pair["input"].shape)
                stats["output_shapes"].add(pair["output"].shape)
                stats["masks"].add(_key(m))
                stats["palettes"].add(tuple(map(int, np.unique(pair["input"]))))
                stats["counts"].add(len(scene["swatches"]))
                stats["layouts"].add((scene["frame_box"], tuple(s["box"] for s in scene["swatches"])))
                stats["phases"].add(match["shift"])
                stats["densities"].add(round(float(m.mean()), 1))
                hidden_test += split == "test" and bool(_hidden_cells(scene, match))
                seam_train += split == "train" and all(scene["aperture"].shape[i] > 2*m.shape[i] for i in (0, 1))
                no_black += 0 not in np.unique(pair["input"])
                different_shapes += len({s["mask"].shape for s in scene["swatches"]}) > 1
                equivalent_phases += len(_matches(scene)) > 1
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list solve failed")
    thresholds = {"puzzles": 42, "input_shapes": 35, "output_shapes": 10, "masks": 100,
                  "palettes": 35, "counts": 3, "layouts": 150, "phases": 15, "densities": 5}
    for name, minimum in thresholds.items():
        _require(len(stats[name]) >= minimum, f"insufficient {name} variation: {len(stats[name])}")
    _require(hidden_test == 42 and seam_train >= 42 and no_black > 5 and different_shapes > 20,
             "missing clipping, seam, palette or geometry coverage")
    _require(_equal(samples[0], generate_puzzle(0)), "nondeterministic default puzzle")
    for nt, ne in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        a = generate_puzzle(101 + nt + ne, nt, ne)
        _require(_equal(a, generate_puzzle(101 + nt + ne, nt, ne)), "nondeterministic explicit counts")
        _require((len(a["train"]), len(a["test"])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    broken = copy.deepcopy(samples[0])
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("accepted a corrupted output")
    # Change just one unseen interior swatch bit. The aperture stays identical,
    # but the completed wallpaper changes in all four copies of that residue.
    causal = False
    for puzzle in samples:
        for pair in puzzle["train"]:
            scene, match = _interpret(pair["input"])
            h, w = match["swatch"]["mask"].shape
            hidden = [(r, c) for r, c in _hidden_cells(scene, match) if 0 < r < h-1 and 0 < c < w-1]
            for r, c in hidden:
                modified = pair["input"].copy()
                r0, _, c0, _ = match["swatch"]["box"]
                bg, ink = scene["background"], match["swatch"]["color"]
                modified[r0+r, c0+c] = bg if modified[r0+r, c0+c] == ink else ink
                try:
                    new = solve(modified)
                except ValueError:
                    # A changed bit can introduce an alternative alignment;
                    # only a still-unambiguous counterfactual is a witness.
                    continue
                _require(np.count_nonzero(new != pair["output"]) == 4,
                         "reference bit has no four-tile causal effect")
                causal = True
                break
            if causal:
                break
        if causal:
            break
    _require(causal, "no hidden-reference causal witness")
    # Color and axis roles are input-local; neither 0 nor a canvas orientation
    # supplies a hidden convention.
    perm = np.array([7, 4, 9, 0, 6, 1, 8, 3, 5, 2], dtype=np.int8)
    for pair in samples[0]["train"]:
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]),
                 "color-role permutation failed")
        _require(np.array_equal(solve(pair["input"].T), pair["output"].T),
                 "axis exchange failed")
    # Demonstrate that the input-only solver also accepts an equivalent-phase
    # periodic member, without requiring a unique numerical phase.
    rng = np.random.default_rng(20261016)
    periodic = np.array([[1, 0, 1, 0], [0, 1, 0, 1]] * 2, dtype=bool)
    other = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1]], dtype=bool)
    wanted = np.tile(~periodic, (2, 2))
    aperture = np.repeat(np.repeat(wanted, 2, 0), 2, 1)[:11, :11]
    grid = _layout(rng, [periodic, other], aperture, (0, 8, 7, 2, 4), 0)
    scene, match = _interpret(grid)
    _require(len(_matches(scene)) > 1, "equivalent-phase regression did not exercise ties")
    _require(np.array_equal(solve(grid), np.where(wanted, 2, 0)), "equivalent-phase member failed")
    return {"seeds": 42, "pairs": 210, "distinct": {k: len(v) for k, v in stats.items()},
            "hidden_tests": int(hidden_test), "seam_train": int(seam_train),
            "nonblack_pairs": int(no_black), "mixed_shape_pairs": int(different_shapes),
            "equivalent_phase_pairs": int(equivalent_phases)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
