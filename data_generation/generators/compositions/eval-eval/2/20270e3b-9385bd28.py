"""Registered window loom: split corners acquire geometry by registration.

Only Python and NumPy are used. Public solving is entirely input-derived.
The header of the isolated table is [scaffold color, registration color];
subsequent rows are [corner cue, opaque paint], highest priority first.
"""

from collections import deque
import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any(g < 0) or np.any(g > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30 and colors 0..9")
    return g.astype(np.int8, copy=False)


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        part.add(q)
                        todo.append(q)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _shift(cells, dr, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _inside(q, box):
    return box[0] <= q[0] <= box[2] and box[1] <= q[1] <= box[3]


def _corners(box, diagonal):
    t, l, b, r = box
    if diagonal == 0:
        return ({(t, l), (t, l + 1), (t + 1, l)},
                {(b, r), (b, r - 1), (b - 1, r)})
    return ({(t, r), (t, r - 1), (t + 1, r)},
            {(b, l), (b, l + 1), (b - 1, l)})


def _recognize_corners(cells):
    if len(cells) != 6:
        raise ValueError("Each registered cue must have six distinct corner cells")
    box = _bbox(cells)
    h, w = box[2] - box[0] + 1, box[3] - box[1] + 1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        raise ValueError("Window corners are too close")
    for diagonal in (0, 1):
        a, b = _corners(box, diagonal)
        if cells == a | b:
            return box, diagonal
    raise ValueError("Registered Ls must be opposite corners")


def _parse(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("Background must be the unique most frequent input color")
    components = _components(map(tuple, np.argwhere(g != bg)))
    if len(components) != 3:
        raise ValueError("Expected an isolated table and two separated carriers")
    candidates = []
    for table_cells in components:
        t, l, b, r = _bbox(table_cells)
        if r - l != 1 or not 3 <= b - t + 1 <= 5:
            continue
        body, guide = map(int, g[t, l:r + 1])
        cues = list(map(int, g[t + 1:b + 1, l]))
        fills = list(map(int, g[t + 1:b + 1, r]))
        if (len({bg, body, guide, *cues}) != 3 + len(cues)
                or any(v in (body, guide) for v in fills)
                or any(g[y, l] == bg for y in range(t, b + 1))):
            continue
        pieces = [p for p in components if p is not table_cells]
        try:
            records, axes, lengths = [], [], []
            for piece in pieces:
                if {int(g[q]) for q in piece} != {body, guide, *cues}:
                    raise ValueError("Carrier colors must agree with the role key")
                bar = {q for q in piece if g[q] == guide}
                if not 2 <= len(bar) <= 5:
                    raise ValueError("Registration bars have lengths 2..5")
                bt, bl, bb, br = _bbox(bar)
                if bt == bb and br - bl + 1 == len(bar):
                    axis = 0
                elif bl == br and bb - bt + 1 == len(bar):
                    axis = 1
                else:
                    raise ValueError("A guide must be a contiguous straight bar")
                nr, nc = (bt, bl) if axis == 0 else (bl, bt)
                coords = {(q[axis] - nr, q[1 - axis] - nc): int(g[q])
                          for q in piece if g[q] != guide}
                negative = any((-1, x) in coords for x in range(len(bar)))
                positive = any((1, x) in coords for x in range(len(bar)))
                if negative == positive:
                    raise ValueError("Backing must identify exactly one mating side")
                for cue in cues:
                    loc = {q for q, v in coords.items() if v == cue}
                    box = _bbox(loc)
                    if len(loc) != 3 or box[2] - box[0] != 1 or box[3] - box[1] != 1:
                        raise ValueError("Every carrier has one three-cell L per cue")
                records.append(dict(negative=negative, pixels=coords,
                                    raw=piece, origin=(nr, nc)))
                axes.append(axis)
                lengths.append(len(bar))
            if len(set(axes)) != 1 or len(set(lengths)) != 1 or records[0]["negative"] == records[1]["negative"]:
                raise ValueError("Bars must be congruent and oppositely backed")
            records.sort(key=lambda p: not p["negative"])
            parsed = dict(background=bg, body=body, guide=guide, cues=cues,
                          fills=fills, pieces=records, axis=axes[0], length=lengths[0],
                          table=(t, l, b, r), input_shape=g.shape)
            base, specs = _registered(parsed)
            extent = _bbox(base)
            if max(extent[2] - extent[0] + 1, extent[3] - extent[1] + 1) > 30:
                raise ValueError("Registered crop exceeds ARC dimensions")
            parsed["specs"] = specs
            candidates.append(parsed)
        except ValueError:
            continue
    if len(candidates) != 1:
        raise ValueError("Input must have exactly one readable table/registration interpretation")
    return candidates[0]


def _registered(p, seam=-1, blend="union"):
    a = p["pieces"][0]["pixels"]
    b = {(r + seam, c): v for (r, c), v in p["pieces"][1]["pixels"].items()}
    # Cue-cue collisions are forbidden. Scaffold can lie under a corner.
    if any(q in a and a[q] != p["body"] and v != p["body"] for q, v in b.items()):
        raise ValueError("Registered corners may not collide")
    if blend == "union":
        base = {q: p["body"] for q in set(a) | set(b)}
        for part in (a, b):
            base.update({q: v for q, v in part.items() if v != p["body"]})
    else:
        front, back = (a, b) if blend == "anchor_opaque" else (b, a)
        box = _bbox(front)
        base = {q: v for q, v in back.items() if not _inside(q, box)}
        base.update(front)
    specs = []
    for cue, fill in zip(p["cues"], p["fills"]):
        cells = {q for part in (a, b) for q, v in part.items() if v == cue}
        if seam == -1:
            box, diagonal = _recognize_corners(cells)
        else:
            box, diagonal = _bbox(cells), -1
        specs.append(dict(cue=cue, fill=fill, cells=cells, bounds=box, diagonal=diagonal))
    return base, specs


def _render(base, specs, bg, axis=0, mode="normal", guides=None):
    """Pixel traversal for solving; generation uses reverse layer compositing."""
    extent = set(base) | {q for spec in specs for q in spec["cells"]}
    t, l, b, r = _bbox(extent)
    out = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
    ordered = list(reversed(specs)) if mode == "reverse" else specs
    for y in range(t, b + 1):
        for x in range(l, r + 1):
            q = (y, x)
            value = base.get(q, bg)
            if mode != "no_windows":
                for spec in ordered:
                    if _inside(q, spec["bounds"]):
                        if q in spec["cells"]:
                            value = spec["cue"]
                        elif mode == "cue_fill":
                            value = spec["cue"]
                        elif mode == "transparent_clear" and spec["fill"] == bg:
                            continue
                        else:
                            value = spec["fill"]
                        break
            if guides and q in guides[0]:
                if mode == "erase_guides":
                    value = bg
                elif mode == "paint_guides":
                    value = guides[1]
            out[y - t, x - l] = value
    return out.T.copy() if axis else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Translate carriers, infer cross-carrier windows, and composite top-first."""
    p = _parse(input_grid)
    base, specs = _registered(p)
    return _render(base, specs, p["background"], p["axis"])


def _alternative(p, mode):
    if mode == "unregistered":
        base = {}
        for part in p["pieces"]:
            nr, nc = part["origin"]
            for (r, c), value in part["pixels"].items():
                base[(r + nr, c + nc)] = value
        specs = [dict(cue=cue, fill=fill, cells={q for q, v in base.items() if v == cue})
                 for cue, fill in zip(p["cues"], p["fills"])]
        for spec in specs:
            spec["bounds"] = _bbox(spec["cells"])
        return _render(base, specs, p["background"], p["axis"])
    seam = 0 if mode == "coincident" else 1 if mode == "opposite_seam" else -1
    blend = mode if mode in ("anchor_opaque", "mover_opaque") else "union"
    try:
        base, specs = _registered(p, seam, blend)
    except ValueError:
        return None
    guides = ({(r, c) for r in (-1, 0) for c in range(p["length"])}, p["body"])
    return _render(base, specs, p["background"], p["axis"], mode, guides)


def _features(p, output):
    base, specs = _registered(p)
    bg, body = p["background"], p["body"]
    t, l, b, r = _bbox(base)
    oriented = output.T if p["axis"] else output
    def visible(q):
        return int(oriented[q[0] - t, q[1] - l])
    guide_cells = {(y, x) for y in (-1, 0) for x in range(p["length"])}
    uncovered = sum(q not in base and _inside(q, (t, l, b, r)) and visible(q) == bg for q in guide_cells)
    covered = sum(q in base and base[q] == body and visible(q) == body for q in guide_cells)
    occlusion = clearing = retention = False
    for i, spec in enumerate(specs):
        for q in spec["cells"]:
            if visible(q) != spec["cue"]:
                occlusion = True
            elif spec["cue"] != spec["fill"]:
                retention = True
        if spec["fill"] == bg:
            for lower in specs[i + 1:]:
                at, al, ab, ar = lower["bounds"]
                for y in range(at, ab + 1):
                    for x in range(al, ar + 1):
                        q = (y, x)
                        if (q not in spec["cells"] and _inside(q, spec["bounds"])
                                and visible(q) == bg
                                and (lower["fill"] != bg or q in lower["cells"])):
                            # Confirm that this clearing layer is the visible one.
                            if not any(_inside(q, upper["bounds"]) for upper in specs[:i]):
                                clearing = True
    alternative_modes = ("unregistered", "coincident", "opposite_seam", "no_windows", "cue_fill",
                         "reverse", "transparent_clear", "erase_guides", "paint_guides",
                         "anchor_opaque", "mover_opaque")
    differences = {mode for mode in alternative_modes
                   if not np.array_equal(output, _alternative(p, mode))}
    a = p["pieces"][0]["pixels"]
    bs = {(y - 1, x): v for (y, x), v in p["pieces"][1]["pixels"].items()}
    boxes = [_bbox(part["raw"]) for part in p["pieces"]]
    x_separate = boxes[0][3] < boxes[1][1] or boxes[1][3] < boxes[0][1]
    y_separate = boxes[0][2] < boxes[1][0] or boxes[1][2] < boxes[0][0]
    stage = "diagonal" if x_separate and y_separate else "horizontal" if x_separate else "vertical" if y_separate else "interleaved"
    sizes = [(s["bounds"][2] - s["bounds"][0] + 1,
              s["bounds"][3] - s["bounds"][1] + 1) for s in specs]
    overlaps = sum(not (a["bounds"][2] < b["bounds"][0] or b["bounds"][2] < a["bounds"][0]
                           or a["bounds"][3] < b["bounds"][1] or b["bounds"][3] < a["bounds"][1])
                   for i, a in enumerate(specs) for b in specs[i + 1:])
    return dict(differences=differences, uncovered_guides=uncovered, covered_guides=covered,
                clearing=clearing, occlusion=occlusion, retention=retention,
                body_overlap=sum(a.get(q) == body and v == body for q, v in bs.items()),
                sizes=sizes, overlaps=overlaps, stage=stage, axis=p["axis"],
                length=p["length"], layers=len(specs), palette=(bg, body, p["guide"], *p["cues"]))


def _path(rng, start, goals, forbidden, bounds):
    """Randomized shortest orthogonal route, with finite local search space."""
    todo, previous = deque([start]), {start: None}
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    directions = [directions[int(i)] for i in rng.permutation(4)]
    end = None
    while todo:
        q = todo.popleft()
        if q in goals:
            end = q
            break
        for dr, dc in directions:
            nxt = (q[0] + dr, q[1] + dc)
            if nxt not in previous and nxt not in forbidden and _inside(nxt, bounds):
                previous[nxt] = q
                todo.append(nxt)
    if end is None:
        return None
    cells = set()
    while end is not None:
        cells.add(end)
        end = previous[end]
    return cells


def _scaffold(rng, cue_cells, length, sign, backed):
    bar = {(0, x) for x in range(length)}
    forbidden = bar | {(-sign, x) for x in range(length)} | {
        (sign, x) for x in range(length) if x not in backed}
    corner_pixels = set().union(*cue_cells.values())
    if corner_pixels & forbidden:
        return None
    body = {(2 * sign, x) for x in range(length)} | {(sign, x) for x in backed}
    # Bounds derive from the payload, so the sampler does not impose one panel.
    t, l, b, r = _bbox(body | corner_pixels | bar)
    bounds = (t - 1, l - 1, b + 1, r + 1)
    for cue in rng.permutation(list(cue_cells)):
        target = min(cue_cells[int(cue)])
        path = _path(rng, target, body | bar, forbidden, bounds)
        if path is None:
            return None
        body.update(path)
    for _ in range(int(rng.integers(2, 8))):
        ordered = sorted(body)
        y, x = ordered[int(rng.integers(len(ordered)))]
        # Spurs and small loops vary shape, cavities and occasional thickness.
        if rng.random() < .35:
            additions = {(y + dy, x + dx) for dy in range(3) for dx in range(3)
                         if dy in (0, 2) or dx in (0, 2)}
        else:
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            additions = {(y + step * dy, x + step * dx) for step in range(int(rng.integers(1, 5)))}
        additions = {q for q in additions if q not in forbidden and _inside(q, bounds)}
        # A clipped spur must still belong to the carrier.
        for part in _components(additions):
            if any(max(abs(q[0] - z[0]), abs(q[1] - z[1])) <= 1 for q in part for z in body):
                body.update(part)
    body -= corner_pixels
    pixels = {q: "body" for q in body}
    for cue, cells in cue_cells.items():
        pixels.update({q: cue for q in cells})
    if len(_components(set(pixels) | bar)) != 1:
        return None
    return pixels


def _random_box(rng, length, guide_demo, thin, elongated):
    rlo, rhi = -5, 6
    clo, chi = (-6, -1) if guide_demo else (-5, 7)
    if guide_demo and rng.random() < .25 and length <= 3:
        clo, chi = length, 8
    hmax, wmax = (10, 10) if elongated else (7, 7)
    if thin:
        if rng.integers(2):
            h, w = 2, int(rng.integers(8 if elongated else 4, min(wmax, chi - clo + 1) + 1))
        else:
            h, w = int(rng.integers(8 if elongated else 4, min(hmax, rhi - rlo + 1) + 1)), 2
    else:
        h = int(rng.integers(3, min(hmax, rhi - rlo + 1) + 1))
        w = int(rng.integers(3, min(wmax, chi - clo + 1) + 1))
        if elongated and max(h, w) < 8:
            if rng.integers(2):
                h = int(rng.integers(8, hmax + 1))
            else:
                w = int(rng.integers(8, wmax + 1))
    y = int(rng.integers(rlo, rhi - h + 2))
    x = int(rng.integers(clo, chi - w + 2))
    return (y, x, y + h - 1, x + w - 1)


def _geometry(rng, profile, cues):
    length = profile["length"]
    backed = []
    for _ in range(2):
        upper = length if not profile["guide_demo"] else length - 1
        number = int(rng.integers(1, upper + 1))
        backed.append(set(map(int, rng.choice(length, number, replace=False))))
    payload = [{}, {}]
    specs, used = [], set()
    for i, cue in enumerate(cues):
        thin = profile["thin"] and (i == 0 or rng.random() < .2)
        box = _random_box(rng, length, profile["guide_demo"], thin, profile["elongated"])
        diagonal = int(rng.integers(2))
        first, second = _corners(box, diagonal)
        if (first | second) & used:
            return None
        used.update(first | second)
        if rng.integers(2):
            first, second = second, first
        payload[0][cue] = first
        # Mover coordinates are before the one-row registration shift.
        payload[1][cue] = _shift(second, 1)
        specs.append(dict(cue=cue, bounds=box, cells=first | second, diagonal=diagonal))
    pieces = [_scaffold(rng, payload[i], length, -1 if i == 0 else 1, backed[i]) for i in range(2)]
    if any(p is None for p in pieces):
        return None
    return pieces, specs


def _rotate(q, rotation):
    y, x = q
    for _ in range(rotation):
        y, x = -x, y
    return y, x


def _construction_output(pieces, specs, bg, body, rotation):
    """Ground truth from constructed masks, by reverse-order rectangle pasting."""
    moved = {(y - 1, x): v for (y, x), v in pieces[1].items()}
    assembled = {q: body for q in set(pieces[0]) | set(moved)}
    for part in (pieces[0], moved):
        assembled.update({q: v for q, v in part.items() if v != body})
    t, l, b, r = _bbox(assembled)
    out = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
    for (y, x), value in assembled.items():
        out[y - t, x - l] = value
    for spec in reversed(specs):
        at, al, ab, ar = spec["bounds"]
        out[at - t:ab - t + 1, al - l:ar - l + 1] = spec["fill"]
        for y, x in spec["cells"]:
            out[y - t, x - l] = spec["cue"]
    return np.rot90(out, rotation).copy()


def _stage(rng, pieces, length, palette, specs, profile):
    bg, body, guide = palette
    normalized = []
    for part in pieces:
        values = dict(part)
        values.update({(0, x): guide for x in range(length)})
        values = {_rotate(q, profile["rotation"]): value for q, value in values.items()}
        t, l, b, r = _bbox(values)
        normalized.append(({(y - t, x - l): v for (y, x), v in values.items()}, b - t + 1, r - l + 1))
    if rng.integers(2):
        normalized.reverse()
    a, ah, aw = normalized[0]
    bp, bh, bw = normalized[1]
    avoid = {(y + dy, x + dx) for y, x in a for dy in (-1, 0, 1) for dx in (-1, 0, 1)}
    modes = [profile["stage"]] * 35 + ["horizontal", "vertical", "diagonal", "interleaved"] * 10
    for mode in modes:
        gap = int(rng.integers(1, 4))
        if mode == "horizontal":
            dy, dx = int(rng.integers(-bh + 1, ah)), aw + gap
        elif mode == "vertical":
            dy, dx = ah + gap, int(rng.integers(-bw + 1, aw))
        elif mode == "diagonal":
            dy, dx = ah + gap, aw + gap
        else:
            dy, dx = int(rng.integers(-bh + 1, ah)), int(rng.integers(-bw + 1, aw))
        shifted = {(y + dy, x + dx): v for (y, x), v in bp.items()}
        if set(shifted) & avoid:
            continue
        values = dict(a)
        values.update(shifted)
        t, l, b, r = _bbox(values)
        h, w = b - t + 1, r - l + 1
        if h > 30 or w > 30:
            continue
        padt, padl = int(rng.integers(min(3, 30 - h) + 1)), int(rng.integers(min(3, 30 - w) + 1))
        h = min(30, h + padt + int(rng.integers(2, 6)))
        w = min(30, w + padl + int(rng.integers(2, 6)))
        if rng.random() < .15:
            if rng.integers(2):
                h = 30
            else:
                w = 30
        g = np.full((h, w), bg, dtype=np.int8)
        for (y, x), value in values.items():
            g[y - t + padt, x - l + padl] = value
        nrows = len(specs) + 1
        locations = []
        for y in range(h - nrows + 1):
            for x in range(w - 1):
                if np.all(g[max(0, y - 1):min(h, y + nrows + 1),
                            max(0, x - 1):min(w, x + 3)] == bg):
                    locations.append((y, x))
        if not locations:
            continue
        y, x = locations[int(rng.integers(len(locations)))]
        g[y, x:x + 2] = (body, guide)
        for i, spec in enumerate(specs, 1):
            g[y + i, x:x + 2] = (spec["cue"], spec["fill"])
        return g
    return None


def _make_pair(rng, profile, used):
    for _ in range(600):
        roles = list(map(int, rng.permutation(10)))
        bg, body, guide = roles[:3]
        cues = roles[3:3 + profile["layers"]]
        geometry = _geometry(rng, profile, cues)
        if geometry is None:
            continue
        raw_pieces, specs = geometry
        pieces = [{q: body if value == "body" else value for q, value in part.items()} for part in raw_pieces]
        specs = [specs[int(i)] for i in rng.permutation(len(specs))]
        fill_choices = [v for v in range(10) if v not in (body, guide)]
        for spec in specs:
            spec["fill"] = int(rng.choice(fill_choices))
        if profile["guide_demo"]:
            # Lookup and priority remain visible, independent of cue identity.
            for spec in specs:
                spec["fill"] = int(rng.choice([v for v in fill_choices if v != bg and v != spec["cue"]]))
        if profile["clearing_demo"]:
            specs[0]["fill"] = bg
            specs[1]["fill"] = int(rng.choice([v for v in fill_choices if v != bg]))
        g = _stage(rng, pieces, profile["length"], (bg, body, guide), specs, profile)
        if g is None or _key(g) in used:
            continue
        output = _construction_output(pieces, specs, bg, body, profile["rotation"])
        if max(output.shape) > 30:
            continue
        try:
            p = _parse(g)
        except ValueError:
            continue
        if not np.array_equal(output, _render(*_registered(p), bg, p["axis"])):
            raise AssertionError("Construction and input-derived solve disagree")
        info = _features(p, output)
        if not {"unregistered", "no_windows", "cue_fill"} <= info["differences"]:
            continue
        if profile["guide_demo"] and not (
                info["covered_guides"] and info["uncovered_guides"] and info["retention"]
                and {"reverse", "anchor_opaque", "mover_opaque", "coincident", "opposite_seam"} <= info["differences"]):
            continue
        if profile["clearing_demo"] and not (info["clearing"] and info["occlusion"]
                and "transparent_clear" in info["differences"]):
            continue
        return {"input": g, "output": output}
    raise RuntimeError("Bounded scaffold/window construction exhausted")


def _key(g):
    a = np.asarray(g)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("Counts must be 2..10 train, 1..4 test, with train > test")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(["layers", "bar_interpolation", "aspect"]))
    mixed_axis = bool(rng.integers(2))
    rotation = int(rng.integers(4))
    thin = bool(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for i in range(count):
            training = split == "train"
            profile = dict(
                layers=int(rng.integers(2, 4)) if training or regime != "layers" else 4,
                length=(2 if i % 2 == 0 else 4) if training and regime == "bar_interpolation"
                       else 3 if regime == "bar_interpolation" else int(rng.integers(2, 6)),
                rotation=(rotation + (i % 2 if mixed_axis else 0)) % 4,
                thin=thin and (not training or i == 0 or rng.random() < .4),
                elongated=not training and regime == "aspect",
                guide_demo=training and i == 0,
                clearing_demo=training and i == 1,
                stage=str(rng.choice(["horizontal", "vertical", "diagonal", "interleaved"])))
            pair = _make_pair(rng, profile, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(g) not in seen, "Inputs must be distinct"
            seen.add(_key(g))
            p = _parse(g)
            base, specs = _registered(p)
            assert np.array_equal(out, _render(base, specs, p["background"], p["axis"])), "Wrong output"
            assert p["guide"] not in set(map(int, np.unique(out))), "Temporary guide leaked into output"
            box = _bbox(base)
            shape = (box[2] - box[0] + 1, box[3] - box[1] + 1)
            assert out.shape == (shape[::-1] if p["axis"] else shape), "Wrong registered crop"
            info = _features(p, out)
            assert {"unregistered", "no_windows", "cue_fill"} <= info["differences"], "A parent is inert"
            information[split].append(info)
    train = information["train"]
    required = {"unregistered", "coincident", "opposite_seam", "no_windows", "cue_fill", "reverse",
                "transparent_clear", "erase_guides", "paint_guides", "anchor_opaque", "mover_opaque"}
    assert required <= set().union(*(i["differences"] for i in train)), "Missing semantic demonstration"
    assert any(i["covered_guides"] and i["uncovered_guides"] for i in train), "Guide transparency not shown"
    assert any(i["clearing"] and i["occlusion"] for i in train), "Opaque clearing/occlusion not shown"
    assert any(i["retention"] for i in train), "Own-corner retention not shown"
    assert {i["axis"] for i in information["test"]} <= {i["axis"] for i in train}
    if any(min(size) == 2 for i in information["test"] for size in i["sizes"]):
        assert any(min(size) == 2 for i in train for size in i["sizes"]), "Thin windows not demonstrated"
    tests = information["test"]
    count_shift = max(i["layers"] for i in train) < min(i["layers"] for i in tests)
    length_shift = ({i["length"] for i in train} == {2, 4}
                    and {i["length"] for i in tests} == {3})
    extent_shift = (max(max(s) for i in train for s in i["sizes"])
                    < min(max(max(s) for s in i["sizes"]) for i in tests))
    assert count_shift or length_shift or extent_shift, "Missing structural train/test shift"


def self_test():
    """Bounded 42-seed variation and semantic mutation checks, no file I/O."""
    shapes, output_shapes, palettes, lengths, axes, counts, stages = [set() for _ in range(7)]
    signatures, structures, differences = set(), set(), set()
    overlaps, clears, thin, within_shapes, within_payload = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        local_shapes, local_payload = set(), set()
        fingerprint = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                p = _parse(pair["input"])
                info = _features(p, pair["output"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                palettes.add(info["palette"])
                lengths.add(info["length"])
                axes.add(info["axis"])
                counts.add(info["layers"])
                stages.add(info["stage"])
                differences.update(info["differences"])
                overlaps += info["body_overlap"] > 0
                clears += info["clearing"]
                thin += any(min(s) == 2 for s in info["sizes"])
                local_shapes.add(pair["input"].shape)
                local_payload.add(tuple(info["sizes"]))
                # Normalize colors while preserving positions and associations.
                roles = {}
                structures.add((pair["input"].shape, tuple(roles.setdefault(int(v), len(roles))
                                                           for v in pair["input"].ravel())))
                fingerprint.append((_key(pair["input"]), _key(pair["output"])))
        signatures.add(tuple(fingerprint))
        within_shapes += len(local_shapes) > 1
        within_payload += len(local_payload) > 1
    assert len(signatures) == 42
    assert len(shapes) >= 40 and len(output_shapes) >= 20 and len(structures) >= 160
    assert len(palettes) >= 150 and lengths == {2, 3, 4, 5} and axes == {0, 1}
    assert counts == {2, 3, 4} and len(stages) >= 3
    assert overlaps >= 30 and clears >= 42 and thin >= 20
    assert within_shapes == 42 and within_payload == 42
    # Every legal explicit count is exercised; no automatic count convention.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            q = generate_puzzle(9047 + a * 10 + b, a, b)
            assert len(q["train"]) == a and len(q["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    a, b = generate_puzzle(93271, 2, 1), generate_puzzle(93271, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for split in a for x, y in zip(a[split], b[split])
               for k in ("input", "output"))
    # A visible role key makes every literal color interchangeable. This also
    # verifies that solving does not mutate the caller's input.
    permutation = np.array([7, 3, 9, 0, 6, 2, 8, 1, 5, 4], dtype=np.int8)
    for split in a:
        for pair in a[split]:
            remapped = permutation[pair["input"]]
            before = remapped.copy()
            assert np.array_equal(solve(remapped), permutation[pair["output"]])
            assert np.array_equal(remapped, before)
    # Change only the visible order; geometry and registration stay unchanged.
    reordered = a["train"][0]["input"].copy()
    parsed = _parse(reordered)
    t, l, bottom, right = parsed["table"]
    reordered[t + 1:bottom + 1, l:right + 1] = reordered[t + 1:bottom + 1, l:right + 1][::-1].copy()
    predicted = solve(reordered)
    assert np.array_equal(predicted, _alternative(parsed, "reverse"))
    assert not np.array_equal(predicted, a["train"][0]["output"])
    corrupted = {split: [{k: v.copy() for k, v in pair.items()} for pair in a[split]] for split in a}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Semantic corruption accepted")
    return dict(seeds=42, input_shapes=len(shapes), output_shapes=len(output_shapes),
                normalized_structures=len(structures), palettes=len(palettes),
                bar_lengths=sorted(lengths), axes=sorted(axes), layer_counts=sorted(counts),
                staging=sorted(stages), body_overlap_pairs=overlaps, clearing_pairs=clears,
                thin_pairs=thin, distinguished_alternatives=sorted(differences))
