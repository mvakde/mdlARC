"""Repair tagged wire assemblies, then dock each reconstructed assembly rigidly.

Only standard-library modules and NumPy are used. No files are read or written by
generation, solving, validation, or self_test. Construction labels come from clean
assembly masks; solve rediscovers geometry and addresses from the input pixels.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 371815990, 797702680, 145345517, 188429892, 752275262,
              913102771, 69964181, 396680462, 76156339, 179214225, 993990648,
              104838723, 638102696, 460645177, 674589856, 505205340, 611774668,
              782399821, 785033172, 910055473, 249090442, 373456449, 461496802,
              433197838, 108503573, 85867572, 59689054, 962538171, 39729004,
              500577556, 788650130, 211992283, 153867356, 702171866, 828863003,
              761509768, 54578322, 933549322, 35837950, 494874929, 587328873)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
          "grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in FOUR:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _at(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _cardinal_support(mask):
    support = np.zeros_like(mask)
    support[:-1] |= mask[1:]
    support[1:] |= mask[:-1]
    support[:, :-1] |= mask[:, 1:]
    support[:, 1:] |= mask[:, :-1]
    return support & mask


def _gaps(mask):
    for vertical in (False, True):
        a = mask.T if vertical else mask
        if a.shape[1] < 5:
            continue
        candidates = a[:, :-4] & a[:, 4:]
        candidates &= ~(a[:, 1:-3] | a[:, 2:-2] | a[:, 3:-1])
        for y, x in np.argwhere(candidates):
            r, c = (int(x), int(y)) if vertical else (int(y), int(x))
            yield r, c, int(vertical), int(not vertical)


def _repair(mask, wire):
    """One simultaneous, input-only repair pass; return clean occupancy and edits.

    Cardinally isolated wire cells can form diagonal evidence/clutter. Tags and
    accepted brackets are protected. No iteration invents new repair evidence.
    """
    support = _cardinal_support(mask)
    weak = wire & ~support
    added, anchors = np.zeros_like(mask), np.zeros_like(mask)
    records = []
    for r, c, dr, dc in _gaps(mask):
        if not (_at(mask, r-dr, c-dc) or _at(mask, r+5*dr, c+5*dc)):
            continue
        matches = []
        for endpoint, toward in ((0, 1), (4, -1)):
            ar, ac = r+endpoint*dr, c+endpoint*dc
            for side in (-1, 1):
                sr, sc = toward*dr+side*dc, toward*dc+side*dr
                spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                if (all(_at(weak, rr, cc) for rr, cc in spur)
                        and not _at(mask, ar+3*sr, ac+3*sc)):
                    matches.append(spur)
        _need(len(matches) <= 1, "ambiguous repair evidence")
        if matches:
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            for p in gap:
                _need(not added[p], "overlapping repairs")
                added[p] = True
            anchors[r, c] = anchors[r+4*dr, c+4*dc] = True
            records.append((gap, matches[0], "v" if dr else "h"))
    weak &= ~anchors
    removed = np.zeros_like(mask)
    for dc in (-1, 1):
        a = weak[:-1, :-1] if dc == 1 else weak[:-1, 1:]
        b = weak[1:, 1:] if dc == 1 else weak[1:, :-1]
        link = a & b
        if dc == 1:
            removed[:-1, :-1] |= link
            removed[1:, 1:] |= link
        else:
            removed[:-1, 1:] |= link
            removed[1:, :-1] |= link
    return (mask & ~removed) | added, added, removed, records


def _bands_in_view(a, background):
    """Recognize the destination array below staging in canonical coordinates."""
    h, w = a.shape
    rows = []
    for row in a:
        nonblank = np.flatnonzero(row != background)
        info = None
        if len(nonblank) >= 8:
            left, right = int(nonblank[0]), int(nonblank[-1])
            if (right-left >= 7 and row[left] == row[right]
                    and row[left] != row[left+1] and row[left+1] != background
                    and np.all(row[left+1:right] == row[left+1])):
                info = (left, right, int(row[left+1]), int(row[left]))
        rows.append(info)
    found = [i for i, row in enumerate(rows) if row is not None]
    _need(found, "missing destination bands")
    first = found[0]
    _need(first >= 3 and np.all(a[first-1] == background), "missing staging separator")
    bands, r = [], first
    while r < h:
        if rows[r] is None:
            _need(np.all(a[r] == background), "foreground beyond destination array")
            r += 1
            continue
        left, right, route, rail = rows[r]
        top = r
        while r+1 < h and rows[r+1] == rows[top]:
            r += 1
        bands.append(dict(top=top, bottom=r, left=left, right=right,
                          route=route, rail=rail))
        r += 1
    _need(len({b["route"] for b in bands}) == len(bands), "duplicate destination address")
    _need(all(y["top"] >= x["bottom"]+2 for x, y in zip(bands, bands[1:])),
          "destinations need blank separators")
    return first, bands


def _interpret_view(a, background):
    first, bands = _bands_in_view(a, background)
    stage = a[:first]
    band_colors = {b[k] for b in bands for k in ("route", "rail")}
    source_colors = set(map(int, np.unique(stage))) - {background}
    candidates = source_colors - band_colors
    _need(len(candidates) == 1, "wire must be the unique staging color absent from bands")
    wire = candidates.pop()
    routes = {b["route"]: b for b in bands}
    _need(source_colors - {wire} <= set(routes), "unrecognized source tag")
    clean, added, removed, records = _repair(stage != background, stage == wire)
    objects = []
    for cells in _components(clean):
        tags = [p for p in cells if stage[p] not in (background, wire)]
        _need(len(tags) == 1, "every repaired assembly needs exactly one address tag")
        tag = int(stage[tags[0]])
        objects.append(dict(cells=cells, tag=tag, band=routes[tag], tag_cell=tags[0]))
    _need(objects and records, "a puzzle needs a supported repair and tagged cargo")
    output, occupied = a.copy(), set()
    output[:first] = background
    for obj in objects:
        b = obj["band"]
        bottom = max(r for r, c in obj["cells"])
        target = {(r+b["bottom"]-bottom, c) for r, c in obj["cells"]}
        _need(all(b["top"] <= r <= b["bottom"] and b["left"] < c < b["right"]
                  for r, c in target), "assembly does not fit its destination")
        _need(not occupied & target, "colliding target cells")
        occupied.update(target)
        obj["target"] = target
        for p in target:
            output[p] = b["rail"]
    return dict(canonical=a, background=background, wire=wire, first=first,
                bands=bands, objects=objects, clean=clean, added=added,
                removed=removed, repairs=records, expected=output)


def _analyze(input_grid):
    a = _grid(input_grid)
    corners = {int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])}
    _need(len(corners) == 1, "the four corners must show background")
    background = corners.pop()
    interpretations = []
    for turn in range(4):
        view = np.rot90(a, turn)
        try:
            info = _interpret_view(view, background)
        except ValueError:
            continue
        info["turn"] = turn
        info["output"] = np.rot90(info["expected"], -turn).copy()
        interpretations.append(info)
    _need(len(interpretations) == 1, "input must have one staging/destination interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every output pixel solely from this input; do not mutate it."""
    return _analyze(input_grid)["output"]


def _draw_tree(rng, height, width, mixed):
    a = np.zeros((height, width), dtype=bool)
    row = int(rng.choice([0, height-1])) if mixed else int(rng.integers(height))
    a[row] = True
    primary = int(rng.choice([0, width-1]))
    a[:, primary] = True
    choices = [c for c in range(width) if abs(c-primary) >= 3]
    for _ in range(int(rng.integers(3))):
        if not choices:
            break
        col = int(rng.choice(choices))
        lo, hi = int(rng.integers(row+1)), int(rng.integers(row, height))
        a[lo:hi+1, col] = True
        choices = [c for c in choices if abs(c-col) >= 3]
    return a


def _damage_options(clean):
    options = []
    for r, c in np.argwhere(clean):
        r, c = int(r), int(c)
        for dr, dc in ((0, 1), (1, 0)):
            if not all(_at(clean, r+k*dr, c+k*dc) for k in range(5)):
                continue
            if not (_at(clean, r-dr, c-dc) or _at(clean, r+5*dr, c+5*dc)):
                continue
            gap = tuple((r+k*dr, c+k*dc) for k in (1, 2, 3))
            if any(_at(clean, rr+dc, cc+dr) or _at(clean, rr-dc, cc-dr) for rr, cc in gap):
                continue
            for endpoint, toward in ((0, 1), (4, -1)):
                ar, ac = r+endpoint*dr, c+endpoint*dc
                for side in (-1, 1):
                    sr, sc = toward*dr+side*dc, toward*dc+side*dr
                    spur = tuple((ar+k*sr, ac+k*sc) for k in (1, 2))
                    # Give deviations a two-cell margin outside the clean box.
                    options.append((gap, spur, "v" if dr else "h"))
    return options


def _damaged_tree(rng, height, width, links, mixed=False, required_axis=None):
    # Padding is part of the staging allocation, not part of the cargo mask.
    for _ in range(60):
        body = _draw_tree(rng, height, width, mixed or required_axis == "v")
        clean = np.pad(body, 2)
        options = _damage_options(clean)
        order = rng.permutation(len(options))
        damaged, selected = clean.copy(), []
        for i in order:
            gap, spur, axis = options[int(i)]
            if required_axis is not None and axis != required_axis:
                continue
            if mixed and selected and axis == selected[0][2]:
                continue
            if not all(damaged[p] for p in gap):
                continue
            if not all(0 <= r < clean.shape[0] and 0 <= c < clean.shape[1]
                       and not clean[r, c] and not damaged[r, c] for r, c in spur):
                continue
            trial = damaged.copy()
            for p in gap:
                trial[p] = False
            for p in spur:
                trial[p] = True
            try:
                recovered, added, _, records = _repair(trial, trial)
            except ValueError:
                continue
            if np.array_equal(recovered, clean) and len(records) == len(selected)+1:
                damaged = trial
                selected.append((gap, spur, axis))
            if len(selected) == links:
                # Repair really couples fragments with different downward extents.
                pieces = _components(clean & damaged)
                if len({max(r for r, c in part) for part in pieces}) >= 2:
                    return damaged, clean
                break
    raise ValueError("no suitable damaged tree in bounded attempts")


def _palette_roles(rng, palette, count):
    background, wire = palette[:2]
    routes = list(map(int, rng.permutation(palette[2:2+count])))
    extras = list(palette[2+count:])
    rails = [int(rng.choice(extras + [c for c in routes if c != route]))
             for route in routes]
    return int(background), int(wire), routes, rails


def _attempt_pair(rng, palette, turn, lesson, links, forced_main_dest=None):
    count = 2 if lesson in ("routing", "independence") else int(rng.choice([1, 2, 2, 2, 3]))
    contrast = lesson == "independence" or count == 3 or rng.random() < .4
    mixed = links == 2 and count <= 2 and rng.random() < .55
    required_axis = "h" if lesson == "routing" else "v" if lesson == "independence" else None
    stage_h = int(rng.integers(10, 13))
    gaps = [int(rng.integers(1, 3)) for _ in range(count)]
    depths = [3]*count
    main_dest = int(rng.integers(count)) if forced_main_dest is None else forced_main_dest
    if mixed or required_axis == "v":
        depths[main_dest] = 6
    budget = 30-stage_h-sum(gaps)-1
    if sum(depths) > budget:
        return None
    for i in rng.integers(count, size=int(rng.integers(budget-sum(depths)+1))):
        depths[int(i)] += 1
    left_margin, right_margin, separation = int(rng.integers(1, 3)), int(rng.integers(1, 3)), int(rng.integers(2, 4))
    if contrast:
        lw, rw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        other_w = lw+3+rw
    else:
        other_w = int(rng.integers(6, 11))
    max_main_w = min(16, 30-left_margin-right_margin-separation-other_w)
    min_main_w = 11 if links == 2 and not mixed else 6
    if max_main_w < min_main_w:
        return None
    main_w = int(rng.integers(min_main_w, max_main_w+1))
    main_max_h = min(depths[main_dest], stage_h-4)
    main_min_h = 6 if mixed or required_axis == "v" else 3
    if main_max_h < main_min_h:
        return None
    main_h = int(rng.integers(main_min_h, main_max_h+1))
    try:
        damaged, clean = _damaged_tree(rng, main_h, main_w, links, mixed, required_axis)
    except ValueError:
        return None
    width = left_margin+main_w+separation+other_w+right_margin
    bg, wire, routes, rails = _palette_roles(rng, palette, count)
    height = stage_h+sum(gaps)+sum(depths)+1
    inp = np.full((height, width), bg, dtype=np.int8)
    bands, cursor = [], stage_h
    for depth, gap, route, rail in zip(depths, gaps, routes, rails):
        cursor += gap
        # Independent inset rails vary usable spans while keeping all cargo inside.
        left = int(rng.integers(left_margin))
        right = width-1-int(rng.integers(right_margin))
        band = dict(top=cursor, bottom=cursor+depth-1, left=left, right=right,
                    route=route, rail=rail)
        bands.append(band)
        inp[cursor:cursor+depth, left:right+1] = route
        inp[cursor:cursor+depth, [left, right]] = rail
        cursor += depth
    clean_stage = np.zeros((stage_h, width), dtype=bool)
    assemblies = []

    def place(local_in, local_clean, top, col, destination):
        source_cells = [(top+int(r), col+int(c)) for r, c in np.argwhere(local_in)]
        cells = tuple((top+int(r), col+int(c)) for r, c in np.argwhere(local_clean))
        if not all(0 <= r < stage_h and 0 < c < width-1 for r, c in source_cells):
            raise ValueError("source outside its staging allocation")
        if any(inp[p] != bg for p in source_cells):
            raise ValueError("source collision")
        for p in source_cells:
            inp[p] = wire
        for p in cells:
            clean_stage[p] = True
        tag = cells[int(rng.integers(len(cells)))]
        inp[tag] = routes[destination]
        assemblies.append((cells, destination))

    try:
        # Padded tree can reach an outer source margin with a deviation. Reject
        # rather than crop, clip, or wrap the evidence.
        top = int(rng.integers(stage_h-clean.shape[0]+1))
        place(damaged, clean, top, left_margin-2, main_dest)
        other_col = left_margin+main_w+separation
        other_dests = [i for i in range(count) if i != main_dest] or [main_dest]
        if contrast:
            d1 = other_dests[0]
            d2 = other_dests[-1]
            maxh = min(depths[d1], depths[d2], stage_h-2)
            h1, h2 = int(rng.integers(2, maxh+1)), int(rng.integers(2, maxh+1))
            if lesson == "independence" and h1 == h2:
                h2 = 2 if h1 != 2 else 3
            y = int(rng.integers(stage_h-max(h1, h2)))
            # A deliberate three-cell break between two separately tagged Ls.
            for ww, hh, x, dest, outer in ((lw, h1, other_col, d1, 0),
                                           (rw, h2, other_col+lw+3, d2, rw-1)):
                shape = np.zeros((hh, ww), bool)
                shape[0] = True
                shape[:, outer] = True
                place(shape, shape, y, x, dest)
        else:
            dest = other_dests[0]
            hh = int(rng.integers(2, min(depths[dest], stage_h-2)+1))
            shape = _draw_tree(rng, hh, other_w, False)
            y = int(rng.integers(stage_h-hh))
            place(shape, shape, y, other_col, dest)
    except ValueError:
        return None
    # Optional intact assemblies occupy genuinely available source/target cells;
    # they can share lateral ranges with larger cargo when their masks allow it.
    target_busy = {(r+bands[d]["bottom"]-max(y for y, x in cells), c)
                   for cells, d in assemblies for r, c in cells}
    for _ in range(int(rng.integers(3))):
        for attempt in range(35):
            dest = int(rng.integers(count))
            hh = int(rng.integers(2, min(depths[dest], 4)+1))
            ww = int(rng.integers(2, 6))
            kind = int(rng.integers(4))
            shape = np.zeros((hh, ww), bool)
            if kind == 0:
                shape = np.ones((1, ww), bool)
            elif kind == 1:
                shape = np.ones((hh, 1), bool)
            elif kind == 2:
                shape[:, 0] = shape[-1] = True
            else:
                shape[0] = True
                shape[:, ww//2] = True
            hh, ww = shape.shape
            top, col = int(rng.integers(stage_h-hh+1)), int(rng.integers(1, width-ww))
            points = [(top+int(r), col+int(c)) for r, c in np.argwhere(shape)]
            if any(inp[r+dr, c+dc] != bg for r, c in points
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                   if 0 <= r+dr < stage_h and 0 <= c+dc < width):
                continue
            b = bands[dest]
            bottom = max(r for r, c in points)
            target = {(r+b["bottom"]-bottom, c) for r, c in points}
            if target & target_busy or not all(b["left"] < c < b["right"] for r, c in target):
                continue
            place(shape, shape, top, col, dest)
            target_busy.update(target)
            break
    # Add detached diagonal clutter only when the complete input still encodes
    # exactly the intended clean assemblies. The loop retains no failed payloads.
    noise_goal = int(rng.integers(1, 4))
    for _ in range(noise_goal):
        for attempt in range(30):
            length = int(rng.choice([2, 3]))
            r, c = int(rng.integers(stage_h)), int(rng.integers(1, width-1))
            dc = int(rng.choice([-1, 1]))
            pts = [(r+k, c+k*dc) for k in range(length)]
            if not all(0 <= rr < stage_h and 0 < cc < width-1 and inp[rr, cc] == bg
                       and not clean_stage[rr, cc] for rr, cc in pts):
                continue
            trial = inp[:stage_h].copy()
            for p in pts:
                trial[p] = wire
            try:
                rebuilt = _repair(trial != bg, trial == wire)[0]
            except ValueError:
                continue
            if np.array_equal(rebuilt, clean_stage):
                inp[:stage_h] = trial
                break
    # Ground truth uses intended clean masks and chosen destinations, never solve.
    out = inp.copy()
    out[:stage_h] = bg
    occupied = set()
    for cells, dest in assemblies:
        b = bands[dest]
        shift = b["bottom"]-max(r for r, c in cells)
        for r, c in cells:
            p = (r+shift, c)
            if p in occupied or not (b["top"] <= p[0] <= b["bottom"] and b["left"] < p[1] < b["right"]):
                return None
            occupied.add(p)
            out[p] = b["rail"]
    if rng.integers(2):
        inp, out = inp[:, ::-1], out[:, ::-1]
    pair = {"input": np.rot90(inp, turn).copy(), "output": np.rot90(out, turn).copy()}
    try:
        a = _analyze(pair["input"])
        if not np.array_equal(a["output"], pair["output"]):
            return None
        stats = _features(a)
        if stats["max_links"] != links or not stats["coupled_depths"]:
            return None
        if lesson == "independence" and not (stats["independent_same_tag"] and stats["open_gaps"]):
            return None
    except ValueError:
        return None
    return pair


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, t = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    fixed_palette = list(map(int, rng.permutation(10)))
    varied_palette = bool(rng.integers(2))
    turns = list(map(int, rng.permutation(4)))
    varied_orientation = bool(rng.integers(2))
    result, seen = {"train": [], "test": []}, set()
    first_main_dest = None
    for split, count in (("train", n), ("test", t)):
        for index in range(count):
            lesson = ("routing" if index == 0 else "independence" if index == 1 else None) if split == "train" else None
            turn = turns[index % 2] if varied_orientation else turns[0]
            palette = list(map(int, rng.permutation(10))) if varied_palette else fixed_palette
            forced = 1-first_main_dest if lesson == "independence" else None
            for _ in range(150):
                pair = _attempt_pair(rng, palette, turn, lesson, 1 if split == "train" else 2, forced)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(pair["input"]))
            result[split].append(pair)
            if lesson == "routing":
                a = _analyze(pair["input"])
                obj = next(o for o in a["objects"] if any(a["added"][p] for p in o["cells"]))
                first_main_dest = a["bands"].index(obj["band"])
    validate_generated_puzzle(result)
    return result


def _features(a):
    depths, link_counts, signatures = [], [], []
    coupled = False
    for obj in a["objects"]:
        cells = obj["cells"]
        depths.append((obj["tag"], max(r for r, c in cells)))
        added_count = sum(bool(a["added"][p]) for p in cells)
        link_counts.append(added_count//3)
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        signatures.append(tuple((r-r0, c-c0) for r, c in cells))
        fragments = np.zeros_like(a["clean"])
        for p in cells:
            fragments[p] = not a["added"][p]
        pieces = _components(fragments)
        coupled |= len({max(r for r, c in part) for part in pieces}) > 1
    return dict(shape=tuple(a["canonical"].shape), bands=len(a["bands"]),
                assemblies=len(a["objects"]), links=len(a["repairs"]), max_links=max(link_counts),
                axes="".join(sorted({x[2] for x in a["repairs"]})),
                open_gaps=sum(1 for _ in _gaps(a["clean"])),
                removed=int(a["removed"].sum()), coupled_depths=bool(coupled),
                independent_same_tag=any(x[0] == y[0] and x[1] != y[1]
                                         for x, y in itertools.combinations(depths, 2)),
                direction=a["turn"], background=a["background"], wire=a["wire"],
                depths=tuple(b["bottom"]-b["top"]+1 for b in a["bands"]),
                reused_rails=sum(b["rail"] in {c["route"] for c in a["bands"]} for b in a["bands"]),
                signatures=tuple(signatures))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _need(np.array_equal(out, a["output"]), "incorrect reconstructed routing output")
            _need(not np.array_equal(inp, out), "identity pair")
            # Exact occupied-cell conservation through repair, tagging and motion.
            stamp_count = sum(len(obj["cells"]) for obj in a["objects"])
            changed_destination = np.count_nonzero(a["expected"][a["first"]:] != a["canonical"][a["first"]:])
            _need(stamp_count == changed_destination, "lost, collided, or invisible stamp cells")
            _need(np.all(a["expected"][:a["first"]] == a["background"]), "uncleared staging")
            for b in a["bands"]:
                _need(np.all(a["expected"][b["top"]:b["bottom"]+1, [b["left"], b["right"]]] == b["rail"]),
                      "modified destination rails")
            info = _features(a)
            _need(info["coupled_depths"], "repair must affect rigid grouping and alignment")
            infos[split].append(info)
    _need(any(x["bands"] >= 2 for x in infos["train"]), "missing color routing demonstration")
    _need(any(x["independent_same_tag"] and x["open_gaps"] for x in infos["train"]),
          "missing unsupported-gap and independent-alignment lesson")
    _need(all(x["max_links"] == 1 for x in infos["train"]), "training curriculum is single-link")
    _need(all(x["max_links"] == 2 for x in infos["test"]), "test must compose two links in one assembly")
    main_destinations = []
    for pair in puzzle["train"][:2]:
        a = _analyze(pair["input"])
        obj = next(o for o in a["objects"] if any(a["added"][p] for p in o["cells"]))
        main_destinations.append(a["bands"].index(obj["band"]))
    _need(len(set(main_destinations)) == 2, "first lessons must change the repaired cargo destination")
    _need({infos["train"][0]["axes"], infos["train"][1]["axes"]} == {"h", "v"},
          "minimum training set must teach both canonical repair axes")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _ablation(a, mode):
    """Explicit wrong rules for bounded teaching checks; never used for labels."""
    out = a["canonical"].copy()
    out[:a["first"]] = a["background"]
    objects = a["objects"]
    if mode in ("no_repair", "fill_all_gaps"):
        objects = []
        mask = a["clean"] & ~a["added"] if mode == "no_repair" else a["clean"].copy()
        if mode == "fill_all_gaps":
            for r, c, dr, dc in list(_gaps(mask)):
                for k in (1, 2, 3):
                    mask[r+k*dr, c+k*dc] = True
        for cells in _components(mask):
            tags = [p for p in cells if a["canonical"][p] not in (a["background"], a["wire"])]
            if tags:
                tag = int(a["canonical"][tags[0]])
                band = next(b for b in a["bands"] if b["route"] == tag)
                objects.append(dict(cells=cells, tag=tag, band=band))
    for obj in objects:
        cells = obj["cells"]
        if mode == "tags_only":
            cells = [obj["tag_cell"]]
        bottom = max(r for r, c in cells)
        if mode == "one_shift_per_tag":
            bottom = max(r for o in objects if o["tag"] == obj["tag"] for r, c in o["cells"])
        for r, c in cells:
            out[r+obj["band"]["bottom"]-bottom, c] = obj["band"]["rail"]
    return out


def _regression_fixture():
    """Hand-specified repair, unsupported gap, tag propagation and docking."""
    inp = np.zeros((22, 24), dtype=np.int8)
    inp[11:15, 1:23], inp[11:15, [0, 23]] = 1, 2
    inp[17:21, 1:23], inp[17:21, [0, 23]] = 3, 4
    inp[2, 2:11], inp[2:6, 10] = 5, 5
    inp[2, 4:7] = 0
    inp[3, 4] = inp[4, 5] = 5
    inp[2, 2] = 1
    inp[4, 14:17], inp[4:6, 14] = 5, 5
    inp[4, 20:23], inp[4:7, 22] = 5, 5
    inp[4, 15] = inp[5, 22] = 3
    inp[7, 5] = inp[8, 6] = inp[9, 7] = 5
    out = inp.copy()
    out[:11] = 0
    out[11, 2:11], out[11:15, 10] = 2, 2
    out[19, 14:17], out[19:21, 14] = 4, 4
    out[18, 20:23], out[18:21, 22] = 4, 4
    return inp, out


def self_test():
    """Bounded 42-seed semantic/diversity checks, plus counts and negative cases.

    Retains aggregate features and hashes, not generated puzzle payloads. Returns
    a compact-JSON-compatible report for optional external recording.
    """
    records, hashes, masks = [], set(), set()
    ablations = dict(no_repair=0, tags_only=0, one_shift_per_tag=0, fill_all_gaps=0)
    inp, out = _regression_fixture()
    for turn in range(4):
        for reflect in (False, True):
            x, y = (inp[:, ::-1], out[:, ::-1]) if reflect else (inp, out)
            assert np.array_equal(solve(np.rot90(x, turn)), np.rot90(y, turn))
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        hashes.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"])
                a = _analyze(original)
                for mode in ablations:
                    ablations[mode] += not np.array_equal(_ablation(a, mode), a["expected"])
                info = _features(a)
                masks.update(info.pop("signatures"))
                records.append((split, info))
        lessons = [_analyze(x["input"]) for x in puzzle["train"][:2]]
        main_destinations = []
        for a in lessons:
            obj = next(o for o in a["objects"] if any(a["added"][p] for p in o["cells"]))
            main_destinations.append(a["bands"].index(obj["band"]))
        assert len(set(main_destinations)) == 2
        assert not np.array_equal(_ablation(lessons[1], "one_shift_per_tag"), lessons[1]["expected"])
        if seed == 0:
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
            # Global rotations and arbitrary palette permutations preserve semantics.
            pair = puzzle["test"][0]
            for turn in range(4):
                assert np.array_equal(solve(np.rot90(pair["input"], turn)), np.rot90(pair["output"], turn))
            recolor = np.array([7, 0, 6, 3, 9, 2, 1, 8, 5, 4], np.int8)
            assert np.array_equal(solve(recolor[pair["input"]]), recolor[pair["output"]])
            # Whole-grid validation must catch one omitted cell of a repair stamp.
            wrong = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            a = _analyze(wrong["test"][0]["input"])
            obj = next(o for o in a["objects"] if any(a["added"][p] for p in o["cells"]))
            p = next(p for p in obj["cells"] if a["added"][p])
            q = (p[0]+obj["band"]["bottom"]-max(r for r, c in obj["cells"]), p[1])
            canonical_bad = np.rot90(wrong["test"][0]["output"], a["turn"]).copy()
            canonical_bad[q] = obj["band"]["route"]
            wrong["test"][0]["output"] = np.rot90(canonical_bad, -a["turn"]).copy()
            try:
                validate_generated_puzzle(wrong)
            except ValueError:
                pass
            else:
                raise AssertionError("omitted repair stamp accepted")
    for n in range(2, 11):
        for t in range(1, min(4, n-1)+1):
            p = generate_puzzle(1234+n*10+t, n, t)
            assert (len(p["train"]), len(p["test"])) == (n, t)
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    all_infos = [x for _, x in records]
    assert len(hashes) == 42 and len(masks) >= 70
    assert len({x["shape"] for x in all_infos}) >= 30
    assert {x["bands"] for x in all_infos} == {1, 2, 3}
    assert {x["assemblies"] for x in all_infos} >= {2, 3}
    assert {x["direction"] for x in all_infos} == {0, 1, 2, 3}
    assert {x["background"] for x in all_infos} == set(range(10))
    assert {x["wire"] for x in all_infos} == set(range(10))
    assert any(x["reused_rails"] for x in all_infos)
    assert any(not x["reused_rails"] for x in all_infos)
    assert any(x["axes"] == "hv" for s, x in records if s == "test")
    assert any(x["axes"] == "v" for s, x in records if s == "train")
    assert ablations["no_repair"] == len(records) and ablations["tags_only"] == len(records)
    assert ablations["one_shift_per_tag"] >= 42
    assert ablations["fill_all_gaps"] >= 42
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(hashes),
                shapes=len({x["shape"] for x in all_infos}), assembly_masks=len(masks),
                band_counts=sorted({x["bands"] for x in all_infos}),
                assembly_counts=sorted({x["assemblies"] for x in all_infos}),
                repair_axes=sorted({x["axes"] for x in all_infos}),
                max_links_train=max(x["max_links"] for s, x in records if s == "train"),
                max_links_test=max(x["max_links"] for s, x in records if s == "test"),
                ablation_disagreements=ablations,
                semantic_verification="construction and input-only generator-owned checks")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
