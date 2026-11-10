"""Port-weighted elections on overlapping, corner-defined rectangular fields.

All generation is in memory, with bounded retries and no file I/O. Local labels
are strict majorities. Spatial ties use the highest ballot card; the highest
supporter of the winning signal supplies the visible corner marker, if any.
"""
from __future__ import annotations

from itertools import combinations
import hashlib

import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer from 1 through 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grid must be a 1..30 by 1..30 integer array of colors 0..9")
    return a


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _cells(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _bounds(cells):
    return (min(p[0] for p in cells), min(p[1] for p in cells),
            max(p[0] for p in cells), max(p[1] for p in cells))


def _ring(box):
    r0, c0, r1, c1 = box
    return ({(r, c) for r in (r0, r1) for c in range(c0, c1 + 1)}
            | {(r, c) for c in (c0, c1) for r in range(r0, r1 + 1)})


def _outside_ports(box):
    r0, c0, r1, c1 = box
    return ([(r0 - 1, c) for c in range(c0, c1 + 1)]
            + [(r1 + 1, c) for c in range(c0, c1 + 1)]
            + [(r, c0 - 1) for r in range(r0, r1 + 1)]
            + [(r, c1 + 1) for r in range(r0, r1 + 1)])


def _neighbors(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in _DIRS}


def _inside(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _marker_cells(box, diagonal):
    r0, c0, r1, c1 = box
    if diagonal == 0:
        return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
                (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}
    return {(r0, c1), (r0, c1 - 1), (r0 + 1, c1),
            (r1, c0), (r1, c0 + 1), (r1 - 1, c0)}


def _recognize_markers(coords):
    """Pattern recognition is separate from the construction marker helper."""
    if len(coords) != 6:
        raise ValueError("a field needs six marker cells")
    box = _bounds(coords)
    r0, c0, r1, c1 = box
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if min(h, w) < 2 or (min(h, w) == 2 and max(h, w) < 4):
        raise ValueError("invalid thin field")
    local = {(r - r0, c - c0) for r, c in coords}
    patterns = (
        {(0, 0), (0, 1), (1, 0), (h-1, w-1), (h-1, w-2), (h-2, w-1)},
        {(0, w-1), (0, w-2), (1, w-1), (h-1, 0), (h-1, 1), (h-2, 0)},
    )
    for diagonal, pattern in enumerate(patterns):
        if local == pattern:
            return box, diagonal
    raise ValueError("cue cells are not opposite L fragments")


def _parse_roles(a, bg, signals):
    signal_mask = (a == signals[0]) | (a == signals[1])
    panels = _components(map(tuple, np.argwhere(signal_mask)))
    if not 2 <= len(panels) <= 4:
        raise ValueError("need two to four cards")
    cards, occupied = [], set()
    for panel in panels:
        pr0, pc0, pr1, pc1 = _bounds(panel)
        if (min(pr1 - pr0 + 1, pc1 - pc0 + 1) < 2
                or len(panel) != (pr1 - pr0 + 1) * (pc1 - pc0 + 1)):
            raise ValueError("panel must be rectangular")
        frame_box = (pr0 - 1, pc0 - 1, pr1 + 1, pc1 + 1)
        if not (0 < pr0 and 0 < pc0 and pr1 < a.shape[0]-1 and pc1 < a.shape[1]-1):
            raise ValueError("missing frame")
        frame = _ring(frame_box)
        frame_colors = {int(a[p]) for p in frame}
        if len(frame_colors) != 1:
            raise ValueError("frame is not monochrome")
        cue = frame_colors.pop()
        if cue in (bg, *signals) or any(o["cue"] == cue for o in cards):
            raise ValueError("cue roles must be distinct")
        n0 = sum(int(a[p]) == signals[0] for p in panel)
        n1 = len(panel) - n0
        if min(n0, n1) == 0 or n0 == n1:
            raise ValueError("local votes must be mixed and strict")
        cue_cells = set(map(tuple, np.argwhere(a == cue)))
        components = _components(cue_cells)
        containing = [c for c in components if c & frame]
        if len(containing) != 1 or not frame <= containing[0]:
            raise ValueError("disconnected frame")
        circuit = containing[0]
        wires = circuit - frame
        if wires & panel or any(_inside(p, frame_box) for p in wires):
            raise ValueError("wire enters card")
        ports = set(_outside_ports(frame_box)) & wires
        if not 1 <= len(ports) <= 4:
            raise ValueError("one to four occupied ports required")
        markers = cue_cells - circuit
        box, diagonal = _recognize_markers(markers)
        card_cells = panel | circuit
        if occupied & card_cells:
            raise ValueError("overlapping cards")
        occupied |= card_cells
        cards.append(dict(cue=cue, panel=panel, frame=frame, frame_box=frame_box,
                          wires=wires, ports=ports, weight=len(ports),
                          majority=signals[int(n1 > n0)], counts=(n0, n1),
                          box=box, markers=markers, diagonal=diagonal))
    expected_colors = {bg, *signals, *(o["cue"] for o in cards)}
    if set(map(int, np.unique(a))) != expected_colors:
        raise ValueError("unexplained color")
    if len({o["frame_box"][0] for o in cards}) != len(cards):
        raise ValueError("card priority is tied")
    for card in cards:
        field = _cells(card["box"])
        if field & occupied:
            raise ValueError("field covers a card or wire")
        if any(not (0 < r < a.shape[0]-1 and 0 < c < a.shape[1]-1) for r, c in field):
            raise ValueError("field reaches protected border")
        for other in cards:
            if card is not other:
                own = card["panel"] | card["frame"] | card["wires"]
                theirs = other["panel"] | other["frame"] | other["wires"]
                if own & _neighbors(theirs):
                    raise ValueError("cards must be separated")
    cards.sort(key=lambda o: o["frame_box"][0])
    return dict(background=bg, signals=signals, cards=cards)


def _parse(input_grid):
    a = _grid(input_grid)
    bg = int(a[0, 0])
    if (np.any(a[0] != bg) or np.any(a[-1] != bg)
            or np.any(a[:, 0] != bg) or np.any(a[:, -1] != bg)):
        raise ValueError("outer border must be background")
    interpretations = []
    values = [int(c) for c in np.unique(a) if c != bg]
    for signals in combinations(values, 2):
        try:
            interpretations.append(_parse_roles(a, bg, signals))
        except ValueError:
            continue
    if len(interpretations) != 1:
        raise ValueError("need a unique card and signal interpretation")
    return interpretations[0]


def _predict(a, info, *, equal_weights=False, reverse=False, minority=False):
    """Input-derived, cell-by-cell electorate reconstruction; no generation data."""
    cards = list(reversed(info["cards"])) if reverse else info["cards"]
    signals = info["signals"]
    label = {o["cue"]: (signals[1-signals.index(o["majority"])]
                         if minority else o["majority"]) for o in cards}
    out = a.copy()
    for o in cards:
        for p in o["wires"]:
            out[p] = label[o["cue"]]
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = r, c
            covering = [o for o in cards if _inside(p, o["box"])]
            if not covering:
                continue
            totals = {s: sum((1 if equal_weights else o["weight"])
                              for o in covering if label[o["cue"]] == s)
                      for s in signals}
            if totals[signals[0]] == totals[signals[1]]:
                winner = label[covering[0]["cue"]]
            else:
                winner = max(signals, key=totals.get)
            leader = next(o for o in covering if label[o["cue"]] == winner)
            out[p] = leader["cue"] if p in leader["markers"] else winner
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    return _predict(a, _parse(a))


def _compose_records(a, records, signals):
    """Construction labels use simultaneous array masks, not the solver traversal."""
    records = sorted(records, key=lambda o: o["frame_box"][0])
    masks = []
    totals = np.zeros((2, *a.shape), dtype=int)
    out = a.copy()
    for o in records:
        mask = np.zeros(a.shape, dtype=bool)
        r0, c0, r1, c1 = o["box"]
        mask[r0:r1+1, c0:c1+1] = True
        masks.append(mask)
        totals[signals.index(o["majority"])] += mask * o["weight"]
        for p in o["wires"]:
            out[p] = o["majority"]
    covered = np.logical_or.reduce(masks)
    winning = np.where(totals[0] > totals[1], signals[0], signals[1])
    tied = covered & (totals[0] == totals[1])
    unresolved = tied.copy()
    for o, mask in zip(records, masks):
        winning[unresolved & mask] = o["majority"]
        unresolved &= ~mask
    out[covered] = winning[covered]
    represented = np.zeros(a.shape, dtype=bool)
    for o, mask in zip(records, masks):
        active = mask & (winning == o["majority"]) & ~represented
        for p in o["markers"]:
            if active[p]:
                out[p] = o["cue"]
        represented |= active
    return out


def _geometry(rng, shape, n, kind, large):
    h, w = shape
    side_lo, side_hi = (15, 19) if large else (8, 13)
    sh = int(rng.integers(side_lo, min(side_hi, h-4)+1))
    sw = int(rng.integers(side_lo, min(side_hi, w-4)+1))
    sr, sc = int(rng.integers(1, h-sh)), int(rng.integers(1, w-sw))
    objects, used = [], set()
    center = sh // 2, sw // 2
    if kind == "chain":
        # Three/four horizontal intervals, neighboring overlap only. Staggered
        # vertical bounds yield several distinct local electorates.
        step = max(2, (sw-5)//(n-1))
        width = step + 1
        for i in range(n):
            c0 = i * step
            c1 = min(sw-1, c0+width)
            r0 = int(rng.integers(0, max(1, sh//3)))
            r1 = int(rng.integers(max(r0+3, 2*sh//3), sh))
            box = sr+r0, sc+c0, sr+r1, sc+c1
            markers = _marker_cells(box, int(rng.integers(2)))
            if markers & used:
                return None
            objects.append(dict(box=box, markers=markers))
            used |= markers
        return objects
    for i in range(n):
        for _ in range(80):
            if kind in ("common", "nested"):
                # Every field contains a common interior point, with freely
                # varying extents; not every field shares the same center.
                r0 = int(rng.integers(0, center[0]))
                c0 = int(rng.integers(0, center[1]))
                r1 = int(rng.integers(center[0]+1, sh))
                c1 = int(rng.integers(center[1]+1, sw))
                if kind == "nested" and i == 0:
                    r0, c0, r1, c1 = 0, 0, sh-1, sw-1
            elif kind == "cross":
                if i % 2:
                    hh, ww = int(rng.integers(2, 4)), int(rng.integers(max(5, sw//2), sw+1))
                else:
                    hh, ww = int(rng.integers(max(5, sh//2), sh+1)), int(rng.integers(2, 4))
                r0, c0 = int(rng.integers(sh-hh+1)), int(rng.integers(sw-ww+1))
                r1, c1 = r0+hh-1, c0+ww-1
            else:
                hh, ww = int(rng.integers(3, sh+1)), int(rng.integers(3, sw+1))
                r0, c0 = int(rng.integers(sh-hh+1)), int(rng.integers(sw-ww+1))
                r1, c1 = r0+hh-1, c0+ww-1
            box = sr+r0, sc+c0, sr+r1, sc+c1
            markers = _marker_cells(box, int(rng.integers(2)))
            if markers & used or any(o["box"] == box for o in objects):
                continue
            objects.append(dict(box=box, markers=markers))
            used |= markers
            break
        else:
            return None
    if kind == "partial":
        a, b = map(lambda o: _cells(o["box"]), objects[:2])
        if len(a & b) < 4 or not a-b or not b-a:
            return None
    return objects


def _wire_system(rng, box, weight, blocked, length):
    h, w = blocked.shape
    all_ports = set(_outside_ports(box))
    available = sorted(p for p in all_ports
                       if 0 < p[0] < h-1 and 0 < p[1] < w-1 and not blocked[p])
    if len(available) < weight:
        return None
    for _ in range(35):
        ports = {available[int(i)] for i in rng.choice(len(available), weight, replace=False)}
        wires = set()
        for start in sorted(ports):
            path, p = {start}, start
            for _ in range(length-1):
                options = []
                for dr, dc in _DIRS:
                    q = p[0]+dr, p[1]+dc
                    if (0 < q[0] < h-1 and 0 < q[1] < w-1 and not blocked[q]
                            and q not in path and q not in all_ports and not _inside(q, box)):
                        options.append(q)
                if not options:
                    break
                p = options[int(rng.integers(len(options)))]
                path.add(p)
            if len(path) < length:
                break
            wires |= path
        else:
            if wires & all_ports == ports:
                return wires
    return None


def _place_cards(rng, shape, objects, sizes, weights, lengths):
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    blocked[0] = blocked[-1] = True
    blocked[:, 0] = blocked[:, -1] = True
    for o in objects:
        r0, c0, r1, c1 = o["box"]
        blocked[max(0,r0-1):min(h,r1+2), max(0,c0-1):min(w,c1+2)] = True
    placed = [None] * len(objects)
    order = list(map(int, rng.permutation(len(objects))))
    for i in order:
        ph, pw = sizes[i]
        fh, fw = ph+2, pw+2
        for _ in range(220):
            r = int(rng.integers(1, h-fh))
            c = int(rng.integers(1, w-fw))
            box = r, c, r+fh-1, c+fw-1
            if any(o is not None and o["frame_box"][0] == r for o in placed):
                continue
            if blocked[r:r+fh, c:c+fw].any():
                continue
            wires = _wire_system(rng, box, weights[i], blocked, lengths[i])
            if wires is None:
                continue
            panel = _cells((r+1, c+1, r+ph, c+pw))
            frame = _ring(box)
            occupied = panel | frame | wires
            for p in occupied | _neighbors(occupied):
                if 0 <= p[0] < h and 0 <= p[1] < w:
                    blocked[p] = True
            placed[i] = dict(objects[i], panel=panel, frame=frame, frame_box=box,
                             wires=wires, weight=weights[i])
            break
        else:
            return None
    return placed


def _panel_pattern(rng, size, majority, minority, strength):
    area = size[0] * size[1]
    if strength == "close":
        count = (area-1)//2
    elif strength == "strong":
        count = 1
    else:
        count = int(rng.integers(1, (area-1)//2+1))
    order_kind = int(rng.integers(3))
    if order_kind == 0:
        order = rng.permutation(area)
    elif order_kind == 1:
        rr, cc = np.indices(size)
        order = np.argsort(((rr+cc) % 2 * area + rr*size[1]+cc).ravel())
    else:
        order = np.arange(area) if rng.random() < .5 else np.arange(area)[::-1]
    flat = np.full(area, majority, dtype=np.int8)
    flat[order[:count]] = minority
    return flat.reshape(size)


def _measure(a, info, out):
    cards, signals = info["cards"], info["signals"]
    marks = set().union(*(o["markers"] for o in cards))
    flags = set()
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = r, c
            covers = [o for o in cards if _inside(p, o["box"])]
            if not covers or p in marks:
                continue
            flags.add("completion")
            if len(covers) < 2:
                continue
            signed = lambda o: 1 if o["majority"] == signals[0] else -1
            vote = sum(signed(o)*o["weight"] for o in covers)
            unweighted = sum(signed(o) for o in covers)
            raw = sum(o["counts"][0]-o["counts"][1] for o in covers)
            wire = sum(signed(o)*len(o["wires"]) for o in covers)
            flags.add("overlap")
            if vote * unweighted < 0:
                flags.add("panel_override")
            if vote * raw < 0:
                flags.add("raw_override")
            if vote * wire < 0:
                flags.add("wire_override")
            if vote * signed(covers[0]) < 0:
                flags.add("front_override")
            strongest = max(covers, key=lambda o: o["weight"])
            if vote * signed(strongest) < 0:
                flags.add("max_weight_override")
            if vote == 0 and len({o["majority"] for o in covers}) == 2:
                flags.add("ordered_tie")
    hidden = sum(int(out[p]) != o["cue"] for o in cards for p in o["markers"])
    retained = sum(int(out[p]) == o["cue"] for o in cards for p in o["markers"])
    if hidden:
        flags.add("occlusion")
    if retained:
        flags.add("retention")
    if any(set(map(int, out[o["box"][0]:o["box"][2]+1,
                                o["box"][1]:o["box"][3]+1].flat)) >= set(signals) for o in cards):
        flags.add("changing_coalition")
    return dict(flags=flags, shape=a.shape, field_count=len(cards),
                weights=tuple(o["weight"] for o in cards),
                field_shapes=tuple((o["box"][2]-o["box"][0]+1,
                                    o["box"][3]-o["box"][1]+1) for o in cards),
                panel_areas=tuple(len(o["panel"]) for o in cards),
                card_positions=tuple(o["frame_box"][:2] for o in cards),
                diagonals=tuple(o["diagonal"] for o in cards),
                wire_cells=sum(len(o["wires"]) for o in cards),
                palette=(info["background"], *signals, *(o["cue"] for o in cards)))


def _candidate(rng, roles, n, kind, witness, large=False, compact=False):
    lo = 25 if n == 4 else (23 if n == 3 else 20)
    hi = 27 if compact and n < 4 else 30
    if large:
        lo = max(lo, 27)
    shape = tuple(map(int, rng.integers(lo, hi+1, size=2)))
    objects = _geometry(rng, shape, n, kind, large)
    if objects is None:
        return None
    if witness == "weighted":
        weights = [int(rng.integers(3, 5)), 1, 1]
        sizes = [(2, int(rng.integers(2, 4)))] + [
            (int(rng.integers(3, 5)), int(rng.integers(3, 6))) for _ in range(2)]
        lengths = [1, int(rng.integers(4, 7)), int(rng.integers(4, 7))]
    elif witness == "tie":
        weight = int(rng.integers(2, 4))
        weights = [weight, weight, int(rng.integers(1, weight))]
        sizes = [(int(rng.integers(2, 4)), int(rng.integers(3, 5))) for _ in range(n)]
        lengths = list(map(int, rng.integers(1, 5, size=n)))
    else:
        weights = list(map(int, rng.integers(1, 5, size=n)))
        sizes = [(int(rng.integers(2, 5)), int(rng.integers(2, 6))) for _ in range(n)]
        lengths = list(map(int, rng.integers(1, 5, size=n)))
    cards = _place_cards(rng, shape, objects, sizes, weights, lengths)
    if cards is None:
        return None
    ranked = sorted(range(n), key=lambda i: cards[i]["frame_box"][0])
    if witness == "weighted" and ranked[0] == 0:
        return None
    bg, s0, s1 = roles[:3]
    signals = tuple(sorted((s0, s1)))
    labels = list(map(int, rng.integers(2, size=n)))
    if witness == "weighted":
        labels = [0, 1, 1]
    elif witness == "tie":
        first = next(i for i in ranked if i in (0, 1))
        labels = [1, 1, 1]
        labels[first] = 0
    elif len(set(labels)) == 1:
        labels[int(rng.integers(n))] ^= 1
    # The semantic winner is randomized even in deliberately contrasting cases.
    if rng.random() < .5:
        labels = [1-x for x in labels]
    a = np.full(shape, bg, dtype=np.int8)
    for i, card in enumerate(cards):
        cue = int(roles[3+i])
        majority = signals[labels[i]]
        minority = signals[1-labels[i]]
        strength = "random"
        if witness == "weighted":
            strength = "close" if i == 0 else "strong"
        elif witness == "tie":
            strength = "close" if labels[i] == labels[first] else "strong"
        pattern = _panel_pattern(rng, sizes[i], majority, minority, strength)
        r0, c0, _, _ = card["frame_box"]
        a[r0+1:r0+1+sizes[i][0], c0+1:c0+1+sizes[i][1]] = pattern
        for p in card["frame"] | card["wires"] | card["markers"]:
            a[p] = cue
        card.update(cue=cue, majority=majority)
    out = _compose_records(a, cards, signals)
    try:
        info = _parse(a)
    except ValueError:
        return None
    m = _measure(a, info, out)
    needed = ({"panel_override", "raw_override", "wire_override", "front_override"}
              if witness == "weighted" else
              {"ordered_tie", "occlusion", "retention", "changing_coalition", "max_weight_override"}
              if witness == "tie" else {"completion"})
    if not needed <= m["flags"]:
        return None
    return dict(input=a, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))
    stable_palette = bool(rng.integers(2))
    palette = list(map(int, rng.choice(10, 7, replace=False)))
    slots = list(map(int, rng.permutation(num_train)))
    obligations = {slots[0]: "weighted", slots[1]: "tie"}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            witness = obligations.get(i, "") if split == "train" else ""
            n = 3 if witness else (
                4 if split == "test" and mode == 0 else
                3 if split == "test" and mode == 1 else int(rng.integers(2, 4)))
            kind = "common" if witness else (
                "chain" if split == "test" and mode == 1 else
                "nested" if split == "test" and mode == 2 else
                str(rng.choice(["common", "partial", "nested", "cross", "free"])))
            large = split == "test" and mode == 2
            compact = split == "train" and mode == 2
            for _ in range(180):
                roles = palette if stable_palette else list(map(int, rng.choice(10, 3+n, replace=False)))
                pair = _candidate(rng, roles, n, kind, witness, large, compact)
                if pair is None:
                    continue
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded card/field packing exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_flags = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, out = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == out.shape
            key = a.shape, a.astype(np.int8).tobytes()
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _parse(a)
            expected = _predict(a, info)
            assert np.array_equal(expected, out), "incorrect spatial election"
            assert not np.array_equal(a, out)
            mutable = set()
            for o in info["cards"]:
                mutable |= _cells(o["box"]) | o["wires"]
                assert all(out[p] == a[p] for p in o["panel"] | o["frame"])
                assert all(out[p] == o["majority"] for p in o["wires"])
                assert all(component & set(_outside_ports(o["frame_box"]))
                           for component in _components(o["wires"]))
            assert all(out[r, c] == a[r, c] for r in range(a.shape[0])
                       for c in range(a.shape[1]) if (r, c) not in mutable)
            m = _measure(a, info, out)
            if split == "train":
                train_flags |= m["flags"]
    required = {"completion", "overlap", "panel_override", "raw_override",
                "wire_override", "front_override", "ordered_tie", "occlusion",
                "retention", "changing_coalition", "max_weight_override"}
    assert required <= train_flags, "missing visible teaching evidence"


def self_test():
    """42 in-memory puzzles; aggregate hashes/sets only, no saved payloads/traces."""
    shapes, fields, weights, positions, palettes, structures, puzzle_hashes = (set() for _ in range(7))
    counts = {2: 0, 3: 0, 4: 0}
    flags, diagonals, field_shapes = set(), set(), set()
    stable, varying, thin, pair_count = 0, 0, 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        per_palettes = set()
        for split in ("train", "test"):
            for pair in p[split]:
                a, out = pair["input"], pair["output"]
                info = _parse(a)
                assert np.array_equal(solve(a), out)
                m = _measure(a, info, out)
                flags |= m["flags"]
                shapes.add(m["shape"])
                fields.add(m["field_count"])
                counts[m["field_count"]] += 1
                weights.update(m["weights"])
                positions.update(m["card_positions"])
                palettes.add(m["palette"])
                per_palettes.add(m["palette"][:3])
                diagonals.update(m["diagonals"])
                field_shapes.update(m["field_shapes"])
                thin += any(min(s) == 2 for s in m["field_shapes"])
                normalized, mapping = [], {}
                for v in a.flat:
                    normalized.append(mapping.setdefault(int(v), len(mapping)))
                structures.add((a.shape, hashlib.sha256(bytes(normalized)).digest()))
                digest.update(str(a.shape).encode())
                digest.update(a.tobytes())
                digest.update(out.tobytes())
                pair_count += 1
        stable += len(per_palettes) == 1
        varying += len(per_palettes) > 1
        puzzle_hashes.add(digest.hexdigest())
    assert len(puzzle_hashes) == 42 and len(structures) == pair_count
    assert fields == {2, 3, 4} and weights == {1, 2, 3, 4}
    assert len(shapes) >= 40 and len(positions) >= 100 and len(field_shapes) >= 35
    assert diagonals == {0, 1} and thin >= 4
    assert stable >= 3 and varying >= 3 and len(palettes) >= 50
    for tr, te in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(730, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(71), generate_puzzle(71)
    assert all(np.array_equal(x[k], y[k]) for s in p
               for x, y in zip(p[s], q[s]) for k in x)
    # A palette bijection changes no geometric convention or priority.
    perm = np.random.default_rng(617).permutation(10)
    a, out = p["test"][0]["input"], p["test"][0]["output"]
    assert np.array_equal(solve(perm[a]), perm[out])
    # Local arrangement does not matter: only the within-card counts vote.
    changed, expected = a.copy(), out.copy()
    info = _parse(a)
    for card in info["cards"]:
        coords = sorted(card["panel"])
        values = sorted(int(a[c]) for c in coords)
        for c, v in zip(coords, values):
            changed[c] = expected[c] = v
    assert np.array_equal(solve(changed), expected)
    # Distinguish the inherited mechanisms on ordinary scene cells.
    assert any(not np.array_equal(pair["output"], _predict(pair["input"],
               _parse(pair["input"]), equal_weights=True)) for pair in p["train"])
    assert any(not np.array_equal(pair["output"], _predict(pair["input"],
               _parse(pair["input"]), reverse=True)) for pair in p["train"])
    assert not np.array_equal(out, _predict(a, info, minority=True))
    mutated = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    card = _parse(mutated["test"][0]["input"])["cards"][0]
    pos = min(card["wires"])
    mutated["test"][0]["output"][pos] = card["cue"]
    try:
        validate_generated_puzzle(mutated)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(puzzle_hashes),
                shapes=len(shapes), color_normalized_structures=len(structures),
                field_counts=counts, port_weights=sorted(weights),
                field_shapes=len(field_shapes), card_positions=len(positions),
                palettes=len(palettes), stable_palette_puzzles=stable,
                varying_palette_puzzles=varying, thin_field_pairs=thin,
                semantic_witnesses=sorted(flags), independent_verification=False)
