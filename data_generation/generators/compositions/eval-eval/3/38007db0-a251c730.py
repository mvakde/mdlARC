"""Replay the edit in each exceptional reference tile at keyed target anchors.

The reference is a lattice; the target is an undivided framed workspace.
Only changed reference cells are instructions. A change to reference fill
means erase to target fill, rather than copy the reference's literal fill.
No files, parent imports, global random state, or saved traces are used.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, todo = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    todo.append(p)
        groups.append(group)
    return groups


def _crop(grid, box):
    r, c, h, w = box
    return grid[r:r+h, c:c+w]


def _border(a, color):
    return bool(np.all(a[[0, -1], :] == color) and np.all(a[:, [0, -1]] == color))


def _panels(grid):
    """Find full frame components, allowing a lattice inside the reference."""
    exterior = int(grid[0, 0])
    _require(_border(grid, exterior), "one uniform exterior margin is required")
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {exterior}):
        for group in _components(grid == color):
            rr, cc = zip(*group)
            r, c = int(min(rr)), int(min(cc))
            h, w = int(max(rr)-r+1), int(max(cc)-c+1)
            if h < 5 or w < 5 or not (r and c and r+h < len(grid) and c+w < grid.shape[1]):
                continue
            box = (r, c, h, w)
            a = _crop(grid, box)
            if not _border(a, color):
                continue
            rows = np.flatnonzero(np.all(a == color, axis=1)).tolist()
            cols = np.flatnonzero(np.all(a == color, axis=0)).tolist()
            lines = np.zeros(a.shape, dtype=bool)
            lines[rows, :] = True
            lines[:, cols] = True
            if len(group) != int(lines.sum()) or not np.array_equal(a == color, lines):
                continue
            if len(rows) == len(cols) == 2:
                candidates.append(dict(role="target", box=box, frame=color))
            elif len(rows) >= 2 and len(cols) >= 4:
                if len(set(np.diff(rows))) != 1 or len(set(np.diff(cols))) != 1:
                    continue
                candidates.append(dict(role="reference", box=box, frame=color,
                                       rows=rows, cols=cols))
    return exterior, candidates


def _interpret(grid, exterior, ref, target):
    _require(ref["frame"] == target["frame"], "frames must have the same role color")
    mask = np.zeros(grid.shape, dtype=np.int8)
    for p in (ref, target):
        r, c, h, w = p["box"]
        mask[r:r+h, c:c+w] += 1
    _require(bool(np.all(mask <= 1) and np.all(grid[mask == 0] == exterior)), "panels must be disjoint on a plain canvas")
    a = _crop(grid, ref["box"])
    rows, cols = ref["rows"], ref["cols"]
    ph, pw = rows[1]-1, cols[1]-1
    _require(4 <= ph <= 7 and 4 <= pw <= 7, "reference interiors must have a fill rim and a small core")
    _require(1 <= len(rows)-1 <= 3 and 3 <= len(cols)-1 <= 5, "unsupported reference geometry")
    templates, all_inks, fills, columns = {}, set(), set(), []
    common_anchor = None
    for r0, r1 in zip(rows, rows[1:]):
        tiles = [a[r0+1:r1, c0+1:c1] for c0, c1 in zip(cols, cols[1:])]
        counts = Counter(t.tobytes() for t in tiles)
        _require(sorted(counts.values()) == [1, len(tiles)-1], "each band needs a strict repeated master and one exception")
        master_bytes = max(counts, key=counts.get)
        master = next(t for t in tiles if t.tobytes() == master_bytes)
        col = next(i for i, t in enumerate(tiles) if t.tobytes() != master_bytes)
        variant = tiles[col]
        fill = int(master[0, 0])
        fills.add(fill)
        _require(all(_border(t, fill) for t in tiles), "all reference tiles need the same fill rim")
        colors, frequencies = np.unique(master, return_counts=True)
        singletons = [int(c) for c, n in zip(colors, frequencies) if n == 1 and c != fill]
        _require(len(singletons) == 1, "one singleton registration key is required per master")
        key = singletons[0]
        _require(key not in templates, "reference bands need different keys")
        ar, ac = map(int, np.argwhere(master == key)[0])
        anchor = ar-1, ac-1
        _require(common_anchor in (None, anchor), "all bands share the same registration point")
        common_anchor = anchor
        _require(np.count_nonzero(variant == key) == 1 and variant[ar, ac] == key, "exception must preserve its key")
        inks = set(map(int, colors)) - {fill, key}
        _require(1 <= len(inks) <= 2, "one or two decoration colors are allowed")
        _require(set(map(int, np.unique(variant))) <= inks | {fill, key}, "exception introduces an unknown role")
        edits = [(int(r-ar), int(c-ac), int(master[r, c]), int(variant[r, c]))
                 for r, c in np.argwhere(master != variant)]
        _require(bool(edits) and len(edits) <= 6, "one to six cells must change")
        _require(all((dr, dc) != (0, 0) for dr, dc, _, _ in edits), "key cannot be edited")
        unchanged = master == variant
        _require(bool(np.any(unchanged[1:-1, 1:-1] & (master[1:-1, 1:-1] == fill))), "retain an unchanged blank core cell")
        _require(bool(np.any(unchanged & np.isin(master, list(inks)))), "retain an unchanged ink cell")
        templates[key] = dict(master=master.copy(), variant=variant.copy(), edits=edits,
                              anchor=(ar, ac), column=col, inks=inks)
        all_inks |= inks
        columns.append(col)
    _require(len(fills) == 1, "one reference fill per scene")
    rf = next(iter(fills))
    keys = set(templates)
    _require(not (keys & all_inks) and len(all_inks) <= 2, "keys and decoration have distinct roles")
    canvas = _crop(grid, target["box"])
    ti = canvas[1:-1, 1:-1]
    tf = int(ti[0, 0])
    _require(_border(ti, tf), "target needs an unmarked fill rim")
    _require(len({exterior, ref["frame"], rf}) == 3 and tf not in (exterior, ref["frame"]), "structural colors are distinct")
    _require(not ((keys | all_inks) & {exterior, ref["frame"], rf, tf}), "payload colors cannot hide in structural roles")
    _require(set(map(int, np.unique(ti))) <= keys | all_inks | {tf}, "unknown target contents")
    _require(np.count_nonzero(ti == tf) > ti.size/2, "target fill must be dominant")
    expected = canvas.copy()
    occupied = np.zeros(ti.shape, dtype=bool)
    anchors, operations, active_old, active_new = [], [], 0, 0
    ch, cw = ph-2, pw-2
    for ar, ac in np.argwhere(np.isin(ti, list(keys))):
        ar, ac = int(ar), int(ac)
        key = int(ti[ar, ac])
        t = templates[key]
        mr, mc = t["anchor"]
        r0, c0 = ar-(mr-1), ac-(mc-1)
        _require(1 <= r0 and 1 <= c0 and r0+ch < ti.shape[0] and c0+cw < ti.shape[1], "target patch envelope must fit inside its fill rim")
        _require(not np.any(occupied[max(0, r0-1):r0+ch+1, max(0, c0-1):c0+cw+1]), "target envelopes need a one-cell gap")
        occupied[r0:r0+ch, c0:c0+cw] = True
        _require(np.count_nonzero(np.isin(ti[r0:r0+ch, c0:c0+cw], list(keys))) == 1, "one key per target envelope")
        anchors.append((ar, ac, key))
        for dr, dc, old, new in t["edits"]:
            old_value, new_value = (tf if old == rf else old), (tf if new == rf else new)
            r, c = ar+dr, ac+dc
            active_old += int(ti[r, c] == old_value)
            active_new += int(ti[r, c] == new_value)
            expected[r+1, c+1] = new_value
            operations.append((r, c, old_value, new_value))
    _require(1 <= len(anchors) <= 6, "one to six target keys are required")
    _require(bool(np.all((ti == tf) | occupied)), "target marks must belong to registered envelopes")
    _require(not np.array_equal(expected, canvas), "at least one edit must remain to apply")
    return expected, dict(reference=ref, target=target, rf=rf, tf=tf, exterior=exterior,
                          templates=templates, anchors=anchors, operations=operations,
                          columns=tuple(columns), core=(ch, cw), active_old=active_old,
                          active_new=active_new, before=canvas.copy())


def _analyze(input_grid):
    grid = _grid(input_grid)
    exterior, panels = _panels(grid)
    interpretations = []
    for ref in panels:
        if ref["role"] != "reference":
            continue
        for target in panels:
            if target["role"] != "target":
                continue
            try:
                interpretations.append(_interpret(grid, exterior, ref, target))
            except ValueError:
                pass
    _require(len(interpretations) == 1, "scene must have exactly one reference/target interpretation")
    return interpretations[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the rule's entire dictionary and registration from this input."""
    return _analyze(input_grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _mask(rng, h, w, anchor, ink_count):
    cells = {anchor}
    while len(cells) < ink_count+1:
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _D4
                           if 0 <= r+dr < h and 0 <= c+dc < w} - cells)
        cells.add(_pick(rng, frontier))
    return sorted(cells - {anchor})


def _make_templates(rng, h, w, ntypes, inks, keys, fill, witness):
    anchor = _pick(rng, [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]) if witness else (int(rng.integers(h)), int(rng.integers(w)))
    templates = []
    for i, key in enumerate(keys):
        count = int(rng.integers(2*len(inks), h*w-3))
        painted = _mask(rng, h, w, anchor, count)
        blank = sorted(set(np.ndindex(h, w)) - set(painted) - {anchor})
        core = np.full((h, w), fill, dtype=np.int8)
        assignments = list(inks)*2 + [_pick(rng, inks) for _ in range(count-2*len(inks))]
        rng.shuffle(assignments)
        for p, color in zip(painted, assignments):
            core[p] = color
        core[anchor] = key
        variant = core.copy()
        if witness and i == 0:
            mode = "exchange"
        elif witness and i == 1:
            mode = "recolor"
        else:
            mode = _pick(rng, ["add", "remove", "exchange", "mixed"] + (["recolor"] if len(inks) == 2 else []))
        exchange_count = (1 if witness else int(rng.integers(1, min(len(blank)-1, len(painted)-1, 3)+1))) if mode == "exchange" else None
        if mode in ("add", "exchange", "mixed"):
            maximum = min(len(blank)-1, 6 if mode == "add" else 3)
            n = exchange_count if mode == "exchange" else int(rng.integers(1, maximum+1))
            for j in rng.choice(len(blank), n, replace=False):
                variant[blank[int(j)]] = _pick(rng, inks)
        if mode in ("remove", "exchange", "mixed"):
            maximum = min(len(painted)-1, 6 if mode == "remove" else 3)
            n = exchange_count if mode == "exchange" else int(rng.integers(1, maximum+1))
            for j in rng.choice(len(painted), n, replace=False):
                variant[painted[int(j)]] = fill
        if mode == "recolor":
            for j in rng.choice(len(painted), int(rng.integers(1, min(6, len(painted)-1)+1)), replace=False):
                p = painted[int(j)]
                variant[p] = next(c for c in inks if c != int(core[p]))
        if witness and i == 0:
            # A corner origin and offsets spanning both axes make orientation
            # observable. Preserve every color count while moving one cell.
            choices = [(p, q) for p in painted for q in blank
                       if (p[0] != anchor[0] or q[0] != anchor[0])
                       and (p[1] != anchor[1] or q[1] != anchor[1])]
            removed, added = _pick(rng, choices)
            variant = core.copy()
            variant[removed] = fill
            variant[added] = core[removed]
        master = np.full((h+2, w+2), fill, dtype=np.int8)
        odd = master.copy()
        master[1:-1, 1:-1], odd[1:-1, 1:-1] = core, variant
        edits = [(int(r), int(c), int(core[r, c]), int(variant[r, c]))
                 for r, c in np.argwhere(core != variant)]
        templates.append(dict(master=master, variant=odd, core=core, edits=edits, anchor=anchor))
    return templates


def _geometry_options(ntypes, total, witness):
    options = []
    for h in range(2 if not witness else 3, 6):
        for w in range(2 if not witness else 3, 6):
            if h*w < (9 if witness else 6):
                continue
            for copies in range(3, 6):
                rh, rw = ntypes*(h+3)+1, copies*(w+3)+1
                if max(rh, rw) > 28:
                    continue
                for across in range(1, total+1):
                    down = (total+across-1)//across
                    th, tw = down*(h+1)+3, across*(w+1)+3
                    for axis in (0, 1):
                        sizes = ((rh, rw), (th, tw))
                        extent = [max(rh, th)+2, max(rw, tw)+2]
                        extent[axis] = sizes[0][axis]+sizes[1][axis]+3
                        if max(extent) <= 30:
                            options.append((h, w, copies, across, axis, rh, rw, th, tw))
    return options


def _generate_pair(rng, ntypes, total, palette, witness=False):
    h, w, copies, across, axis, rh, rw, th, tw = _pick(rng, _geometry_options(ntypes, total, witness))
    exterior, frame, rf, other_fill = map(int, palette[:4])
    tf = other_fill if witness or rng.random() < .65 else rf
    nink = 2 if witness or (h*w >= 9 and rng.random() < .5) else 1
    inks = list(map(int, palette[4:4+nink]))
    keys = list(map(int, palette[6:6+ntypes]))
    templates = _make_templates(rng, h, w, ntypes, inks, keys, rf, witness)
    # Keep the first witness's two exceptions in different columns, to rule
    # out a fixed column. Other bands draw independently, including repeats.
    columns = list(map(int, rng.integers(copies, size=ntypes)))
    if witness and columns[0] == columns[1]:
        columns[1] = (columns[0]+int(rng.integers(1, copies))) % copies
    reference = np.full((rh, rw), frame, dtype=np.int8)
    for b, t in enumerate(templates):
        for c in range(copies):
            r0, c0 = 1+b*(h+3), 1+c*(w+3)
            reference[r0:r0+h+2, c0:c0+w+2] = t["variant"] if c == columns[b] else t["master"]
    target = np.full((th, tw), frame, dtype=np.int8)
    target[1:-1, 1:-1] = tf
    output = target.copy()
    ids = list(range(ntypes)) + list(map(int, rng.integers(ntypes, size=total-ntypes)))
    if witness:
        ids = [0, 0, 1]
    rng.shuffle(ids)
    slots = [(2+(j//across)*(h+1), 2+(j%across)*(w+1)) for j in range(total)]
    rng.shuffle(slots)
    first_for_key = set()
    for type_id, (r0, c0) in zip(ids, slots):
        t = templates[type_id]
        master = t["core"]
        core = master.copy()
        core[core == rf] = tf
        # Target context deliberately differs at cells outside the edit:
        # one old blank gains ink and one unchanged old ink becomes blank.
        changed = {(r, c) for r, c, _, _ in t["edits"]}
        inactive_blank = [p for p in np.ndindex(h, w) if master[p] == rf and p not in changed]
        inactive_ink = [p for p in np.ndindex(h, w) if int(master[p]) in inks and p not in changed]
        private_ink, private_blank = _pick(rng, inactive_blank), _pick(rng, inactive_ink)
        core[private_ink] = _pick(rng, inks)
        core[private_blank] = tf
        # Unchanged context is editable independently; no target baseline is
        # assumed by the solver. Preserve enough structure for visual clarity.
        for p in np.ndindex(h, w):
            if p not in changed and p not in (t["anchor"], private_ink, private_blank) and rng.random() < .12:
                core[p] = _pick(rng, [tf]+inks)
        if type_id not in first_for_key:
            first_for_key.add(type_id)
        else:
            # Idempotent assignments: an edit can already be present.
            r, c, _, new = t["edits"][int(rng.integers(len(t["edits"])))]
            core[r, c] = tf if new == rf else new
        target[r0:r0+h, c0:c0+w] = core
        finished = core.copy()
        for r, c, _, new in t["edits"]:
            finished[r, c] = tf if new == rf else new
        output[r0:r0+h, c0:c0+w] = finished
    # Add small independent padding and offsets where the ARC bound permits.
    shapes = [(rh, rw), (th, tw)]
    minimum = [max(rh, th)+2, max(rw, tw)+2]
    minimum[axis] = shapes[0][axis]+shapes[1][axis]+3
    shape = [int(rng.integers(n, min(30, n+3)+1)) for n in minimum]
    gap = int(rng.integers(1, min(3, shape[axis]-shapes[0][axis]-shapes[1][axis]-2)+1))
    order = [0, 1] if rng.random() < .5 else [1, 0]
    positions = [[0, 0], [0, 0]]
    start = int(rng.integers(1, shape[axis]-shapes[0][axis]-shapes[1][axis]-gap))
    positions[order[0]][axis] = start
    positions[order[1]][axis] = start+shapes[order[0]][axis]+gap
    for i in (0, 1):
        other = 1-axis
        positions[i][other] = int(rng.integers(1, shape[other]-shapes[i][other]))
    grid = np.full(shape, exterior, dtype=np.int8)
    for data, position in zip((reference, target), positions):
        r, c = position
        grid[r:r+data.shape[0], c:c+data.shape[1]] = data
    return {"input": grid, "output": output}


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("counts must be integers: train 2..10 and test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    base_palette = rng.permutation(10)
    stable_palette = rng.random() < .25
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            witness = split == "train" and index == 0
            ntypes = 2 if witness else (3 if split == "test" else int(rng.integers(1, 3)))
            total = 3 if witness else (int(rng.integers(5, 7)) if split == "test" else int(rng.integers(ntypes, 5)))
            for _ in range(32):
                palette = base_palette if stable_palette else rng.permutation(10)
                pair = _generate_pair(rng, ntypes, total, palette, witness)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _whole_template(info):
    """Tempting B-only shortcut: transparently stamp the whole selected motif."""
    out = info["before"].copy()
    for ar, ac, key in info["anchors"]:
        t = info["templates"][key]
        mr, mc = t["anchor"]
        for r, c in np.argwhere(t["variant"] != info["rf"]):
            out[ar+int(r)-mr+1, ac+int(c)-mc+1] = t["variant"][r, c]
    return out


def _unkeyed(info):
    """Tempting A-only shortcut: use the first band's patch for every key."""
    out = info["before"].copy()
    t = next(iter(info["templates"].values()))
    for ar, ac, _ in info["anchors"]:
        for dr, dc, _, new in t["edits"]:
            out[ar+dr+1, ac+dc+1] = info["tf"] if new == info["rf"] else new
    return out


def _fixed_column(info):
    """Omit A's odd-one-out search: read the first tile in each band."""
    out = info["before"].copy()
    for ar, ac, key in info["anchors"]:
        t = info["templates"][key]
        if t["column"] == 0:
            for dr, dc, _, new in t["edits"]:
                out[ar+dr+1, ac+dc+1] = info["tf"] if new == info["rf"] else new
    return out


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            predicted, info = _analyze(inp)
            _require(np.array_equal(predicted, out), "incorrect differential edit or target crop")
            _require(np.array_equal(_crop(inp, info["target"]["box"])[[0, -1]], out[[0, -1]]) and
                     np.array_equal(info["before"][:, [0, -1]], out[:, [0, -1]]), "changed frame")
            # Pixel accounting checks that the entire complement of the edit
            # support is unchanged, including private target context and keys.
            support = np.zeros(out.shape, dtype=bool)
            for r, c, _, new in info["operations"]:
                support[r+1, c+1] = True
                _require(int(out[r+1, c+1]) == new, "wrong assignment")
            _require(np.array_equal(out[~support], info["before"][~support]), "transparent cells were overwritten")
            if split == "train":
                train_info.append((out, info))
    changes = [(old, new, info["rf"]) for _, info in train_info
               for t in info["templates"].values() for _, _, old, new in t["edits"]]
    _require(any(old == rf and new != rf for old, new, rf in changes), "training needs a paint instruction")
    _require(any(new == rf and old != rf for old, new, rf in changes), "training needs an erase instruction")
    _require(any(old != rf and new != rf for old, new, rf in changes), "training needs a recolor instruction")
    _require(any(info["active_new"] and info["active_old"] for _, info in train_info), "show an already-applied edit beside a pending edit")
    _require(any(len(set(info["columns"])) > 1 for _, info in train_info), "show different exception columns")
    _require(any(info["rf"] != info["tf"] for _, info in train_info), "show local-fill erasure")
    _require(any(Counter(t["master"].flat) == Counter(t["variant"].flat)
                 for _, info in train_info for t in info["templates"].values()), "show a histogram-preserving exception")
    _require(any(any(dr for dr, _, _, _ in t["edits"]) and any(dc for _, dc, _, _ in t["edits"])
                 for _, info in train_info for t in info["templates"].values()), "show row and column offsets")
    _require(any(not np.array_equal(out, _fixed_column(info)) for out, info in train_info), "show why fixed-column selection fails")
    _require(any(not np.array_equal(out, _whole_template(info)) for out, info in train_info), "show why whole-template copying fails")
    _require(any(len(info["templates"]) > 1 and not np.array_equal(out, _unkeyed(info))
                 for out, info in train_info), "show why ignoring keys fails")


def self_test():
    """Bounded 42-seed audit; keeps only small sets of feature signatures."""
    shapes, cores, placements, geometries, palettes, template_hashes = set(), set(), set(), set(), set(), set()
    operations, key_counts, target_counts, layouts, hashes = set(), set(), set(), set(), set()
    aligned = balanced = idempotent = equal_fill = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, info = _analyze(pair["input"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
                for role in ("input", "output"):
                    a = pair[role]
                    shapes.add((role, a.shape))
                    digest.update(bytes(a.shape)+a.tobytes())
                ref, tar = info["reference"]["box"], info["target"]["box"]
                layouts.add("vertical" if ref[0]+ref[2] <= tar[0] or tar[0]+tar[2] <= ref[0] else "horizontal")
                cores.add(info["core"])
                placements.add(info["columns"])
                geometries.add((info["core"], len(info["templates"]), len(info["reference"]["cols"])-1, out.shape))
                palettes.add((info["exterior"], info["reference"]["frame"], info["rf"], info["tf"]))
                key_counts.add(len(info["templates"]))
                target_counts.add(len(info["anchors"]))
                aligned += int(len(info["columns"]) > 1 and len(set(info["columns"])) == 1)
                idempotent += int(info["active_new"] > 0)
                equal_fill += int(info["rf"] == info["tf"])
                for t in info["templates"].values():
                    sig = []
                    add = remove = 0
                    for dr, dc, old, new in t["edits"]:
                        op = "paint" if old == info["rf"] else "erase" if new == info["rf"] else "recolor"
                        operations.add(op)
                        sig.append((dr, dc, op))
                        add += int(op == "paint")
                        remove += int(op == "erase")
                    balanced += int(add == remove > 0)
                    template_hashes.add(tuple(sig))
        hashes.add(digest.digest())
    assert len(hashes) == 42 and len(shapes) >= 35 and len(geometries) >= 60
    assert len(cores) >= 8 and len(placements) >= 15 and len(palettes) >= 40
    assert len(template_hashes) >= 100 and key_counts == {1, 2, 3}
    assert target_counts >= {1, 2, 3, 4, 5, 6} and layouts == {"vertical", "horizontal"}
    assert operations == {"paint", "erase", "recolor"} and min(aligned, balanced, idempotent, equal_fill) > 0
    first, second = generate_puzzle(947, 7, 3), generate_puzzle(947, 7, 3)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s]) for k in p)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Family-specific mutations: broken output, missing exceptional tile,
    # damaged key, and interpreting erasure as literal reference fill.
    base = generate_puzzle(0, 2, 1)
    _, info = _analyze(base["train"][0]["input"])
    for defect in ("output", "missing", "key", "erase"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = bad["train"][0]
        if defect == "output":
            pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
        elif defect == "missing":
            ref = info["reference"]
            r, c, _, _ = ref["box"]
            t = next(iter(info["templates"].values()))
            ph, pw = t["master"].shape
            c += 1+t["column"]*(pw+1)
            pair["input"][r+1:r+1+ph, c:c+pw] = t["master"]
        elif defect == "key":
            r, c, _, _ = info["target"]["box"]
            ar, ac, _ = info["anchors"][0]
            pair["input"][r+1+ar, c+1+ac] = info["reference"]["frame"]
        else:
            for r, c, _, new in info["operations"]:
                if new == info["tf"]:
                    pair["output"][r+1, c+1] = info["rf"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted "+defect)
    return {"seeds": 42, "distinct_puzzles": len(hashes), "grid_shapes": len(shapes),
            "core_shapes": len(cores), "geometries": len(geometries),
            "normalized_edit_patterns": len(template_hashes), "role_palettes": len(palettes),
            "aligned_exception_examples": aligned, "balanced_bands": balanced,
            "already_applied_examples": idempotent, "equal_fill_examples": equal_fill}
