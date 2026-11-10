"""Unary run printing followed by independent clockwise color-layer rotation.

The square is a filled sheet on an exterior background. Its majority color is
paper; other colors are source layers. A detached matching bar activates run
synthesis, replacing that source layer. An absent bar preserves its source mask.
No parent modules, files, seeds, or stored examples are consulted by solve().
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= d <= 30 for d in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or np.any(array < 0) or np.any(array > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return array.astype(np.int8, copy=False)


def _key(array):
    array = np.asarray(array, dtype=np.int8)
    return array.shape, array.tobytes()


def _components(mask):
    """Eight-connected components; a blank halo separates every exterior bar."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        result.append(np.asarray(component, dtype=int))
    return result


def _read_input(input_grid):
    """Recover the square, backgrounds, masks, and counters from visible cells."""
    inp = _grid(input_grid)
    h, w = inp.shape
    exterior = int(inp[0, 0])
    if any(int(inp[r, c]) != exterior for r, c in ((0, w-1), (h-1, 0), (h-1, w-1))):
        raise ValueError("all four corners must show the exterior background")
    components = _components(inp != exterior)
    sheets = []
    for index, coords in enumerate(components):
        r, c = coords.min(axis=0)
        e, f = coords.max(axis=0) + 1
        n = int(e-r)
        if n >= 5 and f-c == n and len(coords) == n*n:
            sheets.append((index, int(r), int(c), n))
    if len(sheets) != 1:
        raise ValueError("expected exactly one filled square sheet of size at least 5")
    sheet_index, r, c, n = sheets[0]
    if not (0 < r and r+n < h and 0 < c and c+n < w):
        raise ValueError("sheet must have an exterior margin on every side")
    source = inp[r:r+n, c:c+n]
    colors, amounts = np.unique(source, return_counts=True)
    paper = int(colors[amounts.argmax()])
    if int(amounts.max()) * 2 <= n*n or len(colors) < 2:
        raise ValueError("paper must occupy strictly more than half of the sheet")
    palette = tuple(int(v) for v in colors if v != paper)
    bars = {}
    for index, coords in enumerate(components):
        if index == sheet_index:
            continue
        ys, xs = coords.T
        values = np.unique(inp[ys, xs])
        horizontal = int(np.ptp(ys)) == 0 and int(np.ptp(xs)) + 1 == len(coords)
        vertical = int(np.ptp(xs)) == 0 and int(np.ptp(ys)) + 1 == len(coords)
        if len(values) != 1 or not (horizontal or vertical):
            raise ValueError("exterior components must be monochromatic solid straight bars")
        color = int(values[0])
        if color not in palette or color in bars:
            raise ValueError("each bar must uniquely match a content color")
        length = len(coords)
        if length*length + 2*length - 2 > n*n:
            raise ValueError("run expansion does not fit inside the square")
        bars[color] = {"length": length, "coords": coords,
                       "orientation": "single" if length == 1 else ("horizontal" if horizontal else "vertical")}
    if not 1 <= len(bars) <= 4:
        raise ValueError("expected one through four marked layers")
    # Eight-connected components already separate bars and the sheet by at
    # least one exterior cell, including diagonal contact.
    return {"input": inp, "source": source, "exterior": exterior, "paper": paper,
            "palette": palette, "bars": bars, "size": n, "row": r, "col": c}


def _coordinate_output(info, *, expansion=True, rotation=True, direction=1,
                       square_width=None, skip_unmarked=False):
    """Input-only arithmetic implementation, also used for explicit ablations.

    Position is continuous across row boundaries: separators never restart at a
    row edge. The generation constructor below instead builds a flat bit array.
    """
    n, paper = info["size"], info["paper"]
    result = np.full((n, n), paper, dtype=np.int8)
    occupied = np.zeros((n, n), dtype=bool)
    for color in info["palette"]:
        marker = info["bars"].get(color)
        if marker is None:
            if skip_unmarked:
                continue
            coords = map(tuple, np.argwhere(info["source"] == color).tolist())
            turns = 0
        else:
            count = marker["length"]
            turns = direction * count % 4 if rotation else 0
            if expansion:
                width = n if square_width is None else square_width
                coords_list, position = [], 0
                for j in range(2*count-1):
                    run = count - abs(count-1-j)
                    for _ in range(run):
                        y, x = divmod(position, width)
                        if y >= n or x >= n:
                            raise ValueError("serialized layer leaves the sheet")
                        coords_list.append((y, x))
                        position += 1
                    if j < 2*count-2:
                        position += 1
                coords = coords_list
            else:
                coords = map(tuple, np.argwhere(info["source"] == color).tolist())
        for y, x in coords:
            for _ in range(turns):
                y, x = x, n-1-y
            if occupied[y, x]:
                raise ValueError("different transformed color layers overlap")
            result[y, x] = color
            occupied[y, x] = True
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any well-formed family member, using only the input grid."""
    return _coordinate_output(_read_input(input_grid))


def _run_mask(count, size):
    """A's run encoding, padded to the full sheet before B's rotation."""
    runs = list(range(1, count+1)) + list(range(count-1, 0, -1))
    sequence = []
    for i, length in enumerate(runs):
        if i:
            sequence.append(False)
        sequence.extend([True] * length)
    if len(sequence) > size*size:
        raise ValueError("run sequence exceeds the sheet")
    flat = np.zeros(size*size, dtype=bool)
    flat[:len(sequence)] = sequence
    return flat.reshape(size, size)


def _printed_mask(count, size):
    return np.rot90(_run_mask(count, size), -count % 4)


def _compatible(counts, size):
    used = np.zeros((size, size), dtype=bool)
    for count in counts:
        if count*count + 2*count - 2 > size*size:
            return False
        mask = _printed_mask(count, size)
        if np.any(used & mask):
            return False
        used |= mask
    return True


def _programs(rng, allowed, *, required=None, focal=None, excluded=()):
    """Choose compatible counts by geometry, without a color priority."""
    if required is not None:
        widths = [n for n in range(5, 25) if n not in excluded and _compatible(required, n)]
        if not widths:
            raise ValueError("no compatible teaching width")
        return int(rng.choice(widths)), list(required)
    if focal is None:
        focal = int(rng.choice(allowed))
    widths = [n for n in range(5, 25) if n not in excluded and focal*focal+2*focal-2 <= n*n]
    if not widths:
        raise ValueError("no compatible width for the focal counter")
    size = int(rng.choice(widths))
    masks = {k: _printed_mask(k, size) for k in allowed if k*k+2*k-2 <= size*size}
    # Integer bit sets make at most C(15,4)=1365 small compatibility checks.
    bits = {k: sum(1 << int(i) for i in np.flatnonzero(mask)) for k, mask in masks.items()}
    wanted = int(rng.choice([1, 2, 3, 4], p=[.22, .43, .25, .10]))
    others = [k for k in sorted(masks) if k != focal]
    for count in range(wanted, 0, -1):
        candidates = []
        for rest in itertools.combinations(others, count-1):
            occupied, good = bits[focal], True
            for k in rest:
                if occupied & bits[k]:
                    good = False
                    break
                occupied |= bits[k]
            if good:
                candidates.append((focal,) + rest)
        if candidates:
            selected = candidates[int(rng.integers(len(candidates)))]
            return size, [int(k) for k in rng.permutation(selected)]
    raise AssertionError("a single fitting focal layer is always feasible")


def _shape(rng, available, area):
    """Small patches, bars, diagonal bands, or fragments, clipped to free cells."""
    coords = np.argwhere(available)
    origin = coords[int(rng.integers(len(coords)))]
    delta = np.abs(coords-origin)
    style = int(rng.integers(0, 5))
    if style == 0:
        scores = delta.sum(axis=1) + rng.random(len(coords))*.7
    elif style == 1:
        scores = delta[:, 0]*4 + delta[:, 1] + rng.random(len(coords))*.5
    elif style == 2:
        scores = delta[:, 1]*4 + delta[:, 0] + rng.random(len(coords))*.5
    elif style == 3:
        scores = np.abs((coords[:, 0]-origin[0])-(coords[:, 1]-origin[1]))*4 + delta.sum(axis=1)*.2
        scores = scores + rng.random(len(coords))*.5
    else:
        scores = rng.random(len(coords))
    chosen = coords[np.argsort(scores, kind="stable")[:area]]
    mask = np.zeros(available.shape, dtype=bool)
    mask[chosen[:, 0], chosen[:, 1]] = True
    return mask


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _source_and_target(rng, size, counts, unmarked, paper, colors, witness):
    source = np.full((size, size), paper, dtype=np.int8)
    target = source.copy()
    final_used = np.zeros((size, size), dtype=bool)
    for color, count in zip(colors, counts):
        mask = _printed_mask(count, size)
        assert not np.any(final_used & mask)
        target[mask] = color
        final_used |= mask
    budget = size*size // 3
    marked_count = len(counts)
    # Unmarked masks occupy cells left free by ALL rotated synthesized layers.
    for j in range(unmarked):
        color = colors[marked_count+j]
        available = (source == paper) & ~final_used
        reserve = unmarked-j-1 + marked_count
        maximum = min(size+3, int(available.sum())-(unmarked-j-1)-1, budget-reserve)
        assert maximum >= 1
        area = int(rng.integers(1, maximum+1))
        mask = _shape(rng, available, area)
        if witness and j == 0 and not _asymmetric(mask):
            # A noncentral singleton always fits: these teaching sheets are
            # large and have many cells outside the printed edge layers.
            for y, x in np.argwhere(available):
                single = np.zeros((size, size), dtype=bool)
                single[y, x] = True
                if _asymmetric(single):
                    mask, area = single, 1
                    break
            assert _asymmetric(mask)
        source[mask] = color
        target[mask] = color
        final_used |= mask
        budget -= area
    # Marked source masks are deliberately independent of the generated runs.
    # Their only role after activation is to register the content color.
    for j in range(marked_count):
        maximum = min(9, budget-(marked_count-j-1))
        area = int(rng.integers(1, maximum+1))
        mask = _shape(rng, source == paper, area)
        source[mask] = colors[j]
        budget -= area
    return source, target


def _bar_starts(h, w, length, blocked, orientation):
    bh, bw = (1, length) if orientation == "horizontal" else (length, 1)
    if bh > h or bw > w:
        return np.empty((0, 2), dtype=int)
    rr, cc = np.indices((h-bh+1, w-bw+1))
    clear = np.ones(rr.shape, dtype=bool)
    for i in range(length):
        clear &= ~blocked[rr+(i if bh > 1 else 0), cc+(i if bw > 1 else 0)]
    return np.argwhere(clear)


def _layout(rng, source, exterior, colors, counts, forced, random_attempts=18):
    size = len(source)
    for attempt in range(random_attempts+1):
        if attempt == random_attempts:
            # Guaranteed four-rail fallback. Every fitting count is < size;
            # right-side bars end above the bottom-side bars, so rails cannot
            # intersect. N<=24 leaves room for both rail pairs within 30.
            h = w = size+6
            r = c = 1
        else:
            h = int(rng.integers(size+3, 31))
            w = int(rng.integers(size+3, 31))
            r = int(rng.integers(1, h-size))
            c = int(rng.integers(1, w-size))
        inp = np.full((h, w), exterior, dtype=np.int8)
        inp[r:r+size, c:c+size] = source
        blocked = np.zeros((h, w), dtype=bool)
        blocked[r-1:r+size+1, c-1:c+size+1] = True
        blocked[0, 0] = blocked[0, -1] = blocked[-1, 0] = blocked[-1, -1] = True
        order = sorted(range(len(counts)), key=lambda k: -counts[k])
        if attempt == random_attempts:
            # At most two forced bars of either orientation. Normal generation
            # forces only the repeated teaching count, so one always fits.
            rails = {"horizontal": [size+2, size+4], "vertical": [size+2, size+4]}
            order.sort(key=lambda k: k not in forced)
            for k in order:
                orient = forced.get(k)
                if orient is None:
                    feasible = [v for v in rails if rails[v]]
                    orient = feasible[int(rng.integers(len(feasible)))]
                if not rails[orient]:
                    raise AssertionError("too many forced bars for fallback rails")
                rail = rails[orient].pop(0)
                if orient == "horizontal":
                    inp[rail, 1:1+counts[k]] = colors[k]
                else:
                    inp[1:1+counts[k], rail] = colors[k]
            return inp
        success = True
        for k in order:
            choices = [forced[k]] if k in forced else ["horizontal", "vertical"]
            groups = [(v, _bar_starts(h, w, counts[k], blocked, v)) for v in choices]
            groups = [(v, starts) for v, starts in groups if len(starts)]
            if not groups:
                success = False
                break
            orient, starts = groups[int(rng.integers(len(groups)))]
            y, x = map(int, starts[int(rng.integers(len(starts)))])
            for i in range(counts[k]):
                yy, xx = (y, x+i) if orient == "horizontal" else (y+i, x)
                inp[yy, xx] = colors[k]
                blocked[max(0, yy-1):min(h, yy+2), max(0, xx-1):min(w, xx+2)] = True
        if success:
            return inp
    raise AssertionError("bounded layout fallback was not reached")


def _make_pair(rng, size, counts, unmarked, forced, witness=False):
    palette = [int(v) for v in rng.choice(10, size=len(counts)+unmarked+2, replace=False)]
    exterior, paper, colors = palette[0], palette[1], palette[2:]
    source, target = _source_and_target(rng, size, counts, unmarked, paper, colors, witness)
    inp = _layout(rng, source, exterior, colors, counts, forced)
    return {"input": inp, "output": target}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Deterministically generate one coherent puzzle with a visible curriculum."""
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    width_shift = bool(rng.integers(2))
    opposite = ("horizontal", "vertical") if rng.integers(2) else ("vertical", "horizontal")
    puzzle, seen, train_specs = {"train": [], "test": []}, set(), []
    for split, amount in (("train", num_train), ("test", num_test)):
        for index in range(int(amount)):
            for _ in range(16):
                forced, witness = {}, False
                if split == "train" and index == 0:
                    size, counts = _programs(rng, range(1, 6), required=[1, 2, 3, 4])
                    unmarked, forced, witness = 1, {2: opposite[0]}, True
                elif split == "train" and index == 1:
                    # The repeated n=3 is unwrapped in the first sheet and
                    # wrapped here; changing its bar orientation cannot explain
                    # that geometric change. n=5 also witnesses a full cycle.
                    size, counts = _programs(rng, range(1, 6), required=[3, 5], excluded=range(13, 25))
                    unmarked, forced = int(rng.integers(0, 4)), {0: opposite[1]}
                else:
                    if split == "test" and width_shift:
                        allowed = sorted({k for _, ks in train_specs for k in ks})
                        focal = int(rng.choice([k for k in allowed if k >= 3 and k % 4]))
                        excluded = {n for n, ks in train_specs if focal in ks}
                    else:
                        allowed = list(range(1, 16 if width_shift or split == "test" else 6))
                        focal = int(rng.choice([6, 7, 9, 10, 11, 13, 14, 15])) if split == "test" else None
                        excluded = ()
                    size, counts = _programs(rng, allowed, focal=focal, excluded=excluded)
                    unmarked = int(rng.integers(max(0, 2-len(counts)), min(3, 6-len(counts))+1))
                pair = _make_pair(rng, size, counts, unmarked, forced, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_specs.append((size, counts))
                    break
            else:
                raise RuntimeError("bounded unique-input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise AssertionError("pair must have exactly input and output")
    out = _grid(pair["output"])
    info = _read_input(pair["input"])
    n, paper = info["size"], info["paper"]
    assert out.shape == (n, n), "output must be the complete square sheet"
    assert np.array_equal(out, _coordinate_output(info)), "wrong expansion, rotation, or recombination"
    assert info["exterior"] not in np.unique(out)
    assert set(map(int, np.unique(out))) == set(info["palette"]) | {paper}
    residues = [m["length"] % 4 for m in info["bars"].values()]
    assert len(set(residues)) == len(residues), "shared origin forbids repeated turn residues"
    for color in info["palette"]:
        marker = info["bars"].get(color)
        if marker is None:
            assert np.array_equal(out == color, info["source"] == color)
        else:
            count = marker["length"]
            assert int(np.sum(out == color)) == count*count
            # Undo ONLY this color's turns and read every run and separator.
            unturned = np.rot90(out == color, count % 4).ravel()
            position = 0
            for j in range(2*count-1):
                run = count-abs(count-1-j)
                assert np.all(unturned[position:position+run])
                position += run
                if j < 2*count-2:
                    assert not unturned[position]
                    position += 1
            assert position == count*count+2*count-2
            assert not np.any(unturned[position:]), "all unused sheet cells are padding"
    return info


def _alternative_differs(info, **kwargs):
    try:
        alternative = _coordinate_output(info, **kwargs)
    except ValueError:
        return True  # A conflicting shortcut is also an incorrect answer.
    return not np.array_equal(alternative, _coordinate_output(info))


def validate_generated_puzzle(puzzle) -> None:
    """Check returned cells, input-only localization, causal ablations, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(info["input"])
            assert key not in seen, "inputs must be distinct within the puzzle"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    demonstrated = {m["length"] for f in train for m in f["bars"].values()}
    assert {1, 2, 3, 4, 5}.issubset(demonstrated)
    assert any(_alternative_differs(f, expansion=False) for f in train), "A must change a training answer"
    assert any(_alternative_differs(f, rotation=False) for f in train), "B must change a training answer"
    assert any(_alternative_differs(f, direction=-1) for f in train), "clockwise must be visible"
    assert any(_alternative_differs(f, square_width=f["input"].shape[1]) for f in train)
    independent = False
    for pair, f in zip(puzzle["train"], train):
        common = set(range(4))
        out = np.asarray(pair["output"])
        for color in f["palette"]:
            marker = f["bars"].get(color)
            base = (_run_mask(marker["length"], f["size"]) if marker is not None
                    else f["source"] == color)
            common &= {k for k in range(4) if np.array_equal(np.rot90(base, -k), out == color)}
        independent |= not common
    assert independent, "no single rotation may explain all color layers"
    assert any(_asymmetric(f["source"] == color) for f in train for color in f["palette"]
               if color not in f["bars"]), "show the unmarked zero-turn default"
    repeat = [(f["size"], m["orientation"]) for f in train for m in f["bars"].values() if m["length"] == 3]
    assert any(a[0] != b[0] and {a[1], b[1]} == {"horizontal", "vertical"}
               for a, b in itertools.combinations(repeat, 2)), "repeat a count with new width and orientation"
    assert any(n < 13 for n, _ in repeat) and any(n >= 13 for n, _ in repeat)
    for f in infos["test"]:
        assert _alternative_differs(f, expansion=False), "test must require run generation"
        assert _alternative_differs(f, rotation=False), "test must require layer rotation"
    # Every generated test has either a genuinely larger count, or a familiar
    # count at a width not demonstrated for that count. This is a measured
    # property of returned grids, with no dependency on the sampled mode.
    maximum = max(demonstrated)
    for f in infos["test"]:
        assert any(m["length"] > maximum or
                   (m["length"] in demonstrated and all(
                       t["size"] != f["size"] or all(b["length"] != m["length"] for b in t["bars"].values())
                       for t in train)) for m in f["bars"].values()), "missing structural train/test shift"


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test() -> None:
    """Bounded 42-seed streaming audit; no files, traces, or puzzle bank saved."""
    signatures, shapes, sizes, positions, backgrounds, papers, lengths = set(), set(), set(), set(), set(), set(), set()
    layer_counts, marked_counts, unmarked_counts, orientations = set(), set(), set(), set()
    wrapped = unwrapped = inside_cut = separator_cut = 0
    width_shifts = extrapolations = mixed_bars = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        train = [_read_input(p["input"]) for p in puzzle["train"]]
        demonstrated = {m["length"] for f in train for m in f["bars"].values()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                n = info["size"]
                shapes.add(info["input"].shape)
                sizes.add(n)
                positions.add((info["row"], info["col"]))
                backgrounds.add(info["exterior"])
                papers.add(info["paper"])
                layer_counts.add(len(info["palette"]))
                marked_counts.add(len(info["bars"]))
                unmarked_counts.add(len(info["palette"])-len(info["bars"]))
                here = set()
                for marker in info["bars"].values():
                    k = marker["length"]
                    lengths.add(k)
                    orientations.add((k, marker["orientation"]))
                    here.add(marker["orientation"])
                    total = k*k+2*k-2
                    wrapped += total > n
                    unwrapped += total <= n
                    flat = _run_mask(k, n).ravel()
                    for p in range(n, total, n):
                        inside_cut += bool(flat[p-1] and flat[p])
                        separator_cut += bool(not flat[p-1] or not flat[p])
                mixed_bars += {"horizontal", "vertical"}.issubset(here)
                if split == "test":
                    extrapolations += any(m["length"] > max(demonstrated) for m in info["bars"].values())
                    width_shifts += any(m["length"] in demonstrated and all(
                        t["size"] != n or all(b["length"] != m["length"] for b in t["bars"].values())
                        for t in train) for m in info["bars"].values())
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed in (0, 17):
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(sizes) >= 17 and len(positions) >= 35
    assert backgrounds == papers == set(range(10))
    assert set(range(1, 9)).issubset(lengths) and max(lengths) >= 12
    assert {1, 2, 3, 4}.issubset(marked_counts) and {0, 1, 2, 3}.issubset(unmarked_counts)
    assert len(layer_counts) >= 4
    assert wrapped >= 80 and unwrapped >= 80 and inside_cut >= 30 and separator_cut >= 20
    assert mixed_bars >= 25 and width_shifts >= 8 and extrapolations >= 8
    for k in (2, 3, 4, 5):
        assert {(k, "horizontal"), (k, "vertical")}.issubset(orientations)
    # Exercise the guaranteed layout branch, rather than waiting for sampling
    # to exhaust its retries. Check large same-grammar inputs beyond the
    # generator's sampled N<=24 and count<=15 ranges as well.
    rng = np.random.default_rng(991)
    source, target = _source_and_target(rng, 24, [1, 2, 3, 4], 1, 0, [1, 2, 3, 4, 5], False)
    fallback = _layout(rng, source, 9, [1, 2, 3, 4, 5], [1, 2, 3, 4], {2: "vertical"}, random_attempts=0)
    _analyze_pair({"input": fallback, "output": target})
    for size in (25, 26, 27):
        _analyze_pair(_make_pair(rng, size, [size-1], 1, {0: "horizontal"}))
    # Every legal count combination, including minimal and maximal puzzles.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(42+a*5+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Output corruption and broken color correspondence must be rejected.
    puzzle = generate_puzzle(81, 2, 1)
    pair = puzzle["test"][0]
    bad = {k: v.copy() for k, v in pair.items()}
    bad["output"][0, 0] = (int(bad["output"][0, 0])+1) % 10
    try:
        _analyze_pair(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    info = _read_input(pair["input"])
    bad_input = pair["input"].copy()
    coords = next(iter(info["bars"].values()))["coords"]
    bad_input[coords[:, 0], coords[:, 1]] = info["paper"]
    try:
        solve(bad_input)
    except ValueError:
        pass
    else:
        raise AssertionError("paper-colored exterior bar accepted")
