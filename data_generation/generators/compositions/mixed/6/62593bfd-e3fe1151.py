"""Repair a shared panel multiset before classifying and packing its objects.

Only NumPy and the standard library are used. No file or network I/O occurs here.
The solver uses only the supplied grid; construction metadata never reaches it.
"""

from collections import Counter
from hashlib import sha256

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 478242699, 538365401, 261794780, 422929984, 676906906,
              17011976, 678805222, 216751063, 123406153, 20013822, 900886146,
              424095836, 393738294, 261469944, 413682401, 962909816, 161587939,
              742032436, 926124634, 719744659, 217982567, 371349690, 878501516,
              14623184, 937895061, 786783254, 28352265, 779321465, 460824815,
              842345783, 613322226, 655463796, 203373707, 385493078, 676563247,
              566543900, 87347377, 32195051, 35501111, 912626845, 360742161)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "colors must be 0..9")
    return grid.astype(np.int16, copy=True)


def _panels(shape):
    h, w = (n // 2 for n in shape)
    return ((slice(0, h), slice(0, w)), (slice(0, h), slice(w + 1, 2*w + 1)),
            (slice(h + 1, 2*h + 1), slice(0, w)),
            (slice(h + 1, 2*h + 1), slice(w + 1, 2*w + 1)))


def _hist(panel):
    return np.bincount(panel.ravel().astype(int), minlength=10)


def _normal(mask):
    rows, cols = np.nonzero(mask)
    return np.asarray(mask, dtype=bool)[rows.min():rows.max()+1,
                                        cols.min():cols.max()+1].copy()


def _mask_key(mask):
    mask = _normal(mask)
    return mask.shape, mask.tobytes()


def _canonical(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _connected(mask, directions=DIR8):
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if not cells:
        return False
    seen = {min(cells)}
    pending = list(seen)
    while pending:
        r, c = pending.pop()
        for dr, dc in directions:
            q = r + dr, c + dc
            if q in cells and q not in seen:
                seen.add(q)
                pending.append(q)
    return seen == cells


def _bridges(mask):
    result = []
    for r, c in np.argwhere(mask):
        trial = mask.copy()
        trial[r, c] = False
        if not _connected(trial):
            result.append((int(r), int(c)))
    return result


def _segment(panel, background, panel_id, directions=DIR8):
    seen, objects = set(), []
    h, w = panel.shape
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == background or (r, c) in seen:
                continue
            pending, cells = [(r, c)], []
            seen.add((r, c))
            while pending:
                y, x = pending.pop()
                cells.append((y, x))
                for dy, dx in directions:
                    q = y + dy, x + dx
                    if (0 <= q[0] < h and 0 <= q[1] < w and q not in seen
                            and int(panel[q]) == color):
                        seen.add(q)
                        pending.append(q)
            top = min(y for y, x in cells)
            left = min(x for y, x in cells)
            mask = np.zeros((max(y for y, x in cells)-top+1,
                             max(x for y, x in cells)-left+1), dtype=bool)
            for y, x in cells:
                mask[y-top, x-left] = True
            objects.append(dict(panel=panel_id, color=color, top=top, left=left,
                                mask=mask, key=_canonical(mask), cells=frozenset(cells)))
    return objects


def _repair(input_grid):
    inp = _grid(input_grid)
    h, w = inp.shape
    _require(h >= 3 and w >= 3 and h % 2 == w % 2 == 1,
             "four equal panels require odd dimensions")
    structural = int(inp[h//2, w//2])
    _require(np.all(inp[h//2, :] == structural) and np.all(inp[:, w//2] == structural),
             "missing complete central cross")
    observed = []
    for rs, cs in _panels(inp.shape):
        counts = _hist(inp[rs, cs])
        _require(counts[structural] == 1, "exactly one placeholder is required per panel")
        counts[structural] = 0
        observed.append(counts)
    target = np.maximum.reduce(observed)
    area = (h//2)*(w//2)
    _require(int(target.sum()) == area, "common histogram is not uniquely recoverable")
    background = int(np.argmax(target))
    _require(int(target[background])*2 > area, "panel background must exceed half the area")
    repaired, missing = inp.copy(), []
    for i, (rs, cs) in enumerate(_panels(inp.shape)):
        deficit = target - observed[i]
        _require(np.all(deficit >= 0) and deficit.sum() == 1
                 and np.count_nonzero(deficit) == 1, "non-unique panel repair")
        color = int(np.argmax(deficit))
        panel = repaired[rs, cs]
        panel[panel == structural] = color
        missing.append(color)
    return repaired, structural, background, target, tuple(missing)


def _objects(repaired, background, directions=DIR8):
    return [obj for i, (rs, cs) in enumerate(_panels(repaired.shape))
            for obj in _segment(repaired[rs, cs], background, i, directions)]


def _class_keys(objects, mode):
    if mode == "identity":
        return [_mask_key(o["mask"]) for o in objects]
    if mode == "reflection":
        return [min(o["key"], _canonical(np.fliplr(o["mask"]))) for o in objects]
    if mode == "area":
        return [int(o["mask"].sum()) for o in objects]
    if mode == "color":
        return [o["color"] for o in objects]
    if mode == "local":
        return [(o["panel"], o["key"]) for o in objects]
    return [o["key"] for o in objects]


def _packing(repaired, structural, background, mode="rotation", directions=DIR8, objects=None):
    if objects is None:
        objects = _objects(repaired, background, directions)
    keys = _class_keys(objects, mode)
    counts = Counter(keys)
    out = np.full(repaired.shape, structural, dtype=np.int16)
    for rs, cs in _panels(out.shape):
        out[rs, cs] = background
    for obj, key in zip(objects, keys):
        panel = out[_panels(out.shape)[obj["panel"]]]
        up = counts[key] == 2 if mode == "exactly_two" else counts[key] >= 2
        top = 0 if up else panel.shape[0] - obj["mask"].shape[0]
        for r, c in np.argwhere(obj["mask"]):
            y, x = top + int(r), obj["left"] + int(c)
            _require(panel[y, x] == background, "destination collision")
            panel[y, x] = obj["color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the missing colors, then pool exact rotation classes across panels."""
    repaired, structural, background, _, _ = _repair(input_grid)
    return _packing(repaired, structural, background)


def _random_mask(rng, area, limit):
    cells = {(0, 0)}
    directions = DIR4 if rng.random() < .45 else DIR8
    for _ in range(area - 1):
        candidates = set()
        for r, c in sorted(cells):
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in cells:
                    continue
                trial = cells | {q}
                if (max(y for y, x in trial)-min(y for y, x in trial) < limit
                        and max(x for y, x in trial)-min(x for y, x in trial) < limit):
                    candidates.add(q)
        if not candidates:
            return None
        candidates = sorted(candidates)
        cells.add(candidates[int(rng.integers(len(candidates)))])
    top, left = min(r for r, c in cells), min(c for r, c in cells)
    mask = np.zeros((max(r for r, c in cells)-top+1,
                     max(c for r, c in cells)-left+1), dtype=bool)
    for r, c in cells:
        mask[r-top, c-left] = True
    return mask


def _shape(rng, area, limit, forbidden, chiral=False, bridge=False, diagonal=False):
    for _ in range(220):
        mask = _random_mask(rng, area, limit)
        if mask is None:
            continue
        key, mirror = _canonical(mask), _canonical(np.fliplr(mask))
        if key in forbidden or mirror in forbidden:
            continue
        if chiral and key == mirror:
            continue
        if bridge and not _bridges(mask):
            continue
        if diagonal and _connected(mask, DIR4):
            continue
        return mask
    raise ValueError("bounded shape sampler exhausted")


def _halo(cells):
    return {(r+dr, c+dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _layout(rng, objects, background):
    counts = Counter(o["key"] for o in objects)
    for o in objects:
        o["up"] = counts[o["key"]] >= 2
    groups = [[o for o in objects if o["panel"] == p] for p in range(4)]
    widths, upper, lower = [], [], []
    for group in groups:
        for up in (True, False):
            subset = [o for o in group if o["up"] == up]
            widths.append(sum(o["mask"].shape[1] for o in subset) + max(0, len(subset)-1))
        upper.append(max((o["mask"].shape[0] for o in group if o["up"]), default=0))
        lower.append(max((o["mask"].shape[0] for o in group if not o["up"]), default=0))
    min_h = max(8, max(a+b+3 for a, b in zip(upper, lower)))
    min_w = max(6, max(widths))
    _require(min_h <= 14 and min_w <= 14, "sample does not fit")
    h, w = int(rng.integers(min_h, 15)), int(rng.integers(min_w, 15))
    foreground = sum(int(o["mask"].sum()) for o in groups[0])
    _require(2*foreground < h*w, "background not dominant")
    panels, outputs = [], []
    for group in groups:
        for up in (True, False):
            subset = [o for o in group if o["up"] == up]
            rng.shuffle(subset)
            need = sum(o["mask"].shape[1] for o in subset) + max(0, len(subset)-1)
            spaces = rng.multinomial(w-need, np.full(len(subset)+1, 1/(len(subset)+1)))
            cursor = int(spaces[0])
            for i, o in enumerate(subset):
                o["left"] = cursor
                cursor += o["mask"].shape[1] + 1 + int(spaces[i+1])
        # Random rows with a finite search, then a guaranteed separated edge layout.
        placed = []
        for o in sorted(group, key=lambda ob: -int(ob["mask"].sum())):
            candidates = [int(t) for t in rng.permutation(h-o["mask"].shape[0]+1)]
            for top in candidates:
                cells = {(top+int(r), o["left"]+int(c)) for r, c in np.argwhere(o["mask"])}
                if not any(cells & _halo(other) for other in placed):
                    o["top"] = top
                    placed.append(cells)
                    break
            else:
                for ob in group:
                    ob["top"] = 1 if ob["up"] else h-ob["mask"].shape[0]-1
                break
        panel = np.full((h, w), background, dtype=np.int16)
        output = panel.copy()
        for o in group:
            destination = 0 if o["up"] else h-o["mask"].shape[0]
            for r, c in np.argwhere(o["mask"]):
                panel[o["top"]+int(r), o["left"]+int(c)] = o["color"]
                output[destination+int(r), o["left"]+int(c)] = o["color"]
        _require(len(_segment(panel, background, 0)) == len(group), "input object contact")
        _require(len(_segment(output, background, 0)) == len(group), "output object contact")
        panels.append(panel)
        outputs.append(output)
    return panels, outputs


def _roster(rng, lesson, forbidden):
    # Eight main objects have equal areas but genuinely different exact masks.
    # Assign them to four panels, two colors per panel, independently of class.
    profiles = ((3, 2, 1, 1, 1), (4, 1, 1, 1, 1), (2, 2, 1, 1, 1, 1))
    profile = profiles[lesson if lesson in (0, 1) else int(rng.integers(3))]
    area = int(rng.integers(4, 6 if lesson == 1 else 8))
    limit = 3 if lesson == 1 else int(rng.choice([3, 4, 5]))
    target_class = min((i for i, count in enumerate(profile) if count > 1),
                       key=lambda i: profile[i])
    require_bridge = lesson == 0 or rng.random() < .55
    bases, blocked = [], set(forbidden)
    mirror_index = next(i for i, n in enumerate(profile) if n == 1)
    mirror_negative = lesson in (0, 1) or rng.random() < .4
    for i, count in enumerate(profile):
        if mirror_negative and i == mirror_index+1:
            mask = np.fliplr(bases[mirror_index]).copy()
        else:
            mask = _shape(rng, area, limit, blocked,
                          chiral=mirror_negative and i == mirror_index,
                          bridge=require_bridge and i == target_class,
                          diagonal=lesson == 0 and i == target_class)
        bases.append(mask)
        blocked.update((_canonical(mask), _canonical(np.fliplr(mask))))
    slots = [(p, c) for p in range(4) for c in range(2)]
    labels = [i for i, count in enumerate(profile) for _ in range(count)]
    for _ in range(100):
        rng.shuffle(labels)
        assignment = list(zip(slots, labels))
        if any(len({p for (p, c), j in assignment if j == i}) != count
               for i, count in enumerate(profile) if count > 1):
            continue
        if not any(len({c for (p, c), j in assignment if j == i}) > 1
                   for i, count in enumerate(profile) if count > 1):
            continue
        if not any(len({profile[j] > 1 for (p, c), j in assignment if c == color}) == 2
                   for color in range(2)):
            continue
        break
    else:
        raise ValueError("class assignment budget exhausted")
    objects = []
    turns = {i: int(rng.integers(4)) for i in range(len(profile))}
    uses = Counter()
    for (panel, color), label in assignment:
        turn = int(rng.integers(4))
        if lesson == 0 and label == 0 and uses[label] < 2:
            turn = (turns[label]+uses[label]) % 4
        uses[label] += 1
        mask = _normal(np.rot90(bases[label], turn))
        objects.append(dict(panel=panel, role=color, label=label, mask=mask, key=_canonical(mask)))
    return objects, target_class, area


def _make_pair(rng, lesson=-1, forbidden=frozenset()):
    for _ in range(180):
        try:
            palette = [int(c) for c in rng.permutation(10)]
            structural, background = palette[:2]
            objects, target_class, area = _roster(rng, lesson, forbidden)
            if lesson == -1 and rng.random() < .5:
                # Preserve a color's quota while changing its decomposition.
                # Only singleton classes are split, so targeted partners survive.
                class_counts = Counter(o["label"] for o in objects)
                revised = []
                for o in objects:
                    if class_counts[o["label"]] == 1 and rng.random() < .38:
                        first = int(rng.integers(1, area))
                        for size in (first, area-first):
                            mask = _shape(rng, size, 4, set())
                            revised.append(dict(panel=o["panel"], role=o["role"], label=-2,
                                                mask=mask, key=_canonical(mask)))
                    else:
                        revised.append(o)
                objects = revised
            extra = lesson == 1 or (lesson != 0 and rng.random() < .55)
            if extra:
                quota = int(rng.integers(2, min(5, area-1)+1))
                # Unequal existing-color quotas are also allowed. Teaching the
                # vanished satellite uses a visibly distinct third object color.
                extra_role = 2 if lesson == 1 or rng.random() < .65 else int(rng.integers(2))
                for panel in range(4):
                    parts = [1, quota-1] if lesson == 1 or rng.random() < .55 else [quota]
                    for size in parts:
                        mask = _shape(rng, size, 3, set())
                        mask = _normal(np.rot90(mask, int(rng.integers(4))))
                        objects.append(dict(panel=panel, role=extra_role, label=-1, mask=mask,
                                            key=_canonical(mask)))
            for o in objects:
                o["color"] = palette[2+o["role"]]
            panels, outputs = _layout(rng, objects, background)
            repaired = np.full((2*panels[0].shape[0]+1, 2*panels[0].shape[1]+1),
                               structural, dtype=np.int16)
            output = repaired.copy()
            for (rs, cs), panel, out in zip(_panels(repaired.shape), panels, outputs):
                repaired[rs, cs], output[rs, cs] = panel, out
            damaged = repaired.copy()
            target = [o for o in objects if o["label"] == target_class]
            survivor = target[int(rng.integers(len(target)))]["panel"]
            hole_by_panel = {}
            for o in target:
                if o["panel"] != survivor:
                    choices = _bridges(o["mask"]) or [tuple(map(int, p)) for p in np.argwhere(o["mask"])]
                    r, c = choices[int(rng.integers(len(choices)))]
                    hole_by_panel[o["panel"]] = o["top"]+r, o["left"]+c
            free = [p for p in range(4) if p not in hole_by_panel]
            rng.shuffle(free)
            for index, p in enumerate(free):
                panel = panels[p]
                singletons = [o for o in objects if o["panel"] == p and o["mask"].sum() == 1]
                use_hidden = lesson == 1 or (lesson == -1 and singletons and rng.random() < .35)
                if use_hidden and index == 0:
                    singleton = singletons[int(rng.integers(len(singletons)))]
                    point = singleton["top"], singleton["left"]
                elif index == 0:
                    # A missing blank next to a colored object defeats neighbor filling.
                    foreground = {tuple(map(int, q)) for q in np.argwhere(panel != background)}
                    choices = sorted(q for q in _halo(foreground)
                                     if 0 <= q[0] < panel.shape[0] and 0 <= q[1] < panel.shape[1]
                                     and panel[q] == background)
                    point = choices[int(rng.integers(len(choices)))]
                else:
                    point = int(rng.integers(panel.shape[0])), int(rng.integers(panel.shape[1]))
                hole_by_panel[p] = point
            for p, (r, c) in hole_by_panel.items():
                rs, cs = _panels(damaged.shape)[p]
                damaged[rs, cs][r, c] = structural
            pair = {"input": damaged, "output": output}
            f = _pair_features(pair)
            _require(f["remote_flip"] and f["mixed"], "repair must affect classification remotely")
            if lesson == 0:
                _require(f["bridge"] and f["diagonal"] and f["blank_next_to_color"], "missing teaching witness")
                _require(all(f["alternatives"].values()), "teaching example leaves a named shortcut viable")
            if lesson == 1:
                _require(f["hidden_singleton"], "missing isolated-cell witness")
            return pair, f
        except ValueError:
            continue
    raise RuntimeError("bounded pair construction exhausted")


def _signature(o, top=None):
    return o["panel"], o["color"], o["top"] if top is None else top, o["left"], _mask_key(o["mask"])


def _pair_features(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "output shape changed")
    repaired, structural, background, target, missing = _repair(inp)
    objects = _objects(repaired, background)
    counts = Counter(o["key"] for o in objects)
    _require(np.array_equal(out, _packing(repaired, structural, background, objects=objects)),
             "incorrect output")
    h = inp.shape[0]//2
    expected = sorted(_signature(o, 0 if counts[o["key"]] > 1 else h-o["mask"].shape[0])
                      for o in objects)
    _require(expected == sorted(_signature(o) for o in _objects(out, background)),
             "object merged, changed, or lost")
    for rs, cs in _panels(inp.shape):
        _require(np.array_equal(_hist(out[rs, cs]), target), "panel histogram changed during packing")
    _require(not np.array_equal(repaired, out), "packing has no effect")
    raw = inp.copy()
    holes = []
    for rs, cs in _panels(inp.shape):
        panel = raw[rs, cs]
        holes.append(tuple(map(int, np.argwhere(panel == structural)[0])))
        panel[panel == structural] = background
    broken = _objects(raw, background)
    raw_counts = Counter(o["key"] for o in broken)
    remote_flip = False
    bridge = False
    hidden_singleton = False
    blank_next_to_color = False
    for o in objects:
        hole = holes[o["panel"]]
        if hole not in o["cells"]:
            remote_flip |= (counts[o["key"]] > 1) != (raw_counts[o["key"]] > 1)
        else:
            pieces = [b for b in broken if b["panel"] == o["panel"]
                      and b["color"] == o["color"] and b["cells"] & o["cells"]]
            bridge |= len(pieces) > 1
            hidden_singleton |= len(o["cells"]) == 1
    for p, (rs, cs) in enumerate(_panels(inp.shape)):
        panel = repaired[rs, cs]
        r, c = holes[p]
        if panel[r, c] == background:
            blank_next_to_color |= any(0 <= r+dr < panel.shape[0] and 0 <= c+dc < panel.shape[1]
                                       and panel[r+dr, c+dc] != background for dr, dc in DIR8)
    alternatives = {}
    for mode in ("identity", "reflection", "area", "color", "local", "exactly_two"):
        try:
            alternative = _packing(repaired, structural, background, mode, objects=objects)
            alternatives[mode] = not np.array_equal(alternative, out)
        except ValueError:
            alternatives[mode] = True
    try:
        four = _packing(repaired, structural, background, directions=DIR4)
        alternatives["four_connected"] = not np.array_equal(four, out)
    except ValueError:
        alternatives["four_connected"] = True
    # Same-panel-only repeated classes are not generated: every non-singleton
    # class is visibly represented in another panel, avoiding a scope ambiguity.
    _require(all(n == 1 or len({o["panel"] for o in objects if o["key"] == key}) >= 2
                 for key, n in counts.items()), "same-panel-only repeated class")
    return dict(shape=inp.shape, background=background, structural=structural,
                object_count=len(objects), areas=tuple(sorted({int(o["mask"].sum()) for o in objects})),
                histogram=tuple(map(int, target)), missing=missing, bridge=bool(bridge),
                remote_flip=bool(remote_flip), hidden_singleton=bool(hidden_singleton),
                blank_next_to_color=bool(blank_next_to_color),
                diagonal=any(not _connected(o["mask"], DIR4) for o in objects),
                mixed=any(n == 1 for n in counts.values()) and any(n > 1 for n in counts.values()),
                multiplicities=tuple(sorted(counts.values())), alternatives=alternatives,
                main_keys=frozenset(o["key"] for o in objects if o["mask"].sum() >= 4))


def _integer(value, name, low, high=None):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             f"{name} must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), f"illegal {name}")
    return value


def _grid_key(grid):
    grid = np.asarray(grid, dtype=np.uint8)
    return grid.shape, grid.tobytes()


def _puzzle_digest(puzzle):
    digest = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    puzzle, seen, training_keys, training_histograms = {"train": [], "test": []}, set(), set(), set()
    for i in range(num_train + num_test):
        split = "train" if i < num_train else "test"
        lesson = i if i < 2 else -1
        for _ in range(30):
            pair, f = _make_pair(rng, lesson, training_keys if split == "test" else frozenset())
            key = _grid_key(pair["input"])
            if key in seen or (split == "test" and f["histogram"] in training_histograms):
                continue
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                training_keys.update(f["main_keys"])
                training_histograms.add(f["histogram"])
            break
        else:
            raise RuntimeError("distinct-example budget exhausted")
    # Move teaching examples around: their position is not a convention.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            key = _grid_key(_grid(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _pair_features(pair)
            _require(f["remote_flip"] and f["mixed"], "missing causal repair/packing interaction")
            features[split].append(f)
    train = features["train"]
    _require(any(f["bridge"] and f["diagonal"] for f in train), "bridge/diagonal evidence absent")
    _require(any(f["hidden_singleton"] for f in train), "histogram-only missing-cell evidence absent")
    _require(any(f["blank_next_to_color"] for f in train), "background repair evidence absent")
    _require(any(3 in f["multiplicities"] for f in train), "at-least-two threshold not taught")
    _require(any(4 in f["multiplicities"] for f in train), "four-copy evidence absent")
    for alternative in train[0]["alternatives"]:
        _require(any(f["alternatives"][alternative] for f in train),
                 f"training fails to distinguish {alternative}")
    train_hist = {f["histogram"] for f in train}
    train_keys = set().union(*(f["main_keys"] for f in train))
    for f in features["test"]:
        _require(f["histogram"] not in train_hist, "test histogram is not novel")
        # Small auxiliary fragments may be familiar; main test shapes are fresh.
        _require(bool(f["main_keys"] - train_keys), "test contains no novel main shape")


def self_test():
    """Bounded streaming checks over the requested 42 seeds; no artifacts saved."""
    digests, shapes, backgrounds, markers, object_counts, areas = set(), set(), set(), set(), set(), set()
    bridge_pairs = hidden_pairs = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = _puzzle_digest(puzzle)
        _require(digest == _puzzle_digest(generate_puzzle(seed)), "nondeterministic generation")
        digests.add(digest)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _pair_features(pair)
                shapes.add(f["shape"])
                backgrounds.add(f["background"])
                markers.add(f["structural"])
                object_counts.add(f["object_count"])
                areas.update(f["areas"])
                bridge_pairs += f["bridge"]
                hidden_pairs += f["hidden_singleton"]
                inp = pair["input"].copy()
                _require(np.array_equal(solve(inp), pair["output"]), "array-input solve mismatch")
                _require(np.array_equal(solve(inp.tolist()), pair["output"]), "list-input solve mismatch")
                _require(np.array_equal(inp, pair["input"]), "solve mutated its input")
        del puzzle
    _require(len(digests) == 42 and len(shapes) >= 15, "insufficient structural variation")
    _require(len(backgrounds) == len(markers) == 10, "palette-role variation missing")
    _require(len(object_counts) >= 4 and len(areas) >= 6, "count/shape variation missing")
    _require(bridge_pairs >= 42 and hidden_pairs >= 42, "causal witnesses missing")
    for a in range(2, 11):
        for b in range(1, min(a-1, 4)+1):
            puzzle = generate_puzzle(10000 + 10*a+b, a, b)
            _require(len(puzzle["train"]) == a and len(puzzle["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(0, 2, 1)
    out = bad["test"][0]["output"]
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    # Four identical observed histograms leave the common missing color unknown.
    ambiguous = np.zeros((7, 7), dtype=int)
    ambiguous[3, :] = ambiguous[:, 3] = 9
    for rs, cs in _panels(ambiguous.shape):
        ambiguous[rs, cs][1, 1] = 9
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous histogram accepted")
