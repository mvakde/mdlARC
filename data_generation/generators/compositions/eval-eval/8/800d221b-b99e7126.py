"""Broadcast stencils: microscopic ballots address macroscopic receivers.

Only Python's standard library and NumPy are used. Inference reads input cells;
construction records are used only to construct the independently checked label.
"""
from __future__ import annotations

import numpy as np


_SHAPES = tuple((h, w) for h in range(2, 5) for w in range(2, 5)
                if h * w >= 6)


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or
                not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(grid):
    a = np.asarray(grid)
    return a.shape, a.astype(np.int8).tobytes()


def _runs(indices):
    result = []
    for i in map(int, indices):
        if result and result[-1][1] + 1 == i:
            result[-1] = (result[-1][0], i)
        else:
            result.append((i, i))
    return result


def _layout(grid):
    """Recover complete periodic divider bands, with no color-frequency guess."""
    divider = int(grid[0, 0])
    rr = _runs(np.flatnonzero(np.all(grid == divider, axis=1)))
    cc = _runs(np.flatnonzero(np.all(grid == divider, axis=0)))
    assert len(rr) >= 3 and len(cc) >= 3
    assert rr[0][0] == cc[0][0] == 0
    assert rr[-1][1] == grid.shape[0] - 1
    assert cc[-1][1] == grid.shape[1] - 1
    thickness = {b - a + 1 for a, b in rr + cc}
    assert len(thickness) == 1
    t = thickness.pop()
    heights = {rr[i + 1][0] - rr[i][1] - 1 for i in range(len(rr) - 1)}
    widths = {cc[i + 1][0] - cc[i][1] - 1 for i in range(len(cc) - 1)}
    assert len(heights) == len(widths) == 1
    h, w = heights.pop(), widths.pop()
    assert (h, w) in _SHAPES and t in (1, 2)
    return h, w, t, len(rr) - 1, len(cc) - 1, divider


def _slice(r, c, h, w, t):
    y, x = t + r * (h + t), t + c * (w + t)
    return np.s_[y:y + h, x:x + w]


def _infer(value):
    """Parse source roles and compute arrivals exclusively from the input."""
    grid = _grid(value)
    h, w, t, rows, cols, divider = _layout(grid)
    ar, ac = h // 2, w // 2
    ordinary, sources = [], []
    for r in range(rows):
        for c in range(cols):
            tile = grid[_slice(r, c, h, w, t)]
            assert not np.any(tile == divider)
            if np.all(tile == tile[0, 0]):
                ordinary.append(((r, c), int(tile[0, 0])))
            else:
                sources.append({"pos": (r, c), "tile": tile.copy()})
    assert ordinary and len(sources) >= 2
    neutral_values = {color for _, color in ordinary}
    assert len(neutral_values) == 1
    neutral = neutral_values.pop()
    markers = {int(s["tile"][ar, ac]) for s in sources}
    assert len(markers) == 1
    marker = markers.pop()
    assert len({divider, neutral, marker}) == 3
    signals = tuple(sorted(set(map(int, np.unique(grid))) -
                           {divider, neutral, marker}))
    assert len(signals) == 2
    positions = {s["pos"] for s in sources}
    votes = np.zeros((rows, cols, 2), dtype=int)
    pixel_votes = np.zeros_like(votes)
    clipped = 0
    for s in sources:
        tile = s["tile"]
        assert np.count_nonzero(tile == marker) == 1
        mask = np.isin(tile, signals)
        assert 3 <= int(mask.sum()) <= h * w - 1
        counts = [int(np.count_nonzero(tile == x)) for x in signals]
        assert min(counts) > 0 and counts[0] != counts[1]
        label = int(counts[1] > counts[0])
        s.update(mask=mask, counts=counts, label=label)
        r, c = s["pos"]
        for u, v in np.argwhere(mask):
            y, x = r + int(u) - ar, c + int(v) - ac
            if not (0 <= y < rows and 0 <= x < cols):
                clipped += 1
            elif (y, x) not in positions:
                votes[y, x, label] += 1
                pixel_votes[y, x, signals.index(int(tile[u, v]))] += 1
    expected = grid.copy()
    winners = np.full((rows, cols), -1, dtype=int)
    for (r, c), _ in ordinary:
        a, b = map(int, votes[r, c])
        if a != b:
            label = int(b > a)
            winners[r, c] = label
            expected[_slice(r, c, h, w, t)] = signals[label]
    info = dict(h=h, w=w, t=t, rows=rows, cols=cols, divider=divider,
                neutral=neutral, marker=marker, signals=signals,
                sources=sources, ordinary=[p for p, _ in ordinary],
                votes=votes, pixel_votes=pixel_votes, winners=winners,
                clipped=clipped)
    return expected, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return same-size consensus fills; never mutate the supplied input."""
    return _infer(input_grid)[0]


def _metrics(info):
    v, p, winner = info["votes"], info["pixel_votes"], info["winners"]
    opposed = (v[:, :, 0] > 0) & (v[:, :, 1] > 0)
    strict = winner >= 0
    pixel_strict = p[:, :, 0] != p[:, :, 1]
    pixel_winner = (p[:, :, 1] > p[:, :, 0]).astype(int)
    panel_counts = np.bincount([s["label"] for s in info["sources"]], minlength=2)
    raw_counts = np.sum([s["counts"] for s in info["sources"]], axis=0)
    panel_flip = (panel_counts[0] != panel_counts[1] and
                  np.any(strict & (winner != int(panel_counts[1] > panel_counts[0]))))
    raw_flip = (raw_counts[0] != raw_counts[1] and
                np.any(strict & (winner != int(raw_counts[1] > raw_counts[0]))))
    # Replace each actual stencil by its whole interior (apart from the anchor).
    # This checks that a solid-neighborhood shortcut predicts a different grid.
    box_votes = np.zeros_like(v)
    occupied = {s["pos"] for s in info["sources"]}
    for s in info["sources"]:
        r, c = s["pos"]
        for u in range(info["h"]):
            for x in range(info["w"]):
                y, z = r + u - info["h"] // 2, c + x - info["w"] // 2
                if (0 <= y < info["rows"] and 0 <= z < info["cols"] and
                        (y, z) not in occupied):
                    box_votes[y, z, s["label"]] += 1
    box_winners = np.where(box_votes[:, :, 0] == box_votes[:, :, 1], -1,
                           (box_votes[:, :, 1] > box_votes[:, :, 0]).astype(int))
    return dict(source_count=len(info["sources"]),
                strict_overlap=int(np.count_nonzero(opposed & strict)),
                ties=int(np.count_nonzero(opposed & ~strict)),
                pixel_flip=int(np.count_nonzero(strict & pixel_strict &
                                                 (winner != pixel_winner))),
                panel_flip=bool(panel_flip), raw_flip=bool(raw_flip),
                stencil_matters=bool(np.any(box_winners != winner)),
                both_colors=set(map(int, winner[strict])) == {0, 1},
                filled=int(np.count_nonzero(strict)), clipped=info["clipped"],
                max_arrivals=int(v.sum(axis=2).max()))


def _render(h, w, t, rows, cols, roles, sources):
    """Construction-side painting and votes; does not call inference."""
    divider, neutral, marker, s0, s1 = roles
    inp = np.full((rows * (h + t) + t, cols * (w + t) + t), divider,
                  dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            inp[_slice(r, c, h, w, t)] = neutral
    arrivals = {}
    for (r, c), (tile, label) in sources.items():
        inp[_slice(r, c, h, w, t)] = tile
        for u in range(h):
            for v in range(w):
                if int(tile[u, v]) not in (s0, s1):
                    continue
                pos = (r + u - h // 2, c + v - w // 2)
                if (0 <= pos[0] < rows and 0 <= pos[1] < cols and
                        pos not in sources):
                    arrivals.setdefault(pos, []).append(label)
    out = inp.copy()
    for (r, c), labels in arrivals.items():
        ones = sum(labels)
        if 2 * ones != len(labels):
            out[_slice(r, c, h, w, t)] = (s0, s1)[int(2 * ones > len(labels))]
    return {"input": inp, "output": out}


def _sample_pair(rng, shape, t, roles, kind):
    h, w = shape
    rows = int(rng.integers(4, (30 - t) // (h + t) + 1))
    cols = int(rng.integers(4, (30 - t) // (w + t) + 1))
    count = int(rng.integers(8, 11) if kind == "test" else
                rng.integers(5, 8) if kind == "majority" else rng.integers(2, 8))
    offsets = [(u - h // 2, v - w // 2)
               for u in range(h) for v in range(w)
               if (u, v) != (h // 2, w // 2)]
    # The opposed-majority demonstration shows complete broadcasts. The tie
    # demonstration separately teaches clipping, even at the minimum 2/1 count.
    confine = kind == "majority"
    def available(pos):
        r, c = pos
        return [(u, v) for u in range(h) for v in range(w)
                if (u, v) != (h // 2, w // 2) and
                (not confine or (0 <= r + u - h // 2 < rows and
                                0 <= c + v - w // 2 < cols))]
    target = None
    forced = {}
    winner = int(rng.integers(2))
    k = 0
    if kind in ("majority", "tie"):
        k = (5 if count == 7 and len(offsets) >= 5 and rng.random() < .5 else 3
             ) if kind == "majority" else (4 if count >= 4 and rng.random() < .5 else 2)
        candidates = []
        for y in range(rows):
            for x in range(cols):
                senders = [(y - dy, x - dx) for dy, dx in offsets
                           if 0 <= y - dy < rows and 0 <= x - dx < cols and
                           len(available((y - dy, x - dx))) >= 3]
                if len(senders) >= k:
                    candidates.append(((y, x), senders))
        target, senders = candidates[int(rng.integers(len(candidates)))]
        chosen = rng.choice(len(senders), k, replace=False)
        for i, j in enumerate(chosen):
            pos = senders[int(j)]
            label = winner if i < (k // 2 + 1 if kind == "majority" else k // 2) else 1 - winner
            # All the target-facing pixels can oppose the eventual consensus.
            pixel_label = 1 - winner if kind == "majority" else winner
            local = (target[0] - pos[0] + h // 2,
                     target[1] - pos[1] + w // 2)
            forced[pos] = (label, local, pixel_label)
    positions = list(forced)
    extras = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) in forced or (r, c) == target:
                continue
            cells = available((r, c))
            if target is not None:
                forbidden = (target[0] - r + h // 2, target[1] - c + w // 2)
                cells = [p for p in cells if p != forbidden]
            if len(cells) >= 3:
                extras.append((r, c))
    if len(extras) < count - len(positions):
        return None
    rng.shuffle(extras)
    positions += extras[:count - len(positions)]
    sources = {}
    divider, neutral, marker, s0, s1 = roles
    for pos in positions:
        label = (forced[pos][0] if pos in forced else
                 1 - winner if kind == "majority" else int(rng.integers(2)))
        required = forced[pos][1] if pos in forced else None
        forbidden = None
        if target is not None and pos not in forced:
            u = target[0] - pos[0] + h // 2
            v = target[1] - pos[1] + w // 2
            if 0 <= u < h and 0 <= v < w:
                forbidden = (u, v)
        cells = [p for p in available(pos) if p != forbidden]
        sizes = list(range(3, len(cells) + 1))
        if kind == "majority" and label == winner:
            sizes = [n for n in sizes if n % 2]
        size = int(rng.choice(sizes))
        if required is not None:
            cells.remove(required)
            chosen = [required] + [cells[int(i)] for i in rng.choice(len(cells), size - 1, replace=False)]
        else:
            chosen = [cells[int(i)] for i in rng.choice(len(cells), size, replace=False)]
        if kind == "majority":
            minority_count = (size - 1) // 2 if label == winner else 1
        else:
            minority_count = int(rng.integers(1, (size - 1) // 2 + 1))
        minority = []
        eligible = chosen.copy()
        if required is not None:
            eligible.remove(required)
            if forced[pos][2] != label:
                minority.append(required)
        need = minority_count - len(minority)
        minority += [eligible[int(i)] for i in rng.choice(len(eligible), need, replace=False)]
        tile = np.full((h, w), neutral, dtype=np.int8)
        tile[h // 2, w // 2] = marker
        for u, v in chosen:
            tile[u, v] = (s0, s1)[1 - label if (u, v) in minority else label]
        sources[pos] = (tile, label)
    return _render(h, w, t, rows, cols, roles, sources)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shape = _SHAPES[int(rng.integers(len(_SHAPES)))]
    t = int(rng.choice((1, 2), p=(.7, .3)))
    stable_palette = bool(rng.random() < .35)
    base_roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    # Both witnesses exist even for 2/1. Their order carries no meaning.
    kinds = ["majority", "tie"] + ["ordinary"] * (ntrain - 2)
    rng.shuffle(kinds)
    for split, sequence in (("train", kinds), ("test", ["test"] * ntest)):
        for kind in sequence:
            roles = base_roles if stable_palette else tuple(map(int, rng.choice(10, 5, replace=False)))
            for _ in range(256):
                pair = _sample_pair(rng, shape, t, roles, kind)
                if pair is None:
                    continue
                expected, info = _infer(pair["input"])
                assert np.array_equal(expected, pair["output"])
                m = _metrics(info)
                accept = m["filled"] > 0
                if kind == "majority":
                    accept &= (m["strict_overlap"] > 0 and m["pixel_flip"] > 0 and
                               m["panel_flip"] and m["raw_flip"] and
                               m["stencil_matters"] and m["both_colors"])
                elif kind == "tie":
                    accept &= m["ties"] > 0 and m["clipped"] > 0
                if accept and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded broadcast sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, stats, geometries = set(), {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            expected, info = _infer(inp)
            assert np.array_equal(expected, out), "incorrect consensus or preservation"
            m = _metrics(info)
            assert m["filled"] > 0
            geometries.add((info["h"], info["w"], info["t"]))
            stats[split].append(m)
            # Explicit preservation and whole-receiver invariants.
            divider = inp == info["divider"]
            assert np.array_equal(inp[divider], out[divider])
            for source in info["sources"]:
                r, c = source["pos"]
                sl = _slice(r, c, info["h"], info["w"], info["t"])
                assert np.array_equal(inp[sl], out[sl])
            for r, c in info["ordinary"]:
                tile = out[_slice(r, c, info["h"], info["w"], info["t"])]
                assert np.all(tile == tile[0, 0])
    assert len(geometries) == 1
    tr = stats["train"]
    assert any(m["strict_overlap"] and m["pixel_flip"] and m["panel_flip"] and
               m["raw_flip"] and m["stencil_matters"] and m["both_colors"] for m in tr)
    assert any(m["ties"] and m["clipped"] for m in tr)
    assert any(m["clipped"] == 0 for m in tr)
    assert max(m["source_count"] for m in tr) < min(m["source_count"] for m in stats["test"])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        all(np.array_equal(x[k], y[k]) for k in ("input", "output"))
        for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed coverage plus semantic counterfactual/mutation checks."""
    shapes, geometry, masks, source_counts, margins, signatures = set(), set(), set(), set(), set(), set()
    ties = overlaps = clipped = pixel_flips = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                before = pair["input"].copy()
                prediction, info = _infer(pair["input"])
                assert np.array_equal(prediction, pair["output"])
                assert np.array_equal(before, pair["input"])
                m = _metrics(info)
                shapes.add(before.shape)
                geometry.add((info["h"], info["w"], info["t"]))
                source_counts.add(m["source_count"])
                ties += m["ties"]
                overlaps += m["strict_overlap"]
                clipped += m["clipped"] > 0
                pixel_flips += m["pixel_flip"]
                for s in info["sources"]:
                    masks.add(_key(s["mask"]))
                    margins.add(abs(s["counts"][0] - s["counts"][1]))
    assert len(signatures) == 42 and len(shapes) >= 25
    assert {g[:2] for g in geometry} == set(_SHAPES)
    assert {g[2] for g in geometry} == {1, 2}
    assert len(masks) >= 100 and source_counts == set(range(2, 11))
    assert len(margins) >= 5 and ties >= 42 and overlaps >= 42 and pixel_flips >= 42
    assert clipped >= 42
    assert _equal(generate_puzzle(913, 3, 2), generate_puzzle(913, 3, 2))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(71, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(0)
    pair = next(p for p in puzzle["train"] if _metrics(_infer(p["input"])[1])["pixel_flip"])
    expected, info = _infer(pair["input"])
    # Arbitrary color bijections commute with the rule.
    permutation = np.array([7, 3, 9, 0, 8, 1, 6, 4, 2, 5])
    assert np.array_equal(solve(permutation[pair["input"]]), permutation[expected])
    # Rearranging local ballots while keeping masks and counts changes only
    # the preserved source tiles, never the remote decisions.
    changed, wanted = pair["input"].copy(), expected.copy()
    for s in info["sources"]:
        r, c = s["pos"]
        sl = _slice(r, c, info["h"], info["w"], info["t"])
        tile = s["tile"].copy()
        tile[s["mask"]] = np.roll(tile[s["mask"]], 1)
        changed[sl] = wanted[sl] = tile
    assert np.array_equal(solve(changed), wanted)
    # Flip a close local majority without moving its footprint. At least one
    # transmitter in the planted opposed overlap reverses a receiver decision.
    found = False
    for s in info["sources"]:
        if abs(s["counts"][0] - s["counts"][1]) != 1:
            continue
        changed = pair["input"].copy()
        r, c = s["pos"]
        tile = changed[_slice(r, c, info["h"], info["w"], info["t"])]
        u, v = np.argwhere(tile == info["signals"][s["label"]])[0]
        tile[u, v] = info["signals"][1 - s["label"]]
        new = solve(changed)
        receiver_mask = np.zeros_like(new, dtype=bool)
        for y, x in info["ordinary"]:
            receiver_mask[_slice(y, x, info["h"], info["w"], info["t"])] = True
        found |= bool(np.any(new[receiver_mask] != expected[receiver_mask]))
    assert found, "local majority must affect remote decisions"
    # Move one ink cell into a hole while preserving its color and the local
    # vote. This must be able to change the addressed receivers.
    found = False
    for s in info["sources"]:
        holes = np.argwhere(s["tile"] == info["neutral"])
        for u, v in np.argwhere(s["mask"]):
            for y, x in holes:
                changed = pair["input"].copy()
                r, c = s["pos"]
                tile = changed[_slice(r, c, info["h"], info["w"], info["t"])]
                tile[y, x], tile[u, v] = tile[u, v], info["neutral"]
                new = solve(changed)
                if np.any(new[receiver_mask] != expected[receiver_mask]):
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found, "stencil geometry must affect remote decisions"
    # Reject a wrong fill, a changed protected divider, and filling a tied tile.
    for error in ("missing_fill", "divider", "tie"):
        damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()}
                       for p in puzzle[s]] for s in puzzle}
        p = (next(p for p in damaged["train"] if _metrics(_infer(p["input"])[1])["ties"])
             if error == "tie" else damaged["test"][0])
        _, f = _infer(p["input"])
        if error == "divider":
            p["output"][0, 0] = f["neutral"]
        elif error == "tie":
            y, x = next(pos for pos in f["ordinary"] if
                        f["votes"][pos][0] == f["votes"][pos][1] > 0)
            p["output"][_slice(y, x, f["h"], f["w"], f["t"])] = f["signals"][0]
        else:
            y, x = next(pos for pos in f["ordinary"] if f["winners"][pos] >= 0)
            p["output"][_slice(y, x, f["h"], f["w"], f["t"])] = f["neutral"]
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": 210, "canvas_shapes": len(shapes),
            "tile_geometries": len(geometry), "stencils": len(masks),
            "source_counts": sorted(source_counts), "vote_margins": sorted(margins),
            "tie_receivers": ties, "opposed_majority_receivers": overlaps,
            "pixel_vote_counterexamples": pixel_flips, "pairs_with_clipping": clipped}
