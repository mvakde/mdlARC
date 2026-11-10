"""Rare-cell inspection windows into overlapping, initially hollow plates.

Only NumPy and the standard library are used. All scene recovery in solve is
input-only; generation's anchors and layer order are never passed to it.
"""
from __future__ import annotations

import json
import numpy as np


RING_ROWS = ("001100", "010010", "100001", "100001", "010010", "001100")
RING = tuple((r, c) for r, row in enumerate(RING_ROWS)
             for c, value in enumerate(row) if value == "1")
SOLID = tuple((r, c) for r in range(6)
              for c in range(min(c for rr, c in RING if rr == r),
                             max(c for rr, c in RING if rr == r) + 1))
_RING_SET = frozenset(RING)
OFFSETS = tuple((dr, dc) for dr in range(-5, 6) for dc in range(-5, 6)
                if (dr, dc) != (0, 0)
                and 2 <= len(_RING_SET & {(r + dr, c + dc) for r, c in RING}) <= 6)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _cells(anchor, stencil):
    top, left = anchor
    return {(top + dr, left + dc) for dr, dc in stencil}


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _candidates(a, color, background, probe):
    """Fit the *entire* visible color fragment, including potential occlusion."""
    visible = {tuple(map(int, p)) for p in np.argwhere(a == color)}
    if len(visible) < 4:
        return []
    r, c = min(visible)
    candidates = []
    for dr, dc in RING:
        anchor = (r - dr, c - dc)
        full = _cells(anchor, RING)
        bounded = {p for p in full if _inside(p, a.shape)}
        if (len(bounded) >= 8 and visible <= bounded
                and all(int(a[p]) not in (background, probe) for p in bounded)):
            candidates.append(anchor)
    return sorted(set(candidates))


def _unique_order(edges):
    remaining = set(edges)
    order = []
    while remaining:
        choices = [c for c in sorted(remaining)
                   if not any(c in edges[other] for other in remaining)]
        if len(choices) != 1:
            raise ValueError("crossings must determine one total depth order")
        front = choices[0]
        remaining.remove(front)
        order.append(front)
    return tuple(order)


def _recover(a):
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= a.size // 2:
        raise ValueError("background must occupy a strict majority")
    rare = [c for c in range(10) if c != background and counts[c] == 1]
    if len(rare) != 1:
        raise ValueError("there must be exactly one singleton color")
    probe = rare[0]
    center = tuple(map(int, np.argwhere(a == probe)[0]))
    if not (1 <= center[0] < a.shape[0] - 1 and 1 <= center[1] < a.shape[1] - 1):
        raise ValueError("the inspection window must fit completely")
    placements, noise = {}, []
    for color in range(10):
        if color in (background, probe) or counts[color] == 0:
            continue
        candidates = _candidates(a, color, background, probe)
        if len(candidates) > 1:
            raise ValueError("ambiguous ring placement")
        if candidates:
            placements[color] = candidates[0]
        else:
            noise.append(color)
    if not 2 <= len(placements) <= 5:
        raise ValueError("expected two to five recoverable rings")
    if any(not 2 <= counts[c] <= 8 for c in noise):
        raise ValueError("stray colors must repeat two to eight times")
    occupants = {}
    for color, anchor in placements.items():
        for p in _cells(anchor, RING):
            if _inside(p, a.shape):
                occupants.setdefault(p, []).append(color)
    edges = {color: set() for color in placements}
    for p, owners in occupants.items():
        visible = int(a[p])
        if visible not in owners:
            raise ValueError("a ring boundary is missing or incorrectly occluded")
        edges[visible].update(c for c in owners if c != visible)
    order = _unique_order(edges)
    solids = {c: _cells(anchor, SOLID) for c, anchor in placements.items()}
    if center in occupants or not any(center in s for s in solids.values()):
        raise ValueError("probe must lie in a hollow interior")
    return {"background": background, "probe": probe, "center": center,
            "placements": placements, "order": order, "solids": solids,
            "noise": tuple(noise), "counts": counts, "edges": edges,
            "boundaries": occupants}


def _window(shape, background, probe, center, solids, order):
    result = np.full(shape, background, dtype=np.int8)
    row, col = center
    for r in range(row - 1, row + 2):
        for c in range(col - 1, col + 2):
            for color in order:  # first covering plate is in front
                if (r, c) in solids[color]:
                    result[r, c] = color
                    break
    result[center] = probe
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete solid plates in depth order, visible only in the rare-cell window."""
    a = _grid(input_grid)
    info = _recover(a)
    return _window(a.shape, info["background"], info["probe"], info["center"],
                   info["solids"], info["order"])


def _window_metrics(a, info):
    out = _window(a.shape, info["background"], info["probe"], info["center"],
                  info["solids"], info["order"])
    r, c = info["center"]
    neighbors = [(rr, cc) for rr in range(r - 1, r + 2)
                 for cc in range(c - 1, c + 2) if (rr, cc) != (r, c)]
    colors = {int(out[p]) for p in neighbors} & set(info["order"])
    painted_blank = any(a[p] == info["background"] and out[p] != info["background"]
                        for p in neighbors)
    reversed_out = _window(a.shape, info["background"], info["probe"], info["center"],
                           info["solids"], info["order"][::-1])
    # Demand a strict count inversion, so the teaching witness does not
    # depend on an arbitrary way of breaking equal-frequency ties.
    frequency_wrong = False
    for p in neighbors:
        owners = [color for color in info["order"] if p in info["solids"][color]]
        if owners and any(info["counts"][other] > info["counts"][owners[0]]
                          for other in owners[1:]):
            frequency_wrong = True
    boxes = {color: {(rr, cc) for rr in range(anchor[0], anchor[0] + 6)
                     for cc in range(anchor[1], anchor[1] + 6)}
             for color, anchor in info["placements"].items()}
    box_out = _window(a.shape, info["background"], info["probe"], info["center"],
                      boxes, info["order"])
    return {"colors": colors, "painted_blank": painted_blank,
            "depth_sensitive": not np.array_equal(out, reversed_out),
            "frequency_wrong": frequency_wrong,
            "boxes_wrong": not np.array_equal(out, box_out),
            "blank_neighbors": sum(out[p] == info["background"] for p in neighbors)}


def _geometry(rng, count):
    anchors = [(0, 0)]
    for _ in range(count - 1):
        options = [(anchors[-1][0] + dr, anchors[-1][1] + dc) for dr, dc in OFFSETS]
        options = [p for p in options if p not in anchors]
        anchors.append(options[int(rng.integers(len(options)))])
    minr = min(r for r, _ in anchors)
    minc = min(c for _, c in anchors)
    span = (max(r for r, _ in anchors) - minr + 6,
            max(c for _, c in anchors) - minc + 6)
    margins = [int(v) for v in rng.integers(0, 7, size=4)]
    if rng.random() < 0.18:
        margins[int(rng.integers(4))] = -1
    top, bottom, left, right = margins
    height, width = max(9, span[0] + top + bottom), max(9, span[1] + left + right)
    if rng.random() < 0.25:
        height = width = max(height, width)
    if height > 30 or width > 30:
        return None
    anchors = [(r - minr + top, c - minc + left) for r, c in anchors]
    return (height, width), anchors


def _make_pair(rng, count, palette, require_frequency=False, require_blank=False):
    background, probe = palette[:2]
    colors = palette[2:2 + count]
    noise_colors = palette[2 + count:]
    # The geometry is rejected on visible evidence, before an output is accepted.
    for _ in range(3000):
        geometry = _geometry(rng, count)
        if geometry is None:
            continue
        shape, anchors = geometry
        boundaries = {color: _cells(anchor, RING) for color, anchor in zip(colors, anchors)}
        solids = {color: _cells(anchor, SOLID) for color, anchor in zip(colors, anchors)}
        union = set.union(*boundaries.values())
        a = np.full(shape, background, dtype=np.int8)
        for color in reversed(colors):
            for p in boundaries[color]:
                if _inside(p, shape):
                    a[p] = color
        # Choose a probe in a hollow interior. Its square must
        # expose multiple plate colors and require both filling and depth.
        possible = []
        for r in range(1, shape[0] - 1):
            for c in range(1, shape[1] - 1):
                p = (r, c)
                if p in union or not any(p in s for s in solids.values()):
                    continue
                info = {"background": background, "probe": probe, "center": p,
                        "solids": solids, "order": tuple(colors),
                        "placements": dict(zip(colors, anchors)),
                        "counts": np.bincount(a.ravel(), minlength=10)}
                metrics = _window_metrics(a, info)
                # The corner demonstration reads its depth from remote
                # crossings; none of its outline crossings is in the window.
                local_crossing = any(sum(q in b for b in boundaries.values()) > 1
                                     for q in ((rr, cc) for rr in range(r-1, r+2)
                                               for cc in range(c-1, c+2)))
                if (len(metrics["colors"]) >= 2 and metrics["painted_blank"]
                        and metrics["depth_sensitive"]
                        and (not require_frequency or metrics["frequency_wrong"])
                        and (not require_blank or (metrics["blank_neighbors"] > 0
                             and metrics["boxes_wrong"] and not local_crossing))):
                    possible.append(p)
        if not possible:
            continue
        center = possible[int(rng.integers(len(possible)))]
        a[center] = probe
        # Repeated non-ring colors provide false candidate probe dots. Place
        # one in another hollow interior where possible, and scatter the rest.
        available = sorted(p for p in np.ndindex(shape) if p not in union and p != center)
        noise_count = int(rng.integers(1, min(2, len(noise_colors)) + 1))
        for color in noise_colors[:noise_count]:
            amount = int(rng.integers(2, 9))
            preferred = [p for p in available if p in set.union(*solids.values())]
            sites = []
            if preferred and rng.random() < 0.85:
                sites.append(preferred[int(rng.integers(len(preferred)))])
            pool = [p for p in available if p not in sites]
            sites += [pool[int(i)] for i in rng.choice(len(pool), amount - len(sites), replace=False)]
            for p in sites:
                a[p] = color
            taken = set(sites)
            available = [p for p in available if p not in taken]
        try:
            info = _recover(a)
            if info["order"] != tuple(colors) or info["placements"] != dict(zip(colors, anchors)):
                continue
            metrics = _window_metrics(a, info)
            if not metrics["painted_blank"]:
                continue
            if require_frequency and not metrics["frequency_wrong"]:
                continue
        except ValueError:
            continue
        out = _window(shape, background, probe, center, solids, tuple(colors))
        return {"input": a, "output": out}
    raise RuntimeError("bounded scene sampling exhausted")


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    first_palette = [int(x) for x in rng.permutation(10)]
    second_palette = [int(x) for x in rng.permutation(10)]
    # Swap the first probe and one first-ring color into ring/probe roles in
    # the second demonstration. Neither literal color can identify the probe.
    second_probe = first_palette[2]
    second_ring = first_palette[1]
    remaining = [x for x in second_palette if x not in (second_probe, second_ring)]
    second_palette = [remaining[0], second_probe, second_ring] + remaining[1:]
    plans = [("train", 2, first_palette, False, True),
             ("train", 3, second_palette, True, False)]
    plans += [("train", int(rng.integers(2, 4)),
               [int(x) for x in rng.permutation(10)], False, False)
              for _ in range(num_train - 2)]
    plans += [("test", int(rng.integers(4, 6)),
               [int(x) for x in rng.permutation(10)], False, False)
              for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count, palette, require_frequency, require_blank in plans:
        for _ in range(10):
            pair = _make_pair(rng, count, palette, require_frequency, require_blank)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("could not draw distinct inputs")
        seen.add(_key(pair["input"]))
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check structure, latent geometry recoverability, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos, metrics_by_split = set(), {"train": [], "test": []}, {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            try:
                a, out = _grid(pair["input"]), _grid(pair["output"])
                info = _recover(a)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            assert a.shape == out.shape
            assert _key(a) not in seen
            seen.add(_key(a))
            assert np.array_equal(solve(a), out)
            metrics = _window_metrics(a, info)
            assert len(metrics["colors"]) >= 2
            assert metrics["painted_blank"] and metrics["depth_sensitive"]
            assert len(info["noise"]) >= 1
            # Removing B's interior completion or reversing its layer order
            # changes every accepted answer, even at the smallest pair count.
            r, c = info["center"]
            copy_patch = np.full(a.shape, info["background"], dtype=np.int8)
            copy_patch[r-1:r+2, c-1:c+2] = a[r-1:r+2, c-1:c+2]
            assert not np.array_equal(copy_patch, out)
            infos[split].append(info)
            metrics_by_split[split].append(metrics)
    assert any(m["frequency_wrong"] for m in metrics_by_split["train"])
    assert any(m["blank_neighbors"] > 0 for m in metrics_by_split["train"])
    assert any(m["boxes_wrong"] for m in metrics_by_split["train"])
    assert any(not any(len(owners) > 1 and abs(p[0]-i["center"][0]) <= 1
                      and abs(p[1]-i["center"][1]) <= 1
                      for p, owners in i["boundaries"].items()) for i in infos["train"])
    assert len({i["probe"] for i in infos["train"]}) >= 2
    for info in infos["train"][:2]:
        assert any(info["probe"] in other["order"] for other in infos["train"])
    assert min(len(i["order"]) for i in infos["test"]) > max(len(i["order"]) for i in infos["train"])


def self_test():
    """Bounded 42-seed checks, ablations, equivariance, and measured diversity."""
    shapes, layouts, centers, backgrounds, probes, ring_counts = set(), set(), set(), set(), set(), set()
    clipped = edge_touch = blank_windows = frequency_witnesses = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _recover(a)
                m = _window_metrics(a, info)
                anchors = [info["placements"][color] for color in info["order"]]
                minr, minc = min(r for r, c in anchors), min(c for r, c in anchors)
                layouts.add(tuple((r-minr, c-minc) for r, c in anchors))
                centers.add(tuple((info["center"][0]-r, info["center"][1]-c) for r,c in anchors))
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                probes.add(info["probe"])
                ring_counts.add(len(anchors))
                clipped += any(not all(_inside(p, a.shape) for p in _cells(anchor, RING)) for anchor in anchors)
                edge_touch += any(r == 0 or c == 0 or r+5 == a.shape[0]-1 or c+5 == a.shape[1]-1 for r,c in anchors)
                blank_windows += int(m["blank_neighbors"] > 0)
                frequency_witnesses += m["frequency_wrong"]
        # A's global frequency role is unchanged by a bijection of literal colors.
        permutation = np.random.default_rng(seed + 1000).permutation(10).astype(np.int8)
        sample = puzzle["test"][0]
        assert np.array_equal(solve(permutation[sample["input"]]), permutation[sample["output"]])
        # Reflecting the scene reflects the local output, with no direction convention.
        assert np.array_equal(solve(sample["input"][:, ::-1]), sample["output"][:, ::-1])
        assert np.array_equal(solve(np.rot90(sample["input"])), np.rot90(sample["output"]))
        # Omitting A and retaining every candidate dot makes a different
        # answer: repeated stray colors have no place in the output.
        info = _recover(sample["input"])
        all_dots = sample["output"].copy()
        for color in info["noise"]:
            all_dots[sample["input"] == color] = color
        assert not np.array_equal(all_dots, sample["output"])
        # A controlled B counterfactual: change only the two-ring scene's
        # crossing ownership, keeping the rare cell and ring translations.
        sample = puzzle["train"][0]
        info = _recover(sample["input"])
        flipped = sample["input"].copy()
        reverse = info["order"][::-1]
        for color in reversed(reverse):
            for p in _cells(info["placements"][color], RING):
                if _inside(p, flipped.shape):
                    flipped[p] = color
        expected = _window(flipped.shape, info["background"], info["probe"],
                           info["center"], info["solids"], reverse)
        assert np.array_equal(solve(flipped), expected)
        assert not np.array_equal(expected, sample["output"])
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(layouts) >= 100 and len(centers) >= 70
    assert len(backgrounds) == 10 and len(probes) == 10 and ring_counts == {2, 3, 4, 5}
    assert clipped >= 5 and edge_touch >= 5 and blank_windows >= 10 and frequency_witnesses >= 42
    # Whole-puzzle determinism and the supported legal count extremes.
    for ntrain, ntest in ((2, 1), (10, 4)):
        a, b = generate_puzzle(51, ntrain, ntest), generate_puzzle(51, ntrain, ntest)
        assert all(np.array_equal(x[k], y[k]) for s in a for x,y in zip(a[s],b[s]) for k in x)
    for ntrain, ntest in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # The validator must reject corruption, including losing the unique probe.
    bad = generate_puzzle(9, 2, 1)
    info = _recover(bad["train"][0]["input"])
    bad["train"][0]["input"][info["center"]] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("erased singleton was accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "relative_layer_layouts": len(layouts), "probe_layer_configurations": len(centers),
            "ring_counts": sorted(ring_counts), "background_colors": len(backgrounds),
            "probe_colors": len(probes), "clipped_scenes": clipped,
            "edge_touching_scenes": edge_touch, "windows_with_background": blank_windows,
            "frequency_order_counterexamples": frequency_witnesses}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
