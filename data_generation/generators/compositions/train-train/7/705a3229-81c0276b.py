"""Nearest-edge signal voting on a decoded token lattice.

All semantics are visible in each input.  The generator samples scenes; solve
only decodes a scene and counts the signals arriving at each logical cell.
"""
from __future__ import annotations

from collections import Counter
import json

import numpy as np


PITCH = 5
SELF_TEST_SEEDS = tuple(range(42))


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _decode(value):
    a = _grid(value)
    height, width = a.shape
    if height % PITCH not in (0, 4) or width % PITCH not in (0, 4):
        raise ValueError("the final divider may be present or cropped")
    rows, cols = (height + 1) // PITCH, (width + 1) // PITCH
    if not (4 <= rows <= 6 and 4 <= cols <= 6):
        raise ValueError("expected a 4..6 by 4..6 logical lattice")
    background, divider = int(a[0, 0]), int(a[4, 0])
    if background == divider:
        raise ValueError("divider and background must differ")
    reconstructed = np.full(a.shape, background, dtype=np.int8)
    reconstructed[4::PITCH, :] = divider
    reconstructed[:, 4::PITCH] = divider
    tokens = np.empty((rows, cols), dtype=np.int8)
    sides = np.empty((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y, x = PITCH * r + 1, PITCH * c + 1
            color = int(a[y, x])
            if color == background:
                raise ValueError("every logical cell needs a token")
            side = 2 if int(a[y, x + 1]) == color else 1
            tokens[r, c], sides[r, c] = color, side
            reconstructed[y:y + side, x:x + side] = color
    if not np.array_equal(a, reconstructed):
        raise ValueError("malformed lattice, square token, or token position")
    payload = tokens != divider
    if not (2 <= len(np.unique(tokens[payload])) <= 4 and np.any(~payload)):
        raise ValueError("need 2..4 payload colors and at least one null token")
    for r, c in np.argwhere(payload):
        if r == rows - 1 - r or c == cols - 1 - c:
            raise ValueError("payload tokens cannot have a nearest-edge tie")
    return {"background": background, "divider": divider, "tokens": tokens,
            "sides": sides, "crop": (PITCH * rows - height, PITCH * cols - width)}


def _l_mask(shape, r, c):
    """A token casts one signal per cell, including its corner only once."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    if r < height - 1 - r:
        mask[:r + 1, c] = True
    else:
        mask[r:, c] = True
    if c < width - 1 - c:
        mask[r, :c + 1] = True
    else:
        mask[r, c:] = True
    return mask


def _votes(info, *, area_weighted=False):
    tokens, sides, divider = info["tokens"], info["sides"], info["divider"]
    votes = np.zeros((*tokens.shape, 10), dtype=np.int16)
    for rr, cc in np.argwhere(tokens != divider):
        r, c = int(rr), int(cc)
        weight = int(sides[r, c]) ** 2 if area_weighted else 1
        votes[:, :, int(tokens[r, c])] += _l_mask(tokens.shape, r, c) * weight
    return votes


def _winners(votes, background):
    maxima = votes.max(axis=2)
    unique = (votes == maxima[:, :, None]).sum(axis=2) == 1
    out = np.full(votes.shape[:2], background, dtype=np.int8)
    active = (maxima > 0) & unique
    out[active] = votes.argmax(axis=2)[active]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode tokens, cast nearest-edge L signals, take local plurality.

    A tied largest positive frequency (or no signals) gives background.
    Neither token area, ray length, color ID, nor processing order adds weight.
    """
    info = _decode(input_grid)
    return _winners(_votes(info), info["background"])


def _features(info):
    tokens, divider = info["tokens"], info["divider"]
    votes = _votes(info)
    maximum = votes.max(axis=2)
    support = (votes > 0).sum(axis=2)
    unique = (votes == maximum[:, :, None]).sum(axis=2) == 1
    out = _winners(votes, info["background"])
    weighted = _winners(_votes(info, area_weighted=True), info["background"])
    presence = _winners((votes > 0).astype(np.int16), info["background"])
    global_counts = Counter(int(v) for v in tokens.flat if v != divider)
    priority = np.zeros(10, dtype=np.int16)
    for color, count in global_counts.items():
        priority[color] = count
    global_priority = _winners((votes > 0) * priority, info["background"])
    orientations = set()
    for r, c in np.argwhere(tokens != divider):
        orientations.add((int(r > tokens.shape[0] - 1 - r),
                          int(c > tokens.shape[1] - 1 - c)))
    return {
        "votes": votes, "output": out, "counts": global_counts,
        "orientations": orientations,
        "majority": bool(np.any(unique & (maximum >= 2) & (support >= 2))),
        "tie": bool(np.any((~unique) & (maximum > 0))),
        "null_tie": bool(np.any((tokens == divider) & (~unique) & (maximum > 0))),
        "plurality": bool(np.any(unique & (support >= 3)
                                 & (2 * maximum <= votes.sum(axis=2)))),
        "area_changes": not np.array_equal(out, weighted),
        "presence_changes": not np.array_equal(out, presence),
        "global_changes": not np.array_equal(out, global_priority),
        "new_cells": int(np.count_nonzero((tokens == divider)
                                         & (out != info["background"]))),
        "erased_sources": int(np.count_nonzero((tokens != divider)
                                              & (out != tokens))),
    }


def _render(tokens, sides, background, divider, crop):
    rows, cols = tokens.shape
    a = np.full((PITCH * rows - crop[0], PITCH * cols - crop[1]),
                background, dtype=np.int8)
    a[4::PITCH, :] = divider
    a[:, 4::PITCH] = divider
    for r in range(rows):
        for c in range(cols):
            y, x, side = PITCH * r + 1, PITCH * c + 1, int(sides[r, c])
            a[y:y + side, x:x + side] = tokens[r, c]
    return a


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _make_pair(rng, background, divider, colors, regime, seen):
    # Rejection conditions refer to visible witnesses, never a hidden solution
    # parameter. Sampling all shapes anew prevents an infeasible fixed layout.
    for _ in range(6000):
        rows, cols = (int(v) for v in rng.integers(4, 7, size=2))
        eligible = [(r, c) for r in range(rows) for c in range(cols)
                    if r != rows - 1 - r and c != cols - 1 - c]
        lower, upper = {"two": (8, 10), "three": (10, 12),
                        "extra": (5, 12), "test": (13, 20)}[regime]
        upper = min(upper, len(eligible) - 1)
        if upper < lower:
            continue
        count = int(rng.integers(lower, upper + 1))
        locations = rng.choice(len(eligible), size=count, replace=False)
        values = list(colors) + [int(rng.choice(colors))
                                 for _ in range(count - len(colors))]
        rng.shuffle(values)
        tokens = np.full((rows, cols), divider, dtype=np.int8)
        for index, color in zip(locations, values):
            tokens[eligible[int(index)]] = color
        sides = rng.integers(1, 3, size=(rows, cols), dtype=np.int8)
        info = {"background": background, "divider": divider,
                "tokens": tokens, "sides": sides}
        f = _features(info)
        if (f["new_cells"] < 3 or not f["tie"] or not f["majority"]
                or not f["presence_changes"] or not f["global_changes"]):
            continue
        if regime in ("two", "three"):
            if (len(f["orientations"]) != 4 or not f["null_tie"]
                    or not f["area_changes"] or not f["erased_sources"]
                    or set(np.unique(f["output"])) - {background} != set(colors)):
                continue
            if regime == "three" and not f["plurality"]:
                continue
        crop = tuple(int(v) for v in rng.integers(0, 2, size=2))
        a = _render(tokens, sides, background, divider, crop)
        if _key(a) in seen:
            continue
        seen.add(_key(a))
        return {"input": a, "output": f["output"]}
    raise RuntimeError("failed to sample a scene with the required visible witnesses")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    bg0, bg1, d0, d1, shared, third = palette[:6]
    puzzle, seen = {"train": [], "test": []}, set()
    puzzle["train"].append(_make_pair(rng, bg0, d0, [d1, shared], "two", seen))
    puzzle["train"].append(_make_pair(rng, bg1, d1, [d0, shared, third], "three", seen))
    for split, start, stop in (("train", 2, num_train), ("test", 0, num_test)):
        for _ in range(start, stop):
            role_palette = [int(c) for c in rng.permutation(10)]
            background, divider = role_palette[:2]
            k = int(rng.integers(2, 5))
            puzzle[split].append(_make_pair(
                rng, background, divider, role_palette[2:2 + k],
                "extra" if split == "train" else "test", seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check geometry, exact local counts, conservation, and teaching evidence."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad keys")
    _check(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad counts")
    seen, decoded = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _decode(inp)
            tokens, divider = info["tokens"], info["divider"]
            f = _features(info)
            _check(out.shape == tokens.shape, "output must be the logical lattice")
            _check(np.array_equal(out, f["output"]), "local vote result is incorrect")
            _check(f["new_cells"] >= 3 and f["majority"] and f["tie"],
                   "scene lacks visible extension, multiplicity, or cancellation")
            _check(f["presence_changes"] and f["global_changes"],
                   "local token frequency must affect the answer")
            # Every source contributes exactly one vote along its union of arms.
            signal_total = 0
            for r, c in np.argwhere(tokens != divider):
                signal_total += min(r, tokens.shape[0] - 1 - r)
                signal_total += min(c, tokens.shape[1] - 1 - c) + 1
            _check(int(f["votes"].sum()) == signal_total, "signal mass was double counted")
            # The background and divider never cast signals, including color 0.
            _check(not np.any(f["votes"][:, :, [info["background"], divider]]),
                   "structural or background signals were counted")
            _check(np.all(np.isin(out, [info["background"], *f["counts"]])),
                   "output has a color without a payload source")
            decoded[split].append((info, f))
    first, second = decoded["train"][:2]
    _check(first[0]["background"] != second[0]["background"], "background role not taught")
    _check(first[0]["divider"] != second[0]["divider"], "divider role not taught")
    _check(first[0]["divider"] in second[1]["counts"]
           and second[0]["divider"] in first[1]["counts"], "null/payload roles not exchanged")
    _check(len(first[1]["counts"]) == 2 and len(second[1]["counts"]) == 3,
           "training needs two- and three-color interference")
    for info, f in (first, second):
        _check(len(f["orientations"]) == 4 and f["area_changes"]
               and f["null_tie"] and f["erased_sources"], "anchor witnesses are absent")
        _check(set(np.unique(f["output"])) - {info["background"]} == set(f["counts"]),
               "an anchor payload color never survives")
    _check(second[1]["plurality"], "three-color plurality is not taught")
    max_train = max(sum(f["counts"].values()) for _, f in decoded["train"])
    _check(max_train <= 12, "training source count too high")
    _check(all(13 <= sum(f["counts"].values()) <= 20
               and sum(f["counts"].values()) > max_train
               for _, f in decoded["test"]), "missing source-count shift")


def _serial(puzzle):
    return json.dumps({k: [{q: np.asarray(p[q]).tolist() for q in ("input", "output")}
                           for p in puzzle[k]] for k in ("train", "test")},
                      separators=(",", ":"))


def self_test():
    """A bounded 42-seed audit, with measured structural variation."""
    signatures, shapes, layouts, counts, sides, backgrounds, crops = (set() for _ in range(7))
    all_orientations, pixel_witnesses, pluralities, erased = set(), 0, 0, 0
    for seed in SELF_TEST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_serial(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                f = _features(info)
                _check(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                tokens = info["tokens"]
                shapes.add((pair["input"].shape, pair["output"].shape))
                role_map = {info["divider"]: 0}
                normalized = bytes(role_map.setdefault(int(v), len(role_map)) for v in tokens.flat)
                layouts.add((tokens.shape, normalized))
                counts.add((split, len(f["counts"]), sum(f["counts"].values())))
                sides.add(tuple(sorted(set(int(v) for v in info["sides"].flat))))
                backgrounds.add(info["background"])
                crops.add(info["crop"])
                all_orientations.update(f["orientations"])
                pixel_witnesses += f["area_changes"]
                pluralities += f["plurality"]
                erased += f["erased_sources"] > 0
        # Metamorphic checks change representation, color IDs, and geometry
        # without using generation metadata as a solving convention.
        anchor = puzzle["train"][0]
        info = _decode(anchor["input"])
        resized = _render(info["tokens"], 3 - info["sides"], info["background"],
                          info["divider"], tuple(1 - v for v in info["crop"]))
        _check(np.array_equal(solve(resized), anchor["output"]),
               "patch area or final divider crop changed the logical answer")
        rotated = _render(np.rot90(info["tokens"]), np.rot90(info["sides"]),
                          info["background"], info["divider"], (0, 0))
        _check(np.array_equal(solve(rotated), np.rot90(anchor["output"])),
               "nearest-edge voting did not rotate with the logical scene")
        permutation = np.array([(3 * v + seed) % 10 for v in range(10)], dtype=np.int8)
        _check(np.array_equal(solve(permutation[anchor["input"]]),
                              permutation[anchor["output"]]), "literal color shortcut")
    _check(len(signatures) == 42, "repeated puzzle across seeds")
    _check(len(shapes) >= 25 and len(layouts) >= 180, "insufficient geometric variation")
    _check(len(counts) >= 25 and len(backgrounds) == 10 and len(crops) == 4,
           "insufficient counts, palettes, or boundary crop variation")
    _check((1, 2) in sides and len(all_orientations) == 4, "size or direction coverage missing")
    _check(pixel_witnesses >= 84 and pluralities >= 42 and erased >= 84,
           "family-specific witnesses are missing")
    _check(_serial(generate_puzzle(123, 3, 2)) == _serial(generate_puzzle(123, 3, 2)),
           "non-deterministic puzzle")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            p = generate_puzzle(1000 + train * 10 + test, train, test)
            _check((len(p["train"]), len(p["test"])) == (train, test), "count mismatch")
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A hand-calculated local witness, independent of scene sampling.
    tokens = np.full((6, 6), 9, dtype=np.int8)
    tokens[1, 1] = tokens[1, 2] = 2
    tokens[2, 0] = tokens[2, 2] = 3
    token_sides = np.ones((6, 6), dtype=np.int8)
    token_sides[2, 0] = 2
    a = _render(tokens, token_sides, 0, 9, (0, 0))
    out = solve(a)
    _check(out[1, 0] == 2, "two small signals must beat one large token")
    _check(out[1, 2] == 0 and out[0, 2] == 0, "one-versus-one must cancel")
    _check(out[2, 0] == 3, "same-color signals must coalesce")
    tokens.fill(9)
    tokens[0, 2] = tokens[1, 1] = 2
    tokens[2, 1], tokens[0, 1] = 3, 4
    out = solve(_render(tokens, token_sides, 0, 9, (0, 0)))
    _check(out[0, 1] == 2, "2 versus 1 versus 1 is a unique plurality")
    tokens[0, 1] = 3
    out = solve(_render(tokens, token_sides, 0, 9, (0, 0)))
    _check(out[0, 1] == 0, "2 versus 2 must cancel")
    return {"seeds": 42, "distinct_puzzles": len(signatures),
            "shape_pairs": len(shapes), "normalized_layouts": len(layouts),
            "count_profiles": len(counts), "backgrounds": len(backgrounds),
            "pixel_weight_counterexamples": pixel_witnesses,
            "three_color_pluralities": pluralities, "source_erasure_cases": erased}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
