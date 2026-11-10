"""Synthetic hollow-diamond cue projection puzzles."""

import hashlib

import numpy as np


_SIZES = tuple(range(9, 30, 2))
_LAYOUTS = (
    "canonical_vertical",
    "mirrored_vertical",
    "canonical_horizontal",
    "mirrored_horizontal",
)
_SOURCE_DIGESTS = {
    "759d4e444204e9a8794102c823bb16c93fae3510a816f9f4736c698e5e350941",
    "63db56d11e3a2742da6149280d36a21cc32f1d8096b4f7882d06aa9e28476740",
    "775eb1dc0b90129dc67ef35ccb5de3dea306ab41748ae8031ef058274f8b512a",
    "3b857176a2f25571580d988831eadedbe4f8e38d26d408b0eb8422b2db112f41",
}


def _is_count(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        if not _is_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _is_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _diamond_cells(n):
    m = n // 2
    return {(r, c) for r in range(n) for c in range(n)
            if abs(r - m) + abs(c - m) == m}


def _line_specs(layout, n):
    """Cue, ordered interior chord, and active edge half for each line."""
    m = n // 2
    specs = []
    if layout not in _LAYOUTS:
        raise AssertionError("unknown layout")
    vertical = layout.endswith("vertical")
    mirrored = layout.startswith("mirrored")
    if vertical:
        top_columns = range(0, m) if mirrored else range(m + 1, n)
        bottom_columns = range(m + 1, n) if mirrored else range(0, m)
        for side, columns in ((0, top_columns), (1, bottom_columns)):
            for c in columns:
                top = abs(c - m)
                bottom = n - 1 - top
                rows = (range(top + 1, bottom) if side == 0
                        else range(bottom - 1, top, -1))
                specs.append({"cue": (0 if side == 0 else n - 1, c),
                              "path": [(r, c) for r in rows], "side": side})
    else:
        left_rows = range(m + 1, n) if mirrored else range(0, m)
        right_rows = range(0, m) if mirrored else range(m + 1, n)
        for side, rows in ((0, left_rows), (1, right_rows)):
            for r in rows:
                left = abs(r - m)
                right = n - 1 - left
                columns = (range(left + 1, right) if side == 0
                           else range(right - 1, left, -1))
                specs.append({"cue": (r, 0 if side == 0 else n - 1),
                              "path": [(r, c) for c in columns], "side": side})
    return specs


def _apply_rule(inp, background, layout):
    """Project visible cues inward, stopping before the first input blocker."""
    out = inp.copy()
    for spec in _line_specs(layout, inp.shape[0]):
        value = int(inp[spec["cue"]])
        for cell in spec["path"]:
            if int(inp[cell]) != background:
                break
            out[cell] = value
    return out


def _colors(rng, avoid_background=None, reserved=None):
    backgrounds = [c for c in range(10) if c not in (avoid_background, reserved)]
    background = int(rng.choice(backgrounds))
    outlines = [c for c in range(10) if c not in (background, reserved)]
    outline = int(rng.choice(outlines))
    return background, outline


def _sample_cues(rng, specs, background, outline, style, excluded=(), forced=None):
    available = [c for c in range(10) if c not in (background, outline, *excluded)]
    assert available
    if style == "repeated":
        palette = [int(rng.choice(available))]
    elif style == "alternating":
        palette = [int(x) for x in rng.choice(available, size=2, replace=False)]
    else:
        k = int(rng.integers(2, min(6, len(available)) + 1))
        palette = [int(x) for x in rng.choice(available, size=k, replace=False)]
    values = {spec["cue"]: background for spec in specs}
    for side in (0, 1):
        group = [spec for spec in specs if spec["side"] == side and spec["path"]]
        assert group
        if style == "sparse":
            k = int(rng.integers(1, min(3, len(group)) + 1))
            chosen = [group[int(i)] for i in rng.choice(len(group), size=k, replace=False)]
        elif style == "short_run":
            k = int(rng.integers(2, min(5, len(group)) + 1))
            start = int(rng.integers(0, len(group) - k + 1))
            chosen = group[start:start + k]
        else:
            probability = float(rng.uniform(.35, .9))
            chosen = [spec for spec in group if rng.random() < probability]
            if not chosen:
                chosen = [group[int(rng.integers(len(group)))]]
        for index, spec in enumerate(chosen):
            value = (palette[index % 2] if style == "alternating"
                     else int(rng.choice(palette)))
            values[spec["cue"]] = value
    if forced is not None:
        assert forced not in (background, outline)
        visible = [spec for spec in specs if values[spec["cue"]] != background and spec["path"]]
        target = max(visible, key=lambda spec: len(spec["path"]))
        values[target["cue"]] = forced
    for side in (0, 1):
        long_specs = [s for s in specs if s["side"] == side and len(s["path"]) >= 3]
        if not any(values[s["cue"]] != background for s in long_specs):
            target = max(long_specs, key=lambda s: len(s["path"]))
            values[target["cue"]] = int(rng.choice(palette))
    return values


def _sample_blockers(rng, specs, cues, background, mode):
    """Reserve one live cue chord so even endpoint blocks leave an informative pair."""
    visible = [s for s in specs if s["path"] and cues[s["cue"]] != background]
    assert len(visible) >= 2
    protected = max((s for s in visible if s["side"] == 1),
                    key=lambda s: len(s["path"]))
    eligible = [s for s in visible if s is not protected and len(s["path"]) >= 3]
    primary = max(eligible, key=lambda s: len(s["path"]))
    candidates = [cell for s in specs if s is not protected for cell in s["path"]]
    cells = []
    if mode == "none":
        pass
    elif mode == "few":
        count = int(rng.integers(1, min(3, len(candidates)) + 1))
        cells = [candidates[int(i)] for i in rng.choice(len(candidates), size=count, replace=False)]
    elif mode == "many":
        count = int(rng.integers(4, min(8, len(candidates)) + 1))
        cells = [candidates[int(i)] for i in rng.choice(len(candidates), size=count, replace=False)]
    elif mode == "first":
        cells = [primary["path"][0]]
    elif mode == "last":
        cells = [primary["path"][-1]]
    elif mode == "shadow":
        path = primary["path"]
        first = (1 if len(path) <= 4
                 else int(rng.integers(1, len(path) - 2)))
        cells = [path[first]]
        if len(path) >= 5 and rng.random() < .45:
            second = int(rng.integers(first + 1, len(path)))
            cells.append(path[second])
    elif mode == "double":
        path = primary["path"]
        first = int(rng.integers(0, len(path) - 2))
        second = int(rng.integers(first + 1, len(path)))
        cells = [path[first], path[second]]
    else:
        raise AssertionError("unknown blocker mode")
    return cells, primary


def _materialize(plan):
    n = plan["size"]
    inp = np.full((n, n), plan["background"], dtype=np.int8)
    for cell in _diamond_cells(n):
        inp[cell] = plan["outline"]
    for cue, value in plan["cues"].items():
        inp[cue] = value
    if plan["blockers"]:
        assert plan["blocker_color"] not in (plan["background"], plan["outline"])
        for cell in plan["blockers"]:
            inp[cell] = plan["blocker_color"]
    out = _apply_rule(inp, plan["background"], plan["layout"])
    assert not np.array_equal(inp, out)
    return {"input": inp, "output": out}


def _new_plan(rng, layout, mode, style, excluded=(), forced=None,
              avoid_background=None, reserve_role=None, shared_blocker=None,
              overlap=False):
    size = int(rng.choice(_SIZES))
    background, outline = _colors(rng, avoid_background, reserve_role)
    specs = _line_specs(layout, size)
    cues = _sample_cues(rng, specs, background, outline, style, excluded, forced)
    blockers, primary = _sample_blockers(rng, specs, cues, background, mode)
    blocker_color = None
    if blockers:
        cue_colors = set(cues.values()) - {background}
        if overlap:
            assert mode in ("shadow", "double")
            blocker_color = cues[primary["cue"]]
        elif (shared_blocker is not None and
              shared_blocker not in (background, outline) and
              shared_blocker not in cue_colors):
            blocker_color = shared_blocker
        else:
            choices = [c for c in range(10)
                       if c not in (background, outline) and c not in cue_colors]
            blocker_color = int(rng.choice(choices))
    return {"size": size, "layout": layout, "background": background,
            "outline": outline, "cues": cues, "blockers": blockers,
            "blocker_color": blocker_color}


def _copy_plan(plan):
    return {**plan, "cues": dict(plan["cues"]), "blockers": list(plan["blockers"])}


def _different_cues(rng, plan, excluded):
    revised = _copy_plan(plan)
    available = [c for c in range(10)
                 if c not in (plan["background"], plan["outline"], *excluded)]
    specs = _line_specs(plan["layout"], plan["size"])
    visible = [s for s in specs if s["path"] and
               revised["cues"][s["cue"]] != plan["background"]]
    target = max(visible, key=lambda s: len(s["path"]))
    old = revised["cues"][target["cue"]]
    choices = [c for c in available if c != old and c != plan["blocker_color"]]
    revised["cues"][target["cue"]] = int(rng.choice(choices))
    return revised


def _detect_layout(inp, background):
    n = inp.shape[0]
    diamond = _diamond_cells(n)
    support = {(r, c) for r in range(n) for c in range(n)
               if (r in (0, n - 1) or c in (0, n - 1))
               and (r, c) not in diamond and int(inp[r, c]) != background}
    candidates = []
    for layout in _LAYOUTS:
        specs = _line_specs(layout, n)
        if support <= {s["cue"] for s in specs} and all(
            any(s["side"] == side and s["path"] and
                int(inp[s["cue"]]) != background for s in specs)
            for side in (0, 1)
        ):
            candidates.append(layout)
    assert len(candidates) == 1, "edge cues must identify one layout"
    return candidates[0]


def _analyze_pair(inp, out):
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == out.ndim == 2 and inp.shape == out.shape
    n, width = inp.shape
    assert n == width and n in _SIZES
    assert np.issubdtype(inp.dtype, np.integer) and np.issubdtype(out.dtype, np.integer)
    assert np.all((0 <= inp) & (inp <= 9)) and np.all((0 <= out) & (out <= 9))
    counts = np.bincount(inp.astype(np.int64).ravel(), minlength=10)
    background = int(np.argmax(counts))
    diamond = _diamond_cells(n)
    outlines = [c for c in range(10) if c != background and
                all(int(inp[cell]) == c for cell in diamond)]
    assert len(outlines) == 1
    outline = outlines[0]
    layout = _detect_layout(inp, background)
    specs = _line_specs(layout, n)
    cues = {s["cue"] for s in specs}
    m = n // 2
    interior = {(r, c) for r in range(n) for c in range(n)
                if abs(r - m) + abs(c - m) < m}
    blockers = {cell for cell in interior if int(inp[cell]) != background}
    blocker_colors = {int(inp[cell]) for cell in blockers}
    assert len(blocker_colors) <= 1 and outline not in blocker_colors
    for r in range(n):
        for c in range(n):
            cell = (r, c)
            value = int(inp[cell])
            if cell in diamond:
                assert value == outline
            elif cell in cues:
                assert value != outline
            elif cell in interior:
                assert value == background or cell in blockers
            else:
                assert value == background
    assert np.array_equal(out, _apply_rule(inp, background, layout))
    assert not np.array_equal(inp, out)
    first = last = multiple = shadow = full = overlap_shadow = overlap_on_blocked = False
    for spec in specs:
        path = spec["path"]
        occupied = [i for i, cell in enumerate(path) if cell in blockers]
        cue = int(inp[spec["cue"]])
        if len(occupied) >= 2:
            multiple = True
        if occupied:
            first |= occupied[0] == 0
            last |= occupied[-1] == len(path) - 1
            visible_shadow = cue != background and any(
                int(inp[cell]) == background for cell in path[occupied[0] + 1:])
            shadow |= visible_shadow and occupied[0] > 0
            overlap_on_blocked |= cue in blocker_colors
            overlap_shadow |= visible_shadow and cue in blocker_colors
        elif cue != background and len(path) >= 3:
            full = True
    cue_values = tuple(int(inp[s["cue"]]) for s in specs)
    visible_cues = sum(value != background for value in cue_values)
    return {"size": n, "background": background, "outline": outline,
            "layout": layout, "blockers": blockers,
            "blocker_color": next(iter(blocker_colors)) if blocker_colors else None,
            "cue_values": cue_values, "visible_cues": visible_cues,
            "cue_colors": set(cue_values) - {background},
            "first": first, "last": last, "multiple": multiple,
            "shadow": shadow, "full": full, "overlap_shadow": overlap_shadow,
            "overlap_on_blocked": overlap_on_blocked}


def _same_frame(a, b):
    return all(a[k] == b[k] for k in ("size", "layout", "background", "outline"))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (inp.shape, inp.astype(np.int8).tobytes())
            assert key not in seen
            seen.add(key)
            if inp.shape == (11, 11) and out.shape == (11, 11):
                digest = hashlib.sha256(inp.astype(np.int8).tobytes() +
                                        out.astype(np.int8).tobytes()).hexdigest()
                assert digest not in _SOURCE_DIGESTS
            info = _analyze_pair(inp, out)
            if info["overlap_on_blocked"]:
                # A shared literal color must still visibly cast a shadow.
                assert info["overlap_shadow"]
            analyses[split].append(info)
    train = analyses["train"]
    assert {a["layout"].endswith("vertical") for a in train} == {True, False}
    assert any(a["shadow"] for a in train)
    assert any(a["full"] for a in train)
    assert len({a["background"] for a in train}) >= 2
    if len(train) >= 3:
        assert any(_same_frame(a, b) and a["cue_values"] == b["cue_values"]
                   and a["blockers"] != b["blockers"]
                   for i, a in enumerate(train) for b in train[i + 1:])
    if len(train) >= 5:
        assert any(_same_frame(a, b) and a["blockers"] == b["blockers"]
                   and a["cue_values"] != b["cue_values"]
                   for i, a in enumerate(train) for b in train[i + 1:])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # No hidden color mapping is shared across pairs. The learned convention
    # is the visible edge-to-chord projection and pre-existing-cell stop rule.
    heldout = int(rng.integers(10)) if rng.random() < .55 else None
    excluded = (heldout,) if heldout is not None else ()
    first_layout = _LAYOUTS[int(rng.integers(4))]
    other_axis = [name for name in _LAYOUTS
                  if name.endswith("vertical") != first_layout.endswith("vertical")]
    second_layout = other_axis[int(rng.integers(2))]
    first_style = ("sparse", "short_run", "repeated", "alternating", "irregular")[
        int(rng.integers(5))]
    first = _new_plan(rng, first_layout, "shadow", first_style,
                      excluded=excluded, overlap=rng.random() < .25)
    second_style = ("sparse", "short_run", "repeated", "alternating", "irregular")[
        int(rng.integers(5))]
    second = _new_plan(rng, second_layout, "none", second_style,
                       excluded=excluded, avoid_background=first["background"])
    plans = {"train": [first, second], "test": []}
    modes = ("none", "few", "many", "first", "last", "shadow", "double")
    styles = ("sparse", "short_run", "repeated", "alternating", "irregular")
    for index in range(2, num_train):
        if index == 2:
            plan = _copy_plan(first)
            plan["blockers"] = []
            plan["blocker_color"] = None
        elif index == 4:
            plan = _different_cues(rng, first, excluded)
        else:
            layout = _LAYOUTS[int(rng.integers(4))]
            mode = modes[int(rng.integers(len(modes)))]
            overlap = mode in ("shadow", "double") and rng.random() < .3
            plan = _new_plan(rng, layout, mode, styles[int(rng.integers(5))],
                             excluded=excluded, overlap=overlap)
        plans["train"].append(plan)
    for index in range(num_test):
        layout = _LAYOUTS[int(rng.integers(4))]
        mode = modes[int(rng.integers(len(modes)))]
        overlap = mode in ("shadow", "double") and rng.random() < .3
        forced = heldout if index == 0 else None
        shared = first["blocker_color"] if rng.random() < .5 else None
        plan = _new_plan(rng, layout, mode, styles[int(rng.integers(5))],
                         forced=forced, reserve_role=forced,
                         shared_blocker=shared, overlap=overlap)
        plans["test"].append(plan)
    puzzle = {split: [_materialize(plan) for plan in plans[split]]
              for split in ("train", "test")}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k])
        for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    observed = [(_analyze_pair(pair["input"], pair["output"]), split)
                for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert {a["layout"] for a, _ in observed} == set(_LAYOUTS)
    assert {a["layout"] for a, s in observed if s == "test"} == set(_LAYOUTS)
    assert len({a["size"] for a, _ in observed}) >= 9
    assert any(a["size"] == 11 for a, _ in observed)
    assert any(a["size"] >= 21 and a["visible_cues"] <= 6 for a, _ in observed)
    assert any(a["first"] for a, _ in observed)
    assert any(a["last"] for a, _ in observed)
    assert any(a["multiple"] and s == "train" for a, s in observed)
    assert any(a["blockers"] == set() and s == "test" for a, s in observed)
    assert any(len(a["blockers"]) >= 4 and s == "test" for a, s in observed)
    assert any(a["overlap_shadow"] for a, _ in observed)
    assert any(len(p["train"]) >= 5 for p in puzzles)
    train_cues = [{c for pair in p["train"]
                   for c in _analyze_pair(pair["input"], pair["output"])["cue_colors"]}
                  for p in puzzles]
    assert any(any(_analyze_pair(pair["input"], pair["output"])["cue_colors"] - train_cues[i]
                   for pair in p["test"]) for i, p in enumerate(puzzles))
    coverage = {name: 0 for name in (
        "blocker_control", "cue_control", "train_recolor", "test_unblocked",
        "train_multiple", "shared_blocker", "novel_test_cue", "same_layout",
        "new_layout", "mixed_test_axes", "large_sparse", "one_cue_color",
        "full_strip", "short_runs", "alternating", "overlap", "zero_length",
        "last_cell", "last_and_multiple", "mirror_endpoint",
    )}
    for puzzle in puzzles:
        info = {split: [_analyze_pair(pair["input"], pair["output"])
                        for pair in puzzle[split]] for split in ("train", "test")}
        train, test = info["train"], info["test"]
        coverage["blocker_control"] += any(
            _same_frame(a, b) and a["cue_values"] == b["cue_values"]
            and a["blockers"] != b["blockers"]
            for i, a in enumerate(train) for b in train[i + 1:])
        coverage["cue_control"] += any(
            _same_frame(a, b) and a["blockers"] == b["blockers"]
            and a["cue_values"] != b["cue_values"]
            for i, a in enumerate(train) for b in train[i + 1:])
        coverage["train_recolor"] += len({a["background"] for a in train}) > 1
        coverage["test_unblocked"] += any(not a["blockers"] for a in test)
        coverage["train_multiple"] += any(a["multiple"] for a in train)
        coverage["shared_blocker"] += bool(
            ({a["blocker_color"] for a in train} - {None}) &
            ({a["blocker_color"] for a in test} - {None}))
        coverage["novel_test_cue"] += bool(
            set().union(*(a["cue_colors"] for a in test)) -
            set().union(*(a["cue_colors"] for a in train)))
        train_layouts = {a["layout"] for a in train}
        test_layouts = {a["layout"] for a in test}
        coverage["same_layout"] += bool(train_layouts & test_layouts)
        coverage["new_layout"] += bool(test_layouts - train_layouts)
        coverage["mixed_test_axes"] += len({
            a["layout"].endswith("vertical") for a in test}) == 2
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], info[split]):
                inp = pair["input"]
                background = a["background"]
                specs = _line_specs(a["layout"], a["size"])
                coverage["large_sparse"] += a["size"] >= 21 and a["visible_cues"] <= 6
                coverage["one_cue_color"] += len(a["cue_colors"]) == 1
                coverage["full_strip"] += a["visible_cues"] >= a["size"] // 2
                coverage["overlap"] += a["overlap_shadow"]
                coverage["mirror_endpoint"] += (
                    a["layout"] == "mirrored_horizontal" and (a["first"] or a["last"]))
                short_runs = alternating = True
                for side in (0, 1):
                    group = [s for s in specs if s["side"] == side and s["path"]]
                    indices = [i for i, s in enumerate(group)
                               if int(inp[s["cue"]]) != background]
                    values = [int(inp[group[i]["cue"]]) for i in indices]
                    short_runs &= (2 <= len(indices) <= 5 and
                                   indices == list(range(indices[0], indices[-1] + 1)))
                    alternating &= (len(values) >= 4 and len(set(values)) == 2
                                    and all(value == values[i % 2]
                                            for i, value in enumerate(values)))
                coverage["short_runs"] += short_runs
                coverage["alternating"] += alternating
                for spec in specs:
                    occupied = [i for i, cell in enumerate(spec["path"])
                                if cell in a["blockers"]]
                    if occupied and int(inp[spec["cue"]]) != background:
                        coverage["zero_length"] += occupied[0] == 0
                        coverage["last_cell"] += occupied[-1] == len(spec["path"]) - 1
                        coverage["last_and_multiple"] += (
                            len(occupied) >= 2 and occupied[-1] == len(spec["path"]) - 1)
    for name in ("blocker_control", "train_recolor", "novel_test_cue",
                 "same_layout", "large_sparse", "one_cue_color", "full_strip",
                 "short_runs", "zero_length", "last_cell", "mirror_endpoint"):
        assert coverage[name] >= 20, (name, coverage[name])
    for name in ("cue_control", "test_unblocked", "train_multiple",
                 "shared_blocker", "new_layout", "mixed_test_axes",
                 "alternating", "overlap", "last_and_multiple"):
        assert coverage[name] >= 5, (name, coverage[name])
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + train_count * 10 + test_count,
                            train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        p = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(p["train"]) == train_count
        if test_count is not None:
            assert len(p["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _puzzles_equal(generate_puzzle(777, 6, 2), generate_puzzle(777, 6, 2))
    assert not _puzzles_equal(generate_puzzle(777, 6, 2), generate_puzzle(778, 6, 2))
