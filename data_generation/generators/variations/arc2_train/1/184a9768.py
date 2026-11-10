"""Synthetic puzzles about translating a complete inventory of rectangular tiles."""

import hashlib

import numpy as np


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError("illegal pair count")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("training count must exceed test count")
    return int(num_train), int(num_test)


def _components(mask, values=None):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            value = None if values is None else int(values[r, c])
            todo = [(r, c)]
            seen[r, c] = True
            cells = []
            for rr, cc in todo:
                cells.append((rr, cc))
                for nr, nc in ((rr-1, cc), (rr+1, cc), (rr, cc-1), (rr, cc+1)):
                    if (0 <= nr < height and 0 <= nc < width and mask[nr, nc]
                            and not seen[nr, nc]
                            and (values is None or int(values[nr, nc]) == value)):
                        seen[nr, nc] = True
                        todo.append((nr, nc))
            groups.append(cells)
    return groups


def _bbox(cells):
    return (min(r for r, _ in cells), min(c for _, c in cells),
            max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1)


def _layout(rng, regime):
    count = int(rng.choice((1, 1, 2, 2, 2, 3) if regime == "source_train"
                           else (1, 2, 2, 2, 3, 3)))
    if count == 1:
        height = int(rng.integers(17, 31))
        width = int(rng.integers(17, 31))
        ph = int(rng.integers(9, min(18, height - 2) + 1))
        pw = int(rng.integers(9, min(18, width - 2) + 1))
        top = int(rng.integers(1, height - ph))
        left = int(rng.integers(1, width - pw))
        boxes = [(top, left, top + ph, left + pw)]
    elif count == 2:
        orientation = int(rng.integers(0, 2))
        if orientation == 0:
            height = int(rng.integers(19, 31))
            width = int(rng.integers(23, 31))
            cut = int(rng.integers(10, width - 9))
            spans = ((1, cut - 1), (cut + 1, width - 1))
            boxes = []
            for lo, hi in spans:
                pw = int(rng.integers(7, min(14, hi-lo) + 1))
                ph = int(rng.integers(6, min(17, height-2) + 1))
                left = int(rng.integers(lo, hi-pw+1))
                top = int(rng.integers(1, height-ph))
                boxes.append((top, left, top+ph, left+pw))
        else:
            height = int(rng.integers(23, 31))
            width = int(rng.integers(19, 31))
            cut = int(rng.integers(10, height - 9))
            spans = ((1, cut - 1), (cut + 1, height - 1))
            boxes = []
            for lo, hi in spans:
                ph = int(rng.integers(7, min(14, hi-lo) + 1))
                pw = int(rng.integers(6, min(17, width-2) + 1))
                top = int(rng.integers(lo, hi-ph+1))
                left = int(rng.integers(1, width-pw))
                boxes.append((top, left, top+ph, left+pw))
    else:
        height = int(rng.integers(25, 31))
        width = int(rng.integers(25, 31))
        split_r = int(rng.integers(12, height-10))
        split_c = int(rng.integers(12, width-10))
        slots = [(1, 1, split_r-1, split_c-1),
                 (1, split_c+1, split_r-1, width-1),
                 (split_r+1, 1, height-1, split_c-1),
                 (split_r+1, split_c+1, height-1, width-1)]
        omit = int(rng.integers(0, 4))
        boxes = []
        for i, (rlo, clo, rhi, chi) in enumerate(slots):
            if i == omit:
                continue
            ph = int(rng.integers(6, min(11, rhi-rlo) + 1))
            pw = int(rng.integers(6, min(11, chi-clo) + 1))
            top = int(rng.integers(rlo, rhi-ph+1))
            left = int(rng.integers(clo, chi-pw+1))
            boxes.append((top, left, top+ph, left+pw))
    return height, width, boxes


def _touching(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    if ar+ah == br or br+bh == ar:
        return max(ac, bc) < min(ac+aw, bc+bw)
    if ac+aw == bc or bc+bw == ac:
        return max(ar, br) < min(ar+ah, br+bh)
    return False


def _regions(rng, boxes, regime):
    regions = []
    have_touch = False
    for panel, (r0, c0, r1, c1) in enumerate(boxes):
        ph, pw = r1-r0, c1-c0
        if len(boxes) == 1:
            desired = int(rng.integers(2, 7 if regime == "source_train" else 8))
        else:
            desired = int(rng.integers(1, 5 if regime == "source_train" else 6))
        desired = min(desired, 9-len(regions)-(len(boxes)-panel-1))
        local = []
        occupied = np.zeros((ph, pw), dtype=bool)
        make_touch = desired >= 2 and ph >= 8 and pw >= 8 and (
            not have_touch and int(rng.integers(0, 3)) != 0)
        if make_touch:
            vh = int(rng.integers(3, min(6, ph-2)+1))
            hw = int(rng.integers(3, min(6, pw-3)+1))
            vr = int(rng.integers(1, ph-vh))
            vc = int(rng.integers(1, pw-hw-1))
            hr = int(rng.integers(vr, vr+vh))
            local.extend(((vr, vc, vh, 1), (hr, vc+1, 1, hw)))
            occupied[vr:vr+vh, vc] = True
            occupied[hr, vc+1:vc+1+hw] = True
            have_touch = True
        for _ in range(desired-len(local)):
            candidates = []
            for h in range(1, min(6, ph-2)+1):
                for w in range(1, min(6, pw-2)+1):
                    if h*w > 18:
                        continue
                    for r in range(1, ph-h):
                        for c in range(1, pw-w):
                            if np.any(occupied[r:r+h, c:c+w]):
                                continue
                            if int(occupied.sum()) + h*w > int((ph-2)*(pw-2)*0.55):
                                continue
                            rect = (r, c, h, w)
                            contact = any(_touching(rect, x) for x in local)
                            candidates.append((rect, contact))
            if not candidates:
                break
            want_touch = bool(local) and int(rng.integers(0, 5)) == 0
            preferred = [x for x in candidates if x[1] == want_touch]
            if not preferred:
                preferred = candidates
            rect = preferred[int(rng.integers(0, len(preferred)))][0]
            r, c, h, w = rect
            occupied[r:r+h, c:c+w] = True
            local.append(rect)
            have_touch |= any(_touching(rect, x) for x in local[:-1])
        regions.extend((panel, r0+r, c0+c, h, w) for r, c, h, w in local)
    return regions


def _split_region(rng, h, w, count):
    parts = [(h, w)]
    for _ in range(count-1):
        choices = [i for i, (a, b) in enumerate(parts) if a > 1 or b > 1]
        if not choices:
            break
        i = int(choices[int(rng.integers(0, len(choices)))])
        a, b = parts.pop(i)
        axes = (0, 1) if a > 1 and b > 1 else ((0,) if a > 1 else (1,))
        axis = axes[int(rng.integers(0, len(axes)))]
        if axis == 0:
            cut = int(rng.integers(1, a))
            parts.extend(((cut, b), (a-cut, b)))
        else:
            cut = int(rng.integers(1, b))
            parts.extend(((a, cut), (a, b-cut)))
    return parts


def _place_exterior(rng, inp, boxes, pieces, background, distractor, noise_count):
    height, width = inp.shape
    panel_mask = np.zeros(inp.shape, dtype=bool)
    for r0, c0, r1, c1 in boxes:
        panel_mask[r0:r1, c0:c1] = True
    for color, h, w in sorted(pieces, key=lambda x: -x[1]*x[2]):
        candidates = []
        for r in range(height-h+1):
            for c in range(width-w+1):
                if np.any(panel_mask[r:r+h, c:c+w]) or np.any(
                    inp[r:r+h, c:c+w] != background
                ):
                    continue
                rr0, rr1 = max(0, r-1), min(height, r+h+1)
                cc0, cc1 = max(0, c-1), min(width, c+w+1)
                if np.any(inp[rr0:rr1, cc0:cc1] == color):
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        inp[r:r+h, c:c+w] = color
    for _ in range(noise_count):
        candidates = []
        for r in range(height):
            for c in range(width):
                if panel_mask[r, c] or inp[r, c] != background:
                    continue
                if any(0 <= rr < height and 0 <= cc < width
                       and inp[rr, cc] == distractor
                       for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))):
                    continue
                candidates.append((r, c))
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        inp[r, c] = distractor
    return True


def _colorings(holes, pieces, limit=2, node_limit=50000):
    """Exact cover of actual open cells; only different output colors count."""
    if sum(h*w for _, h, w in pieces) != len(holes):
        return set()
    positions = {cell: i for i, cell in enumerate(holes)}
    choices = [[] for _ in holes]
    descriptors = tuple(pieces)
    for i, (color, h, w) in enumerate(descriptors):
        for r, c in holes:
            covered = [(rr, cc) for rr in range(r, r+h) for cc in range(c, c+w)]
            if not all(cell in positions for cell in covered):
                continue
            bitmask = sum(1 << positions[cell] for cell in covered)
            option = (i, color, bitmask, tuple(positions[cell] for cell in covered))
            for cell in covered:
                choices[positions[cell]].append(option)
    if any(not x for x in choices):
        return set()
    full = (1 << len(holes)) - 1
    all_used = (1 << len(pieces)) - 1
    painted = [-1] * len(holes)
    results = set()
    nodes = 0

    def search(filled, used):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            results.add(None)
            return
        if len(results) >= limit:
            return
        if filled == full:
            if used == all_used:
                results.add(tuple(painted))
            return
        first = ((full ^ filled) & -(full ^ filled)).bit_length()-1
        seen = set()
        for i, color, mask, indices in choices[first]:
            if used & (1 << i) or mask & filled:
                continue
            if descriptors[i] in seen:
                continue
            seen.add(descriptors[i])
            for cell in indices:
                painted[cell] = color
            search(filled | mask, used | (1 << i))
            for cell in indices:
                painted[cell] = -1
            if len(results) >= limit:
                return

    search(0, 0)
    return results


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    background = int(out[0, 0])
    assert np.all(out[0, :] == background) and np.all(out[:, 0] == background)
    assert np.all(out[-1, :] == background) and np.all(out[:, -1] == background)
    panel_groups = _components(out != background)
    assert 1 <= len(panel_groups) <= 3
    panel_mask = np.zeros(inp.shape, dtype=bool)
    panels = []
    holes = []
    for cells in panel_groups:
        r0, c0, r1, c1 = _bbox(cells)
        assert len(cells) == (r1-r0)*(c1-c0)
        assert 0 < r0 and 0 < c0 and r1 < inp.shape[0] and c1 < inp.shape[1]
        panel_mask[r0:r1, c0:c1] = True
        border = np.concatenate((inp[r0, c0:c1], inp[r1-1, c0:c1],
                                 inp[r0:r1, c0], inp[r0:r1, c1-1]))
        assert len(np.unique(border)) == 1
        carrier = int(border[0])
        assert carrier != background
        iv = inp[r0:r1, c0:c1]
        ov = out[r0:r1, c0:c1]
        assert np.all((iv == background) | (iv == carrier))
        assert np.all(ov[iv == carrier] == carrier)
        assert np.all(ov[iv == background] != background)
        local_holes = np.argwhere(iv == background)
        assert len(local_holes) > 0
        holes.extend((r0+int(r), c0+int(c)) for r, c in local_holes)
        panels.append((r0, c0, r1, c1, carrier))
    assert np.all(out[~panel_mask] == background)
    assert holes
    holes.sort()
    expected = tuple(int(out[r, c]) for r, c in holes)
    only_input = set(int(c) for c in np.unique(inp)) - set(int(c) for c in np.unique(out))
    assert len(only_input) <= 1
    distractor = next(iter(only_input)) if only_input else None
    outside = ~panel_mask & (inp != background)
    pieces = []
    noise_count = 0
    for cells in _components(outside, inp):
        r0, c0, r1, c1 = _bbox(cells)
        color = int(inp[r0, c0])
        assert len(cells) == (r1-r0)*(c1-c0)
        if color == distractor:
            assert len(cells) == 1
            noise_count += 1
        else:
            pieces.append((color, r1-r0, c1-c0))
    assert pieces and len(pieces) <= 14
    assert _colorings(holes, pieces) == {expected}
    components = _components(np.isin(inp, [background]) & panel_mask)
    touching_multicolor = any(len({int(out[r,c]) for r,c in cells}) > 1
                              for cells in components)
    return {"background": background, "distractor": distractor,
            "panels": panels, "holes": holes, "pieces": pieces,
            "noise_count": noise_count, "touching_multicolor": touching_multicolor,
            "hole_components": len(components),
            "max_span": max(max(h,w) for _,h,w in pieces)}


def _make_pair(rng, regime, distractor, novel_color, force_novel,
               first_train, fixed_background):
    for attempt in range(320):
        background = (int(fixed_background) if fixed_background is not None
                      else int(rng.choice([c for c in range(10)
                                           if c not in (distractor, novel_color)])))
        height, width, boxes = _layout(rng, regime)
        regions = _regions(rng, boxes, regime)
        if len(regions) < 2:
            continue
        available = [c for c in range(10) if c not in (background, distractor)
                     and (force_novel or c != novel_color)]
        available = [int(x) for x in rng.permutation(available)]
        carrier_options = [c for c in available if c != novel_color]
        carriers = [carrier_options[int(x)] for x in
                    rng.integers(0, len(carrier_options), size=len(boxes))]
        if len(set(carriers)) < len(carriers):
            carriers = carrier_options[:len(boxes)]
        colors = []
        for index, (panel, r, c, h, w) in enumerate(regions):
            twins = [j for j, (_, _, _, hh, ww) in enumerate(regions[:index])
                     if (hh, ww) == (h, w)]
            if twins:
                color = colors[twins[0]]
            elif index == 0 and force_novel:
                color = novel_color
            elif index == 0 and int(rng.integers(0, 3)) == 0:
                color = carriers[int(rng.integers(0, len(carriers)))]
            elif colors and int(rng.integers(0, 5)) == 0:
                color = colors[int(rng.integers(0, len(colors)))]
            else:
                unused = [x for x in available if x not in colors]
                color = unused[int(rng.integers(0, len(unused)))] if unused else colors[0]
            colors.append(color)
        inp = np.full((height, width), background, dtype=np.int8)
        out = inp.copy()
        for box, carrier in zip(boxes, carriers):
            r0, c0, r1, c1 = box
            inp[r0:r1, c0:c1] = carrier
            out[r0:r1, c0:c1] = carrier
        pieces = []
        for (_, r, c, h, w), color in zip(regions, colors):
            inp[r:r+h, c:c+w] = background
            out[r:r+h, c:c+w] = color
            fragment = int(rng.integers(0, 100))
            desired = (3 if fragment < (12 if regime == "source_train" else 20)
                       else 2 if fragment < (45 if regime == "source_train" else 56)
                       else 1)
            for ph, pw in _split_region(rng, h, w, desired):
                pieces.append((color, ph, pw))
        if len(pieces) > 14:
            continue
        noise_count = (int(rng.integers(1, 7)) if first_train
                       else int(rng.integers(0, 8)))
        if not _place_exterior(rng, inp, boxes, pieces, background,
                               distractor, noise_count):
            continue
        rotation = int(rng.integers(0, 4))
        inp = np.rot90(inp, rotation).copy()
        out = np.rot90(out, rotation).copy()
        try:
            analysis = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if first_train and not analysis["touching_multicolor"]:
            continue
        return {"input": inp, "output": out}
    raise ValueError("could not construct an unambiguous pair")


def _pair_digest(inp, out):
    h = hashlib.sha256()
    for grid in (inp, out):
        h.update(bytes(grid.shape))
        h.update(np.asarray(grid, dtype=np.uint8).tobytes())
    return h.hexdigest()


# Exact source-pair digests use shapes and cell values, not layout categories.
_SOURCE_DIGESTS = {
    "64e6f7f90d73287b16e1cf2f005fe672e5146320612695980c78359d98c6119e",
    "3a90ce5ab02976b658a4be1edac45af751d85b6e788db745b4fd3e3e2c78c156",
    "7e0c5357a1d0ba30c8ae763dc9b35c804df90fa3b676ca7374c5c1bfbffe95df",
    "7cef32420cea500a1a60323816ebbc0bd3fc6691765ac2b886b612687b1e30fa",
}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    distractors = set()
    training_noise = 0
    training_touch = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            ikey = (inp.shape, inp.tobytes())
            pkey = (ikey, out.shape, out.tobytes())
            assert ikey not in seen_inputs and pkey not in seen_pairs
            assert _pair_digest(inp, out) not in _SOURCE_DIGESTS
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            analysis = _analyze_pair(inp, out)
            if analysis["distractor"] is not None:
                distractors.add(analysis["distractor"])
            if split == "train":
                training_noise += analysis["noise_count"]
                training_touch |= analysis["touching_multicolor"]
    assert len(distractors) == 1 and training_noise > 0 and training_touch


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    distractor = int(rng.integers(0, 10))
    novel_color = int(rng.choice([c for c in range(10) if c != distractor]))
    base_background = int(rng.choice([c for c in range(10)
                                      if c not in (distractor, novel_color)]))
    fixed_background = (base_background if int(rng.integers(0, 2)) else None)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count, regime in (("train", train_count, "source_train"),
                                  ("test", test_count, "target_test")):
        for index in range(count):
            for _ in range(20):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63-1)))
                force_novel = (split == "test" and int(rng.integers(0, 3)) == 0)
                pair = _make_pair(pair_rng, regime, distractor,
                                  novel_color, force_novel,
                                  split == "train" and index == 0,
                                  fixed_background)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not make a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal_puzzles(a, b):
    return all(np.array_equal(x[role], y[role])
               for split in ("train", "test")
               for x, y in zip(a[split], b[split])
               for role in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _equal_puzzles(generate_puzzle(7), generate_puzzle(7))
    assert not _equal_puzzles(generate_puzzle(7), generate_puzzle(8))
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 2)):
        validate_generated_puzzle(generate_puzzle(8000+train+test, train, test))
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(9000+(train or 0)+(test or 0),
                                                  train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(3, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    analyses = [_analyze_pair(pair["input"], pair["output"])
                for p in puzzles for split in ("train", "test") for pair in p[split]]
    assert {len(a["panels"]) for a in analyses} == {1, 2, 3}
    assert any(a["touching_multicolor"] for a in analyses)
    assert any(a["noise_count"] == 0 for a in analyses)
    assert any(a["max_span"] >= 5 for a in analyses)
    assert any(len(a["pieces"]) >= 10 for a in analyses)
    assert any(a["background"] == 0 for a in analyses)
    assert len({pair["input"].shape for p in puzzles
                for split in ("train", "test") for pair in p[split]}) >= 20
    fixed_backgrounds = 0
    changing_backgrounds = 0
    familiar_tests = 0
    novel_tests = 0
    role_reuse = 0
    panel_sharing = 0
    varying_panel_counts = 0
    varying_piece_counts = 0
    varying_noise_counts = 0
    repeated_destinations = 0
    repeated_singletons = 0
    three_piece_fills = 0
    horizontal_joins = 0
    vertical_joins = 0
    nonrect_openings = 0
    edge_touch_tiles = 0
    layouts = set()
    region_dimensions = set()
    for puzzle in puzzles:
        pair_analyses = [_analyze_pair(pair["input"], pair["output"])
                         for split in ("train", "test") for pair in puzzle[split]]
        backgrounds = {a["background"] for a in pair_analyses}
        fixed_backgrounds += len(backgrounds) == 1
        changing_backgrounds += len(backgrounds) > 1
        varying_panel_counts += len({len(a["panels"]) for a in pair_analyses}) > 1
        varying_piece_counts += len({len(a["pieces"]) for a in pair_analyses}) > 1
        varying_noise_counts += len({a["noise_count"] for a in pair_analyses}) > 1
        train_colors = {int(c) for pair in puzzle["train"]
                        for role in ("input", "output")
                        for c in np.unique(pair[role])}
        for pair in puzzle["test"]:
            novel = set(map(int, np.unique(pair["input"]))) - train_colors
            familiar_tests += not novel
            novel_tests += bool(novel)
        for pair, a in zip(
            puzzle["train"] + puzzle["test"], pair_analyses
        ):
            inp, out = pair["input"], pair["output"]
            bg = a["background"]
            carriers = {p[4] for p in a["panels"]}
            hole_colors = {int(out[r, c]) for r, c in a["holes"]}
            role_reuse += bool(carriers & hole_colors)
            boxes = [p[:4] for p in a["panels"]]
            panel_mask = np.zeros(inp.shape, dtype=bool)
            regions = []
            for r0, c0, r1, c1 in boxes:
                panel_mask[r0:r1, c0:c1] = True
                local_holes = inp[r0:r1, c0:c1] == bg
                local_out = out[r0:r1, c0:c1]
                for cells in _components(local_holes):
                    hr0, hc0, hr1, hc1 = _bbox(cells)
                    nonrect_openings += len(cells) < (hr1-hr0)*(hc1-hc0)
                for cells in _components(local_holes, local_out):
                    hr0, hc0, hr1, hc1 = _bbox(cells)
                    regions.append((int(local_out[cells[0]]),
                                    hr1-hr0, hc1-hc0, len(cells)))
            region_dimensions.update((h, w) for _, h, w, _ in regions)
            dimensions = [(h, w) for _, h, w, _ in regions]
            repeated_destinations += len(dimensions) != len(set(dimensions))
            singleton_colors = [color for color, _, _, area in regions if area == 1]
            repeated_singletons += len(singleton_colors) != len(set(singleton_colors))
            by_color = {}
            for color, h, w in a["pieces"]:
                by_color.setdefault(color, []).append((h, w))
            for color, h, w, area in regions:
                if sum(entry[0] == color for entry in regions) != 1:
                    continue
                parts = by_color.get(color, [])
                if sum(ph*pw for ph, pw in parts) != area:
                    continue
                three_piece_fills += len(parts) >= 3
                if len(parts) == 2:
                    horizontal_joins += (parts[0][1] == w and parts[1][1] == w
                                         and parts[0][0]+parts[1][0] == h)
                    vertical_joins += (parts[0][0] == h and parts[1][0] == h
                                       and parts[0][1]+parts[1][1] == w)
            for i, first in enumerate(boxes):
                for second in boxes[i+1:]:
                    row_gap = first[2] <= second[0] or second[2] <= first[0]
                    col_gap = first[3] <= second[1] or second[3] <= first[1]
                    layouts.add("diagonal" if row_gap and col_gap else
                                "stacked" if row_gap else "side_by_side")
            for r, c in zip(*np.nonzero((~panel_mask) & (inp != bg))):
                if a["distractor"] is not None and int(inp[r, c]) == a["distractor"]:
                    continue
                if any(0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                       and panel_mask[rr, cc] and int(inp[rr, cc]) != int(inp[r, c])
                       for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))):
                    edge_touch_tiles += 1
                    break
            if len(a["panels"]) >= 2:
                panel_colors = []
                for r0, c0, r1, c1, _ in a["panels"]:
                    local = pair["input"][r0:r1, c0:c1] == a["background"]
                    panel_colors.append(set(map(int, np.unique(
                        pair["output"][r0:r1, c0:c1][local]
                    ))))
                panel_sharing += any(panel_colors[i] & panel_colors[j]
                                     for i in range(len(panel_colors))
                                     for j in range(i+1, len(panel_colors)))
    assert fixed_backgrounds > 0 and changing_backgrounds > 0
    assert familiar_tests > 0 and novel_tests > 0
    assert role_reuse > 0 and panel_sharing > 0
    assert varying_panel_counts >= 20
    assert varying_piece_counts >= 20 and varying_noise_counts >= 20
    assert repeated_destinations > 0 and repeated_singletons > 0
    assert three_piece_fills > 0 and horizontal_joins > 0 and vertical_joins > 0
    assert nonrect_openings > 0 and edge_touch_tiles > 0
    assert layouts == {"stacked", "side_by_side", "diagonal"}
    assert {(2, 5), (3, 6), (2, 6), (6, 2)} <= region_dimensions
