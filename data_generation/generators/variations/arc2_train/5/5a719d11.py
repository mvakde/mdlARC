import numpy as np


# Rejecting every original source input is stronger than merely rejecting the
# corresponding complete input/output pair: an exact copied pair must have an
# exact copied input.
_SOURCE_INPUT_ROWS = (
    tuple(
        """66666666044444444
66666666044444444
66466466044884844
66444466044484844
66666666044488844
66666666044444444
66666666044444444
00000000000000000
00000000000000000
22222222022222222
22233322022111222
22332222022121222
22232222022121222
22233222022211122
22222222022212222
22222222022222222
00000000000000000
00000000000000000
33333333011111111
33323333014411111
33223333011441111
33323333011141111
33323333011141111
33222233011144411
33333333011111111""".splitlines()
    ),
    tuple(
        """88888888088888888
88888888088888888
88111188088882888
81188888082222228
88118188088882888
88888888088888888
88888888088888888
00000000000000000
00000000000000000
44444444088888888
44144444088838888
44141444083338888
44111444088838888
44441144088833888
44444444088888888
44444444088888888
00000000000000000
00000000000000000
44444444022222222
44242444022882222
44224444022282222
44422444022288822
44424444022228822
44444444022222222
44444444022222222""".splitlines()
    ),
    tuple(
        """22222222044444444
22222222044444444
22212222049444444
21112222049449444
22212222049449444
22212222049999994
22211112044449444
22222222044444444
00000000000000000
00000000000000000
33333333088888888
34443333088866888
33343333088886868
33444333088886668
33343333088888868
33333333088888868
33333333088888868
33333333088888888""".splitlines()
    ),
    tuple(
        """8888888004444444
8181118004774474
8111888004477774
8181118004444474
8188818004444474
8888888004444444
0000000000000000
0000000000000000
2222222003333333
2882222003999993
2282222003333393
2288882003333393
2282222003999993
2222222003333333
0000000000000000
0000000000000000
4444444009999999
4664464009199919
4644444009111119
4444464009991999
4644664009991999
4444444009999999
0000000000000000
0000000000000000
1111111001111111
1131111001555111
1131111001151151
1333331001151151
1131131001555111
1111111001111111""".splitlines()
    ),
)


def _check(condition: object, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _checked_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_optional_count("num_train", num_train, 2, 10)
    test_count = _checked_optional_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        span = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((seed_value // 4 + 3 * seed_value) % span)
        )
    elif train_count is None:
        span = 10 - test_count
        train_count = (
            test_count
            + 1
            + ((seed_value // 3 + seed_value) % span)
        )
    elif test_count is None:
        upper = min(4, train_count - 1)
        test_count = 1 + ((seed_value // 5 + seed_value) % upper)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _layout(rng: np.random.Generator, regime: str) -> tuple[int, int, int, int, int]:
    """Return panel height/width, group count, column/row separator widths."""
    weights = ([18, 30, 29, 17, 6] if regime == "source_train"
               else [12, 23, 27, 28, 10])
    groups = int(rng.choice(np.arange(1, 6), p=np.array(weights) / sum(weights)))
    column_gap = int(rng.integers(1, 4))
    row_gap = int(rng.integers(1, 4))
    max_height = min(12, (30 - (groups - 1) * row_gap) // groups)
    if max_height < 4:
        row_gap = 1
        max_height = (30 - (groups - 1)) // groups
    height = int(rng.integers(4, max_height + 1))
    width = int(rng.integers(4, (30 - column_gap) // 2 + 1))
    return height, width, groups, column_gap, row_gap


def _mask(rng: np.random.Generator, h: int, w: int, style: str,
          edge: bool = False) -> np.ndarray:
    a, b = h - 2, w - 2
    core = np.zeros((a, b), dtype=bool)
    area = a * b
    if style == "tiny":
        core.flat[rng.choice(area, size=min(area, int(rng.integers(1, 3))), replace=False)] = True
    elif style == "scatter":
        count = int(rng.integers(2, max(3, min(area, area // 3 + 2))))
        core.flat[rng.choice(area, size=min(count, area), replace=False)] = True
    elif style == "dense":
        count = min(area, max(2, int(area * rng.uniform(.38, .67))))
        core.flat[rng.choice(area, size=count, replace=False)] = True
    elif style == "hollow":
        rh = int(rng.integers(2, a + 1))
        rw = int(rng.integers(2, b + 1))
        r = int(rng.integers(a - rh + 1))
        c = int(rng.integers(b - rw + 1))
        core[r, c:c + rw] = True
        core[r + rh - 1, c:c + rw] = True
        core[r:r + rh, c] = True
        core[r:r + rh, c + rw - 1] = True
    elif style == "thin":
        if bool(rng.integers(2)):
            r = int(rng.integers(a))
            c = int(rng.integers(b))
            core[r, :] = True
            core[:, c] = True
        else:
            for r in range(a):
                core[r, int((r * b / max(a, 1) + rng.integers(2)) % b)] = True
    else:  # An irregular connected branch, with optional satellites.
        target = min(area, max(3, int(rng.integers(3, max(4, area // 2 + 2)))))
        start = (int(rng.integers(a)), int(rng.integers(b)))
        cells = {start}
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc)
                               for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r + dr < a and 0 <= c + dc < b
                               and (r + dr, c + dc) not in cells})
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        for r, c in cells:
            core[r, c] = True
        if area > 8 and rng.random() < .3:
            core[int(rng.integers(a)), int(rng.integers(b))] = True
    result = np.zeros((h, w), dtype=bool)
    result[1:-1, 1:-1] = core
    if edge:
        side = int(rng.integers(4))
        if side == 0:
            result[0, int(rng.integers(1, w - 1))] = True
        elif side == 1:
            result[-1, int(rng.integers(1, w - 1))] = True
        elif side == 2:
            result[int(rng.integers(1, h - 1)), 0] = True
        else:
            result[int(rng.integers(1, h - 1)), -1] = True
    return result


def _second_mask(rng: np.random.Generator, first: np.ndarray, style: str,
                 relation: str, edge: bool) -> np.ndarray:
    h, w = first.shape
    if relation == "identical":
        return first.copy()
    if relation == "contained" and np.count_nonzero(first) >= 2:
        indexes = np.flatnonzero(first)
        take = int(rng.integers(1, len(indexes)))
        second = np.zeros_like(first)
        second.flat[rng.choice(indexes, size=take, replace=False)] = True
        return second
    if relation == "disjoint":
        candidates = np.flatnonzero(~first[1:-1, 1:-1])
        if len(candidates):
            second = np.zeros_like(first)
            count = int(rng.integers(1, min(len(candidates), max(2, (h - 2) * (w - 2) // 2)) + 1))
            sub = second[1:-1, 1:-1]
            sub.flat[rng.choice(candidates, size=count, replace=False)] = True
            return second
    second = _mask(rng, h, w, style, edge)
    if relation == "overlap":
        common = np.flatnonzero(first)
        second.flat[int(rng.choice(common))] = True
    if np.array_equal(first, second):
        candidates = np.flatnonzero(~second[1:-1, 1:-1])
        if len(candidates):
            second[1:-1, 1:-1].flat[int(candidates[0])] = True
        else:
            second[1, 1] = False
    return second


_STYLES = ("branch", "scatter", "thin", "hollow", "dense", "tiny")
_RELATIONS = ("free", "disjoint", "overlap", "contained", "identical")


def _paint(rng: np.random.Generator, base: int, opposite_base: int,
           mask: np.ndarray, pool: list[int], multi: bool,
           donor_ink: bool) -> np.ndarray:
    panel = np.full(mask.shape, base, dtype=np.int8)
    choices = [v for v in pool if v != base]
    if not choices:
        choices = [v for v in range(1, 10) if v != base]
    first = opposite_base if donor_ink and opposite_base != base else int(rng.choice(choices))
    panel[mask] = first
    if multi and np.count_nonzero(mask) >= 2:
        alternatives = [v for v in choices if v != first]
        if alternatives:
            second = int(rng.choice(alternatives))
            indexes = np.flatnonzero(mask)
            count = max(1, len(indexes) // 3)
            panel.flat[rng.choice(indexes, size=count, replace=False)] = second
    return panel


def _make_pair(rng: np.random.Generator, layout: tuple[int, int, int, int, int],
               separator: int, pool: list[int], specs: list[dict],
               novel: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w, groups, cg, rg = layout
    total_h = groups * h + (groups - 1) * rg
    input_grid = np.full((total_h, 2 * w + cg), separator, dtype=np.int8)
    output_grid = input_grid.copy()
    bases = [v for v in pool if v != separator]
    if len(bases) < 2:
        bases = [v for v in range(1, 10) if v != separator]
    for index, spec in enumerate(specs):
        left_base = int(rng.choice(bases))
        if novel is not None and index == 0:
            left_base = novel
        right_base = (left_base if spec.get("same", False) else
                      int(rng.choice([v for v in bases if v != left_base])))
        left_style = str(rng.choice(_STYLES))
        right_style = str(rng.choice(_STYLES))
        relation = spec.get("relation", str(rng.choice(_RELATIONS[:-1])))
        edge = bool(spec.get("edge", False))
        left_mask = _mask(rng, h, w, left_style, edge and bool(rng.integers(2)))
        if spec.get("multi", False) and np.count_nonzero(left_mask) < 2:
            available = np.flatnonzero(~left_mask[1:-1, 1:-1])
            left_mask[1:-1, 1:-1].flat[int(available[0])] = True
        right_mask = _second_mask(rng, left_mask, right_style, relation,
                                  edge and not np.any(left_mask[0]))
        if edge and not (np.any(left_mask[0]) or np.any(left_mask[-1]) or
                        np.any(left_mask[:, 0]) or np.any(left_mask[:, -1]) or
                        np.any(right_mask[0]) or np.any(right_mask[-1]) or
                        np.any(right_mask[:, 0]) or np.any(right_mask[:, -1])):
            right_mask[0, 1] = True
        if not np.any(right_mask):
            right_mask[1, 1] = True
        multi = bool(spec.get("multi", False))
        left = _paint(rng, left_base, right_base, left_mask, bases,
                      multi and bool(rng.integers(2)), bool(spec.get("donor_ink", False)))
        right = _paint(rng, right_base, left_base, right_mask, bases,
                       multi and len(np.unique(left[left_mask])) == 1,
                       bool(spec.get("donor_ink", False)))
        if multi and len(np.unique(left[left_mask])) == 1 and len(np.unique(right[right_mask])) == 1:
            # Ensure the promised demonstration has at least two marker colors.
            target = left if np.count_nonzero(left_mask) >= 2 else right
            mask = left_mask if target is left else right_mask
            options = [v for v in bases if v != int(target.flat[np.flatnonzero(mask)[0]])
                       and v != int(target[0, 0])]
            if options and np.count_nonzero(mask) >= 2:
                target.flat[np.flatnonzero(mask)[-1]] = int(rng.choice(options))
        left_out = np.full((h, w), left_base, dtype=np.int8)
        right_out = np.full((h, w), right_base, dtype=np.int8)
        left_out[right_mask] = right_base
        right_out[left_mask] = left_base
        r0 = index * (h + rg)
        input_grid[r0:r0+h, :w] = left
        input_grid[r0:r0+h, w+cg:] = right
        output_grid[r0:r0+h, :w] = left_out
        output_grid[r0:r0+h, w+cg:] = right_out
    return input_grid, output_grid


def _array_key(array: np.ndarray) -> tuple[tuple[int, int], bytes]:
    value = np.asarray(array, dtype=np.int8)
    return tuple(value.shape), value.tobytes()


def _is_source_input(array: np.ndarray) -> bool:
    rows = tuple("".join(str(int(v)) for v in row) for row in array)
    return rows in _SOURCE_INPUT_ROWS


def _read_layout(grid: np.ndarray) -> tuple[int, int, int, int, int, int, list[int]]:
    height, width = grid.shape
    possible = []
    for cg in (1, 2, 3):
        if (width - cg) % 2:
            continue
        w = (width - cg) // 2
        if w < 4 or w > 14:
            continue
        separator = int(grid[0, w])
        if not np.all(grid[:, w:w+cg] == separator):
            continue
        blank_rows = np.all(grid == separator, axis=1)
        starts = [i for i in range(height) if blank_rows[i] and (i == 0 or not blank_rows[i-1])]
        ends = [i for i in range(height) if blank_rows[i] and (i == height-1 or not blank_rows[i+1])]
        if starts:
            if starts[0] == 0 or ends[-1] == height - 1:
                continue
            gaps = [b-a+1 for a,b in zip(starts, ends)]
            if len(set(gaps)) != 1 or gaps[0] not in (1,2,3):
                continue
            row_starts = [0] + [e + 1 for e in ends]
            row_ends = starts + [height]
            lengths = [b-a for a,b in zip(row_starts, row_ends)]
            if len(set(lengths)) != 1 or lengths[0] < 4:
                continue
            h, groups, rg = lengths[0], len(lengths), gaps[0]
        else:
            h, groups, rg, row_starts = height, 1, 0, [0]
        if h > 12 or groups > 5:
            continue
        if all(np.all(grid[r:r+h, 0] != separator) and
               np.all(grid[r:r+h, w+cg] != separator) for r in row_starts):
            possible.append((h, w, groups, cg, rg, separator, row_starts))
    _check(len(possible) == 1, "ambiguous or malformed panel layout")
    return possible[0]


def _panel_info(panel: np.ndarray, separator: int) -> tuple[int, np.ndarray, int, bool]:
    corners = (int(panel[0,0]), int(panel[0,-1]), int(panel[-1,0]), int(panel[-1,-1]))
    _check(len(set(corners)) == 1, "panel corners do not identify the base")
    base = corners[0]
    _check(1 <= base <= 9 and base != separator, "invalid panel base")
    mask = panel != base
    _check(np.count_nonzero(mask) >= 1, "empty motif")
    _check(np.count_nonzero(mask) <= .7 * panel.size, "base not clear")
    _check(not np.any(panel == separator), "separator inside a panel")
    border = np.concatenate((panel[0], panel[-1], panel[1:-1,0], panel[1:-1,-1]))
    _check(np.count_nonzero(border != base) <= 2, "panel base boundary unclear")
    inks = np.unique(panel[mask])
    _check(1 <= len(inks) <= 3 and np.all((inks >= 1) & (inks <= 9)),
           "invalid marker colors")
    return base, mask, len(inks), bool(np.any(border != base))


def _mask_traits(mask: np.ndarray) -> set[str]:
    traits = set()
    if np.count_nonzero(mask) <= 2:
        traits.add("tiny")
    if np.count_nonzero(mask) >= .35 * (mask.shape[0]-2) * (mask.shape[1]-2):
        traits.add("dense")
    if not np.array_equal(mask, mask[:, ::-1]) and not np.array_equal(mask, mask[::-1]):
        traits.add("asymmetric")
    if np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:,0]) or np.any(mask[:,-1]):
        traits.add("edge")
    h,w = mask.shape
    visited = np.zeros_like(mask)
    components = 0
    for r,c in zip(*np.where(mask)):
        if visited[r,c]:
            continue
        components += 1
        stack = [(int(r),int(c))]
        visited[r,c] = True
        while stack:
            x,y = stack.pop()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx,ny = x+dx,y+dy
                if 0 <= nx < h and 0 <= ny < w and mask[nx,ny] and not visited[nx,ny]:
                    visited[nx,ny] = True
                    stack.append((nx,ny))
    if components > 1:
        traits.add("multicomponent")
    if any(sum(mask[x,y] for x,y in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
               if 0 <= x < h and 0 <= y < w) >= 3 for r,c in zip(*np.where(mask))):
        traits.add("branch")
    if (any(np.any(mask[r,c:c+3]) and np.all(mask[r,c:c+3])
            for r in range(h) for c in range(w-2)) or
        any(np.all(mask[r:r+3,c]) for r in range(h-2) for c in range(w))):
        if not np.any(mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:] & mask[1:,1:]):
            traits.add("thin")
    # A background component enclosed by the mask is a genuine hollow motif.
    outside = np.zeros((h+2,w+2), dtype=bool)
    padded = np.pad(mask, 1)
    stack = [(0,0)]
    outside[0,0] = True
    while stack:
        x,y = stack.pop()
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx,ny = x+dx,y+dy
            if 0 <= nx < h+2 and 0 <= ny < w+2 and not padded[nx,ny] and not outside[nx,ny]:
                outside[nx,ny] = True
                stack.append((nx,ny))
    if np.any(~padded & ~outside):
        traits.add("hollow")
    return traits


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _check(input_grid.shape == output_grid.shape, "dimensions changed")
    h, w, groups, cg, rg, separator, row_starts = _read_layout(input_grid)
    _check(np.array_equal(input_grid[:, w:w+cg], output_grid[:, w:w+cg]),
           "column gutter changed")
    records = []
    for r in row_starts:
        left = input_grid[r:r+h, :w]
        right = input_grid[r:r+h, w+cg:]
        lb, lm, li, le = _panel_info(left, separator)
        rb, rm, ri, re = _panel_info(right, separator)
        left_expected = np.full((h,w), lb, dtype=np.int8)
        right_expected = np.full((h,w), rb, dtype=np.int8)
        left_expected[rm] = rb
        right_expected[lm] = lb
        _check(np.array_equal(output_grid[r:r+h, :w], left_expected),
               "left mask transfer or donor-base color incorrect")
        _check(np.array_equal(output_grid[r:r+h, w+cg:], right_expected),
               "right mask transfer or donor-base color incorrect")
        relation = ("identical" if np.array_equal(lm, rm) else
                    "disjoint" if not np.any(lm & rm) else
                    "contained" if np.all(lm <= rm) or np.all(rm <= lm) else
                    "overlap")
        records.append({"same": lb == rb, "identical": relation == "identical",
                        "relation": relation, "multi": li > 1 or ri > 1,
                        "edge": le or re,
                        "donor_ink": bool(np.any(left[lm] == rb) or np.any(right[rm] == lb)),
                        "nonidentical_visible": lb != rb and relation != "identical",
                        "mask_sizes": (int(lm.sum()), int(rm.sum())),
                        "masks": (_array_key(lm.astype(np.int8)), _array_key(rm.astype(np.int8)))})
    for i, r in enumerate(row_starts[:-1]):
        a = r + h
        b = row_starts[i+1]
        _check(np.all(input_grid[a:b] == separator) and
               np.all(output_grid[a:b] == separator), "row gutter changed")
    _check(not np.array_equal(input_grid, output_grid), "unchanged pair")
    return {"layout": (h,w,groups,cg,rg), "separator": separator,
            "groups": records}


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
           and len(puzzle) == 2, "puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "pair lists required")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen = set()
    statistics = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}
                   and len(pair) == 2, "pair schema")
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "grid must be a 2D array")
                _check(np.issubdtype(grid.dtype, np.integer), "integer grid required")
                _check(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                       "ARC dimension bound")
                _check(np.all((grid >= 0) & (grid <= 9)), "ARC color bound")
            key = _array_key(pair["input"])
            _check(key not in seen, "duplicate input or input/output pair")
            seen.add(key)
            _check(not _is_source_input(pair["input"]), "copied source input")
            statistics[section].append(_analyze_pair(pair["input"], pair["output"]))
    train_groups = [g for x in statistics["train"] for g in x["groups"]]
    test_groups = [g for x in statistics["test"] for g in x["groups"]]
    _check(sum(g["nonidentical_visible"] for g in train_groups) >= 2,
           "training lacks two visible nonidentical swaps")
    _check(any(g["donor_ink"] and not g["same"] for g in train_groups),
           "training lacks marker/donor-base role exchange")
    for feature in ("same", "identical", "multi", "edge"):
        if any(g[feature] for g in test_groups):
            _check(any(g[feature] for g in train_groups),
                   f"test {feature} case absent from training")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    # The transformation has no hidden color map: colors are local roles.
    train_pool = [int(v) for v in rng.choice(np.arange(1,10),
                  size=int(rng.integers(4,9)), replace=False)]
    unused = [v for v in range(1,10) if v not in train_pool]
    novel = int(rng.choice(unused)) if unused and rng.random() < .6 else None
    enable_identical = bool(rng.random() < .55)
    enable_multi = bool(rng.random() < .6)
    enable_edge = bool(rng.random() < .55)
    layouts = [_layout(rng, "source_train") for _ in range(train_count)]
    minimum = 4 if enable_identical else 3
    if sum(x[2] for x in layouts) < minimum:
        h,w,g,cg,rg = layouts[-1]
        target = min(5, g + minimum - sum(x[2] for x in layouts))
        h = min(h, (30 - (target-1)*rg)//target)
        if h < 4:
            rg, h = 1, 4
        layouts[-1] = (h,w,target,cg,rg)
    slots = [(i,j) for i,x in enumerate(layouts) for j in range(x[2])]
    rng.shuffle(slots)
    specs = [[{"same": False} for _ in range(x[2])] for x in layouts]
    for i,j in slots:
        if rng.random() < .22:
            specs[i][j]["same"] = True
    for i,j in slots[:2]:
        specs[i][j].update(same=False, relation="free")
    i,j = slots[2]
    specs[i][j]["same"] = True
    if enable_identical:
        i,j = slots[3]
        specs[i][j].update(same=False, relation="identical")
    i,j = slots[0]
    specs[i][j]["donor_ink"] = True
    multi_slot = None
    if enable_multi:
        i,j = slots[int(rng.integers(len(slots)))]
        specs[i][j]["multi"] = True
        multi_slot = (i,j)
    if enable_edge:
        i,j = (multi_slot if multi_slot is not None and rng.random() < .65
               else slots[int(rng.integers(len(slots)))])
        specs[i][j]["edge"] = True
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train":
                layout = layouts[index]
                group_specs = specs[index]
                pool = train_pool
            else:
                if rng.random() < .2:
                    layout = layouts[int(rng.integers(len(layouts)))]
                else:
                    layout = _layout(rng, "target_test")
                groups = layout[2]
                # A zero-cancellation test remains fully visible; one or more
                # cancellations are allowed only after the training example.
                same_count = int(rng.integers(0, min(groups - 1, 2) + 1))
                same_indices = set(int(v) for v in np.atleast_1d(
                    rng.choice(groups, size=same_count, replace=False)))
                group_specs = []
                for group_index in range(groups):
                    spec = {"same": group_index in same_indices}
                    if enable_identical and not spec["same"] and rng.random() < .16:
                        spec["relation"] = "identical"
                    if enable_multi and rng.random() < .2:
                        spec["multi"] = True
                    if enable_edge and rng.random() < .2:
                        spec["edge"] = True
                    group_specs.append(spec)
                pool = train_pool + ([novel] if novel is not None else [])
            for attempt in range(20):
                separator_options = [0] if rng.random() < .7 else [v for v in range(1,10)
                                      if novel is None or section == "train" or v != novel]
                separator = int(rng.choice(separator_options))
                pair = _make_pair(rng, layout, separator, pool, group_specs,
                                  novel if section == "test" and index == 0 else None)
                key = _array_key(pair[0])
                if (key not in seen and not _is_source_input(pair[0])
                        and not np.array_equal(pair[0], pair[1])):
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            seen.add(key)
            puzzle[section].append({"input": pair[0], "output": pair[1]})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, tuple((_array_key(p["input"]), _array_key(p["output"]))
                                 for p in puzzle[section])) for section in ("train", "test"))


def self_test() -> None:
    layouts = set()
    panel_shapes = set()
    separators = set()
    colors = set()
    relations = set()
    masks = set()
    train_counts = set()
    test_counts = set()
    cancellation_counts = set()
    cancellations_at = set()
    novel_tests = 0
    shared_layouts = 0
    reverse_counts = 0
    signatures = set()
    multi = edge = identical = donor_ink = 0
    motif_traits: dict[str, int] = {}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        train_layouts = set()
        train_group_counts = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                h,w,g,cg,rg = info["layout"]
                _,_,_,_,_,_,row_starts = _read_layout(pair["input"])
                for r in row_starts:
                    for panel in (pair["input"][r:r+h,:w],
                                  pair["input"][r:r+h,w+cg:]):
                        base = int(panel[0,0])
                        for trait in _mask_traits(panel != base):
                            motif_traits[trait] = motif_traits.get(trait,0) + 1
                layouts.add(info["layout"])
                panel_shapes.add((h,w))
                separators.add(info["separator"])
                colors.update(int(v) for v in np.unique(pair["input"]))
                if section == "train":
                    train_layouts.add(info["layout"])
                    train_group_counts.append(g)
                    train_colors.update(int(v) for v in np.unique(pair["input"]))
                else:
                    shared_layouts += info["layout"] in train_layouts
                    reverse_counts += g < max(train_group_counts)
                    novel_tests += bool(set(int(v) for v in np.unique(pair["input"])) - train_colors)
                if section == "test":
                    cancellation_counts.add(sum(item["same"] for item in info["groups"]))
                for index,item in enumerate(info["groups"]):
                    if item["same"]:
                        cancellations_at.add(index)
                    relations.add(item["relation"])
                    masks.update(item["masks"])
                    multi += item["multi"]
                    edge += item["edge"]
                    identical += item["identical"]
                    donor_ink += item["donor_ink"]
    _check(len(signatures) == 42, "cross-seed puzzle duplication")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic count diversity")
    _check(len(layouts) >= 60 and len(panel_shapes) >= 20,
           "layout or panel aspect diversity")
    _check(len(separators) >= 4 and len(colors) == 10, "color or gutter diversity")
    _check(len(relations) == 4 and len(masks) >= 200, "mask or relation diversity")
    _check({0,1,2} <= cancellation_counts and len(cancellations_at) >= 3,
           "cancellation count or position diversity")
    _check(novel_tests >= 8 and shared_layouts >= 1 and reverse_counts >= 8,
           "train/test palette or layout regime diversity")
    _check(min(multi,edge,identical,donor_ink) >= 8, "stronger motif case diversity")
    _check(all(motif_traits.get(trait,0) >= 8 for trait in
               ("tiny","dense","asymmetric","multicomponent","branch","thin","hollow","edge")),
           f"motif morphology lacks diversity: {motif_traits}")
    for a,b in ((2,1),(10,1),(10,4),(6,3)):
        p = generate_puzzle(1000 + a*10+b, a, b)
        _check((len(p["train"]),len(p["test"])) == (a,b), "explicit counts")
        validate_generated_puzzle(p)
    for arguments in ({"num_train":None,"num_test":1},
                      {"num_train":None,"num_test":4},
                      {"num_train":2,"num_test":None},
                      {"num_train":10,"num_test":None}):
        p = generate_puzzle(2000, **arguments)
        validate_generated_puzzle(p)
        if arguments["num_train"] is not None:
            _check(len(p["train"]) == arguments["num_train"], "partial train count")
        if arguments["num_test"] is not None:
            _check(len(p["test"]) == arguments["num_test"], "partial test count")
    for arguments in ({"num_train":1}, {"num_train":11}, {"num_test":0},
                      {"num_test":5}, {"num_train":2,"num_test":2},
                      {"num_train":3,"num_test":4}):
        try:
            generate_puzzle(3, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {arguments}")
    _check(_puzzle_key(generate_puzzle(777,7,3)) ==
           _puzzle_key(generate_puzzle(777,7,3)), "nondeterministic generation")
    _check(_puzzle_key(generate_puzzle(777,7,3)) !=
           _puzzle_key(generate_puzzle(778,7,3)), "seed has no effect")
