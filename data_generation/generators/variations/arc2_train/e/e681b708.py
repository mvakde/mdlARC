import numpy as np


def _checked_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(rng, num_train, num_test):
    nt = _checked_count("num_train", num_train)
    ns = _checked_count("num_test", num_test)

    if nt is not None and not 2 <= nt <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if ns is not None and not 1 <= ns <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if nt is None and ns is None:
        ns = int(rng.integers(1, 5))
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif nt is None:
        nt = int(rng.integers(max(2, ns + 1), 11))
    elif ns is None:
        ns = int(rng.integers(1, min(4, nt - 1) + 1))

    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")

    return nt, ns


def _sample_spaced(rng, low, high, count, gap):
    if count < 1 or low + (count - 1) * gap > high:
        raise ValueError("impossible spaced-coordinate request")
    compressed = rng.choice(
        np.arange(low, high - (count - 1) * (gap - 1) + 1),
        size=count,
        replace=False,
    )
    return [int(x) + i * (gap - 1) for i, x in enumerate(np.sort(compressed))]


def _network_nodes(network):
    h, w = network.shape
    nodes = []

    for item in np.argwhere(network):
        r, c = int(item[0]), int(item[1])

        up = r > 0 and bool(network[r - 1, c])
        down = r + 1 < h and bool(network[r + 1, c])
        left = c > 0 and bool(network[r, c - 1])
        right = c + 1 < w and bool(network[r, c + 1])

        degree = int(up) + int(down) + int(left) + int(right)
        corner = degree == 2 and not (
            (up and down) or (left and right)
        )

        if degree != 2 or corner:
            nodes.append((r, c))

    return nodes


def _enumerate_faces(network):
    h, w = network.shape
    face_id = np.full((h, w), -1, dtype=np.int16)
    faces = []

    for r0 in range(h):
        for c0 in range(w):
            if network[r0, c0] or face_id[r0, c0] >= 0:
                continue

            index = len(faces)
            stack = [(r0, c0)]
            face_id[r0, c0] = index
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))

                for dr, dc in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    rr, cc = r + dr, c + dc

                    if (
                        0 <= rr < h
                        and 0 <= cc < w
                        and not network[rr, cc]
                        and face_id[rr, cc] < 0
                    ):
                        face_id[rr, cc] = index
                        stack.append((rr, cc))

            faces.append(cells)

    return face_id, faces


def _face_data(network):
    h, w = network.shape
    nodes = _network_nodes(network)
    face_id, faces = _enumerate_faces(network)

    incident = [[] for _ in faces]

    for node_index, (r, c) in enumerate(nodes):
        touched = set()

        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                face = int(face_id[rr, cc])
                if face >= 0:
                    touched.add(face)

        for face in touched:
            incident[face].append(node_index)

    dot_candidates = []

    for cells in faces:
        usable = []

        for r, c in cells:
            clear = True

            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if network[rr, cc]:
                        clear = False
                        break

                if not clear:
                    break

            if clear:
                usable.append((r, c))

        dot_candidates.append(usable)

    return nodes, face_id, faces, incident, dot_candidates


def _line_runs(network, vertical, minimum=4):
    h, w = network.shape
    runs = []

    outer = w if vertical else h
    inner = h if vertical else w

    for fixed in range(outer):
        position = 0

        while position < inner:
            value = (
                network[position, fixed]
                if vertical
                else network[fixed, position]
            )

            if not value:
                position += 1
                continue

            start = position

            while position + 1 < inner:
                nxt = (
                    network[position + 1, fixed]
                    if vertical
                    else network[fixed, position + 1]
                )

                if not nxt:
                    break

                position += 1

            end = position

            if end - start + 1 >= minimum:
                runs.append((fixed, start, end))

            position += 1

    return runs


def _topology_stats(network):
    h, w = network.shape

    vertical = _line_runs(network, True)
    horizontal = _line_runs(network, False)

    full_vertical = tuple(
        sorted(
            run[0]
            for run in vertical
            if run[1] == 0 and run[2] == h - 1
        )
    )

    bounded_vertical = tuple(
        run
        for run in vertical
        if run[1] > 0 and run[2] < h - 1
    )

    horizontal_rows = tuple(
        sorted(set(run[0] for run in horizontal))
    )

    left_branch = any(
        start == 0 and end < w - 1
        for _, start, end in horizontal
    )

    right_branch = any(
        start > 0 and end == w - 1
        for _, start, end in horizontal
    )

    return {
        "full_vertical": full_vertical,
        "bounded_vertical": bounded_vertical,
        "horizontal_rows": horizontal_rows,
        "left_branch": left_branch,
        "right_branch": right_branch,
    }


import hashlib


_SOURCE_INPUT_HASHES = frozenset({
    'c09ab859921e0e057a49b0f3d56e77dc87a633d1a5a08d82aadcde0a9fbe35e1',
    'f4b00f6cdb4c1cecdc44ea907794f176b31071c8770142f9c3a3323b8c3f80f5',
    'c7e738a91663a3c611127ad56bfba000f3ecd5b14ff8281e4cb649c7b747923b',
    '5c4762685c86d6be65e8f6266ca2efd9d9c185fc93f0663e94d2c43007efbf84',
})


def _source_hash(grid):
    h, w = grid.shape
    return hashlib.sha256(bytes((h, w)) + np.asarray(grid, dtype=np.uint8).tobytes()).hexdigest()


def _sample_network(rng, regime, index):
    dense_test = regime == 'target_test' and index == 0
    h = int(rng.integers(22 if dense_test else 20, 31))
    w = int(rng.integers(26 if dense_test else 20, 31))
    spine_count = 2 if dense_test or (regime == 'source_train' and index == 1) else int(rng.integers(1, 3))
    if spine_count == 1:
        spines = [int(rng.integers(5, w - 5))]
    else:
        spines = _sample_spaced(rng, 5, w - 6, 2, 8)
    max_rows = min(5 if regime == 'target_test' else 4, 1 + (h - 7) // 5)
    min_rows = 3 if regime == 'target_test' else 2
    row_count = int(rng.integers(min_rows, max_rows + 1))
    rows = _sample_spaced(rng, 3, h - 4, row_count, 5)
    net = np.zeros((h, w), dtype=bool)
    for x in spines:
        net[:, x] = True
    full_count = 2 if row_count >= 3 and rng.random() < (0.60 if dense_test else 0.30) else 1
    full_rows = set(int(v) for v in rng.choice(rows, size=full_count, replace=False))
    if spine_count == 1:
        x = spines[0]
        spans = [(0, x), (x, w - 1), (0, w - 1)]
    else:
        x1, x2 = spines
        spans = [(0, x1), (x1, x2), (x2, w - 1), (0, x2), (x1, w - 1), (0, w - 1)]
    drawn = {}
    for row in rows:
        if row in full_rows:
            start, end = 0, w - 1
        else:
            start, end = spans[int(rng.integers(len(spans)))]
        net[row, start:end + 1] = True
        drawn[row] = (start, end)
    used_x = list(spines)
    want_partial = (regime == 'source_train' and index == 0) or dense_test or rng.random() < 0.58
    partial_count = (1 + int(regime == 'target_test' and not dense_test and rng.random() < 0.25)) if want_partial else 0
    for _ in range(partial_count):
        options = [x for x in range(4, w - 4) if all(abs(x - prev) >= 4 for prev in used_x)]
        if not options:
            break
        x = int(rng.choice(options))
        row = int(rng.choice(sorted(full_rows)))
        if rng.random() < 0.5:
            net[:row + 1, x] = True
        else:
            net[row:, x] = True
        used_x.append(x)
    want_bridge = dense_test or (regime == 'source_train' and index == 1 and rng.random() < 0.55) or rng.random() < 0.30
    if want_bridge and len(rows) >= 2:
        options = [x for x in range(4, w - 4) if all(abs(x - prev) >= 4 for prev in used_x)]
        rng.shuffle(options)
        for x in options:
            covering = [r for r in rows if drawn[r][0] <= x <= drawn[r][1]]
            if len(covering) >= 2:
                a, b = sorted(int(v) for v in rng.choice(covering, size=2, replace=False))
                net[a:b + 1, x] = True
                break
    return net


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        groups.append(cells)
    return groups


def _interpret_input(inp):
    h, w = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert int(counts.max()) > inp.size // 2
    remaining = [(int(n), int(v)) for v, n in zip(values, counts) if int(v) != bg]
    structural = max(remaining)[1]
    components = _components(inp != bg)
    largest = max(components, key=len)
    assert len(largest) >= 12 and sum(len(g) == len(largest) for g in components) == 1
    net = np.zeros(inp.shape, dtype=bool)
    for r, c in largest:
        net[r, c] = True
    queries = []
    for group in components:
        if group is largest:
            continue
        assert len(group) == 1
        r, c = group[0]
        assert int(inp[r, c]) == structural
        queries.append((r, c))
    assert queries
    assert not np.any(net[:-1, :-1] & net[1:, :-1] & net[:-1, 1:] & net[1:, 1:])
    for r, c in queries:
        assert not np.any(net[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)])
    for i, (r, c) in enumerate(queries):
        assert all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in queries[:i])
    nodes, face_id, faces, incident, candidates = _face_data(net)
    node_set = set(nodes)
    marker_colors = set()
    for r, c in np.argwhere(net):
        r, c = int(r), int(c)
        v = int(inp[r, c])
        if (r, c) in node_set:
            assert v not in (bg, structural)
            marker_colors.add(v)
        else:
            assert v == structural
    assert 2 <= len(marker_colors) <= 5
    assert set(int(v) for v in values) == {bg, structural} | marker_colors
    expected = inp.copy()
    queried_faces = set()
    votes = []
    for r, c in queries:
        face = int(face_id[r, c])
        assert face >= 0 and (r, c) in candidates[face]
        queried_faces.add(face)
    for face in sorted(queried_faces):
        tally = {v: 0 for v in marker_colors}
        for j in incident[face]:
            r, c = nodes[j]
            tally[int(inp[r, c])] += 1
        maximum = max(tally.values())
        winners = [v for v, n in tally.items() if n == maximum]
        assert maximum > 0 and len(winners) == 1
        winner = winners[0]
        for r, c in queries:
            if int(face_id[r, c]) == face:
                expected[r, c] = winner
        votes.append((winner, tuple(sorted(tally.values(), reverse=True))))
    return expected, {
        'background': bg, 'structural': structural, 'markers': frozenset(marker_colors),
        'network': net, 'faces': faces, 'face_id': face_id, 'queries': queries,
        'queried_faces': queried_faces, 'votes': votes, 'topology': _topology_stats(net),
    }


def _choose_assignment(rng, net, labels, dense, dominant, require_multi, losing=None, balanced=False):
    nodes, face_id, faces, incident, candidates = _face_data(net)
    eligible = [i for i, pool in enumerate(candidates) if pool]
    if len(eligible) < 3 or (dense and len(eligible) != len(faces)):
        return None
    n = len(nodes)
    m = len(labels)
    matrix = np.zeros((len(eligible), n), dtype=np.int16)
    for row, face in enumerate(eligible):
        matrix[row, incident[face]] = 1
    primary_weight = 0.43 if balanced else 0.65
    weights = np.array([primary_weight if v == dominant else (1 - primary_weight) / (m - 1) for v in labels])
    if losing is not None:
        weights[labels.index(losing)] = 0.07
        weights /= weights.sum()
    for _ in range(8):
        assignment = rng.choice(m, size=(128, n), p=weights)
        onehot = np.eye(m, dtype=np.int16)[assignment]
        tally = np.einsum('fn,bnk->bfk', matrix, onehot)
        maximum = tally.max(axis=2)
        unique = (tally == maximum[:, :, None]).sum(axis=2) == 1
        winner = tally.argmax(axis=2)
        for batch in range(128):
            valid_faces = np.flatnonzero(unique[batch])
            if dense and len(valid_faces) != len(faces):
                continue
            if len(valid_faces) < (len(faces) if dense else 3):
                continue
            if len(set(int(v) for v in assignment[batch])) != m:
                continue
            active = set(int(winner[batch, f]) for f in valid_faces)
            if require_multi and len(active) < 2:
                continue
            if dominant is not None and labels.index(dominant) not in active:
                continue
            if losing is not None and labels.index(losing) in active:
                continue
            rich = any(maximum[batch, f] >= 2 and np.count_nonzero(tally[batch, f]) >= 2 for f in valid_faces)
            if not rich:
                continue
            return nodes, faces, candidates, eligible, valid_faces, assignment[batch]
    return None


def _place_dots(rng, candidates, chosen_faces, dense, shape):
    dots = []
    for face in chosen_faces:
        pool = list(candidates[face])
        rng.shuffle(pool)
        if dense:
            edge = [p for p in pool if p[0] in (0, shape[0] - 1) or p[1] in (0, shape[1] - 1)]
            if edge and rng.random() < 0.55:
                pool = [edge[0]] + [p for p in pool if p != edge[0]]
        wanted = int(rng.integers(2, 6)) if dense else int(rng.integers(1, 4))
        local = []
        for r, c in pool:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in dots):
                dots.append((r, c))
                local.append((r, c))
                if len(local) == wanted:
                    break
        if not local:
            return None
    return dots


def _generate_pair(rng, regime, index, colors):
    backgrounds, structurals, label_pool = colors[:2], colors[2:4], colors[4:9]
    if index == 0:
        bg, structural = int(backgrounds[0]), int(structurals[0])
    else:
        bg = int(rng.choice(backgrounds))
        structural = int(rng.choice(structurals))
    if regime == 'source_train':
        if index == 0:
            labels = [int(label_pool[j]) for j in (0, 1, 2)]
            dominant, losing = labels[0], None
        elif index == 1:
            labels = [int(label_pool[j]) for j in (0, 1, 3)]
            dominant, losing = labels[1], labels[2]
        else:
            labels = [int(v) for v in rng.choice(label_pool[:4], size=int(rng.integers(2, 5)), replace=False)]
            dominant, losing = labels[0], None
    else:
        if index == 0:
            labels = [int(label_pool[j]) for j in (0, 1, 4)]
            dominant, losing = labels[2], None
        else:
            choices = [int(label_pool[j]) for j in (0, 1, 2, 4)]
            label_count = 4 if rng.random() < 0.5 else int(rng.integers(2, 4))
            labels = [int(v) for v in rng.choice(choices, size=label_count, replace=False)]
            dominant, losing = labels[0], None
    dense = index == 0 or (index != 1 and (regime == 'target_test' and rng.random() < 0.45 or rng.random() < 0.28))
    require_multi = regime == 'source_train' and index == 0
    for _ in range(90):
        net = _sample_network(rng, regime, index)
        selected = _choose_assignment(rng, net, labels, dense, dominant, require_multi, losing,
                                      balanced=regime == 'target_test' and index > 0)
        if selected is None:
            continue
        nodes, faces, candidates, eligible, valid, assignment = selected
        if dense:
            chosen = [eligible[f] for f in valid]
        else:
            choices = [eligible[f] for f in valid]
            upper = len(choices) - (regime == 'source_train' and index == 1)
            k = int(rng.integers(max(2, len(choices) // 2), upper + 1))
            chosen = [int(v) for v in rng.choice(choices, size=k, replace=False)]
        dots = _place_dots(rng, candidates, chosen, dense, net.shape)
        if dots is None:
            continue
        inp = np.full(net.shape, bg, dtype=np.int8)
        inp[net] = structural
        for j, (r, c) in enumerate(nodes):
            inp[r, c] = labels[int(assignment[j])]
        for r, c in dots:
            inp[r, c] = structural
        try:
            out, info = _interpret_input(inp)
        except AssertionError:
            continue
        if not np.any(out != inp) or _source_hash(inp) in _SOURCE_INPUT_HASHES:
            continue
        if regime == 'source_train' and index == 1 and losing in set(int(v) for v in out[inp != out]):
            continue
        if regime == 'target_test' and index == 0 and dominant not in set(int(v) for v in out[inp != out]):
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('could not construct a valid pair')


def _grid_key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen and _source_hash(inp) not in _SOURCE_INPUT_HASHES
            seen.add(key)
            expected, info = _interpret_input(inp)
            assert np.array_equal(expected, out)
            difference = inp != out
            assert np.count_nonzero(difference) == len(info['queries'])
            assert set(int(v) for v in inp[difference]) == {info['structural']}
            assert set(int(v) for v in out[difference]) <= info['markers']
            assert any(votes[0] > votes[1] for _, votes in info['votes'] if len(votes) > 1)
            analyses[split].append(info)
    assert len(analyses['train'][0]['queried_faces']) == len(analyses['train'][0]['faces'])
    assert len(analyses['test'][0]['queried_faces']) == len(analyses['test'][0]['faces'])
    assert len(analyses['train'][1]['queried_faces']) < len(analyses['train'][1]['faces'])
    first_network = analyses['train'][0]['network']
    assert any((start == 0) != (end == first_network.shape[axis] - 1)
               for axis, vertical in ((0, True), (1, False))
               for _, start, end in _line_runs(first_network, vertical))
    train_colors = set().union(*(a['markers'] for a in analyses['train']))
    test_colors = set().union(*(a['markers'] for a in analyses['test']))
    assert train_colors - test_colors
    assert analyses['train'][0]['markers'] != analyses['train'][1]['markers']
    test_first = analyses['test'][0]
    assert any(v not in train_colors for v in test_first['markers'])
    novel = test_first['markers'] - train_colors
    assert any(int(v) in novel for v in puzzle['test'][0]['output'][puzzle['test'][0]['input'] != puzzle['test'][0]['output']])
    # Semantic color roles are local: the modal background and long connected divider
    # identify them even when literal colors change between pairs.
    for info in analyses['train'] + analyses['test']:
        assert 20 <= info['network'].shape[0] <= 30
        assert 20 <= info['network'].shape[1] <= 30


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(rng, num_train, num_test)
    colors = [int(v) for v in rng.permutation(10)]
    rotation = int(rng.choice((0, 0, 0, 0, 1, 2, 3)))
    reflected = bool(rng.random() < 0.30)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count, regime in (('train', nt, 'source_train'), ('test', ns, 'target_test')):
        for index in range(count):
            for _ in range(12):
                pair = _generate_pair(rng, regime, index, colors)
                if rotation or reflected:
                    pair = {name: np.fliplr(np.rot90(grid, rotation)).copy()
                            if reflected else np.rot90(grid, rotation).copy()
                            for name, grid in pair.items()}
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate generation failure')
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output')
    ) for s in ('train', 'test'))


def self_test() -> None:
    for nt, ns in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(3000 + (nt or 0) * 10 + (ns or 0), nt, ns))
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                   {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(9, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    train_counts, test_counts, shapes, topologies, palettes, coverage = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                out, info = _interpret_input(pair['input'])
                assert np.array_equal(out, pair['output'])
                shapes.add(pair['input'].shape)
                t = info['topology']
                topologies.add((len(t['full_vertical']), len(t['bounded_vertical']),
                                len(t['horizontal_rows']), len(info['faces']),
                                bool(t['left_branch']), bool(t['right_branch'])))
                palettes.add((info['background'], info['structural'], tuple(sorted(info['markers']))))
                coverage.add((split, len(info['queried_faces']), len(info['faces'])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 25 and len(topologies) >= 20 and len(palettes) >= 30
    assert any(a == b for _, a, b in coverage) and any(a < b for _, a, b in coverage)
