"""Color-coordinate stamping, with visible masks and source-order priority.

The only semantic conventions are the source-supported left-to-right layout,
odd bounding-box centers, paired coordinates, and top-to-bottom priority.
Palette choices and curriculum parameters never change that transformation.
"""
import numpy as np


def _count_value(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    a, b = _count_value(num_train, "num_train"), _count_value(num_test, "num_test")
    if a is not None and not 2 <= a <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if b is not None and not 1 <= b <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if a is None and b is None:
        b = 1 + seed % 4
        a = b + 1 + (seed // 4) % (10 - b)
    elif a is None:
        a = b + 1 + (seed // 7) % (10 - b)
    elif b is None:
        b = 1 + seed % min(4, a - 1)
    if a <= b:
        raise ValueError("num_train must exceed num_test")
    return a, b


def _key(grid):
    # Numeric equality must not depend on the caller's integer dtype.
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _connected(mask, diagonal=True):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    seen = {next(iter(cells))}
    todo = list(seen)
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while todo:
        r, c = todo.pop()
        for dr, dc in steps:
            p = r + dr, c + dc
            if p in cells and p not in seen:
                seen.add(p)
                todo.append(p)
    return seen == cells


def _sample_mask(rng, h, w):
    """Finite source motifs plus a much larger connected, compositional space."""
    mode = int(rng.integers(10))
    mask = np.zeros((h, w), dtype=bool)
    if mode == 0 or min(h, w) == 1:
        mask[:] = True
    elif mode == 1:
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2 and h == w:
        yy, xx = np.indices((h, w))
        mask = abs(yy - h // 2) + abs(xx - w // 2) == h // 2
    elif mode == 3:
        for r in range(h):
            mask[r, round(r * (w - 1) / (h - 1))] = True
        for c in range(w):
            mask[round(c * (h - 1) / (w - 1)), c] = True
    elif mode == 4 and h == w:
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        # An unadorned cross remains possible, but is not an eligibility cue.
        mask |= rng.random((h, w)) < float(rng.uniform(0, .25))
    elif mode == 5 and h == w == 3:
        mask[:] = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool)
    else:
        # Connect independently selected extremal points with 8-neighbor paths.
        points = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(points)
        r, c = points[0]
        mask[r, c] = True
        for tr, tc in points[1:]:
            while (r, c) != (tr, tc):
                dr, dc = int(np.sign(tr - r)), int(np.sign(tc - c))
                if dr and dc and rng.random() < .45:
                    if rng.random() < .5:
                        dr = 0
                    else:
                        dc = 0
                r, c = r + dr, c + dc
                mask[r, c] = True
        target = int(rng.uniform(.18, .98) * h * w)
        while int(mask.sum()) < target:
            frontier = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                        and mask[max(0, r-1):r+2, max(0, c-1):c+2].any()]
            r, c = frontier[int(rng.integers(len(frontier)))]
            mask[r, c] = True
    if rng.random() < .5:
        mask = mask[::-1]
    if rng.random() < .5:
        mask = mask[:, ::-1]
    if h == w:
        mask = np.rot90(mask, int(rng.integers(4)))
    return mask.copy()


def _geometry(inp):
    """Find the layout from its empty interior and border, not literal colors."""
    h, w = inp.shape
    candidates = []
    for d in range(1, w - 4):
        if not np.all(inp[:, d] == inp[0, d]):
            continue
        interior = inp[1:-1, d+2:-1]
        if not interior.size or not np.all(interior == interior[0, 0]):
            continue
        bg, divider = int(interior[0, 0]), int(inp[0, d])
        edge = np.concatenate((inp[0, d+1:], inp[-1, d+1:],
                               inp[1:-1, d+1], inp[1:-1, -1]))
        counts = np.bincount(edge, minlength=10)
        frame = int(np.argmax(counts))
        corners = inp[np.array([0, 0, h-1, h-1]), np.array([d+1, w-1, d+1, w-1])]
        if len({bg, divider, frame}) != 3 or np.count_nonzero(corners == frame) < 3:
            continue
        if counts[frame] <= max(counts[c] for c in range(10) if c != frame):
            continue
        stage = inp[:, :d]
        if np.any(stage != bg) and not np.any((stage == frame) | (stage == divider)):
            candidates.append((d, bg, divider, frame))
    assert len(candidates) == 1, "layout must be uniquely visible"
    return candidates[0]


def _parse(inp):
    d, bg, divider, frame = _geometry(inp)
    h, w = inp.shape
    rows, cols = {}, {}
    for r in range(1, h - 1):
        color = int(inp[r, d+1])
        if color == int(inp[r, -1]) and color not in (bg, frame):
            rows.setdefault(color, []).append(r)
    for c in range(d+2, w-1):
        color = int(inp[0, c])
        if color == int(inp[-1, c]) and color not in (bg, frame):
            cols.setdefault(color, []).append(c)
    templates = []
    for color in np.unique(inp[:, :d]):
        color = int(color)
        if color == bg:
            continue
        cells = np.argwhere(inp[:, :d] == color)
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        bh, bw = map(int, hi - lo + 1)
        assert bh % 2 == bw % 2 == 1, "centers require odd bounding boxes"
        mask = inp[lo[0]:hi[0]+1, lo[1]:hi[1]+1] == color
        assert _connected(mask), "one identifiable 8-connected template per color"
        # Irregular marks are harmless only for colors without templates.
        assert np.array_equal(inp[0, d+2:-1] == color, inp[-1, d+2:-1] == color)
        assert np.array_equal(inp[1:-1, d+1] == color, inp[1:-1, -1] == color)
        assert all(int(inp[r, c]) != color for r, c in ((0,d+1),(0,w-1),(h-1,d+1),(h-1,w-1)))
        rs, cs = rows.get(color, []), cols.get(color, [])
        state = "A" if rs and cs else "R" if rs else "C" if cs else "N"
        center = (lo + hi) // 2
        offsets = cells - center
        stamps = []
        boxes = []
        if state == "A":
            for r in rs:
                for c in cs:
                    translated = offsets + np.array([r, c])
                    assert np.all((translated[:, 0] >= 1) & (translated[:, 0] < h-1))
                    assert np.all((translated[:, 1] >= d+2) & (translated[:, 1] < w-1))
                    stamps.append(set(map(tuple, translated)))
                    boxes.append({(y, x) for y in range(r-bh//2, r+bh//2+1)
                                  for x in range(c-bw//2, c+bw//2+1)})
        templates.append(dict(color=color, cells=cells, lo=tuple(map(int, lo)),
                              shape=(bh,bw), mask=mask.copy(), rows=rs, cols=cs,
                              state=state, stamps=stamps, boxes=boxes))
    templates.sort(key=lambda t: t["lo"])
    assert len({t["lo"][0] for t in templates}) == len(templates), "no source-order ties"
    return dict(d=d, bg=bg, divider=divider, frame=frame, templates=templates,
                rows=rows, cols=cols, height=h, width=w)


def _predict(inp, parsed=None, order="source", erase_boxes=False):
    """Visible-input predictor: forward traversal and an explicit occupancy set."""
    p = _parse(inp) if parsed is None else parsed
    out = inp.copy()
    active = [t for t in p["templates"] if t["state"] == "A"]
    for t in active:
        if erase_boxes:
            r, c = t["lo"]
            h, w = t["shape"]
            out[r:r+h, c:c+w] = p["bg"]
        else:
            out[t["cells"][:,0], t["cells"][:,1]] = p["bg"]
    if order == "destination":
        active.sort(key=lambda t: (min(t["rows"]), min(t["cols"])))
    elif order == "size":
        active.sort(key=lambda t: t["shape"][0] * t["shape"][1])
    elif order == "reverse":
        active.reverse()
    occupied = set()
    for t in active:
        for stamp in t["stamps"]:
            for r, c in stamp - occupied:
                out[r, c] = t["color"]
            occupied.update(stamp)
    return out


def solve(train_pairs, test_input):
    """Prediction API accepts only demonstrations and the visible query.

    This is a local diagnostic solver, not a protected independent reference.
    Select priority from demonstrated outputs, without access to test labels.
    """
    orders = ["source", "destination", "size", "reverse"]
    consistent = [o for o in orders if all(np.array_equal(_predict(p["input"], order=o), p["output"])
                                         for p in train_pairs)]
    assert "source" in consistent, "inconsistent training transformation"
    predictions = [_predict(test_input, order=o) for o in consistent]
    assert all(np.array_equal(predictions[0], x) for x in predictions[1:]), "priority is underdemonstrated"
    return predictions[0]


def _features(inp, out=None):
    p = _parse(inp)
    ts = p["templates"]
    active = [t for t in ts if t["state"] == "A"]
    correct = _predict(inp, p)
    unions = [set().union(*t["stamps"]) for t in active]
    conflicts = sum(len(a & b) for i, a in enumerate(unions) for b in unions[i+1:])
    merged = sum(sum(map(len, t["stamps"])) - len(u) for t, u in zip(active, unions))
    touch = sum(not (a & b) and any((r+dr,c+dc) in b for r,c in a
                                   for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
                for t in active for i,a in enumerate(t["stamps"]) for b in t["stamps"][i+1:])
    directions = tuple((int(np.sign(min(a["rows"])-min(b["rows"]))),
                        int(np.sign(min(a["cols"])-min(b["cols"]))))
                       for i,a in enumerate(active) for b in active[i+1:])
    spacing = tuple(int(gap) for t in active for name in ("rows","cols")
                    for gap in np.diff(sorted(t[name])))
    holes = 0
    for t, u in zip(active, unions):
        transparent = set().union(*t["boxes"]) - u
        holes += sum(correct[r,c] not in (p["bg"], t["color"]) for r,c in transparent)
    source_overlap = sum(max(a["lo"][0], b["lo"][0]) < min(a["lo"][0]+a["shape"][0], b["lo"][0]+b["shape"][0])
                         and max(a["lo"][1], b["lo"][1]) < min(a["lo"][1]+a["shape"][1], b["lo"][1]+b["shape"][1])
                         for i,a in enumerate(ts) for b in ts[i+1:])
    border = np.concatenate((inp[0,p["d"]+1:], inp[-1,p["d"]+1:], inp[1:-1,p["d"]+1], inp[1:-1,-1]))
    absent = set(map(int, border)) - {p["bg"],p["frame"]} - {t["color"] for t in ts}
    irregular = int(np.count_nonzero(inp[0,p["d"]+2:-1] != inp[-1,p["d"]+2:-1]))
    irregular += int(np.count_nonzero(inp[1:-1,p["d"]+1] != inp[1:-1,-1]))
    corners = [int(inp[r,c]) for r,c in ((0,p["d"]+1),(0,inp.shape[1]-1),(inp.shape[0]-1,p["d"]+1),(inp.shape[0]-1,inp.shape[1]-1))]
    return dict(height=p["height"], width=p["width"], strip=p["d"],
                roles=(p["bg"],p["divider"],p["frame"]), count=len(ts), active=len(active),
                states="".join(t["state"] for t in ts), absent=len(absent),
                shapes=tuple(t["shape"] for t in ts),
                active_shapes=tuple(t["shape"] for t in active),
                motifs=tuple((t["shape"],tuple(map(int,t["mask"].flat))) for t in ts),
                active_motifs=tuple((t["shape"],tuple(map(int,t["mask"].flat))) for t in active),
                inactive_shapes=tuple(t["shape"] for t in ts if t["state"]!="A"),
                multiplicities=tuple((len(t["rows"]),len(t["cols"])) for t in active),
                source_overlap=int(source_overlap),
                erase_box_difference=int(np.count_nonzero(correct != _predict(inp,p,erase_boxes=True))),
                conflicts=conflicts, same_color_merge=merged, hole_exposure=holes,
                touch=int(touch), directions=directions, spacing=spacing,
                destination_difference=int(np.count_nonzero(correct != _predict(inp,p,order="destination"))),
                size_difference=int(np.count_nonzero(correct != _predict(inp,p,order="size"))),
                reverse_difference=int(np.count_nonzero(correct != _predict(inp,p,order="reverse"))),
                empty_centers=sum(not t["mask"][t["shape"][0]//2,t["shape"][1]//2] for t in active),
                solid=sum(t["mask"].all() for t in active),
                sparse=sum(t["mask"].mean() <= .4 for t in active),
                dense=sum(t["mask"].mean() >= .75 for t in active),
                diagonal=sum(not _connected(t["mask"],False) for t in active),
                asymmetric=sum(not np.array_equal(t["mask"],np.rot90(t["mask"],2)) for t in active),
                irregular=irregular, corner_noise=sum(c != p["frame"] for c in corners),
                divider_label=p["divider"] in absent,
                black_template=any(t["color"]==0 for t in ts))


def _dimensions(rng, parameters, regime, witness=False):
    if parameters["shift"] == "extrapolate":
        low, high = (9,21) if regime == "source_train" else (17,31)
    else:
        low, high = (7,31)
    h = int(rng.integers(low, high))
    sw = int(rng.integers(3,10))
    fw = int(rng.integers(7,30-sw))
    if witness:
        h, sw, fw = max(h,15), max(sw,5), max(fw,13)
        fw = min(fw,29-sw)
    return h, sw, fw


def _new_template(mask, top, left, state="A"):
    return dict(mask=mask, top=int(top), left=int(left), state=state, rows=[], cols=[])


def _place_templates(rng, h, sw, count, max_size, bank=(), fixed=()):
    ts = list(fixed)
    occupied = np.zeros((h,sw),dtype=bool)
    tops = set()
    for t in ts:
        bh,bw=t["mask"].shape
        occupied[t["top"]:t["top"]+bh,t["left"]:t["left"]+bw] |= t["mask"]
        tops.add(t["top"])
    allow_interleave = rng.random() < .4
    for _ in range(count-len(ts)):
        placed=False
        for trial in range(40):
            sizes=[(a,b) for a in (1,3,5,7) for b in (1,3,5,7)
                   if a<=min(h-2,max_size) and b<=min(sw,max_size)
                   and (a,b)!=(1,1)]
            weights=np.array([5 if a==b==3 else 2 if max(a,b)<=5 else .6 for a,b in sizes],float)
            bh,bw=sizes[int(rng.choice(len(sizes),p=weights/weights.sum()))]
            compatible=[m for m in bank if m.shape[0]<=h-2 and m.shape[1]<=sw]
            mask=compatible[int(rng.integers(len(compatible)))].copy() if compatible else _sample_mask(rng,bh,bw)
            bh,bw=mask.shape
            candidates=[]
            for r in range(h-bh+1):
                if r in tops:
                    continue
                for c in range(sw-bw+1):
                    blocked=occupied[r:r+bh,c:c+bw]
                    if not np.any(blocked & mask) and (allow_interleave or not blocked.any()):
                        candidates.append((r,c))
            if candidates:
                r,c=candidates[int(rng.integers(len(candidates)))]
                placed=True
                break
        if not placed:
            if bank:
                # Recombination must not invent a fallback motif. There is no
                # output-relevant count until source placement is complete.
                break
            # A free top row and an unoccupied cell always exist with <=6 colors.
            choices=[(r,c) for r in range(h) if r not in tops for c in range(sw) if not occupied[r,c]]
            if not choices:
                break
            r,c=choices[int(rng.integers(len(choices)))]
            mask=np.ones((1,1),dtype=bool)
            bh,bw=mask.shape
        t=_new_template(mask,r,c)
        ts.append(t)
        occupied[r:r+bh,c:c+bw] |= mask
        tops.add(r)
    ts.sort(key=lambda t:(t["top"],t["left"]))
    return ts


def _priority_templates(rng):
    """Variable masks distinguish source priority, destination order and size."""
    for _ in range(100):
        dims=[(5,5,3,3),(5,3,3,3),(3,5,3,3),(7,5,3,3)]
        bh,bw,sh,sw=dims[int(rng.integers(len(dims)))]
        a,b=_sample_mask(rng,bh,bw),_sample_mask(rng,sh,sw)
        dc=int(rng.choice([-1,1]))
        aa={(r-bh//2+1,c-bw//2+dc) for r,c in np.argwhere(a)}
        bb={(r-sh//2,c-sw//2) for r,c in np.argwhere(b)}
        box={(r-bh//2+1,c-bw//2+dc) for r in range(bh) for c in range(bw)}
        if aa & bb and (box-aa) & bb and aa-bb and bb-aa:
            return a,b,dc
    a=np.ones((5,5),bool)
    a[1:-1,1:-1]=False
    return a,np.ones((3,3),bool),1


def _assign_coordinates(rng, ts, h, fw, separated=False, repeat=True):
    used=[set(),set()]
    for t in ts:
        for axis in (0,1):
            used[axis].update(t["rows" if axis==0 else "cols"])
    for axis, (length, name) in enumerate(((h,"rows"),(fw,"cols"))):
        needed=[t for t in ts if t["state"] in ("A", "R" if axis==0 else "C") and not t[name]]
        needed.sort(key=lambda t:-t["mask"].shape[axis])
        for t in needed:
            rad=t["mask"].shape[axis]//2 if t["state"]=="A" else 0
            choices=[v for v in range(1+rad,length-1-rad) if v not in used[axis]]
            if not choices:
                return False
            v=int(rng.choice(choices))
            t[name]=[v]
            used[axis].add(v)
    # Extras are assigned in shuffled color order, independently on each axis.
    if repeat:
        for idx in rng.permutation(len(ts)):
            t=ts[int(idx)]
            if t["state"]!="A" or t.get("single",False):
                continue
            for axis,(length,name) in enumerate(((h,"rows"),(fw,"cols"))):
                desired=t.get("counts",(None,None))[axis]
                if desired is None:
                    desired=int(rng.choice([1,2,3],p=[.57,.29,.14]))
                rad=t["mask"].shape[axis]//2
                choices=[v for v in range(1+rad,length-1-rad) if v not in used[axis]]
                if rng.random()<.5:
                    choices.sort(key=lambda v:abs(v-t[name][0]))
                else:
                    rng.shuffle(choices)
                for v in choices[:max(0,desired-len(t[name]))]:
                    t[name].append(v)
                    used[axis].add(v)
    if separated:
        claimed=set()
        for t in ts:
            if t["state"]!="A":
                continue
            bh,bw=t["mask"].shape
            cells={(r+int(y)-bh//2,c+int(x)-bw//2) for r in t["rows"] for c in t["cols"] for y,x in np.argwhere(t["mask"])}
            if claimed & cells:
                return False
            claimed.update(cells)
    return True


def _assemble(rng, h, sw, fw, ts, parameters, noise=None):
    """Ground truth uses construction records and reverse-order mask painting."""
    palette=list(map(int,rng.permutation(10)))
    bg=parameters["background"] if not parameters["vary_background"] else palette[0]
    palette.remove(bg)
    divider,frame=palette[:2]
    colors=palette[2:]
    inp=np.full((h,sw+1+fw),bg,dtype=np.int8)
    inp[:,sw]=divider
    inp[[0,-1],sw+1:]=frame
    inp[:,[sw+1,-1]]=frame
    for t,color in zip(ts,colors):
        t["color"]=color
        bh,bw=t["mask"].shape
        view=inp[t["top"]:t["top"]+bh,t["left"]:t["left"]+bw]
        assert np.all(view[t["mask"]]==bg)
        view[t["mask"]]=color
        for r in t["rows"]:
            inp[r,sw+1]=inp[r,-1]=color
        for c in t["cols"]:
            inp[0,sw+1+c]=inp[-1,sw+1+c]=color
    absent=[divider]+colors[len(ts):]
    rng.shuffle(absent)
    n=int(rng.choice([0,1,2,3],p=[.26,.30,.28,.16])) if noise is None else min(noise,len(absent))
    for color in absent[:n]:
        typ=int(rng.integers(5))
        for axis in (0,1):
            if typ==1 and axis==1 or typ==2 and axis==0:
                continue
            free=[r for r in range(1,h-1) if inp[r,sw+1]==inp[r,-1]==frame] if axis==0 else [c for c in range(sw+2,inp.shape[1]-1) if inp[0,c]==inp[-1,c]==frame]
            if not free:
                continue
            pos=int(rng.choice(free))
            if axis==0:
                inp[pos,sw+1]=color
                if typ!=3:
                    inp[pos,-1]=color
                elif len(free)>1:
                    inp[int(rng.choice([v for v in free if v!=pos])),-1]=color
            else:
                inp[0,pos]=color
                if typ!=3:
                    inp[-1,pos]=color
                elif len(free)>1:
                    inp[-1,int(rng.choice([v for v in free if v!=pos]))]=color
        if typ==4:
            r=int(rng.choice([0,h-1]))
            c=int(rng.choice([sw+1,inp.shape[1]-1]))
            # At most one corner anomaly; the regular frame stays identifiable.
            inp[0,sw+1]=inp[0,-1]=inp[-1,sw+1]=inp[-1,-1]=frame
            inp[r,c]=color
    out=inp.copy()
    for t in ts:
        if t["state"]=="A":
            bh,bw=t["mask"].shape
            view=out[t["top"]:t["top"]+bh,t["left"]:t["left"]+bw]
            view[t["mask"]]=bg
    for t in reversed(ts):
        if t["state"]!="A":
            continue
        bh,bw=t["mask"].shape
        for r in t["rows"]:
            for c in t["cols"]:
                view=out[r-bh//2:r+bh//2+1,sw+1+c-bw//2:sw+1+c+bw//2+1]
                view[t["mask"]]=t["color"]
    return {"input":inp,"output":out}


def _make_pair(rng, parameters, regime, kind="random", bank=()):
    assert regime in ("source_train","target_test")
    for attempt in range(100):
        h,sw,fw=_dimensions(rng,parameters,regime,kind!="random")
        if kind=="priority":
            a,b,dc=_priority_templates(rng)
            sw=max(sw,a.shape[1])
            fw=min(fw,29-sw)
            top=int(rng.integers(max(1,h-a.shape[0]-b.shape[0])))
            first=_new_template(a,top,int(rng.integers(sw-a.shape[1]+1)))
            second=_new_template(b,top+a.shape[0]+int(rng.integers(2)),int(rng.integers(sw-b.shape[1]+1)))
            ts=[first,second]
            r=int(rng.integers(1+a.shape[0]//2,h-2-a.shape[0]//2))
            c=int(rng.integers(2+a.shape[1]//2,fw-2-a.shape[1]//2))
            first["rows"],first["cols"]=[r+1],[c+dc]
            second["rows"],second["cols"]=[r],[c]
            first["single"]=second["single"]=True
        elif kind=="coverage":
            # Source interleaving is constructed from unoccupied mask cells.
            # Its shape and placement are visible; no bounding-box erasure works.
            a=np.ones((5,5),bool)
            a[1:-1,1:-1]=False
            if rng.random()<.5:
                a=np.eye(5,dtype=bool)
                if rng.random()<.5:
                    a=a[:,::-1].copy()
                line_row=4
                line_col=0 if a[4,4] else 2
            else:
                line_row=int(rng.integers(1,4))
                line_col=1
            top=int(rng.integers(0,max(1,h-13)))
            left=int(rng.integers(sw-4))
            first=_new_template(a,top,left)
            states=list(rng.permutation(["N","R","C"]))
            nested=_new_template(np.ones((1,3),bool),top+line_row,left+line_col,states[0])
            small=_sample_mask(rng,3,3)
            # Demonstrate a hollow center even when the first mask is a diagonal.
            small=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=bool) if a[2,2] else small
            second=_new_template(small,top+5+int(rng.integers(2)),int(rng.integers(sw-2)))
            r=second["top"]+3
            tail=[]
            for state in states[1:]:
                m=_sample_mask(rng,1,int(rng.choice([1,3])))
                tail.append(_new_template(m,r,int(rng.integers(sw-m.shape[1]+1)),str(state)))
                r+=1
            ts=sorted([first,nested,second]+tail,key=lambda t:t["top"])
            # The later template repeats on both axes; the earlier one uses an
            # independently chosen unequal product, breaking the position cue.
            first["counts"]=tuple([(2,1),(1,2),(3,1),(1,3)][int(rng.integers(4))])
            second["counts"]=(2,int(rng.choice([2,3])))
        else:
            max_count=3 if parameters["simple"] else 6
            if parameters["shift"]=="extrapolate" and regime=="source_train":
                max_count=min(max_count,3)
            count=int(rng.choice(np.arange(1,max_count+1),p=np.array([2,4,5,4,2,1][:max_count])/sum([2,4,5,4,2,1][:max_count])))
            # Enough coordinates for independent eligibility of all colors.
            count=min(count,h-4,fw-4)
            max_size=min(7,h-3,fw-3)
            if parameters["shift"]=="extrapolate" and regime=="source_train":
                max_size=min(5,max_size)
            ts=_place_templates(rng,h,sw,count,max_size,bank)
            all_active=rng.random()<.23
            for t in ts:
                t["state"]="A" if all_active else str(rng.choice(["A","N","R","C"],p=[.60,.16,.12,.12]))
            if not any(t["state"]=="A" for t in ts):
                ts[int(rng.integers(len(ts)))]["state"]="A"
        # Coordinate failures resample geometry, never silently change eligibility.
        if not _assign_coordinates(rng,ts,h,fw,separated=parameters["simple"],repeat=not parameters["single"]):
            continue
        pair=_assemble(rng,h,sw,fw,ts,parameters)
        try:
            p=_parse(pair["input"])
        except AssertionError:
            continue
        assert np.array_equal(_predict(pair["input"],p),pair["output"])
        if kind=="priority":
            f=_features(pair["input"])
            if not (f["destination_difference"] and f["size_difference"] and f["hole_exposure"]):
                continue
        if kind=="coverage":
            f=_features(pair["input"])
            if not f["erase_box_difference"] or not any(a>=2 and b>=2 for a,b in f["multiplicities"]):
                continue
        return pair
    # Guaranteed, color-parameterized constructions after bounded random search.
    if kind=="priority":
        a=np.ones((5,5),bool); a[1:-1,1:-1]=False
        ts=[_new_template(a,0,0),_new_template(np.ones((3,3),bool),7,0)]
        ts[0]["rows"],ts[0]["cols"]=[7],[7]
        ts[1]["rows"],ts[1]["cols"]=[6],[6]
        return _assemble(rng,17,5,16,ts,parameters,noise=0)
    if kind=="coverage":
        a=np.ones((5,5),bool); a[1:-1,1:-1]=False
        ts=[_new_template(a,0,0),_new_template(np.ones((1,3),bool),2,1,"N"),
            _new_template(np.array([[0,1,0],[1,0,1],[0,1,0]],bool),6,0),
            _new_template(np.ones((1,3),bool),10,0,"R"),_new_template(np.ones((1,3),bool),12,0,"C")]
        ts[0]["rows"],ts[0]["cols"]=[4,12],[4]
        ts[2]["rows"],ts[2]["cols"]=[3,11],[9,13]
        ts[3]["rows"],ts[4]["cols"]=[7],[7]
        return _assemble(rng,18,5,17,ts,parameters,noise=0)
    mask=min(bank,key=lambda m:m.size).copy() if bank else _sample_mask(rng,3,3)
    bh,bw=mask.shape
    t=_new_template(mask,int(rng.integers(11-bh)),0)
    t["rows"]=[int(rng.integers(1+bh//2,9-bh//2))]
    t["cols"]=[int(rng.integers(1+bw//2,9-bw//2))]
    return _assemble(rng,10,max(3,bw),10,[t],parameters,noise=0)


def _coverage(train, test):
    """Machine-checkable evidence for every test-side convention."""
    if any(f["conflicts"] for f in test):
        for feature in ("destination_difference","size_difference","reverse_difference","hole_exposure"):
            assert any(f[feature] for f in train), f"missing training witness: {feature}"
    for state in "NRC":
        if any(state in f["states"] for f in test):
            assert any(state in f["states"] for f in train)
    for axis in (0,1):
        if any(any(m[axis]>1 for m in f["multiplicities"]) for f in test):
            assert any(any(m[axis]>1 for m in f["multiplicities"]) for f in train)
    if any(any(a>1 and b>1 for a,b in f["multiplicities"]) for f in test):
        assert any(any(a>1 and b>1 for a,b in f["multiplicities"]) for f in train)
    for feature in ("erase_box_difference","empty_centers"):
        if any(f[feature] for f in test):
            assert any(f[feature] for f in train)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed=_count_value(seed,"seed")
    if seed is None:
        raise ValueError("seed must be an integer")
    seed=seed % (1<<64)
    nt,ne=_resolve_counts(seed,num_train,num_test)
    rng=np.random.default_rng(seed)
    simple=bool(rng.random()<.18)
    parameters=dict(simple=simple,single=simple and rng.random()<.45,
                    shift=str(rng.choice(["extrapolate","overlap","recombine"])),
                    background=int(rng.integers(10)),vary_background=bool(rng.random()<.55))
    # Tests are sampled independently of their answers; feature needs determine
    # which training evidence must be supplied. No hidden semantic choice varies.
    tests=[_make_pair(rng,parameters,"target_test") for _ in range(ne)]
    test_features=[_features(p["input"]) for p in tests]
    train=[]
    need_priority=any(f["conflicts"] for f in test_features)
    train.append(_make_pair(rng,parameters,"source_train","priority" if need_priority else "random"))
    # A compact, variable coverage scene teaches all three reasons for inactivity,
    # Cartesian replication, transparent centers, and selective source erasure.
    # Single-placement puzzles instead receive a one-placement coverage variant.
    coverage_parameters=dict(parameters,single=False)
    train.append(_make_pair(rng,coverage_parameters,"source_train","coverage"))
    for _ in range(nt-2):
        train.append(_make_pair(rng,parameters,"source_train"))
    if parameters["shift"]=="recombine":
        bank=[t["mask"] for pair in train for t in _parse(pair["input"])["templates"]]
        tests=[_make_pair(rng,parameters,"target_test",bank=bank) for _ in range(ne)]
        if any(_features(p["input"])["conflicts"] for p in tests) and not need_priority:
            train[0]=_make_pair(rng,parameters,"source_train","priority")
            # Keep the familiar vocabulary genuinely present in training.
            bank=[t["mask"] for pair in train for t in _parse(pair["input"])["templates"]]
            tests=[_make_pair(rng,parameters,"target_test",bank=bank) for _ in range(ne)]
    # Duplicates and source copies are repaired by a bijective recoloring. This
    # bounded fallback retains geometry and all visible equality relationships.
    seen=set()
    source=_source_keys()
    for pair in train+tests:
        originals={role:grid.copy() for role,grid in pair.items()}
        p=_parse(pair["input"])
        for offset in range(20):
            color_map=np.arange(10,dtype=np.int8)
            if offset>=10:
                color_map[p["bg"]],color_map[p["divider"]]=p["divider"],p["bg"]
            color_map=(color_map+offset%10)%10
            for role in ("input","output"):
                pair[role]=color_map[originals[role]]
            key=_key(pair["input"])
            pk=(key,_key(pair["output"]))
            if key not in seen and pk not in source:
                break
        else:
            # Twenty distinct recolorings exceed thirteen prior inputs plus
            # five source pairs, so this branch is mathematically unreachable.
            raise AssertionError("exhausted distinct bijective recolorings")
        seen.add(_key(pair["input"]))
    puzzle={"train":train,"test":tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle,dict) and set(puzzle)=={"train","test"}
        assert all(isinstance(puzzle[s],list) for s in ("train","test"))
        assert 2<=len(puzzle["train"])<=10 and 1<=len(puzzle["test"])<=4
        assert len(puzzle["train"])>len(puzzle["test"])
        seen=set(); infos={"train":[],"test":[]}
        source=_source_keys()
        for split in ("train","test"):
            for pair in puzzle[split]:
                assert isinstance(pair,dict) and set(pair)=={"input","output"}
                for grid in pair.values():
                    assert isinstance(grid,np.ndarray) and grid.ndim==2
                    assert np.issubdtype(grid.dtype,np.integer)
                    assert all(1<=s<=30 for s in grid.shape)
                    assert np.all((grid>=0)&(grid<=9))
                inp,out=pair["input"],pair["output"]
                assert inp.shape==out.shape and not np.array_equal(inp,out)
                assert _key(inp) not in seen
                seen.add(_key(inp))
                assert (_key(inp),_key(out)) not in source
                assert np.array_equal(_predict(inp),out), "wrong mask placement or preservation"
                infos[split].append(_features(inp))
        _coverage(infos["train"],infos["test"])
        for pair in puzzle["test"]:
            assert np.array_equal(solve(puzzle["train"],pair["input"].copy()),pair["output"])
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def _decode(rows):
    return np.array([[int(c) for c in row] for row in rows.split()],dtype=np.int8)


def _source_keys():
    return {(_key(_decode(a)),_key(_decode(b))) for a,b in _SOURCE_CASES}


def _puzzle_key(p):
    return tuple(tuple((_key(x["input"]),_key(x["output"])) for x in p[s]) for s in ("train","test"))


def self_test() -> None:
    # Expected source outputs are literal staged evidence, not parser products.
    for inp,out in _SOURCE_CASES:
        a,b=_decode(inp),_decode(out)
        assert np.array_equal(_predict(a),b)
        for shift in range(10):
            assert np.array_equal(_predict((a+shift)%10),(b+shift)%10)
    # Source training alone does not distinguish every alternative ordering;
    # the fixed source reading-order rule still predicts both source tests.
    for inp,out in _SOURCE_CASES[3:]:
        assert np.array_equal(_predict(_decode(inp)),_decode(out))
    p=generate_puzzle(912345,6,3)
    assert _puzzle_key(p)==_puzzle_key(generate_puzzle(912345,6,3))
    assert _puzzle_key(generate_puzzle(101,5,2))!=_puzzle_key(generate_puzzle(102,5,2))
    for seed,(a,b) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p=generate_puzzle(seed+500,a,b)
        if a is not None:
            assert len(p["train"])==a
        if b is not None:
            assert len(p["test"])==b
    for a,b in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1),(2,1.5)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    fs=[]; counts=[set(),set()]; shapes=set(); motifs=set(); inputs=set(); varied=0
    rejected_shortcuts=set()
    within_background=0
    for seed in range(42):
        p=generate_puzzle(seed)
        counts[0].add(len(p["train"])); counts[1].add(len(p["test"]))
        local=[]
        for split in ("train","test"):
            for pair in p[split]:
                f=_features(pair["input"])
                fs.append(f); local.append(f)
                shapes.add(pair["input"].shape); motifs.update(f["motifs"]); inputs.add(_key(pair["input"]))
                for shortcut,feature in (("destination","destination_difference"),
                                         ("size","size_difference"),
                                         ("erase","erase_box_difference")):
                    if shortcut in rejected_shortcuts or not f[feature]:
                        continue
                    wrong=_predict(pair["input"],order=shortcut if shortcut!="erase" else "source",
                                   erase_boxes=shortcut=="erase")
                    bad={s:[{k:v.copy() for k,v in x.items()} for x in p[s]] for s in ("train","test")}
                    index=next(i for i,x in enumerate(p[split]) if x is pair)
                    bad[split][index]["output"]=wrong
                    try:
                        validate_generated_puzzle(bad)
                    except AssertionError:
                        rejected_shortcuts.add(shortcut)
                    else:
                        raise AssertionError("validator accepts a reviewed shortcut")
        varied+=len({f["count"] for f in local})>1 and len({f["roles"] for f in local})>1
        within_background+=len({f["roles"][0] for f in local})>1
    assert len(counts[0])>=4 and len(counts[1])>=3
    assert len(shapes)>=70 and len(motifs)>=100 and len(inputs)>=200
    assert varied>=35
    assert within_background>=10 and rejected_shortcuts=={"destination","size","erase"}
    for feature in ("source_overlap","erase_box_difference","same_color_merge","hole_exposure",
                    "destination_difference","size_difference","empty_centers","solid","sparse",
                    "dense","diagonal","asymmetric","irregular","corner_noise","divider_label","black_template","touch"):
        assert sum(bool(f[feature]) for f in fs)>=5, feature
    mult={m for f in fs for m in f["multiplicities"]}
    assert {(1,1),(2,1),(1,2),(2,2),(3,1),(1,3),(2,3),(3,2)}<=mult
    assert {1,2,3,4,5,6}<={f["count"] for f in fs}
    assert {0,1,2,3}<={f["absent"] for f in fs}
    assert {0,1,2,3}<={f["roles"][0] for f in fs}
    assert any(f["height"]==30 for f in fs) and any(f["width"]==30 for f in fs)
    assert any(f["height"]<10 for f in fs)
    assert any(f["height"]>f["width"] for f in fs)
    assert any(f["height"]*2<f["width"] for f in fs)
    assert any(f["active"]==f["count"] for f in fs)
    assert any(f["states"].startswith("N") and "A" in f["states"] for f in fs)
    assert any("ANA" in f["states"] for f in fs)
    assert any(f["conflicts"]==0 and f["active"]>1 for f in fs)
    assert any(sum(a*b>1 for a,b in f["multiplicities"])>=2 for f in fs)
    assert any(f["source_overlap"] and f["irregular"] and f["same_color_merge"] and f["roles"][0]!=0 for f in fs)
    assert len({m for f in fs for m in f["active_motifs"] if m[0]==(3,3)})>30
    for shape in ((1,3),(3,1),(3,5),(5,3),(5,5)):
        assert any(shape in f["active_shapes"] for f in fs)
        assert any(shape in f["inactive_shapes"] for f in fs)
    assert {(1,1),(1,-1),(-1,1),(-1,-1)}=={d for f in fs for d in f["directions"]}
    assert len({gap for f in fs for gap in f["spacing"]})>=10
    for n in (2,3,4):
        assert len({f["active"] for f in fs if f["count"]==n})>=2


# Five solved source regressions, in train-then-test order.
_SOURCE_CASES = (
    ('040233633473333333 444230000000000003 040230000000000003 000230000000000003 000240000000000004 000230000000000003 000270000000000007 000230000000000003 777230000000000003 777260000000000006 777230000000000003 000233633473333333',
     '000233633473333333 000230000000000003 000230000000000003 000230000400000003 000240004440000004 000230000477000003 000270000777000007 000230000777000003 000230000000000003 000260000000000006 000230000000000003 000233633473333333'),
    ('050034474547444 505034000000004 050037000000007 800034000000004 080034000000004 008035000000005 070034000000004 777037000000007 070034000000004 000034474547444',
     '000034474547444 000034070007004 000037777077707 800034070007004 080034000500004 008035005050005 000034070507004 000037777077707 000034070007004 000034474547444'),
    ('03012223222622 33312000000002 03012000000002 00013000000003 06012000000002 60612000000002 06016000000006 00012000000002 77712000000002 00012223222622',
     '00012223222622 00012000000002 00012003000002 00013033300003 00012003000002 00012000000602 00016000006066 00012000000602 77712000000002 00012223222622'),
    ('01003991993999469989 11103900000000000009 01003900000000000009 00003100000000000001 40003900000000000009 04003900000000000009 00403300000000000003 00003400000000000004 66603400000000000004 66603600000000000006 60603900000000000009 00003900000000000009 07073100000000000001 00703900000000000009 07073981993999469999',
     '00003991993999469989 00003900000000000009 00003901000000000009 00003111100000000001 00003901000000000009 00003900000000000009 00003300000004000003 00003400000004400004 00003400000000446004 00003600000000646006 00003900000000606009 00003901000000000009 07073111100000000001 00703901000000000009 07073981993999469999'),
    ('600002917173146111 060002100000000001 006002700000000007 000602100000000001 888062300000000003 000002400000000004 004402100000000001 004002100000000001 044002600000000006 007002100000000001 077702100000000001 007002117173146111',
     '000002917173146111 000002107070000001 000002777777000007 000002107070000001 888002300000044003 000002400000040004 000002100000640001 000002100000060001 000002600000006006 000002100000000601 000002100000000061 000002117173146111'),
)
