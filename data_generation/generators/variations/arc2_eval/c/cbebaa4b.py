"""Translate spatially separated rigid pieces into one anchored assembly.

The source has five-piece training chains and seven-piece test assemblies,
including a branch and repeated movable colors. Connector cells are geometric
constraints, not necessarily connected blobs or signatures with unique areas.
The sampler makes trees; the solver also considers cyclic alternative placements
when proving that all connectors have exactly two owners and the output is unique.
"""
import hashlib

import numpy as np


_SOURCE_PAIR_DIGESTS = {
    (
        "22333b5ff1083ec8d405f66856641be6216a82a8d11a4590de4a4afc80b2e78a",
        "889f49d000151765bbd17554568d36566946d2b71e3c24f043aca696dc2bbe56",
    ),
    (
        "4396ee62430a257889aaf0b8970dc6fa12591c833d00f81f43096fc1b8d5f869",
        "b53f24d1e6167aac63e34f18bb4bebad09860d7d3680c03775992a6c7433be4c",
    ),
    (
        "807da84892d3bb657c4ead605eabc41f724f7b6267a736d24a4d35b353e8e3ea",
        "6d4c40f63571d1f6f33b18a69124d6e7a36a00395df3901fb7fb4fda12cb51ca",
    ),
    (
        "3cef8d70c60a99cadd6d9125c7977bda93a99bcf3ae7920aebe0c7a44948acc9",
        "477837fc96ebefbe5f7f3a55e4678d75d3de5de36ec00604f5a4d16db6e7cc10",
    ),
}

def _is_plain_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        if not _is_plain_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_plain_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    code = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + code % 4
        low = num_test + 1
        num_train = low + (code // 4) % (11 - low)
    elif num_train is None:
        low = num_test + 1
        num_train = low + code % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + code % high

    if not (2 <= num_train <= 10 and 1 <= num_test <= 4):
        raise ValueError("pair counts are outside their legal ranges")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test



N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack = [start]
        found = {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def shifted(cells, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in cells}


def bbox(cells):
    return (min(r for r,c in cells), min(c for r,c in cells),
            max(r for r,c in cells), max(c for r,c in cells))


def analyze(grid):
    bg = int(np.bincount(grid.ravel(), minlength=10).argmax())
    cc = components(map(tuple, np.argwhere(grid != bg).tolist()))
    assert len(cc) >= 2
    palettes = [{int(grid[q]) for q in cells} for cells in cc]
    common = set.intersection(*palettes)
    assert len(common) == 1 and all(len(p) == 2 for p in palettes)
    marker = common.pop()
    pieces = []
    for cells, colors in zip(cc, palettes):
        color = (colors - {marker}).pop()
        payload = {q for q in cells if grid[q] == color}
        pins = cells - payload
        assert len(components(payload)) == 1
        pieces.append(dict(cells=cells, payload=payload, pins=pins,
                           color=color, box=bbox(cells)))
    return bg, marker, pieces


class SearchLimit(Exception):
    pass


def reconstruct(grid, anchor_color, limit=40000):
    """Exact search on individual marker cells; no preassigned joint groups."""
    bg, marker, pieces = analyze(grid)
    n = len(pieces)
    anchors = [i for i,p in enumerate(pieces) if p['color'] == anchor_color]
    assert len(anchors) == 1
    anchor = anchors[0]
    h,w = grid.shape
    # All possible relative translations sharing at least one marker. Reject
    # payload intersections before searching, but do not assume joint areas.
    links = {}
    for i in range(n):
        a = pieces[i]
        for j in range(n):
            if i == j:
                continue
            b = pieces[j]
            candidates = {}
            offsets = sorted({(r-s,c-t) for r,c in a['pins'] for s,t in b['pins']})
            for d in offsets:
                bp = shifted(b['payload'],d)
                bm = shifted(b['pins'],d)
                if a['cells'] & bp or a['payload'] & bm:
                    continue
                overlap = a['pins'] & bm
                for q in sorted(overlap):
                    candidates.setdefault(q,[]).append((j,d))
            links[i,j] = candidates
    solutions = {}
    visited = set()
    nodes = 0

    def visit(positions, bodies, singles, doubles, owners):
        nonlocal nodes
        state = tuple(sorted(positions.items()))
        if state in visited:
            return
        visited.add(state)
        nodes += 1
        if nodes > limit:
            raise SearchLimit('uniqueness search exhausted its node budget')
        if not singles:
            if len(positions) != n:
                return
            out = np.full(grid.shape,bg,dtype=np.int8)
            for i,d in positions.items():
                for r,c in shifted(pieces[i]['payload'],d):
                    out[r,c] = pieces[i]['color']
            for q in doubles:
                out[q] = marker
            solutions.setdefault(out.tobytes(),(out,dict(positions)))
            return
        if len(positions) == n:
            return
        best = None
        # Every unmatched cell must eventually overlap an unplaced piece.
        # MRV is also a sound forward check on every outstanding marker.
        for q in sorted(singles):
            i = owners[q]
            di = positions[i]
            local = q[0]-di[0], q[1]-di[1]
            choices = []
            for j in range(n):
                if j in positions:
                    continue
                b = pieces[j]
                for _,relative in links[i,j].get(local,[]):
                    d = di[0]+relative[0],di[1]+relative[1]
                    t,l,bottom,right = b['box']
                    if t+d[0]<0 or l+d[1]<0 or bottom+d[0]>=h or right+d[1]>=w:
                        continue
                    bp = shifted(b['payload'],d)
                    bm = shifted(b['pins'],d)
                    if bp & (bodies|singles|doubles) or bm & (bodies|doubles):
                        continue
                    overlap = bm & singles
                    # Do not assume a chain, a signature partition, or even
                    # acyclicity while ruling out alternative assemblies.
                    choices.append((j,d,bp,bm,overlap))
            if not choices:
                return
            if best is None or len(choices)<len(best):
                best=choices
                if len(best)==1:
                    break
        for j,d,bp,bm,overlap in best:
            new_owners = dict(owners)
            new_owners.update({q:j for q in bm-overlap})
            positions[j]=d
            visit(positions,bodies|bp,(singles|bm)-overlap,doubles|overlap,new_owners)
            del positions[j]
            if len(solutions)>1:
                return

    a=pieces[anchor]
    visit({anchor:(0,0)},a['payload'],a['pins'],set(),{q:anchor for q in a['pins']})
    return list(solutions.values()), nodes


def learn_anchor(train):
    possible = None
    for pair in train:
        inp,out = pair['input'],pair['output']
        _,_,pieces = analyze(inp)
        colors = [p['color'] for p in pieces]
        fixed = {p['color'] for p in pieces if colors.count(p['color']) == 1
                 and all(out[q] == p['color'] for q in p['payload'])
                 and np.count_nonzero(out == p['color']) == len(p['payload'])}
        possible = fixed if possible is None else possible & fixed
    assert len(possible)==1, ('anchor not demonstrated',possible)
    return possible.pop()


def solve(train, query):
    answers,_=reconstruct(query,learn_anchor(train))
    assert len(answers)==1, ('ambiguous or impossible',len(answers))
    return answers[0][0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _rotate(q, k):
    r,c = q
    for _ in range(k % 4):
        r,c = c,-r
    return r,c


def _rectangle(t,l,h,w):
    return {(r,c) for r in range(t,t+h) for c in range(l,l+w)}


def _body_size(rng, scale):
    if rng.random()<0.18:
        a,b = int(rng.integers(1,3)),int(rng.integers(3,8))
        return (a,b) if rng.random()<0.5 else (b,a)
    return int(rng.integers(2,scale+1)), int(rng.integers(2,scale+1))


def _layout(rng, count, branched, scale, complexity):
    """Grow rectangles and interlocking boundary arms in unrestricted coordinates."""
    for restart in range(48):
        h,w = _body_size(rng,scale)
        nodes = [dict(rect=_rectangle(0,0,h,w), arms=set(), pins=set(), required=set(), sides=set())]
        edges=[]
        for j in range(1,count):
            added=False
            for attempt in range(160):
                parent = int(rng.integers(j)) if branched else j-1
                p=nodes[parent]
                k=int(rng.integers(4))
                if k in p['sides']:
                    continue
                rect={_rotate(q,k) for q in p['rect']}
                t,l,b,r=bbox(rect)
                h,w=_body_size(rng,scale)
                top=int(rng.integers(t-h+1,b+1))
                gap=int(rng.integers(1,4))
                left=r+gap+1
                overlap=list(range(max(t,top),min(b,top+h-1)+1))
                if not overlap:
                    continue
                probs = [0.02,0.93,0.05] if complexity else [0.03,0.95,0.02]
                size=int(rng.choice([1,2,3],p=probs))
                if size>len(overlap):
                    continue
                rows=sorted(int(x) for x in rng.choice(overlap,size,replace=False))
                # Sparse keys predominate. Adjacent lines remain available as
                # a simple subfamily, with no unique-cardinality convention.
                offsets=[int(rng.integers(gap)) if complexity or rng.random()<0.5 else 0 for _ in rows]
                pins={(a,r+1+o) for a,o in zip(rows,offsets)}
                if size>1 and len(components(pins))==1 and rng.random()<(0.90 if complexity else 0.65):
                    continue
                pa={(a,c) for a,o in zip(rows,offsets) for c in range(r+1,r+1+o)}
                ca={(a,c) for a,o in zip(rows,offsets) for c in range(r+2+o,left)}
                pr={(a,r) for a in rows}
                cr={(a,left) for a in rows}
                childrect=_rectangle(top,left,h,w)
                pins,pa,ca,pr,cr,childrect=[{_rotate(q,-k) for q in cells}
                                           for cells in (pins,pa,ca,pr,cr,childrect)]
                if pins & (p['rect']|p['arms']|p['pins']) or pa & p['pins']:
                    continue
                childall=childrect|ca|pins
                if childall & (p['rect']|p['arms']|p['pins']|pa):
                    continue
                if any((childall|pa|pins)&(z['rect']|z['arms']|z['pins'])
                       for i,z in enumerate(nodes) if i!=parent):
                    continue
                allcells=set().union(*(z['rect']|z['arms']|z['pins'] for z in nodes))|childall|pa
                bt,bl,bb,br=bbox(allcells)
                if bb-bt+1>28 or br-bl+1>28:
                    continue
                p['arms']|=pa
                p['pins']|=pins
                p['required']|=pr
                p['sides'].add(k)
                nodes.append(dict(rect=childrect,arms=ca,pins=set(pins),required=cr,sides={(k+2)%4}))
                edges.append((parent,j,pins))
                added=True
                break
            if not added:
                break
        if len(nodes)==count:
            degrees=[sum(i in (a,b) for a,b,_ in edges) for i in range(count)]
            if not branched or max(degrees)>=3:
                return nodes,edges
    return None


def _make_body(rng, node):
    t,l,b,r=bbox(node['rect'])
    h,w=b-t+1,r-l+1
    kind=int(rng.integers(6))
    thickness=int(rng.integers(1,min(2,(min(h,w)+1)//2)+1))
    if kind==0 or min(h,w)==1:
        body=set(node['rect'])
    elif kind==1:
        body={q for q in node['rect'] if min(q[0]-t,b-q[0],q[1]-l,r-q[1])<thickness}
    elif kind in (2,3):
        side=int(rng.integers(4))
        body={q for q in node['rect'] if any((q[0]-t,b-q[0],q[1]-l,r-q[1])[i]<thickness
                                           for i in range(4) if i!=side)}
    else:
        start=(int(rng.integers(t,b+1)),int(rng.integers(l,r+1)))
        body={start}
        target=int(rng.integers(max(2,(h*w)//4),max(3,(h*w*4)//5+1)))
        while len(body)<min(target,h*w):
            frontier=sorted({(a+dr,c+dc) for a,c in body for dr,dc in N4}&node['rect']-body)
            if not frontier:
                break
            body.add(_pick(rng,frontier))
    # Connect actual attachment sites to the sampled skeleton. This preserves
    # open frames and holes without turning missing cells into a repair task.
    targets=sorted(node['required'])
    rng.shuffle(targets)
    for target in targets:
        start=min(body,key=lambda q:(abs(q[0]-target[0])+abs(q[1]-target[1]),q))
        a,c=start
        axes=[0,1] if rng.random()<0.5 else [1,0]
        for axis in axes:
            if axis==0:
                while a!=target[0]:
                    a+=1 if target[0]>a else -1
                    body.add((a,c))
            else:
                while c!=target[1]:
                    c+=1 if target[1]>c else -1
                    body.add((a,c))
    body |= node['arms']
    assert len(components(body))==1
    assert not body & node['pins']
    return body


def _scatter(rng, shape, pieces, anchor, bg, marker):
    h,w=shape
    for restart in range(24):
        occupied=pieces[anchor]['payload']|pieces[anchor]['pins']
        forbidden={(r+dr,c+dc) for r,c in occupied for dr in (-1,0,1) for dc in (-1,0,1)}
        placements={anchor:(0,0)}
        order=[i for i in range(len(pieces)) if i!=anchor]
        rng.shuffle(order)
        order.sort(key=lambda i: -len(pieces[i]['payload']|pieces[i]['pins']))
        for i in order:
            cells=pieces[i]['payload']|pieces[i]['pins']
            t,l,b,r=bbox(cells)
            for attempt in range(500):
                d=(int(rng.integers(0,h-(b-t))),int(rng.integers(0,w-(r-l))))
                d=d[0]-t,d[1]-l
                if d==(0,0):
                    continue
                moved=shifted(cells,d)
                if moved & forbidden:
                    continue
                placements[i]=d
                forbidden|={(r+dr,c+dc) for r,c in moved for dr in (-1,0,1) for dc in (-1,0,1)}
                break
            else:
                break
        if len(placements)==len(pieces):
            inp=np.full(shape,bg,dtype=np.int8)
            for i,d in placements.items():
                for q in shifted(pieces[i]['payload'],d):
                    inp[q]=pieces[i]['color']
                for q in shifted(pieces[i]['pins'],d):
                    inp[q]=marker
            return inp
    return None


def _fallback_layout(rng,count):
    # Finite constructive backup: a narrow stack of bars, with distinct
    # geometric two-cell spacings and equal areas. It fits even eight pieces.
    spacings=[int(x) for x in rng.permutation(np.arange(1,count))]
    nodes=[dict(rect=_rectangle(2*i,0,1,count+i%2),arms=set(),pins=set(),required=set(),sides=set()) for i in range(count)]
    edges=[]
    for i,s in enumerate(spacings):
        pins={(2*i+1,0),(2*i+1,s)}
        nodes[i]['pins']|=pins
        nodes[i+1]['pins']|=pins
        edges.append((i,i+1,pins))
    return nodes,edges


def _generate_pair(rng, count, params, index, regime):
    anchor_color=params['anchor']
    bg=params['backgrounds'][index % len(params['backgrounds'])]
    marker=params['markers'][index % len(params['markers'])]
    if marker==bg:
        marker=next(c for c in range(10) if c not in (bg,anchor_color))
    available=[c for c in range(10) if c not in (bg,marker,anchor_color)]
    difficulty=params['difficulty']
    extended=regime=='target_test'
    branch_prob=(0.18,0.50,0.70)[difficulty]
    if params['regime']=='branch' and extended:
        branch_prob=0.90
    branched=count>=4 and rng.random()<branch_prob
    if params['regime']=='branch' and index==1:
        branched=True
    scale=5 if difficulty==0 else 7
    source_scale=params['regime']=='source_scale'
    if source_scale:
        scale=6 if extended else 5
    complexity=difficulty>0 or (extended and params['regime']=='markers')
    for attempt in range(96):
        layout=_layout(rng,count,branched,scale,complexity) if attempt<92 else (_fallback_layout(rng,count))
        if layout is None:
            continue
        nodes,edges=layout
        anchor=int(rng.integers(count))
        colors_count=int(rng.integers(1,min(len(available),count-1)+1))
        if difficulty==0 and rng.random()<0.65:
            colors_count=min(count-1,len(available))
        palette=[int(c) for c in rng.choice(available,colors_count,replace=False)]
        movable=palette+[_pick(rng,palette) for _ in range(count-1-colors_count)]
        rng.shuffle(movable)
        pieces=[]
        for i,node in enumerate(nodes):
            color=anchor_color if i==anchor else movable.pop()
            pieces.append(dict(payload=_make_body(rng,node),pins=node['pins'],color=color))
        cells=set().union(*(p['payload']|p['pins'] for p in pieces))
        t,l,b,r=bbox(cells)
        bh,bw=b-t+1,r-l+1
        if source_scale:
            canvas_size=26 if extended else 22
            if max(bh,bw)>canvas_size:
                continue
            if extended and max(bh,bw)<1.5*min(bh,bw):
                continue
        h=min(30,max(14,bh)+int(rng.integers(0,8)))
        w=min(30,max(14,bw)+int(rng.integers(0,8)))
        area=sum((bbox(p['payload']|p['pins'])[2]-bbox(p['payload']|p['pins'])[0]+2)*
                 (bbox(p['payload']|p['pins'])[3]-bbox(p['payload']|p['pins'])[1]+2) for p in pieces)
        while h*w<1.6*area and min(h,w)<30:
            if h<w and h<30:
                h+=1
            elif w<30:
                w+=1
            else:
                h+=1
        if source_scale:
            h=w=canvas_size
        for enlarge in range(1 if source_scale else 2):
            if enlarge:
                h,w=30,30
            offset=(int(rng.integers(h-bh+1))-t,int(rng.integers(w-bw+1))-l)
            placed=[dict(payload=shifted(p['payload'],offset),pins=shifted(p['pins'],offset),color=p['color']) for p in pieces]
            inp=_scatter(rng,(h,w),placed,anchor,bg,marker)
            if inp is not None:
                break
        if inp is None:
            continue
        out=np.full((h,w),bg,dtype=np.int8)
        for p in placed:
            for q in p['payload']:
                assert out[q]==bg
                out[q]=p['color']
        for p in placed:
            for q in p['pins']:
                assert out[q] in (bg,marker)
                out[q]=marker
        # Whole-scene symmetries preserve translation-only piece motion.
        k=int(rng.integers(4))
        if source_scale and extended:
            k=2*int(rng.integers(2))+(0 if bh>=bw else 1)
        inp,out=np.rot90(inp,k).copy(),np.rot90(out,k).copy()
        if rng.random()<0.5:
            inp,out=inp[:,::-1].copy(),out[:,::-1].copy()
        try:
            answers,_=reconstruct(inp,anchor_color,limit=6000)
        except SearchLimit:
            continue
        if len(answers)==1 and np.array_equal(answers[0][0],out):
            return dict(input=inp,output=out)
    raise RuntimeError('bounded construction could not produce a unique assembly')


def _grid_digest(grid):
    return hashlib.sha256(np.ascontiguousarray(grid,dtype=np.uint8).tobytes()).hexdigest()


def _sample_puzzle_parameters(rng):
    anchor=int(rng.integers(1,10))
    colors=[c for c in range(10) if c!=anchor]
    mode=int(rng.integers(3))
    if mode==0:
        backgrounds=[0]
    elif mode==1:
        backgrounds=[_pick(rng,colors)]
    else:
        backgrounds=[int(c) for c in rng.choice(colors,int(rng.integers(2,4)),replace=False)]
    remaining=[c for c in colors if c not in backgrounds]
    marker_count=int(rng.integers(2,4)) if rng.random()<0.35 else 1
    markers=[int(c) for c in rng.choice(remaining,marker_count,replace=False)]
    return dict(anchor=anchor,backgrounds=backgrounds,markers=markers,
                difficulty=int(rng.choice([0,1,2],p=[0.25,0.4,0.35])),
                regime=_pick(rng,['comparable','count','branch','markers','source_scale']))


def _placement_features(inp, out, anchor, positions):
    """Check complete rigid preservation and measure visible, reconstructed facts."""
    bg,marker,pieces=analyze(inp)
    assert inp.shape==out.shape
    assert not np.array_equal(inp,out)
    assert set(positions)==set(range(len(pieces)))
    payloads=[]
    pins=[]
    expected=np.full(inp.shape,bg,dtype=np.int8)
    multiplicity={}
    anchor_indices=[i for i,p in enumerate(pieces) if p['color']==anchor]
    assert len(anchor_indices)==1
    ai=anchor_indices[0]
    assert positions[ai]==(0,0)
    fixed=[]
    for i,p in enumerate(pieces):
        if positions[i]==(0,0):
            fixed.append(i)
        body=shifted(p['payload'],positions[i])
        markers=shifted(p['pins'],positions[i])
        for r,c in body|markers:
            assert 0<=r<inp.shape[0] and 0<=c<inp.shape[1]
        assert not set().union(*payloads) & body
        for q in body:
            assert expected[q]==bg
            expected[q]=p['color']
        for q in markers:
            multiplicity[q]=multiplicity.get(q,0)+1
        payloads.append(body)
        pins.append(markers)
    assert fixed==[ai]
    assert set(multiplicity.values())=={2}
    assert not set(multiplicity)&set().union(*payloads)
    for q in multiplicity:
        expected[q]=marker
    assert np.array_equal(out,expected)
    assert np.count_nonzero(inp==marker)==2*np.count_nonzero(out==marker)
    adjacency=[[] for _ in pieces]
    joints=[]
    displacements=[]
    for i in range(len(pieces)):
        for j in range(i):
            joint=pins[i]&pins[j]
            if joint:
                adjacency[i].append(j)
                adjacency[j].append(i)
                t,l,_,_=bbox(joint)
                joints.append(tuple(sorted((r-t,c-l) for r,c in joint)))
                a,b=bbox(payloads[i]),bbox(payloads[j])
                displacements.append((a[0]-b[0],a[1]-b[1]))
    assert len(joints)==len(pieces)-1
    seen={ai}
    todo=[ai]
    while todo:
        for j in adjacency[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    assert len(seen)==len(pieces)
    boxes=[]
    areas=[]
    holes=[]
    thicknesses=[]
    corner_patterns=[]
    normalized_bodies=[]
    for p in pieces:
        t,l,b,r=bbox(p['payload'])
        h,w=b-t+1,r-l+1
        boxes.append((h,w))
        areas.append(len(p['payload']))
        missing=_rectangle(t,l,h,w)-p['payload']
        holes.append(sum(not any(a in (t,b) or c in (l,r) for a,c in comp)
                         for comp in components(missing)))
        largest=1
        for side in range(2,min(h,w)+1):
            if any(_rectangle(a,c,side,side)<=p['payload']
                   for a in range(t,b-side+2) for c in range(l,r-side+2)):
                largest=side
            else:
                break
        thicknesses.append(largest)
        corner_patterns.append(tuple(int(q in p['payload']) for q in [(t,l),(t,r),(b,l),(b,r)]))
        normalized_bodies.append(tuple(sorted((a-t,c-l) for a,c in p['payload'])))
    t,l,b,r=bbox(set().union(*payloads,*pins))
    degrees=[len(x) for x in adjacency]
    sizes=[len(x) for x in joints]
    categories=[]
    for joint in joints:
        if len(joint)==1:
            categories.append('singleton')
        elif len(components(joint))==1:
            categories.append('connected')
        elif len({a for a,c in joint})==1:
            categories.append('horizontal')
        elif len({c for a,c in joint})==1:
            categories.append('vertical')
        elif len(joint)==2 and abs(joint[1][0]-joint[0][0])==abs(joint[1][1]-joint[0][1]):
            categories.append('diagonal')
        else:
            categories.append('offset')
    distances={ai:0}
    queue=[ai]
    while queue:
        i=queue.pop()
        for j in adjacency[i]:
            if j not in distances:
                distances[j]=distances[i]+1
                queue.append(j)
    arm_lengths=[]
    for i,neighbors in enumerate(adjacency):
        if len(neighbors)>=3:
            for j in neighbors:
                prev,cur,length=i,j,1
                while len(adjacency[cur])==2:
                    nxt=next(x for x in adjacency[cur] if x!=prev)
                    prev,cur=cur,nxt
                    length+=1
                arm_lengths.append(length)
    payload_colors=[p['color'] for p in pieces]
    collision=len(sizes)-len(set(sizes))
    unequal_keys=any(len(x)==len(y) and x!=y for i,x in enumerate(joints) for y in joints[:i])
    reuse=len(pieces)-len(set(payload_colors))
    return dict(piece_count=len(pieces),background=bg,connector=marker,anchor=anchor,
                palette=tuple(sorted(set(payload_colors))),reuse=reuse,
                body_boxes=boxes,body_areas=areas,body_shapes=normalized_bodies,
                occupancies=[round(a/(h*w),3) for a,(h,w) in zip(areas,boxes)],
                holes=holes,thicknesses=thicknesses,corner_patterns=corner_patterns,
                degrees=degrees,anchor_degree=degrees[ai],anchor_box=boxes[ai],
                anchor_position=bbox(pieces[ai]['payload'])[:2],
                branch_distances=[distances[i] for i,d in enumerate(degrees) if d>=3],
                arm_lengths=arm_lengths,joint_sizes=sizes,joint_keys=joints,
                marker_categories=categories,cardinality_collisions=collision,
                equal_area_different_keys=unequal_keys,displacements=displacements,
                assembly_box=(b-t+1,r-l+1),margins=(t,l,inp.shape[0]-b-1,inp.shape[1]-r-1),
                canvas=tuple(inp.shape),scatter=list(positions.values()),
                combined=(max(degrees)>=3 and reuse>0 and len(set(boxes))>1
                          and unequal_keys and any(c in categories for c in ['horizontal','vertical','diagonal','offset'])
                          and inp.shape[0]!=inp.shape[1]))


def puzzle_features(puzzle):
    """Analysis hook used by self-tests and scratch coverage reports, never generation metadata."""
    anchor=learn_anchor(puzzle['train'])
    records=[]
    for split in ['train','test']:
        for pair in puzzle[split]:
            answers,nodes=reconstruct(pair['input'],anchor)
            assert len(answers)==1, 'the visible input has no unique reconstruction'
            predicted,positions=answers[0]
            assert np.array_equal(predicted,pair['output'])
            record=_placement_features(pair['input'],pair['output'],anchor,positions)
            record.update(split=split,search_nodes=nodes)
            records.append(record)
    return records


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle,dict) and set(puzzle)=={'train','test'}
        assert all(isinstance(puzzle[s],list) for s in ['train','test'])
        assert 2<=len(puzzle['train'])<=10
        assert 1<=len(puzzle['test'])<=4 and len(puzzle['train'])>len(puzzle['test'])
        inputs=set()
        pairs=set()
        for split in ['train','test']:
            for pair in puzzle[split]:
                assert isinstance(pair,dict) and set(pair)=={'input','output'}
                for grid in pair.values():
                    assert isinstance(grid,np.ndarray) and grid.ndim==2
                    assert all(1<=v<=30 for v in grid.shape)
                    assert np.issubdtype(grid.dtype,np.integer)
                    assert np.all((grid>=0)&(grid<=9))
                inp,out=pair['input'],pair['output']
                ik=(inp.shape,_grid_digest(inp))
                pk=(ik,out.shape,_grid_digest(out))
                assert ik not in inputs and pk not in pairs
                assert (_grid_digest(inp),_grid_digest(out)) not in _SOURCE_PAIR_DIGESTS
                inputs.add(ik)
                pairs.add(pk)
        # Infer the sole learned convention from demonstrations before examining
        # any query output. Marker and background roles are visible per input.
        features=puzzle_features(puzzle)
        assert len({x['anchor'] for x in features})==1
        assert len({x['piece_count'] for x in features})>=2
        assert len({x['assembly_box'] for x in features})>=2
        assert len({x['palette'] for x in features})>=2
        assert len({x['canvas'] for x in features})>=2
        for x in features:
            assert 3<=x['piece_count']<=8
            assert len(set(x['body_boxes']))>=2
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('malformed or unreconstructable puzzle') from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng=np.random.default_rng(seed)
    train_count,test_count=_resolve_counts(seed,num_train,num_test)
    params=_sample_puzzle_parameters(rng)
    puzzle={'train':[],'test':[]}
    seen=set()
    first=None
    index=0
    for split,count,regime in [('train',train_count,'source_train'),('test',test_count,'target_test')]:
        for local_index in range(count):
            choices=list(range(3,6 if params['difficulty']==0 else 9))
            if params['regime']=='count':
                choices=list(range(3,7)) if split=='train' else [6,7,8]
            elif params['regime']=='branch':
                choices=list(range(4,9))
            elif params['regime']=='source_scale':
                choices=[5] if split=='train' else [6]
            if split=='train' and local_index==1 and params['regime']!='source_scale':
                choices=[c for c in choices if c!=first['piece_count']]
            piece_count=_pick(rng,choices)
            for attempt in range(64):
                pair=_generate_pair(rng,piece_count,params,index,regime)
                key=(pair['input'].shape,_grid_digest(pair['input']))
                if key in seen:
                    continue
                answers,_=reconstruct(pair['input'],params['anchor'])
                info=_placement_features(pair['input'],pair['output'],params['anchor'],answers[0][1])
                if len(set(info['body_boxes']))<2:
                    continue
                varying=['assembly_box','palette']
                if params['regime']!='source_scale':
                    varying.append('canvas')
                if split=='train' and local_index==1 and any(info[k]==first[k] for k in varying):
                    continue
                if first is None:
                    first=info
                break
            else:
                raise RuntimeError('bounded distinct-pair construction failed')
            puzzle[split].append(pair)
            seen.add(key)
            index+=1
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a,b):
    return all(len(a[s])==len(b[s]) and all(np.array_equal(x[k],y[k])
               for x,y in zip(a[s],b[s]) for k in ['input','output']) for s in ['train','test'])


def self_test() -> None:
    puzzles=[generate_puzzle(seed) for seed in range(42)]
    records=[]
    groups=[]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        group=puzzle_features(puzzle)
        groups.append(group)
        records.extend(group)
    assert len({len(p['train']) for p in puzzles})>=4
    assert len({len(p['test']) for p in puzzles})>=3
    assert _puzzles_equal(puzzles[17],generate_puzzle(17))
    assert not _puzzles_equal(puzzles[17],puzzles[18])
    for train,test in [(2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None)]:
        p=generate_puzzle(8123,train,test)
        assert train is None or len(p['train'])==train
        assert test is None or len(p['test'])==test
        assert _puzzles_equal(p,generate_puzzle(8123,train,test))
        validate_generated_puzzle(p)
    for train,test in [(1,1),(11,1),(2,0),(6,5),(2,2),(3,4),(True,1),(3,1.5)]:
        try:
            generate_puzzle(5,train,test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    assert {r['piece_count'] for r in records}==set(range(3,9))
    assert len({r['anchor'] for r in records})>=7
    assert len({r['connector'] for r in records})>=7
    assert len({r['background'] for r in records})>=7
    assert len({r['canvas'] for r in records})>=50
    assert len({r['assembly_box'] for r in records})>=80
    assert len({b for r in records for b in r['body_boxes']})>=35
    assert len({b for r in records for b in r['body_shapes']})>=250
    assert len({b for r in records for b in r['joint_keys']})>=25
    assert {c for r in records for c in r['marker_categories']}=={'singleton','connected','horizontal','vertical','diagonal','offset'}
    assert len({d for r in records for d in r['displacements']})>=100
    assert {d for r in records for d in r['degrees']} >= {1,2,3,4}
    assert {r['anchor_degree'] for r in records} >= {1,2,3}
    assert max(max(r['arm_lengths'],default=0) for r in records)>=3
    assert len({r['anchor_position'] for r in records})>=100
    assert len({r['anchor_box'] for r in records})>=25
    assert sum(any(x==0 for x in r['margins']) for r in records)>=20
    assert sum(r['equal_area_different_keys'] for r in records)>len(records)//2
    assert sum(r['reuse']>0 for r in records)>len(records)//3
    assert sum(r['combined'] for r in records)>=20
    assert sum(len(set(r['joint_keys']))<len(r['joint_keys']) for r in records)>=20
    assert max(max(r['joint_sizes']) for r in records)==3
    assert sum(any(h>0 for h in r['holes']) for r in records)>=20
    assert len({t for r in records for t in r['thicknesses']})>=5
    assert len({c for r in records for c in r['corner_patterns']})>=8
    assert any(min(b)==1 for r in records for b in r['body_boxes'])
    assert any(r['assembly_box'][0]>2*r['assembly_box'][1] for r in records)
    assert any(r['assembly_box'][1]>2*r['assembly_box'][0] for r in records)
    assert sum(all(r['piece_count']==(5 if r['split']=='train' else 6)
                   and r['canvas']==((22,22) if r['split']=='train' else (26,26))
                   and (r['split']=='train' or r['assembly_box'][0]>=1.5*r['assembly_box'][1])
                   for r in group) for group in groups)>=2
    for role in ['background','connector']:
        assert sum(len({r[role] for r in group})>1 for group in groups)>=4
        assert sum(len({r[role] for r in group})==1 for group in groups)>=4
    for split in ['train','test']:
        subset=[r for r in records if r['split']==split]
        assert any(max(r['degrees'])==2 for r in subset)
        assert any(max(r['degrees'])>=3 for r in subset)
        assert any(r['piece_count']==7 for r in subset)
    # Negative checks exercise semantic constraints, not only the schema.
    p=puzzles[0]
    bad={s:[{k:v.copy() for k,v in q.items()} for q in p[s]] for s in ['train','test']}
    bg,marker,ps=analyze(bad['test'][0]['input'])
    q=tuple(np.argwhere(bad['test'][0]['output']==marker)[0])
    bad['test'][0]['output'][q]=bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('erased connector accepted')
    ambiguous=np.zeros((15,15),dtype=np.int8)
    ambiguous[8,7]=1
    ambiguous[8,6]=ambiguous[8,8]=2
    ambiguous[2,2],ambiguous[1,2]=3,2
    ambiguous[2,12],ambiguous[1,12]=4,2
    alternatives,_=reconstruct(ambiguous,1)
    assert len(alternatives)==2, 'interchangeable singleton leaves must be rejected'
