"""Graph evaluation with explicit encodings and exact, certified SID extrema.

Canonical PDAGs use row->column 1 for directed edges and symmetric 1 for
undirected edges. Never infer endpoint encoding from signs: pass it explicitly.
"""
from __future__ import annotations
from collections import deque
from functools import lru_cache
from time import monotonic
import os
import numpy as np
import networkx as nx


class InvalidGraph(ValueError):
    pass


def adjacency(matrix, *, endpoints=False):
    a = np.asarray(matrix)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or np.diag(a).any() or not np.isfinite(a).all():
        raise InvalidGraph('Expected a finite square matrix without self-loops')
    b = np.zeros(a.shape, dtype=np.int8)
    for i in range(len(a)):
        for j in range(i+1,len(a)):
            x,y = a[i,j],a[j,i]
            if endpoints:
                if (x,y)==(1,-1): b[i,j]=1
                elif (x,y)==(-1,1): b[j,i]=1
                elif (x,y)==(-1,-1): b[i,j]=b[j,i]=1
                elif (x,y)!=(0,0): raise InvalidGraph(f'Unsupported endpoint pair {(x,y)}')
            elif x == -1 or y == -1:
                if x == 1 or y == 1: raise InvalidGraph('Mixed endpoint/adjacency encoding')
                b[i,j]=b[j,i]=1
            else:
                if x not in (0,1) or y not in (0,1): raise InvalidGraph('Adjacency must contain only 0, 1, -1')
                b[i,j],b[j,i]=x,y
    return b


def require_dag(b):
    if not nx.is_directed_acyclic_graph(nx.from_numpy_array(b,create_using=nx.DiGraph)):
        raise InvalidGraph('Not a DAG')


def complete_cpdag(dag):
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.DAG2CPDAG import dag2cpdag
    dag=adjacency(dag); require_dag(dag)
    nodes=[GraphNode(str(i)) for i in range(len(dag))]
    graph=Dag(nodes)
    for i,j in zip(*np.where(dag==1)): graph.add_directed_edge(nodes[i],nodes[j])
    return adjacency(dag2cpdag(graph).graph.T,endpoints=True)


def consistent_extension(pdag, seed=None):
    """Dor--Tarsi sink elimination; reject instead of looping on invalid PDAGs."""
    p=adjacency(pdag)
    d=((p==1)&(p.T==0)).astype(np.int8)
    require_dag(d)
    remaining=set(range(len(p)))
    priority=list(range(len(p))) if seed is None else np.random.default_rng(seed).permutation(len(p)).tolist()
    while remaining:
        for i in (v for v in priority if v in remaining):
            children={j for j in remaining if p[i,j] and not p[j,i]}
            if children: continue
            neighbors={j for j in remaining if p[i,j] and p[j,i]}
            parents={j for j in remaining if p[j,i] and not p[i,j]}
            if any(not(p[x,y] or p[y,x]) for x in neighbors for y in (neighbors|parents)-{x}): continue
            for j in neighbors: d[j,i]=1
            remaining.remove(i)
            break
        else:
            raise InvalidGraph('PDAG has no consistent DAG extension')
    require_dag(d)
    return d


def decode_estimate(raw,method):
    """Decode the learner's recorded representation before any binarization."""
    if method in {'mpc','pc','pc_max','cpc','spc'}:
        p=adjacency(raw,endpoints=True)
    elif method=='ges':
        p=adjacency(np.asarray(raw).T,endpoints=True)
    elif method=='nt':
        p=adjacency((np.asarray(raw)!=0).astype(np.int8))
    else:
        p=adjacency(raw)
    return p


def prepare_estimate(raw,method,seed=None):
    p=decode_estimate(raw,method)
    d=consistent_extension(p,seed=seed) if (p*p.T).any() else p
    require_dag(d)
    c=complete_cpdag(d)
    # A native partially directed output must already encode this equivalence
    # class; do not silently discard extra directions to force validity.
    if (p*p.T).any() and not np.array_equal(c,p):
        raise InvalidGraph('Native PDAG is not completed; reevaluate its representation before scoring')
    return d,c


def graph_sets(pdag):
    p=adjacency(pdag)
    skeleton=set(); arrows=set(); edges=set()
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if not(p[i,j] or p[j,i]): continue
            skeleton.add((i,j))
            if p[i,j] and p[j,i]: edges.add(('undirected',i,j))
            elif p[i,j]: arrows.add((i,j)); edges.add(('directed',i,j))
            else: arrows.add((j,i)); edges.add(('directed',j,i))
    return skeleton,arrows,edges


def prf(pred,true):
    tp=len(pred&true)
    precision=tp/len(pred) if pred else float('nan')
    recall=tp/len(true) if true else float('nan')
    f1=2*tp/(len(pred)+len(true)) if pred or true else float('nan')
    return precision,recall,f1


def structural_scores(estimate,reference):
    sets=graph_sets(estimate); truth=graph_sets(reference)
    out={}
    for prefix,p,t in zip(('adjacency_','arrowhead_',''),sets,truth):
        out.update(zip((prefix+'precision',prefix+'recall',prefix+'F1'),prf(p,t)))
    # One edit per differing pair state: absent, undirected, i->j or j->i.
    a=adjacency(estimate);b=adjacency(reference)
    out['shd']=sum((a[i,j],a[j,i])!=(b[i,j],b[j,i]) for i in range(len(a)) for j in range(i+1,len(a)))
    out['nnz']=len(sets[0])
    return out


def dag_sid(truth,estimate):
    os.environ.setdefault('RAYON_NUM_THREADS','1')
    from gadjid import sid
    require_dag(estimate);require_dag(truth)
    return int(sid(np.asarray(truth,dtype=np.int8),np.asarray(estimate,dtype=np.int8),edge_direction='from row to column')[1])


def enumerate_sid_bounds(truth,cpdag, *, max_dags=100000, timeout=60.):
    """Enumerate each chain component by covered edge reversals.

    SID is additive over intervention nodes, whose scores depend only on their
    estimated parent sets. Independent component extrema can therefore be added
    without enumerating the Cartesian product of component orientations.
    Every visited orientation is a DAG in the same MEC. No approximate fallback.
    """
    started=monotonic();cpdag=adjacency(cpdag)
    base=consistent_extension(cpdag)
    if not np.array_equal(complete_cpdag(base),cpdag):
        raise InvalidGraph('SID extrema require a completed PDAG')
    base_sid=dag_sid(truth,base);lo=hi=base_sid;counts=[]
    for component in nx.connected_components(nx.from_numpy_array(cpdag*cpdag.T)):
        if len(component)==1: continue
        nodes=sorted(component);idx=np.ix_(nodes,nodes)
        queue=deque([base.copy()]);seen={base[idx].tobytes()};values=[]
        while queue:
            if monotonic()-started > timeout or len(seen)>max_dags:
                return {'sid_low':np.nan,'sid_high':np.nan,'sid_status':'enumeration_limit',
                        'sid_component_dags':counts+[len(seen)]}
            d=queue.popleft();values.append(dag_sid(truth,d))
            parents=[set(np.flatnonzero(d[:,i])) for i in range(len(d))]
            for i in nodes:
                for j in nodes:
                    if d[i,j] and parents[j]==parents[i]|{i}:
                        new=d.copy();new[i,j]=0;new[j,i]=1;key=new[idx].tobytes()
                        if key not in seen: seen.add(key);queue.append(new)
        counts.append(len(seen));lo+=min(values)-base_sid;hi+=max(values)-base_sid
    assert 0 <= lo <= hi <= len(base)*(len(base)-1)
    return {'sid_low':lo,'sid_high':hi,'sid_status':'exact','sid_component_dags':counts}


def exact_sid_bounds(truth,cpdag, *, max_states=1000000, timeout=60.):
    """Exact dynamic program over simplicial-sink eliminations in each component.

    A state is the remaining vertex subset. Removing a simplicial sink fixes
    its parents; SID's intervention-node contribution is additive. All perfect
    elimination orders are covered, with shared subproblems memoized. Witness
    DAGs are independently scored and must attain both returned extrema.
    """
    os.environ.setdefault('RAYON_NUM_THREADS','1')
    from gadjid import sid
    started=monotonic();p=adjacency(cpdag);truth=adjacency(truth)
    base=consistent_extension(p)
    if not np.array_equal(complete_cpdag(base),p): raise InvalidGraph('Expected a CPDAG')
    empty=np.zeros(p.shape,dtype=np.int8)
    def score(d): return int(sid(truth,d,edge_direction='from row to column')[1])
    empty_sid=score(empty)
    @lru_cache(None)
    def cost(i,parents):
        star=empty.copy();star[list(parents),i]=1
        return score(star)-empty_sid
    total_lo=total_hi=empty_sid
    low_witness=base.copy();high_witness=base.copy();state_counts=[]
    directed=(p==1)&(p.T==0)
    for comp in nx.connected_components(nx.from_numpy_array(p*p.T)):
        nodes=sorted(comp);n=len(nodes)
        outside={i:tuple(np.flatnonzero(directed[:,i])) for i in nodes}
        neighbors=[sum(1<<j for j in range(n) if p[i,nodes[j]] and p[nodes[j],i]) for i in nodes]
        choices={};states=0
        @lru_cache(None)
        def solve(mask):
            nonlocal states
            states+=1
            if states>max_states or monotonic()-started>timeout: raise TimeoutError('Exact SID state/time limit')
            if not mask: return 0,0
            low=float('inf');high=-float('inf')
            for k,i in enumerate(nodes):
                if not mask & (1<<k): continue
                nb=neighbors[k]&mask
                # A sink's remaining undirected neighbors must form a clique.
                if any((nb&~(1<<j)) & ~neighbors[j] for j in range(n) if nb&(1<<j)): continue
                parents=tuple(sorted((*outside[i],*(nodes[j] for j in range(n) if nb&(1<<j)))))
                delta=cost(i,parents);slo,shi=solve(mask&~(1<<k))
                if delta+slo<low: low=delta+slo;choose_lo=k
                if delta+shi>high: high=delta+shi;choose_hi=k
            if low==float('inf'): raise InvalidGraph('Nonchordal component')
            choices[mask]=(choose_lo,choose_hi)
            return low,high
        try: lo,hi=solve((1<<n)-1)
        except TimeoutError:
            return {'sid_low':np.nan,'sid_high':np.nan,'sid_status':'enumeration_limit','sid_states':state_counts+[states]}
        total_lo+=lo;total_hi+=hi;state_counts.append(states)
        for index,witness in enumerate((low_witness,high_witness)):
            mask=(1<<n)-1
            while mask:
                k=choices[mask][index];i=nodes[k]
                for j in range(n):
                    if mask&(1<<j) and neighbors[k]&(1<<j):
                        witness[nodes[j],i]=1;witness[i,nodes[j]]=0
                mask &= ~(1<<k)
    for witness,value in ((low_witness,total_lo),(high_witness,total_hi)):
        require_dag(witness)
        if not np.array_equal(complete_cpdag(witness),p) or score(witness)!=value:
            raise AssertionError('SID witness certification failed')
    return {'sid_low':total_lo,'sid_high':total_hi,'sid_status':'exact','sid_states':state_counts,
            'sid_low_witness':low_witness,'sid_high_witness':high_witness}
