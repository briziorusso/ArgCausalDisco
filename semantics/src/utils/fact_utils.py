from src.utils.enums import RelationEnum, Fact
import networkx as nx
import pandas as pd


def parse_fact_line(line):
    line = line.strip()
    relation, rest = line.split('(', 1)
    rest = rest.split(')')[0]
    node1, node2, node_set = rest.split(',')
    node1 = int(node1.strip())
    node2 = int(node2.strip())
    node_set = node_set.strip()
    if node_set == 'empty':
        node_set = {}
    else:
        node_set = node_set.strip()[1:].split('y')
        node_set = {int(x) for x in node_set if x.strip()}

    return Fact(
        relation=RelationEnum[relation.strip()],
        node1=node1,
        node2=node2,
        node_set=node_set,
        score=1,
    )


def facts_from_file(filename):
    facts = []
    with open(filename, 'r') as f:
        for line in f:
            facts.append(parse_fact_line(line))
    return facts


def generate_fact_tuple(fact):
    X, Y, S, dep_type_PC, I = fact.node1, fact.node2, fact.node_set, fact.relation.value, fact.score
    s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in S])
    return(X,S,Y,dep_type_PC, f"{dep_type_PC}({X},{Y},{s_str}).", I)


def get_fact_location(facts, base_location='./facts.lp'):

    facts_location = base_location
    facts_location_wc = base_location.replace('.lp', '_wc.lp')
    facts_location_I = base_location.replace('.lp', '_I.lp')

    facts = [generate_fact_tuple(fact) for fact in facts]


    ### Save external statements
    with open(facts_location, "w") as f:
        for n, s in enumerate(facts):
            f.write(f"#external ext_{s[4]}\n")
    ### Save weak constraints
    with open(facts_location_wc, "w") as f:
        for n, s in enumerate(facts):
            f.write(f":~ {s[4]} [-{int(s[5]*1000)}]\n")
    ### Save inner strengths
    with open(facts_location_I, "w") as f:
        for n, s in enumerate(facts):
            f.write(f"{s[4]} I={s[5]}, NA\n")
        
    return facts_location


def check_if_fact_is_true(fact, B_true):
    """
    Check if the fact is true in the true graph B_true.
    """
    G_true = nx.DiGraph(pd.DataFrame(B_true,
                                     columns=[f"X{i+1}" for i in range(B_true.shape[1])],
                                     index=[f"X{i+1}" for i in range(B_true.shape[1])]
                                     )
                        )
    x, y, s = fact.node1, fact.node2, fact.node_set
    s_text = [f"X{r+1}" for r in s]
    is_d_separated = nx.algorithms.d_separated(G_true, {f"X{x+1}"}, {f"X{y+1}"}, set(s_text))
    is_indep = fact.relation == RelationEnum.indep

    return is_d_separated == is_indep
