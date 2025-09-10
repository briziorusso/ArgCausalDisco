from itertools import combinations

import pyagrum as gum

PROMPT_GRAPH_DESC = """You are tasked with generating descriptions for causal variables in a randomly generated causal graph used for synthetic dataset evaluations of causal discovery algorithms. Your goal is to create meaningful descriptions for each variable that accurately reflect their role in the graph without revealing their relationships with other variables.

Here is the description of the randomly generated causal graph:

<causal_graph>
{CAUSAL_GRAPH_DESCRIPTION}
</causal_graph>

To complete this task, follow these steps:

1. Carefully read and analyze the causal graph description.

2. For each variable mentioned in the graph, create a description that:
   a) Precisely describes its meaning within the context of the graph
   b) Does NOT spoil its relationships with other variables, either explicitly or implicitly
   c) Maintains a consistent context or scenario across all variable descriptions

3. If a variable has different contextual meanings in its relationships with other variables, provide a general description that could encompass these different meanings without revealing the specific relationships.

4. After generating all variable descriptions, assess the quality of the random causal graph based on how well the variables' meanings align with each other and how realistic the overall scenario is.

5. Present your output in the following format:

<title>
[Provide a concise title for the causal graph]
</title>

<variable_descriptions>
[List each variable and its description]
</variable_descriptions>

<graph_quality_assessment>
[Provide your assessment of the graph quality, including how well the variables' meanings align and how realistic the overall scenario is]
</graph_quality_assessment>

Remember, your primary goal is to create meaningful descriptions without revealing any causal relationships. Be creative in developing a consistent context that could plausibly connect all the variables."""

PROMPT_PRIOR = """You are an expert in causal inference tasked with identifying pairs of discrete causal variables that are conditionally independent given the knowledge of all other variables. This task is crucial for understanding the underlying structure of causal relationships in various fields such as statistics, machine learning, and causal inference.

Here are the variables in the system:

<variable_list>
{VARIABLES}
</variable_list>

And here are all possible pair combinations of these variables:

<pair_combinations>
{PAIR_COMBINATIONS}
</pair_combinations>

Your goal is to identify which pairs of variables are conditionally independent. Two variables A and B are conditionally independent given a set of variables Z if, once we know the values of all variables in Z, knowing the value of A provides no additional information about the value of B, and vice versa.

Instructions:
1. Analyze each pair of variables (A, B) from the pair_combinations list.
2. Consider all other variables as the conditioning set Z.
3. Determine if A and B are conditionally independent given Z.
4. Include pairs in your final list if they are conditionally independent or if there's reasonable doubt about their conditional dependence.

Conduct your analysis inside <conditional_independence_analysis> tags. Follow these steps:

1. List all variables and their potential relationships.
2. For each pair:
   a. State the pair being analyzed.
   b. List all other variables as the conditioning set.
   c. Consider direct relationships between the pair.
   d. Consider indirect relationships through other variables.
   e. Evaluate how the conditioning set might affect the relationship.
   f. Make a decision with a confidence level.

Consider both direct and indirect relationships between variables. Be less conservative in your analysis - if there's a possibility of conditional independence, include the pair. It's OK for this section to be quite long.

<conditional_independence_analysis>
[Your detailed analysis here, following the steps outlined above]
</conditional_independence_analysis>

After your analysis, present your results in the following format:

<conditionally_independent_pairs>
(Variable1, Variable2): [Explanation for why this pair is conditionally independent]
(Variable3, Variable4): [Explanation for why this pair is conditionally independent]
...
</conditionally_independent_pairs>

Remember, while precision is important, don't be overly conservative. If there's a reasonable argument for conditional independence, include the pair in your list. This task requires balancing between identifying as many conditionally independent pairs as possible while maintaining a high level of confidence in your selections.

Begin your analysis now, using the provided variables and pair combinations to identify conditionally independent pairs."""


def prepare_graph_description(
    bn: gum.BayesNet,
) -> str:
    topological_order = {id: rank for rank, id in enumerate(bn.topologicalOrder())}
    variables = "\n".join(
        sorted(bn.names(), key=lambda name: topological_order[bn.idFromName(name)])
    )
    arcs = "\n".join(
        [
            f"{bn.variable(id1).name()} -> {bn.variable(id2).name()}"
            for id1, id2 in sorted(bn.arcs(), key=lambda arc: topological_order[arc[0]])
        ]
    )
    return PROMPT_GRAPH_DESC.format(
        CAUSAL_GRAPH_DESCRIPTION="\n".join(
            [
                "Variables:",
                "```",
                variables,
                "```",
                "Causal relationships:",
                "```",
                arcs,
                "```",
            ]
        )
    )


def prepare_priors(bn: gum.BayesNet, descriptions: dict[str, str] | None = None) -> str:
    pairs_desc = "\n".join(
        [f"{var1}, {var2}" for var1, var2 in combinations(bn.names(), 2)]
    )
    if descriptions is None:
        variables_desc = "\n".join(bn.names())
    else:
        variables_desc = "\n".join(
            [f"{var}: {descriptions[var]}" for var in bn.names()]
        )
    return PROMPT_PRIOR.format(VARIABLES=variables_desc, PAIR_COMBINATIONS=pairs_desc)
