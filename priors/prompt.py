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

SCRATCHPAD = """Before providing your final answer, use the scratchpad below to work through your reasoning:

<scratchpad>
[Analyze each potential causal relationship between the variables, considering:
- Temporal relationships
- Logical dependencies  
- Definitional relationships
- Well-established causal mechanisms
- Potential contradictions or impossibilities]
</scratchpad>
"""

PROMPT_PRIOR = """
You will be analyzing a set of causal variables to determine the required and forbidden causal directions between them. Your goal is to achieve very high precision in identifying these relationships.

<causal_variables>
{CAUSAL_VARIABLES}
</causal_variables>

Your task is to analyze these causal variables and generate two sets:
1. **Required directions**: Causal relationships that must exist based on logical necessity, temporal ordering, or fundamental causal principles
2. **Forbidden directions**: Causal relationships that cannot exist due to logical impossibility, temporal constraints, or definitional contradictions

Here are the key principles to follow:

**Required directions** should include:
- Relationships where one variable definitionally or logically must cause another
- Temporal precedence relationships (causes must precede effects)
- Relationships where the causal mechanism is well-established and unavoidable

**Forbidden directions** should include:
- Relationships that would violate temporal ordering (effects cannot cause their own causes)
- Relationships that are logically contradictory or definitionally impossible
- Relationships where the direction would violate established causal mechanisms

**Important guidelines:**
- Only include relationships where you have very high confidence
- When in doubt about a relationship, do not include it in either set
- Consider both direct and indirect causal pathways
- Be precise about the direction of causality (A → B is different from B → A)

Format your final answer as follows:

**Required Directions:**
- [Variable A] → [Variable B]: [Brief justification]
- [Continue for all required directions]

**Forbidden Directions:**  
- [Variable C] → [Variable D]: [Brief justification]
- [Continue for all forbidden directions]

Your final answer should only include the Required Directions and Forbidden Directions sections with their respective causal relationships and justifications. Aim for very high precision - only include relationships where you are highly confident in the causal direction requirement or prohibition.
"""

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
    if descriptions is None:
        variables_desc = "\n".join(bn.names())
    else:
        variables_desc = "\n".join(
            [f"{var}: {descriptions[var]}" for var in bn.names()]
        )
    return PROMPT_PRIOR.format(CAUSAL_VARIABLES=variables_desc)
