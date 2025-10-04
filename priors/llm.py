import re
from typing import Annotated, Literal, TypeVar

import instructor
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel, Field, computed_field

from priors.schema import Constraints

T_Model = TypeVar("T_Model", bound=BaseModel)

client = instructor.from_openai(
    AsyncOpenAI(base_url="http://localhost:4000", api_key=""), mode=instructor.Mode.MD_JSON
)


class GraphDescriptionBase(BaseModel):
    title: str
    variable_descriptions: dict[str, str]
    graph_quality: Literal[
        "Very Poor", "Poor", "Moderate", "Good", "Very Good", "Unknown"
    ]

    @computed_field
    @property
    def identifier(self) -> str:
        return re.sub("\W+|^(?=\d)", "_", self.title.lower())


async def extract(
    prompt: str,
    pydantic_model: type[T_Model] | None,
    model="gemini-2.5-flash-lite",
    temperature=1,
    max_retries=5,
) -> T_Model | ChatCompletion:
    return await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_retries=max_retries,
        response_model=pydantic_model,
    )


async def parse_graph_description(
    graph_response: str,
    model: str,
    valid_vars: set[str] | None = None,
) -> dict[str, str] | GraphDescriptionBase:
    valid_var_pattern = (
        r"|".join(re.escape(var) for var in valid_vars) if valid_vars else r"\w+"
    )
    
    class GraphDescription(GraphDescriptionBase):
        variable_descriptions: dict[
            Annotated[str, Field(pattern=valid_var_pattern)], str
        ]

    return await extract(graph_response, GraphDescription, model=model)


async def parse_priors(
    prior_response: str,
    model: str | None = None,
) -> set[tuple[str, str]]:
    return await extract(
        prompt=prior_response,
        pydantic_model=Constraints,
        model=model,
    )
