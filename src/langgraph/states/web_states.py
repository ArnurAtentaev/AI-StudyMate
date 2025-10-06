import operator
from typing import Annotated
from pydantic import BaseModel


class StatesWebSearcher(BaseModel):
    messages: Annotated[str, operator.add]
