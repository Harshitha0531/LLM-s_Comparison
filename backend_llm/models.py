from pydantic import BaseModel
from typing import Optional


class QueryBody(BaseModel):
    query: str
    mode: Optional[str] = "compare_fast"
    page_start: Optional[int] = None
    page_end: Optional[int] = None


class QueryRequest(BaseModel):
    query: str
