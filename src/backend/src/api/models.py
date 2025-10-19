from pydantic import BaseModel, Field
from typing import List, Dict


class SearchResult(BaseModel):
    filename: str
    genre: str
    distance: float = Field(ge=0.0, description="Cosine distance (lower is more similar)")


class QueryInfo(BaseModel):
    filename: str
    duration: float


class SearchResponse(BaseModel):
    query: QueryInfo
    results: List[SearchResult]


class GenreStats(BaseModel):
    total_samples: int
    genres: Dict[str, int]
    embedding_dimension: int


class HealthResponse(BaseModel):
    status: str
    message: str
    endpoints: List[str]


class ErrorResponse(BaseModel):
    detail: str