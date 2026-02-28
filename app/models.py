"""Data models for the transactions API."""

from __future__ import annotations

from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Transaction(BaseModel):
    """A single financial transaction."""

    id: int
    amount: float
    currency: str = "USD"
    description: str
    timestamp: datetime
    category: str


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated API response wrapper."""

    items: list[T]
    total: int = Field(description="Total number of items across all pages")
    page: int = Field(description="Current page number (1-indexed)")
    per_page: int = Field(description="Number of items per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")
