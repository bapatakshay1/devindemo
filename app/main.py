"""FastAPI application with a paginated /transactions endpoint.

This module provides a REST API for querying financial transactions
with proper offset-based pagination.

Fix for GitHub Issue #9: broken pagination on /transactions endpoint.
"""

from __future__ import annotations

import math

from fastapi import FastAPI, HTTPException, Query

from app.models import PaginatedResponse, Transaction
from app.sample_data import SAMPLE_TRANSACTIONS

app = FastAPI(
    title="FinServ Co Transactions API",
    description="A simple transactions API with proper pagination.",
    version="1.0.0",
)


@app.get("/transactions", response_model=PaginatedResponse[Transaction])
def list_transactions(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(
        10, ge=1, le=100, description="Number of items per page (max 100)"
    ),
) -> PaginatedResponse[Transaction]:
    """Return a paginated list of transactions.

    Supports page-based pagination with configurable page size.

    - **page**: The page number to retrieve (starts at 1).
    - **per_page**: How many transactions per page (1-100, default 10).
    """
    total = len(SAMPLE_TRANSACTIONS)
    total_pages = max(1, math.ceil(total / per_page))

    if page > total_pages:
        raise HTTPException(
            status_code=404,
            detail=f"Page {page} not found. Total pages: {total_pages}.",
        )

    start = (page - 1) * per_page
    end = start + per_page
    items = SAMPLE_TRANSACTIONS[start:end]

    return PaginatedResponse[Transaction](
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
