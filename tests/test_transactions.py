"""Tests for the /transactions endpoint pagination."""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sample_data import SAMPLE_TRANSACTIONS

client = TestClient(app)

TOTAL = len(SAMPLE_TRANSACTIONS)


class TestTransactionsPagination:
    """Tests for correct pagination behaviour on GET /transactions."""

    def test_default_pagination(self) -> None:
        """Default request returns page 1 with 10 items."""
        resp = client.get("/transactions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["per_page"] == 10
        assert data["total"] == TOTAL
        assert len(data["items"]) == 10
        assert data["has_prev"] is False
        assert data["has_next"] is True
        assert data["total_pages"] == math.ceil(TOTAL / 10)

    def test_first_page_explicit(self) -> None:
        """Explicitly requesting page=1 works."""
        resp = client.get("/transactions", params={"page": 1, "per_page": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 1
        assert data["per_page"] == 5
        assert len(data["items"]) == 5
        assert data["items"][0]["id"] == 1
        assert data["has_prev"] is False
        assert data["has_next"] is True

    def test_second_page(self) -> None:
        """Second page returns the correct offset of items."""
        resp = client.get("/transactions", params={"page": 2, "per_page": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == 2
        assert len(data["items"]) == 5
        assert data["items"][0]["id"] == 6
        assert data["has_prev"] is True
        assert data["has_next"] is True

    def test_last_page(self) -> None:
        """Last page returns remaining items and has_next=False."""
        per_page = 7
        total_pages = math.ceil(TOTAL / per_page)
        resp = client.get(
            "/transactions", params={"page": total_pages, "per_page": per_page}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["page"] == total_pages
        assert data["has_next"] is False
        assert data["has_prev"] is True
        expected_remaining = TOTAL - (total_pages - 1) * per_page
        assert len(data["items"]) == expected_remaining

    def test_single_item_per_page(self) -> None:
        """per_page=1 returns exactly one item per page."""
        resp = client.get("/transactions", params={"page": 3, "per_page": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == 3
        assert data["total_pages"] == TOTAL

    def test_all_items_single_page(self) -> None:
        """Requesting per_page >= total returns everything in one page."""
        resp = client.get("/transactions", params={"page": 1, "per_page": 100})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == TOTAL
        assert data["total_pages"] == 1
        assert data["has_next"] is False
        assert data["has_prev"] is False

    def test_page_beyond_total_returns_404(self) -> None:
        """Requesting a page beyond total_pages returns 404."""
        total_pages = math.ceil(TOTAL / 10)
        resp = client.get("/transactions", params={"page": total_pages + 1})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_page_zero_returns_422(self) -> None:
        """page=0 is invalid and should return 422."""
        resp = client.get("/transactions", params={"page": 0})
        assert resp.status_code == 422

    def test_negative_page_returns_422(self) -> None:
        """Negative page numbers should return 422."""
        resp = client.get("/transactions", params={"page": -1})
        assert resp.status_code == 422

    def test_per_page_zero_returns_422(self) -> None:
        """per_page=0 is invalid and should return 422."""
        resp = client.get("/transactions", params={"per_page": 0})
        assert resp.status_code == 422

    def test_per_page_exceeds_max_returns_422(self) -> None:
        """per_page > 100 should return 422."""
        resp = client.get("/transactions", params={"per_page": 101})
        assert resp.status_code == 422

    def test_total_pages_calculation(self) -> None:
        """total_pages is correctly calculated for various per_page values."""
        for per_page in (1, 3, 5, 7, 10, 15, 20):
            resp = client.get(
                "/transactions", params={"page": 1, "per_page": per_page}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_pages"] == math.ceil(TOTAL / per_page)

    def test_items_are_consistent_across_pages(self) -> None:
        """All items across all pages match the full dataset (no duplicates or gaps)."""
        per_page = 3
        total_pages = math.ceil(TOTAL / per_page)
        all_ids: list[int] = []
        for page in range(1, total_pages + 1):
            resp = client.get(
                "/transactions", params={"page": page, "per_page": per_page}
            )
            assert resp.status_code == 200
            data = resp.json()
            all_ids.extend(item["id"] for item in data["items"])

        expected_ids = [t.id for t in SAMPLE_TRANSACTIONS]
        assert all_ids == expected_ids

    def test_response_includes_all_pagination_fields(self) -> None:
        """Response includes all expected pagination metadata fields."""
        resp = client.get("/transactions")
        assert resp.status_code == 200
        data = resp.json()
        required_keys = {
            "items",
            "total",
            "page",
            "per_page",
            "total_pages",
            "has_next",
            "has_prev",
        }
        assert required_keys.issubset(data.keys())
