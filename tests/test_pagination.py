from __future__ import annotations

from unittest.mock import MagicMock
from artifact_mgmt._pagination import PageIterator


def test_single_page_returns_all_items() -> None:
    fetch = MagicMock(return_value=(["a", "b", "c"], None))
    result = list(PageIterator(fetch))
    assert result == ["a", "b", "c"]
    fetch.assert_called_once_with(None)


def test_three_pages_fetched_exactly_three_times() -> None:
    fetch = MagicMock(side_effect=[
        (["a", "b"], "tok1"),
        (["c", "d"], "tok2"),
        (["e"], None),
    ])
    result = list(PageIterator(fetch))
    assert result == ["a", "b", "c", "d", "e"]
    assert fetch.call_count == 3
    fetch.assert_any_call(None)
    fetch.assert_any_call("tok1")
    fetch.assert_any_call("tok2")


def test_second_page_not_fetched_until_first_exhausted() -> None:
    calls: list[str | None] = []

    def fetch(token: str | None) -> tuple[list[str], str | None]:
        calls.append(token)
        if token is None:
            return (["a", "b"], "tok1")
        return (["c"], None)

    iterator = PageIterator(fetch)

    assert next(iterator) == "a"
    assert len(calls) == 1  # second page not yet fetched

    assert next(iterator) == "b"
    assert len(calls) == 1  # still only one fetch

    assert next(iterator) == "c"
    assert len(calls) == 2  # now second page fetched


def test_empty_first_page_returns_empty_iterator() -> None:
    fetch = MagicMock(return_value=([], None))
    result = list(PageIterator(fetch))
    assert result == []
    fetch.assert_called_once_with(None)


def test_iterator_works_for_list_models_items_key() -> None:
    # Simulates list_models where response key is "items"
    pages = [
        {"items": ["model1", "model2"], "nextPageToken": "tok1"},
        {"items": ["model3"], "nextPageToken": None},
    ]
    page_iter = iter(pages)

    def fetch(token: str | None) -> tuple[list[str], str | None]:
        page = next(page_iter)
        return page["items"], page["nextPageToken"]

    result = list(PageIterator(fetch))
    assert result == ["model1", "model2", "model3"]


def test_iterator_works_for_list_versions_versions_key() -> None:
    # Simulates list_versions where response key is "versions"
    pages = [
        {"versions": ["1.0", "1.1"], "nextPageToken": "tok1"},
        {"versions": ["2.0"], "nextPageToken": None},
    ]
    page_iter = iter(pages)

    def fetch(token: str | None) -> tuple[list[str], str | None]:
        page = next(page_iter)
        return page["versions"], page["nextPageToken"]

    result = list(PageIterator(fetch))
    assert result == ["1.0", "1.1", "2.0"]


def test_iterator_is_reusable_as_for_loop() -> None:
    fetch = MagicMock(return_value=(["x", "y"], None))
    items = [item for item in PageIterator(fetch)]
    assert items == ["x", "y"]
