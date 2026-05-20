from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class PageIterator(Generic[T]):
    """Lazy iterator that fetches the next page only when the current page is exhausted."""

    def __init__(
        self,
        fetch_page: Callable[[str | None], tuple[list[T], str | None]],
    ) -> None:
        self._fetch_page = fetch_page
        self._buffer: list[T] = []
        self._next_token: str | None = None
        self._exhausted = False

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        while not self._buffer:
            if self._exhausted:
                raise StopIteration
            items, self._next_token = self._fetch_page(self._next_token)
            self._buffer = items
            if self._next_token is None:
                self._exhausted = True
            if not self._buffer:
                raise StopIteration
        return self._buffer.pop(0)
