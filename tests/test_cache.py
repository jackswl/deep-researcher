import json
import os
import tempfile
import time

import pytest

import deep_researcher.cache as cache_module
from deep_researcher.cache import invalidate, load, save
from deep_researcher.models import Paper


@pytest.fixture(autouse=True)
def isolated_cache_dir(monkeypatch, tmp_path):
    """Redirect all cache I/O to a temp directory for every test."""
    monkeypatch.setattr(cache_module, "_CACHE_DIR", str(tmp_path))
    return tmp_path


def _sample_papers() -> dict[str, Paper]:
    p1 = Paper(
        title="Deep Learning for Bridges",
        authors=["Alice Smith", "Bob Jones"],
        year=2023,
        abstract="A study on bridges.",
        doi="10.1234/bridges",
        source="scopus",
        citation_count=42,
        journal="Nature",
        volume="10",
        pages="1-10",
        publisher="Springer",
        arxiv_id=None,
        pmid=None,
        open_access_url="https://example.com/paper",
        keywords=["bridges", "deep learning"],
    )
    p2 = Paper(
        title="Crack Detection via CNN",
        authors=["Carol White"],
        year=2022,
        arxiv_id="2201.00001",
        source="arxiv",
    )
    return {p1.unique_key: p1, p2.unique_key: p2}


class TestSaveAndLoad:
    def test_roundtrip_basic(self):
        papers = _sample_papers()
        save("test query", papers)
        result = load("test query")
        assert result is not None
        assert len(result) == len(papers)

    def test_roundtrip_preserves_fields(self):
        papers = _sample_papers()
        save("test query", papers)
        result = load("test query")
        doi_key = "doi:10.1234/bridges"
        assert doi_key in result
        p = result[doi_key]
        assert p.title == "Deep Learning for Bridges"
        assert p.authors == ["Alice Smith", "Bob Jones"]
        assert p.year == 2023
        assert p.abstract == "A study on bridges."
        assert p.doi == "10.1234/bridges"
        assert p.citation_count == 42
        assert p.journal == "Nature"
        assert p.volume == "10"
        assert p.pages == "1-10"
        assert p.publisher == "Springer"
        assert p.open_access_url == "https://example.com/paper"
        assert p.keywords == ["bridges", "deep learning"]

    def test_roundtrip_arxiv_key(self):
        papers = _sample_papers()
        save("test query", papers)
        result = load("test query")
        assert "arxiv:2201.00001" in result

    def test_papers_without_title_are_excluded(self):
        papers = {"k": Paper(title="")}
        save("query", papers)
        result = load("query")
        assert result is not None
        assert len(result) == 0


class TestCacheMiss:
    def test_returns_none_when_no_cache(self):
        assert load("nonexistent query") is None

    def test_returns_none_after_ttl_expired(self, isolated_cache_dir, monkeypatch):
        papers = _sample_papers()
        save("old query", papers)

        # Make the cache file look 8 days old
        path = cache_module._cache_path("old query")
        old_mtime = time.time() - (8 * 24 * 60 * 60)
        os.utime(path, (old_mtime, old_mtime))

        assert load("old query") is None

    def test_returns_data_within_ttl(self, isolated_cache_dir):
        papers = _sample_papers()
        save("fresh query", papers)

        # Make it look 6 days old (within TTL)
        path = cache_module._cache_path("fresh query")
        recent_mtime = time.time() - (6 * 24 * 60 * 60)
        os.utime(path, (recent_mtime, recent_mtime))

        assert load("fresh query") is not None


class TestCacheKeyNormalization:
    def test_same_key_for_different_case(self):
        papers = _sample_papers()
        save("Theology of Dispensationalism", papers)
        result = load("theology of dispensationalism")
        assert result is not None

    def test_same_key_for_leading_trailing_whitespace(self):
        papers = _sample_papers()
        save("  my query  ", papers)
        result = load("my query")
        assert result is not None

    def test_different_queries_have_different_cache_files(self, isolated_cache_dir):
        p1 = cache_module._cache_path("query one")
        p2 = cache_module._cache_path("query two")
        assert p1 != p2


class TestInvalidate:
    def test_invalidate_existing_returns_true(self):
        save("query", _sample_papers())
        assert invalidate("query") is True

    def test_invalidate_removes_file(self, isolated_cache_dir):
        save("query", _sample_papers())
        invalidate("query")
        assert load("query") is None

    def test_invalidate_nonexistent_returns_false(self):
        assert invalidate("never cached") is False


class TestCorruptCache:
    def test_corrupt_json_returns_none(self, isolated_cache_dir):
        path = cache_module._cache_path("bad query")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("this is not valid json {{{")
        assert load("bad query") is None

    def test_save_to_unwritable_dir_does_not_raise(self, monkeypatch, tmp_path):
        unwritable = tmp_path / "locked"
        unwritable.mkdir()
        unwritable.chmod(0o444)
        monkeypatch.setattr(cache_module, "_CACHE_DIR", str(unwritable / "cache"))
        # Should not raise — errors are swallowed
        save("query", _sample_papers())
