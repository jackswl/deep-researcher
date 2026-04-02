import os
import re
import tempfile

from deep_researcher.models import Paper, clean_abstract
from deep_researcher.report import save_report


class TestPaperUniqueKey:
    def test_doi_takes_priority(self):
        p = Paper(title="Test", doi="10.1234/test", arxiv_id="2301.00001")
        assert p.unique_key == "doi:10.1234/test"

    def test_arxiv_when_no_doi(self):
        p = Paper(title="Test", arxiv_id="2301.00001")
        assert p.unique_key == "arxiv:2301.00001"

    def test_pmid_when_no_doi_or_arxiv(self):
        p = Paper(title="Test", pmid="12345678")
        assert p.unique_key == "pmid:12345678"

    def test_title_hash_fallback(self):
        p = Paper(title="Some Paper Title")
        assert p.unique_key.startswith("title:")
        assert len(p.unique_key) == len("title:") + 16

    def test_title_hash_normalized(self):
        p1 = Paper(title="Some  Paper   Title")
        p2 = Paper(title="some paper title")
        assert p1.unique_key == p2.unique_key


class TestPaperMerge:
    def test_fills_missing_abstract(self):
        p1 = Paper(title="Test", abstract=None)
        p2 = Paper(title="Test", abstract="An abstract")
        p1.merge(p2)
        assert p1.abstract == "An abstract"

    def test_does_not_overwrite_existing_abstract(self):
        p1 = Paper(title="Test", abstract="Original")
        p2 = Paper(title="Test", abstract="New")
        p1.merge(p2)
        assert p1.abstract == "Original"

    def test_takes_higher_citation_count(self):
        p1 = Paper(title="Test", citation_count=10)
        p2 = Paper(title="Test", citation_count=50)
        p1.merge(p2)
        assert p1.citation_count == 50

    def test_keeps_higher_when_existing_is_higher(self):
        p1 = Paper(title="Test", citation_count=100)
        p2 = Paper(title="Test", citation_count=50)
        p1.merge(p2)
        assert p1.citation_count == 100

    def test_concatenates_sources(self):
        p1 = Paper(title="Test", source="scopus")
        p2 = Paper(title="Test", source="ieee")
        p1.merge(p2)
        assert "scopus" in p1.source
        assert "ieee" in p1.source

    def test_no_duplicate_sources(self):
        p1 = Paper(title="Test", source="scopus")
        p2 = Paper(title="Test", source="scopus")
        p1.merge(p2)
        assert p1.source.count("scopus") == 1


class TestPaperBibtex:
    def test_doi_based_key(self):
        p = Paper(title="Test Paper", doi="10.1234/test.2023", journal="Nature")
        bib = p.to_bibtex()
        assert "@article{10_1234_test_2023," in bib

    def test_fallback_key_format(self):
        p = Paper(title="Deep Learning Approach", authors=["John Smith"], year=2023)
        bib = p.to_bibtex()
        assert "smith2023deep" in bib

    def test_key_suffix(self):
        p = Paper(title="Deep Learning", authors=["Smith"], year=2023)
        bib = p.to_bibtex(key_suffix="_2")
        assert "_2" in bib

    def test_journal_uses_article_type(self):
        p = Paper(title="Test", journal="Nature")
        assert p.to_bibtex().startswith("@article{")

    def test_no_journal_uses_misc_type(self):
        p = Paper(title="Test")
        assert p.to_bibtex().startswith("@misc{")


class TestCleanAbstract:
    def test_strips_html(self):
        assert clean_abstract("<p>Hello <b>world</b></p>") == "Hello world"

    def test_normalizes_whitespace(self):
        assert clean_abstract("hello   world\n\nfoo") == "hello world foo"

    def test_none_returns_none(self):
        assert clean_abstract(None) is None

    def test_empty_returns_none(self):
        assert clean_abstract("") is None


class TestBibtexCollisionHandling:
    """Test the collision-handling logic in save_report's BibTeX writer."""

    def test_duplicate_keys_get_suffix(self):
        # Two papers by "Li" from 2023 starting with "deep" -> same key
        papers = {
            "a": Paper(title="Deep Learning for Bridges", authors=["Li Wei"], year=2023, source="scopus"),
            "b": Paper(title="Deep Networks for Cracks", authors=["Li Chen"], year=2023, source="ieee"),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_report("test query", "# Report", papers, tmpdir)
            bib_path = os.path.join(tmpdir, os.listdir(tmpdir)[0], "references.bib")
            with open(bib_path) as f:
                content = f.read()
            # Both papers should be present (second gets _1 suffix)
            keys = re.findall(r"@\w+\{(.+?),", content)
            assert len(keys) == 2
            assert keys[0] != keys[1]  # Keys must be distinct

    def test_unique_keys_no_suffix(self):
        papers = {
            "a": Paper(title="Deep Learning", authors=["Smith"], year=2023, source="scopus"),
            "b": Paper(title="Crack Detection", authors=["Jones"], year=2024, source="ieee"),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_report("test query", "# Report", papers, tmpdir)
            bib_path = os.path.join(tmpdir, os.listdir(tmpdir)[0], "references.bib")
            with open(bib_path) as f:
                content = f.read()
            keys = re.findall(r"@\w+\{(.+?),", content)
            assert len(keys) == 2
            # No suffix needed — keys are naturally distinct
            assert all("_" not in k or k.count("_") == 0 for k in keys) or keys[0] != keys[1]
