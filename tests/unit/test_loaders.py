from citerag.ingestion.loaders import load_text


def test_load_text_returns_single_page(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nSome body text.", encoding="utf-8")
    pages = load_text(f)
    assert len(pages) == 1
    assert pages[0].source == "note.md"
    assert pages[0].page is None
    assert "Some body text." in pages[0].text


def test_load_text_skips_empty_file(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n", encoding="utf-8")
    assert load_text(f) == []