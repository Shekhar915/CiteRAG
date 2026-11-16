from pypdf import PdfWriter

from citerag.ingestion.loaders import load_pdf, load_text


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

def test_load_pdf_skips_pages_without_text(tmp_path):
    f = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with f.open("wb") as fh:
        writer.write(fh)
    assert load_pdf(f) == []