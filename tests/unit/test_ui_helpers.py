from citerag.schemas.qa import Citation
from citerag.ui.helpers import format_citation, ingest_uploads


class FakePipeline:
    def __init__(self, indexed=()):
        self.seen = []
        self.indexed = set(indexed)

    def is_up_to_date(self, path):
        return path.name in self.indexed

    def ingest(self, path):
        if path.suffix == ".exe":
            raise ValueError("unsupported file type: .exe")
        self.seen.append((path.name, path.read_text(encoding="utf-8")))
        return 3


def test_ingest_uploads_indexes_each_file():
    pipeline = FakePipeline()
    results = ingest_uploads(pipeline, [("a.txt", b"hello"), ("b.md", b"world")])
    assert [(r.name, r.chunks, r.error) for r in results] == [
        ("a.txt", 3, None),
        ("b.md", 3, None),
    ]
    assert pipeline.seen == [("a.txt", "hello"), ("b.md", "world")]


def test_unchanged_file_is_not_reindexed():
    pipeline = FakePipeline(indexed={"a.txt"})
    results = ingest_uploads(pipeline, [("a.txt", b"hello")])
    assert results[0].unchanged is True
    assert pipeline.seen == []


def test_bad_file_is_reported_without_stopping_others():
    results = ingest_uploads(FakePipeline(), [("bad.exe", b"x"), ("ok.txt", b"fine")])
    assert results[0].error == "unsupported file type: .exe"
    assert results[1].error is None


def test_upload_name_cannot_escape_the_temp_folder():
    pipeline = FakePipeline()
    results = ingest_uploads(pipeline, [("../../evil.txt", b"data")])
    assert results[0].name == "evil.txt"
    assert pipeline.seen == [("evil.txt", "data")]


def test_format_citation_with_and_without_page():
    assert format_citation(Citation(source="a.pdf", chunk_index=2, page=5)) == (
        "a.pdf · page 5 · chunk 2"
    )
    assert format_citation(Citation(source="a.md", chunk_index=0)) == "a.md · chunk 0"


def test_filename_with_path_is_reduced_to_its_basename():
    pipeline = FakePipeline()
    results = ingest_uploads(pipeline, [("../../evil/notes.txt", b"hi")])
    assert results[0].name == "notes.txt"
    assert pipeline.seen == [("notes.txt", "hi")]
