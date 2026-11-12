from trustquery.ingestion.loader import load_pdf, load_text_file


def test_load_text_file():
    result = load_text_file("data/sample/access_control.txt")

    assert result["source"] == "access_control.txt"
    assert "CloudDesk reviews user access permissions every quarter." in result["text"]


def test_load_pdf():
    result = load_pdf("data/sample/access_control_policy.pdf")

    assert len(result) == 2

    assert result[0]["source"] == "access_control_policy.pdf"
    assert result[0]["page"] == 1
    assert "Multi-factor authentication" in result[0]["text"]

    assert result[1]["page"] == 2
    assert "reviews user access permissions every quarter" in result[1]["text"]