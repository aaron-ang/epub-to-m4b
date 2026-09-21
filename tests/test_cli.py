from epub_to_m4b.cli import build_parser


def test_parser_builds() -> None:
    assert build_parser().prog == "epub-to-m4b"
