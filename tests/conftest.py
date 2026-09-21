from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

CONTENT_OPF = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:title>Tiny Book</dc:title>
    <dc:creator opf:role="aut">Ada Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">urn:uuid:1234</dc:identifier>
    <meta name="cover" content="cover-img"/>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="cover-img" href="cover.jpg" media-type="image/jpeg"/>
    <item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx">
    <itemref idref="title"/>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>
"""

TOC_NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:1234"/></head>
  <docTitle><text>Tiny Book</text></docTitle>
  <navMap>
    <navPoint id="n1" playOrder="1">
      <navLabel><text>One: The Beginning</text></navLabel>
      <content src="ch1.xhtml"/>
    </navPoint>
    <navPoint id="n2" playOrder="2">
      <navLabel><text>Two: The End</text></navLabel>
      <content src="ch2.xhtml#top"/>
    </navPoint>
  </navMap>
</ncx>
"""


def _xhtml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )


LONG = " ".join(["Words fill the page and the narrator reads them aloud."] * 6)

DOCS = {
    "title.xhtml": _xhtml("<h1>Tiny Book</h1><p>Ada Author</p>"),
    "ch1.xhtml": _xhtml(f"<h1>Chapter One</h1><p>{LONG}</p><p>{LONG}</p>"),
    "ch2.xhtml": _xhtml(f'<h1 id="top">Chapter Two</h1><p>{LONG}</p>'),
}

COVER_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"


def build_epub(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", CONTENT_OPF)
        zf.writestr("OEBPS/toc.ncx", TOC_NCX)
        zf.writestr("OEBPS/cover.jpg", COVER_BYTES)
        for name, content in DOCS.items():
            zf.writestr(f"OEBPS/{name}", content)
    return path


@pytest.fixture
def tiny_epub(tmp_path: Path) -> Path:
    return build_epub(tmp_path / "tiny.epub")


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No test may read or write the real ~/.cache/epub-to-m4b.
    monkeypatch.setenv("E2M_CACHE_DIR", str(tmp_path / "e2m-cache"))
