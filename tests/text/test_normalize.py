"""Normalization: NeMo's English output for the forms books use."""

from __future__ import annotations

from pathlib import Path

import pytest

from epub_to_m4b.text.normalize import (
    nemo_cache_dir,
    nemo_normalizer,
    normalize,
    normalize_all,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        # years and decades
        ("in 1997 he moved", "in nineteen ninety seven he moved"),
        ("It's the year 2024.", "It's the year twenty twenty four."),
        ("the 1980s", "the nineteen eighties"),
        ("The 2000s and the 1910s.", "The two thousands and the nineteen tens."),
        (
            "The '60s and the 1800s differ; in the 80s, 3 out of 4 agreed.",
            "The sixties and the eighteen hundreds differ; in the eighties, three out of four"
            " agreed.",
        ),
        (
            "From 1914-1918 the war raged.",
            "From nineteen fourteen to nineteen eighteen the war raged.",
        ),
        # currency
        ("It cost $3.50.", "It cost three dollars fifty cents."),
        ("It's 20% of $1.2 billion.", "It's twenty percent of one point two billion dollars."),
        ("It costs \u00a35.50 or \u20ac3.", "It costs five pounds fifty pence or three euros."),
        # clock
        ("The meeting is at 18:00.", "The meeting is at eighteen o'clock."),
        ("At 3 a.m. he woke.", "At three AM he woke."),
        # ordinals
        ("He was 2nd and she was 21st.", "He was second and she was twenty first."),
        # decimals and cardinals
        ("It rained 2.5 inches.", "It rained two point five inches."),
        ("100,000,000 people.", "one hundred million people."),
        # titles
        ("Mr. Smith met Dr. Jones.", "mister Smith met doctor Jones."),
        ("Mrs. Brown at St. Paul's.", "misses Brown at Saint Paul's."),
        # sentences from the evaluation set
        (
            "In 1999 he paid $2,000 for 3.5 kg at 9:30 p.m. on March 3rd.",
            "In nineteen ninety nine he paid two thousand dollars for three point five kilograms"
            " at nine thirty PM on march third.",
        ),
        (
            "The 1990s saw 1,997 people; page 1066.",
            "The nineteen nineties saw one thousand nine hundred and ninety seven people; page ten"
            " sixty six.",
        ),
        (
            "He ran 26.2 miles in 3:05:12 and finished 2nd of 1,200.",
            "He ran twenty six point two miles in three hours five minutes and twelve seconds and"
            " finished second of one thousand two hundred.",
        ),
        (
            "The U.S. GDP grew 2.3% in Q3; NASA and the FBI disagreed.",
            "The US GDP grew two point three percent in Q three; NASA and the FBI disagreed.",
        ),
        # typographic punctuation passes through
        (
            "\u201cHello,\u201d she said. \u2018Hi\u2019 \u2014 he replied\u2026 then left.",
            "\u201cHello,\u201d she said. \u2018Hi\u2019 \u2014 he replied\u2026 then left.",
        ),
    ],
)
def test_nemo_speaks_written_forms(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Chapter IV", "Chapter IV"),
        (
            "Henry VIII married six times; World War II ended in 1945.",
            "Henry VIII married six times; World War two ended in nineteen forty five.",
        ),
        (
            "From 1914-1918, Dr. Smith lived on St. James St. No. 5.",
            "From nineteen fourteen to nineteen eighteen, doctor Smith lived on Saint James St."
            " No. five.",
        ),
        (
            "She was born c. 1850 and died ca. 1920, aged 70.",
            "She was born c. eighteen fifty and died ca. nineteen twenty, aged seventy.",
        ),
        (
            "Mt. Everest is 8,849 m (29,032 ft) high; temperatures hit -40\u00b0C.",
            "Mount Everest is eight thousand eight hundred and forty nine M (twenty nine thousand"
            " and thirty two feet) high; temperatures hit minus forty degrees Celsius.",
        ),
        ("self- pity", "self- pity"),
        ("He waited . . . and waited.", "He waited . . . and waited."),
    ],
)
def test_nemo_output_is_used_as_is(text: str, expected: str) -> None:
    # Forms NeMo leaves as written or reads other than a narrator would; no
    # rule of ours rewrites them.
    assert normalize(text) == expected


def test_one_normalizer_per_language() -> None:
    assert nemo_normalizer("en") is nemo_normalizer("en")


def test_normalize_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="xx"):
        normalize("hello", lang="xx")


def test_nemo_grammars_live_under_the_cache_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("E2M_NEMO_CACHE_DIR")
    monkeypatch.setenv("E2M_CACHE_DIR", str(tmp_path))
    assert nemo_cache_dir("en").parent.parent == tmp_path / "nemo"
    assert nemo_cache_dir("en").name == "en"


def test_nemo_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("E2M_NEMO_CACHE_DIR", str(tmp_path / "grammars"))
    assert nemo_cache_dir("de").parent.parent == tmp_path / "grammars"
    assert nemo_cache_dir("de").name == "de"


def test_normalize_all_keeps_order_across_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    # Three workers over five texts: chunks of two, two and one.
    monkeypatch.setattr("epub_to_m4b.text.normalize.os.process_cpu_count", lambda: 3)
    texts = [f"Room {n} has {n + 1} chairs." for n in range(5)]
    assert normalize_all(texts) == [normalize(text) for text in texts]


def test_normalize_all_of_nothing_is_empty() -> None:
    assert normalize_all([]) == []
