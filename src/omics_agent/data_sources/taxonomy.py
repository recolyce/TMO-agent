"""Tiny built-in NCBI taxon map. Unknown species are not guessed."""

from __future__ import annotations

KNOWN_TAXA: dict[str, int] = {
    "homo sapiens": 9606,
    "human": 9606,
    "mus musculus": 10090,
    "mouse": 10090,
    "rattus norvegicus": 10116,
    "rat": 10116,
    "danio rerio": 7955,
    "zebrafish": 7955,
    "drosophila melanogaster": 7227,
    "saccharomyces cerevisiae": 4932,
    "arabidopsis thaliana": 3702,
    "caenorhabditis elegans": 6239,
}


def taxon_id_or_none(name: str | None) -> int | None:
    """Return a taxon ID only for a known scientific or common name."""

    if not name:
        return None
    return KNOWN_TAXA.get(name.strip().lower())
