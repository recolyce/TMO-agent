"""PubMed E-utilities and Europe PMC REST adapters.

All HTTP goes through :class:`HttpTransport`. CI injects a fake transport
and never opens a socket. Paper text is stored as untrusted metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from omics_agent.data_sources.http import Downloader, HttpTransport, UrllibTransport
from omics_agent.errors import LiteratureError
from omics_agent.schemas.ingest import DownloadPolicy

PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass(frozen=True)
class PaperHit:
    """One bibliographic hit. Fields may be empty; do not invent them."""

    source_name: str
    pmid: str | None
    doi: str | None
    title: str | None
    year: str | None
    abstract: str | None
    raw: dict[str, Any]


def pubmed_esearch_url(query: str, *, retmax: int) -> str:
    return f"{PUBMED_ESEARCH}?db=pubmed&term={quote(query)}&retmode=json&retmax={retmax}"


def pubmed_esummary_url(pmids: list[str]) -> str:
    return f"{PUBMED_ESUMMARY}?db=pubmed&id={','.join(pmids)}&retmode=json"


def europepmc_search_url(query: str, *, page_size: int) -> str:
    return f"{EUROPEPMC_SEARCH}?query={quote(query)}&format=json&pageSize={page_size}"


class PubMedAdapter:
    """NCBI E-utilities esearch + esummary."""

    name = "pubmed_eutils"

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def search(self, query: str, *, retmax: int = 5) -> list[PaperHit]:
        response = self.downloader.fetch_json(pubmed_esearch_url(query, retmax=retmax))
        if response.status >= 400:
            raise LiteratureError(
                f"PubMed esearch returned HTTP {response.status}.",
                how_to_fix="Retry later or inject a mock HttpTransport in tests.",
            )
        payload = _as_dict(response.body)
        ids = list((payload.get("esearchresult") or {}).get("idlist") or [])
        if not ids:
            return []
        return self.summarize(ids)

    def summarize(self, pmids: list[str]) -> list[PaperHit]:
        response = self.downloader.fetch_json(pubmed_esummary_url(pmids))
        if response.status >= 400:
            raise LiteratureError(
                f"PubMed esummary returned HTTP {response.status}.",
                how_to_fix="Retry later or inject a mock HttpTransport in tests.",
            )
        payload = _as_dict(response.body)
        result = payload.get("result") or {}
        hits: list[PaperHit] = []
        for pmid in pmids:
            rec = result.get(str(pmid))
            if not isinstance(rec, dict):
                continue
            doi = _doi_from_elocation(rec.get("elocationid"))
            hits.append(
                PaperHit(
                    source_name=self.name,
                    pmid=str(rec.get("uid") or pmid),
                    doi=doi,
                    title=rec.get("title") or None,
                    year=_year(rec.get("pubdate")),
                    abstract=None,
                    raw=rec,
                )
            )
        return hits


class EuropePmcAdapter:
    """Europe PMC REST search. Returns abstracts when the API includes them."""

    name = "europepmc"

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        policy: DownloadPolicy | None = None,
    ) -> None:
        self.downloader = Downloader(transport or UrllibTransport(), policy or DownloadPolicy())

    def search(self, query: str, *, page_size: int = 5) -> list[PaperHit]:
        response = self.downloader.fetch_json(europepmc_search_url(query, page_size=page_size))
        if response.status >= 400:
            raise LiteratureError(
                f"Europe PMC search returned HTTP {response.status}.",
                how_to_fix="Retry later or inject a mock HttpTransport in tests.",
            )
        payload = _as_dict(response.body)
        rows = ((payload.get("resultList") or {}).get("result")) or []
        hits: list[PaperHit] = []
        for rec in rows:
            if not isinstance(rec, dict):
                continue
            hits.append(
                PaperHit(
                    source_name=self.name,
                    pmid=str(rec["pmid"]) if rec.get("pmid") else None,
                    doi=str(rec["doi"]) if rec.get("doi") else None,
                    title=rec.get("title") or None,
                    year=str(rec["pubYear"]) if rec.get("pubYear") else None,
                    abstract=rec.get("abstractText") or None,
                    raw=rec,
                )
            )
        return hits


def _as_dict(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiteratureError(
            "Literature adapter received non-JSON bytes.",
            how_to_fix="The HTTP body is stored as metadata only; check the mock or the API.",
        ) from exc
    if not isinstance(payload, dict):
        raise LiteratureError(
            "Literature adapter JSON root is not an object.",
            how_to_fix="Check the PubMed / Europe PMC payload.",
        )
    return payload


def _doi_from_elocation(value: object) -> str | None:
    text = str(value or "")
    if "10." in text:
        start = text.find("10.")
        return text[start:].strip() or None
    return None


def _year(value: object) -> str | None:
    text = str(value or "").strip()
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return None
