"""Citation counts from OpenAlex.

Deliberately not asked of the language model. A citation count is not printed
in the paper, so a model has nothing to read it off and would supply a
plausible number instead -- and here that number would drive the sort order of
the public site. Either it comes from something that actually knows, or it
stays empty and a person types it in.

OpenAlex is free, needs no key, and asks only for a contact address in the
User-Agent so they can get in touch about traffic. Lookup is by DOI where the
paper printed one, which is exact; the title search fallback is not, so it is
held to a stricter match.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("paperpod.citations")

API = "https://api.openalex.org/works"
SOURCE = "openalex"
TIMEOUT_S = 15


def normalize_doi(raw: str | None) -> str | None:
    """Strip the prefixes people paste around DOIs. Returns None if it does not
    look like one at all, rather than sending nonsense to the API."""
    if not raw:
        return None
    doi = str(raw).strip()
    doi = re.sub(r"^\s*(?:doi:\s*)?(?:https?://(?:dx\.)?doi\.org/)?", "", doi, flags=re.I)
    doi = doi.strip().rstrip(".").strip()
    return doi if re.match(r"^10\.\d{4,9}/\S+$", doi) else None


def _get(url: str, cfg: dict) -> dict | None:
    """One GET returning parsed JSON, or None. Never raises: a citation count
    is a nice-to-have and must not be able to fail an episode."""
    mailto = (cfg.get("site", {}) or {}).get("contact_email", "")
    req = urllib.request.Request(url, headers={
        # OpenAlex asks for a contact address; supplying one keeps us in their
        # faster pool and out of the anonymous bucket.
        "User-Agent": f"paperpod/1.0 (mailto:{mailto})" if mailto else "paperpod/1.0",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            log.warning("citation lookup failed (%s) for %s", e.code, url)
        return None
    except Exception as e:
        log.warning("citation lookup failed (%s) for %s", type(e).__name__, url)
        return None
    return data if isinstance(data, dict) else None


def _count(work: dict) -> int | None:
    n = work.get("cited_by_count")
    return int(n) if isinstance(n, int) and n >= 0 else None


def _titles_match(a: str, b: str) -> bool:
    """Title search returns near-misses, and a near-miss here would attach some
    other paper's citation count to this episode. Only an exact match on
    letters and digits counts."""
    norm = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").casefold())
    return bool(norm(a)) and norm(a) == norm(b)


def lookup(doi: str | None, title: str | None, cfg: dict) -> tuple[int, str] | None:
    """(count, source) for a paper, or None if it cannot be established."""
    clean = normalize_doi(doi)
    if clean:
        work = _get(f"{API}/doi:{urllib.parse.quote(clean, safe='/.')}", cfg)
        n = _count(work) if work else None
        if n is not None:
            return n, SOURCE
        log.info("no OpenAlex record for doi %s", clean)

    if not (title or "").strip():
        return None
    # No DOI printed, so fall back to the title -- but only accept an exact
    # match, since a plausible-looking wrong paper is worse than no number.
    url = f"{API}?{urllib.parse.urlencode({'search': title, 'per-page': 5})}"
    data = _get(url, cfg)
    for work in (data or {}).get("results") or []:
        if not isinstance(work, dict):
            continue
        found = work.get("display_name") or work.get("title") or ""
        if _titles_match(title, found):
            n = _count(work)
            if n is not None:
                return n, SOURCE
    log.info("no exact OpenAlex title match for %r", (title or "")[:80])
    return None
