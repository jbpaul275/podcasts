"""Turning stored paper metadata into readable English.

Shared between the web layer and the pipeline: the attribution line under an
episode, the credit in the feed, and the spoken disclosure at the top of every
episode all have to name the same paper the same way. Keeping the rules here
means fixing "SMITH ET AL" once fixes it everywhere.
"""

SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in",
               "nor", "of", "on", "or", "the", "to", "via", "with"}

# Acronyms that must survive title-casing an all-caps string. Without these,
# "NBER WORKING PAPER SERIES" reads back as "Nber Working Paper Series".
ACRONYMS = {
    "NBER", "IZA", "CEPR", "SSRN", "IMF", "OECD", "ECB", "BLS", "BEA", "IRS",
    "FDA", "EPA", "CDC", "WHO", "UN", "EU", "US", "USA", "UK", "MIT", "UCLA",
    "NYU", "LSE", "AER", "QJE", "JPE", "JEL", "PNAS", "GDP", "GNP", "CPI",
    "RCT", "RCTS", "OLS", "IV", "GMM", "AI", "ML", "LLM", "COVID", "HIV",
    "CEO", "CFO", "GPS", "STEM", "PISA", "NAEP", "SAT", "GED", "K-12",
}


def decaps(text: str) -> str:
    """NBER and many journals print titles in all caps. Left alone they shout
    on the page, so title-case them — but only when they really are all caps,
    so deliberate capitalization (RCT, GDP) in mixed-case titles survives."""
    if not text or not text.isupper():
        return text
    out = []
    for i, raw in enumerate(text.split()):
        if raw.strip(".,:;()").upper() in ACRONYMS:
            out.append(raw)          # already correctly capitalized
            continue
        w = raw.lower()
        out.append(w if (w in SMALL_WORDS and i) else w[:1].upper() + w[1:])
    return " ".join(out)


def _has_upper(word: str) -> bool:
    return any(c.isupper() for c in word)


def title_case(text: str) -> str:
    """Title Case, applied conservatively.

    Asking the model for a capitalization style gets you that style most of the
    time, which is exactly the failure here: one title in sentence case next to
    one in title case reads as sloppiness rather than variety. So the rule is
    enforced after the fact instead of only requested.

    Conservative in one specific way: a word that already contains a capital is
    left completely alone. Blindly upper-casing first letters turns "iPhone"
    into "IPhone" and "eBay" into "EBay", and those are exactly the words a
    reader notices. Only all-lowercase words are touched, which can add
    capitals but never move one.
    """
    words = text.split()
    out = []
    for i, word in enumerate(words):
        last = i == len(words) - 1
        core = word.strip(".,:;!?()[]'\"“”‘’").casefold()
        if 0 < i and not last and core in SMALL_WORDS and not _has_upper(word):
            out.append(word.casefold())
        elif _has_upper(word):
            out.append(word)          # iPhone, GDP, McKinsey — already right
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def author_credit(authors: list[str], max_named: int = 3) -> str:
    if not authors:
        return "an uncredited author"
    if len(authors) > max_named:
        return f"{authors[0]} et al."
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + " and " + authors[-1]
