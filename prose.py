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


def author_credit(authors: list[str], max_named: int = 3) -> str:
    if not authors:
        return "an uncredited author"
    if len(authors) > max_named:
        return f"{authors[0]} et al."
    if len(authors) == 1:
        return authors[0]
    return ", ".join(authors[:-1]) + " and " + authors[-1]
