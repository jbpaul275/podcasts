Extract bibliographic metadata from the attached academic paper.

Respond with strict JSON only — no prose, no markdown, no code fences. Use exactly this shape:

{"title": "...", "authors": ["...", "..."], "year": 2020, "abstract": "...", "summary": "...", "venue_or_series": "..."}

Rules:
- "title": the paper's full title as printed.
- "authors": every author, in order, as "Firstname Lastname". Empty array if truly absent.
- "year": publication or working-paper year as an integer, or null if not determinable.
- "abstract": the paper's own abstract verbatim. If there is no labeled abstract, write a one-paragraph neutral summary of the introduction and set it anyway.
- "summary": one or two sentences, maximum 45 words, for a listing page. Plain language a curious non-specialist would understand — no jargon, no methodology names, no citation. Say what question the paper asks and what it found. Write it as a teaser, not an abstract: lead with the finding or the stake, never with "This paper examines".
- "venue_or_series": journal, conference, or working-paper series (e.g. "NBER Working Paper"), or null.
- Output nothing before or after the JSON object.
