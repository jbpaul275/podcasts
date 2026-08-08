Several works are attached. Read all of them and work out how they actually stand to each other.

Your job is not to summarise them and it is not to decide who is right. It is to locate the relationship precisely enough that a later stage can build an honest episode on it.

The thing you are guarding against: works that look like they disagree but are answering different questions. Different outcome variable, different population, different period, different definition of the contested term, different unit of analysis. This happens constantly and it is the single most common way a comparison goes wrong, because a confident adjudication of a disagreement that was never real is worse than no episode at all.

So be strict. Two works conflict only if there is a claim one asserts and the other denies, about the same thing, measured comparably.

Return strict JSON, no code fences:

{
  "question": "the one question these works can be read as answering, in a sentence",
  "papers": [
    {
      "title": "the work's title as it appears in the PDF",
      "claim": "its answer to that question, put as strongly as its own author would put it",
      "construct": "what it actually measures or argues about",
      "population": "who or what it covers",
      "period": "when the evidence is from, or the period it argues about"
    }
  ],
  "commensurable": true,
  "why": "why these are or are not answering the same question — cite the specific mismatch if there is one",
  "crux": "the single thing a real disagreement rests on: if the sides agreed here, would the disagreement survive? empty string if they do not really disagree",
  "relation": "conflict | convergent | extension",
  "relation_why": "one sentence on why that is the right description"
}

Rules:
- One entry in `papers` per attached work, in the order they were attached.
- `commensurable` is false whenever the works are not really addressing the same claim. Say so plainly in `why`; that is a useful finding, not a failure.
- `relation` is `conflict` when they give incompatible answers, `convergent` when they reach the same conclusion by separate routes, `extension` when a later work takes up something an earlier one left open. If they are not commensurable, still choose the closest description and explain the mismatch in `why`.
- Every claim you attribute to a work must be in that work. Do not import what you know about it from elsewhere, and do not invent quotations.
- Put the strongest version of each position, including the one you find least convincing. A later stage decides the verdict and it cannot do that on a straw man.
