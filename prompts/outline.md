You are planning one episode of a two-host podcast that discusses a single academic paper for a curious lay audience. Read the attached paper and return a beat sheet.

You are not writing dialogue. You are deciding what this episode is made of, and therefore how long it is.

HOW LONG:
- The episode should be as long as the paper gives you material for, between $MIN_MINUTES and $MAX_MINUTES minutes. Do not aim for the middle.
- A short methods note with one clean result is an eight-minute episode and padding it out would be obvious. A paper with three experiments, a mechanism worth explaining and a real weakness is twenty-five minutes, and compressing it would throw away the interesting parts.
- Length is the consequence of your beats, not a target you write towards: plan what is worth saying, then let the total fall where it falls.
- Speaking rate is about $WORDS_PER_MINUTE words a minute.

THE ARC — every episode follows this order. Give each segment at least one beat; give a segment several when the paper earns it, and keep a segment brief when it does not.
1. Cold open — the stake, not the paper. A concrete question a non-specialist would care about.
2. Setup — what was believed or unknown before, and why the question is hard.
3. Identification — how the authors got leverage on it.
4. Findings — the results, with real magnitudes.
5. Pressure — where the result is weakest, specific to this paper.
6. Context — how it sits against other work.
7. So what — implications the paper does not itself claim.

FOR EACH BEAT:
- `segment` — one of the seven names above, exactly.
- `covers` — one sentence on what this beat does. Concrete, about this paper.
- `facts` — the specific numbers, names, comparisons or definitions the hosts must get in. Every one must come from the paper. This is the part that stops the script wandering, so be specific: "the effect is 2.8 percent, about a fifth of the raw gap" beats "the main result".
- `words` — how many words of dialogue this beat is worth. Most beats are 80 to 400. A beat worth fewer than 60 words is not a beat; fold it into its neighbour.

Return ONLY a JSON object, no code fence, no commentary:

{"minutes": 18, "why": "one sentence on why this paper is worth that long", "beats": [{"segment": "Cold open", "covers": "...", "facts": ["..."], "words": 160}]}
