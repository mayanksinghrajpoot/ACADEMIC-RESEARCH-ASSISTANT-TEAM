# Citation Constraints Cut Unsupported Scientific Claims

Authors: Ada Okonkwo. Follow-up Study in Scientific Question Answering, 2025.

A follow-up to the 2023 retrieval study asked whether the model would use a passage if the output format made unused passages obvious. We kept the same scientific question-answering set and the same retrieved passages. The only change was a constraint: every retained claim had to name a retrieved passage identifier and include a quote copied from that passage.

Requiring each claim to name a retrieved passage identifier cut unsupported claims from 31 percent to 6 percent on the same scientific question-answering set. Claims that could not offer a quote present in the cited passage were dropped rather than rewritten into something smoother. The remaining errors were mostly scope slips, where the quote was real and the claim generalized past it. Those were caught when a checker compared the quote to the claim and rejected the generalization.

This finding conflicts with the report that retrieved passages leave the unsupported-claim rate unchanged because the model answers from parametric memory. In our run the rate changed only after the citation constraint. Retrieval without that constraint was not enough. The passages had to be named, quoted, and checked. Both positions should stay visible: one study finds no reduction from showing passages, and this study finds a reduction when unsupported claims are removed for lack of a quote.
