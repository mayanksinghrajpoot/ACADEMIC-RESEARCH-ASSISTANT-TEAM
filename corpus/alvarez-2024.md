# When Evidence Is Split: Multi-hop Failure in Scientific Retrieval

Authors: Mateo Alvarez. Journal of Computational Literature, 2024.

Many scientific questions need two passages at once. A reader may have to join a measurement in one paper with a caveat in another. We built a small multi-hop set on battery electrolytes. Each question was written so that neither paper alone contained the full answer, and both papers were in the corpus.

Single-passage retrieval looked healthy. The top hit usually contained one half of the evidence. Answers were still often unsupported, because the model stated the joined conclusion without the second passage. In 42 of 80 questions the generated claim combined a real number from the first paper with a condition that the second paper had explicitly ruled out. The result was a sentence that cited a real quantity and an invented scope.

Splitting the query into sub-queries, one per hop, raised the chance that both passages appeared in the retrieved set from 36 percent to 71 percent. Even then, the summary step sometimes dropped the second passage and wrote a claim that only the first source could support. Multi-hop scientific questions remain fragile. Retrieval must surface every required passage, and the writer must not merge them into a claim that neither passage states on its own.
