# Retrieval Grounding and Unsupported Answers in Scientific Question Answering

Authors: Lina Chen and Ada Okonkwo. Workshop on Scientific Language Systems, 2023.

Scientific question answering often sounds fluent while citing nothing the reader can check. We studied a closed set of materials-science abstracts and asked whether retrieval-augmented generation reduces unsupported claims when the needed passage is actually present in the index. Annotators marked a claim unsupported when no retrieved sentence entailed it.

On questions whose evidence sentence was among the top retrieved passages, unsupported claims fell from 28 percent with a parametric baseline to 11 percent with retrieval. The gain disappeared when the index missed the relevant abstract: the model then filled the gap with a plausible mechanism that no passage stated. We therefore treat retrieval as a conditional benefit. It lowers unsupported answers only when the supporting sentence is retrieved and shown to the model.

The practical recommendation is narrow. Build the index from the paper set the question is allowed to use, keep passage text intact rather than compressing it into a keyword list, and refuse an answer when retrieval returns nothing. In that empty case the honest output is that the corpus does not contain support, not a fluent guess. Our count of 11 percent is the rate of unsupported claims after a successful retrieval, not a claim that retrieval removes every error.
