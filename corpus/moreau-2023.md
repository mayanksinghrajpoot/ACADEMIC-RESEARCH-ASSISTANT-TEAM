# Recall Limits of Retrieval for Scientific Question Answering

Authors: Camille Moreau. Notes on Research Infrastructure, 2023.

Retrieval quality sets a ceiling on grounded scientific answers. We measured recall of gold evidence sentences for questions about catalysis, using a corpus of short papers indexed by overlapping chunks. When recall at the cutoff used by the reader was below one half, unsupported claims returned at roughly the same rate as a model that saw no passages at all.

The failure mode was consistent. The model treated the retrieved neighborhood as if it were the whole literature and answered with a mechanism that the missed paper had qualified. Raising the cutoff recovered some of those sentences and cut unsupported claims, at the cost of a longer prompt. A hybrid of lexical overlap and embedding similarity was more stable on this small corpus than embeddings alone, because several gold sentences shared rare technical terms with the question and sat near each other in wording rather than in a broad semantic cluster.

Low recall does not merely omit detail. It reintroduces unsupported claims by letting the model speak for papers that never reached the prompt. Any pipeline that keeps a claim should be able to show the retrieved passage that entails it, and should treat a thin retrieved set as a reason to say less.
