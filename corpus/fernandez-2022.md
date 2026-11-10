# A Parametric Baseline for Scientific Answers Before Retrieval

Authors: Mateo Fernandez. Notes on Closed-Book Question Answering, 2022.

Before any retrieval step was added, we measured what the same model asserted from parametric memory alone. The questions were scientific, the index was empty, and the prompt contained only the question. Annotators marked a claim unsupported when no sentence in the source abstracts entailed it.

On this closed-book setup the unsupported-claim rate was 34 percent. The answers were fluent, and many named a real paper, but the named paper often did not contain the sentence the claim required. Numbers were the weakest point: a result from one abstract was stated as if it were the result of another.

This baseline is the comparison point for later retrieval studies. A retrieval system that does not beat 34 percent has not shown a gain. A system that reports a lower rate must still show that the remaining claims quote a retrieved sentence, because a lower rate can also come from refusing to answer.
