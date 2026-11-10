# Domain Shift: Retrieval Helps Less Outside the Training Field

Authors: Kenji Nakamura. Cross-Domain Scientific Question Answering, 2024.

The retrieval studies above used scientific questions close to the abstracts in the index. We kept the same model and the same citation constraint, and we replaced half of the questions with ones whose evidence lived in a neighboring field: clinical measurement rather than materials science. The index still contained only the original abstracts.

On in-domain questions, requiring a passage id and a quote cut unsupported claims from 31 percent to 9 percent. On the shifted questions, the same constraint cut the rate only from 33 percent to 24 percent. The model often quoted a materials sentence that shared a method word and then answered the clinical question from parametric memory.

Retrieval reduces unsupported claims when the index can supply the evidence sentence. A citation constraint does not repair a corpus that never contained the answer. Reports of a large reduction should name the domain of both the questions and the papers.
