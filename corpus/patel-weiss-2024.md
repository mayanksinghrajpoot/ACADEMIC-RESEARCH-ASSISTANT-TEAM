# Repeated Passages Inflate Confidence Without New Evidence

Authors: Rohan Patel and Miriam Weiss. Retrieval Duplicates and Claim Confidence, 2024.

The index sometimes contained the same sentence twice: once in an abstract and once in a later citation of that abstract. Hybrid search returned both, and the summary treated the pair as two independent sources. Readers then saw a claim with two passage ids and took the repetition as agreement.

We marked near-duplicate passages before retrieval and kept only the earliest one. Supported claims that had cited both copies lost the second id. The text of those claims did not change, and the quote still matched the remaining passage. What changed was the appearance of corroboration. In a reader study, confidence ratings dropped when the duplicate id was removed, even though the evidence sentence was the same.

Two passage ids are not two findings when the text is the same sentence. A summary should not present a repeated chunk as a second source.
