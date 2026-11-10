# Tables Hide Evidence from Sentence Retrieval

Authors: Amina Diallo. Workshop on Scientific Document Structure, 2025.

Several of the source papers stated their main numbers in a table and used the surrounding prose only to point at that table. Sentence retrieval returned the pointer and missed the cell. Claims about those numbers were then marked unsupported, or the model filled the number from memory.

We split each table row into a passage of its own, with the column names copied into the passage text, and we indexed those passages beside the prose. On the same questions, the unsupported-claim rate for numeric claims fell from 22 percent to 8 percent. Prose-only claims barely moved. The gain was confined to questions whose answer was a cell rather than a sentence.

Sentence retrieval is blind to evidence that lives in a table. Turning rows into passages made those numbers citable. It did not improve questions that were already answered by a sentence in the abstract.
