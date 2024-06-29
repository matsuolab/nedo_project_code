We are currently developing a dataset to enhance the recognition of entailment relations (RTE) for a Japanese large language model that we are building particularly tailored to the Japanese business context. The criteria and methodology for recognising entailment relations are as follows:

### Criteria and Methodology for Recognising Entailment Relations ###

- Select 'entailment' and respond with *entailment* only, if a hypothesis can be inferred as true from the premise.
- Select 'contradiction' and respond with *contradiction* only, if a hypothesis can be inferred as false from the premise.
- Select 'neutral' and respond with *neutral* only, if the truth or falsehood of a hypothesis cannot be inferred from the premise.

Please incorporate your new dataset that include example sentences typically and frequently used in the business domain by strictly following the requirements below.

### Requirements ###

- Use the JSONL format with keys for `id`, `entailment_instance`, `premise`, `hypothesis`, and `label`.
- Create at least fifteen pairs of `id` and `entailment_instance` in total, which means you MUST newly add five pairs. More ideas are appreciated, thus 16 or 17 pairs in total are preferable, and over 20 pairs are even more desirable.
- For the same `id`, prepare one common `premise` and four `hypothesis`es, each with `label`s of `entailment`, `contradict`, `neutral`, and `neutral`, respectively, to create instances of `entailment`. For instance, from a premise *A*, create hypotheses *AE* with the label `entailment`, *AC* with `contradict`, and *AN1* and *AN2* with `neutral`.
- Each `premise` and `hypothesis` **MUST contain only one Japanese sentence**. More than one sentence is prohibited.
- **Your entire response MUST be in JSONL format**. DO NOT include any text or comment other than the dataset, such as "Here are examples of the new dataset...".
- Imagine various and distinct business scenarios such as management, sales, planning, engineering, customer support, accounting, legal, and human resources. Utilise a diverse range of verbs, nouns, and adjectives to write realistic business scenes in the sentences of `premise` and `hypothesis`.

**Please continue your response in Japanese.**

### YOUR ANSWER ###

$FewShotExamples