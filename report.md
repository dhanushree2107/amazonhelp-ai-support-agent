# AmazonHelp AI Support Agent
## Hiver SDE Intern — Take-Home Assignment

### 1. Executive Summary

This project builds an AI-assisted customer support agent for AmazonHelp using the Customer Support on Twitter dataset. The system classifies incoming customer messages into 11 support intents, retrieves historically similar AmazonHelp conversations, drafts a grounded response, and decides whether the case can be auto-handled or should be escalated to a human.

I evaluated the system on a manually labelled 200-example golden set. The hybrid system achieved 57.50% intent accuracy, compared with 55.50% for a TF-IDF + Logistic Regression baseline and 35.50% for a majority-class baseline.

The result shows that combining rules with a lightweight ML classifier provides a measurable improvement over the baselines. However, the 57.50% number should not be interpreted as production-ready performance. The evaluation set is relatively small, several examples are inherently context-dependent, and the current response-generation evaluation is limited by API quota.

---

### 2. Problem Framing

Customer-support messages are short, noisy and often contain multiple issues. A useful support agent therefore needs to do more than classify text.

For each incoming customer message, the system should:

1. Identify the customer's main support intent.
2. Retrieve similar historical customer-support interactions.
3. Draft a response based on previously observed support patterns.
4. Decide whether the case is safe to auto-handle or should be reviewed by a human.

The system focuses on AmazonHelp because it has a large number of linked customer-support interactions in the dataset, providing sufficient examples for both intent learning and historical retrieval.

#### What good means

A good system should:

- identify the customer's main issue;
- avoid confidently assigning unrelated intents;
- retrieve relevant historical examples;
- produce a useful and conservative response;
- escalate sensitive or uncertain cases;
- avoid unsupported claims.

#### What was not built

This prototype is not intended to replace human customer-support agents. It does not connect to Amazon's internal order, payment or account systems, and therefore cannot actually perform refunds, cancellations, account recovery or order changes.

---

### 3. System Approach

The pipeline is:

**Customer message → cleaning → intent classification → historical retrieval → response generation → risk-aware escalation**

The intent classifier combines two approaches:

- deterministic rules for high-signal patterns;
- TF-IDF + Logistic Regression for cases not confidently covered by rules.

Historical support messages are represented using TF-IDF and retrieved using cosine similarity. Retrieval is also made intent-aware by restricting candidate examples using the predicted intent.

The response layer uses an LLM when available and falls back to conservative templates when generation is unavailable or unsuitable.

The escalation layer sends higher-risk cases to a human, particularly where the message involves account security, fraud, weak historical evidence, or other situations where an automated response could be unsafe.

---

### 4. Intent Taxonomy

The final taxonomy contains 11 intents:

1. delivery_issue
2. order_issue
3. return_refund
4. wrong_damaged_item
5. payment_billing
6. prime_membership
7. account_access
8. technical_issue
9. cancellation_change
10. customer_service
11. other

The taxonomy was derived from patterns observed in AmazonHelp conversations. Keyword-based labels were used only as development/silver labels; they were not treated as ground truth.

A separate 200-example golden set was manually labelled for final evaluation.

---

### 5. Data and Evaluation

The original dataset contains approximately 3 million customer-support tweets. For this prototype, AmazonHelp conversations were linked using tweet-response relationships.

After cleaning and filtering, the AmazonHelp subset contained approximately 151,745 usable customer-support examples.

The evaluation golden set contains 200 manually labelled examples. It was designed to cover the 11-intent taxonomy while including difficult and ambiguous support messages.

The evaluation uses:

- Accuracy
- Macro F1
- Weighted F1
- Auto-handle vs human escalation counts
- Error analysis
- High-confidence error analysis

---

### 6. Results

| System | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority Class | 35.50% | 0.0476 | 0.1860 |
| TF-IDF + Logistic Regression | 55.50% | 0.5564 | 0.5749 |
| Hybrid AI Agent | 57.50% | 0.5591 | 0.5965 |

The hybrid agent improves accuracy by 2.0 percentage points over the TF-IDF baseline and by 22.0 percentage points over the majority baseline.

On the 200-example golden set:

- Correct predictions: 115
- Incorrect predictions: 85
- High-confidence incorrect predictions: 31
- Auto-handle decisions: 98
- Human escalations: 102

The results indicate that the hybrid approach adds value, but also expose substantial remaining classification ambiguity.

---

### 7. Top Five Failure Modes

#### Failure Mode 1 — Multi-intent messages

Some customer messages contain several issues at once.

Example:

> “Why do I pay for a Prime account if my package still gets delayed 3-5 days every time I order something?”

This contains both a Prime-related complaint and a delivery complaint. The classifier may focus on the strongest keyword rather than the customer's primary issue.

**Hypothesis:** A single-label classifier is insufficient for multi-intent support messages.

**Potential improvement:** Use hierarchical or multi-label intent classification.

---

#### Failure Mode 2 — Delivery vs refund/return confusion

Messages about late or missing deliveries frequently mention refunds.

Example:

> “I ordered an item that didn't arrive. Seller refunded the money. The item finally arrived today. I don't know how to pay the seller now.”

The message contains delivery history, a completed refund and a payment-related question. A simple keyword-based system can assign the wrong intent.

**Hypothesis:** The classifier is sensitive to words such as “refund” and “arrived” without sufficiently modelling the customer's final request.

**Potential improvement:** Explicitly extract the customer's requested action before assigning the final intent.

---

#### Failure Mode 3 — Prime membership overlaps with unrelated issues

Prime is frequently mentioned in messages whose actual problem is delivery, payment or content.

Example:

> “Why not add Kannada movies into Prime Video?”

The presence of “Prime” can pull the classifier toward `prime_membership`, even though the underlying request concerns content availability.

**Hypothesis:** Product/entity mentions are being confused with intent.

**Potential improvement:** Separate entity detection from intent classification.

---

#### Failure Mode 4 — Short and context-dependent messages

Some messages provide very little information.

Examples include:

> “Still no information”

and

> “Your site broken”

These messages can require the previous conversation context to classify correctly.

**Hypothesis:** Tweet-level classification loses conversational context.

**Potential improvement:** Classify the complete conversation thread rather than an isolated customer tweet.

---

#### Failure Mode 5 — Broad complaints and sarcasm

Customer-service messages often contain sarcasm, frustration or broad complaints rather than explicit requests.

Example:

> “Representatives from ur side are dumb and know nothing. Only lying and misguiding”

The message expresses dissatisfaction but provides little operational information.

**Hypothesis:** Intent categories such as `customer_service` and `other` overlap when the customer does not state a concrete action.

**Potential improvement:** Add sentiment/frustration and requested-action signals while keeping them separate from the main intent.

---

### 8. What Is Misleading About My Headline Number?

The headline accuracy of **57.50%** is useful for comparing this prototype with the baselines, but it is not a production-quality estimate.

There are several reasons:

1. The golden set contains only 200 examples.
2. Some messages are genuinely ambiguous without conversation context.
3. The taxonomy forces every example into one primary intent even when multiple intents are present.
4. The dataset is historical Twitter support data and may not represent modern customer-support traffic.
5. The hybrid system contains deterministic rules tuned using observed examples, so the result should not be interpreted as an unbiased estimate of performance on every possible customer message.
6. A correct intent prediction does not automatically mean the generated reply is useful.
7. The LLM-as-a-judge calibration was completed on a 20-example matched human-reviewed sample, with agreement measured using exact agreement and Cohen’s kappa.

Therefore, 57.50% should be interpreted as a **prototype benchmark on this 200-example golden set**, not as an expected production accuracy.

---

### 9. Response Quality and LLM-as-a-Judge

An LLM-as-a-judge harness was implemented to evaluate:

- relevance;
- groundedness;
- helpfulness;
- hallucination;
- overall response quality.

A 20-example human calibration set was prepared using the same response examples, and the LLM judge successfully evaluated all 20 examples. Human–LLM agreement was then calculated on the 20 matched examples. Exact agreement was 25% for relevance, 10% for groundedness, 55% for helpfulness, 70% for hallucination, and 35% for overall quality. Cohen’s kappa values were 0.060, 0.058, 0.368, 0.000, and 0.169 respectively. The low agreement indicates that the LLM judge should be treated as a supplementary evaluation tool rather than a replacement for human review.

Therefore, no fabricated agreement statistic is reported.

The next evaluation run should complete the 20 matched examples and calculate agreement on the same examples using the same rubric.

---

### 10. Automation and Escalation

The agent produced:

- 98/200 auto-handle decisions;
- 102/200 human-escalation decisions.

The escalation mechanism is intentionally conservative. Cases involving account security, fraud, weak evidence or higher-risk customer situations are more likely to be routed to a human.

This is important because an imperfect classifier should not automatically convert every prediction into an irreversible customer action.

---

### 11. What I Would Build With One More Week

If given another week, I would focus on five improvements.

**1. Conversation-level classification**

Instead of classifying only one tweet, include the previous customer-support turns. This would reduce ambiguity in short messages.

**2. Multi-intent detection**

Allow a message to contain more than one intent and separately identify the customer's requested action.

**3. Better retrieval**

Improve retrieval using intent-aware semantic embeddings and reranking rather than relying mainly on TF-IDF similarity.

**4. Stronger response evaluation**

Extend the human–LLM calibration to a larger independently reviewed matched sample and refine the rubric based on the observed disagreements.

**5. Safer production architecture**

Add structured confidence thresholds, PII redaction, audit logs, rate limiting and explicit human approval for sensitive actions.

---

### 12. Limitations

This prototype has several limitations:

- The original dataset is historical Twitter support data.
- The golden evaluation set is relatively small.
- Some intent boundaries are inherently ambiguous.
- The system does not have access to real Amazon order/account systems.
- The LLM judge was calibrated on 20 matched examples; the observed low agreement shows that judge scores should be interpreted cautiously and validated with additional human review.
- Retrieval quality depends on the quality and coverage of historical examples.
- Rule-based classification can be brittle when customer wording changes.

These limitations are important when interpreting the benchmark results.

---

### 13. Conclusion

The project demonstrates a practical AI-assisted customer-support workflow rather than treating the task as classification alone.

The hybrid classifier achieved **57.50% accuracy** on the 200-example golden set, compared with **55.50%** for TF-IDF + Logistic Regression and **35.50%** for the majority baseline.

The main lesson from the evaluation is that support automation is not only a classification problem. Multi-intent messages, missing context, overlapping product and intent terms, and broad customer complaints remain important challenges.

The current system is therefore best viewed as a prototype for **AI-assisted support with human escalation**, rather than a fully autonomous customer-support replacement.
