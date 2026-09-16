# Decision Log

## 1. Brand Selection — AmazonHelp
Selected AmazonHelp because it had the largest number of customer messages linked to brand responses among the major support accounts examined, giving us a large historical support corpus.

## 2. Real Customer-Support Data
Used the Customer Support on Twitter dataset because the assignment focuses on noisy, real-world support conversations rather than synthetic examples.

## 3. Conversation Linking
Linked customer messages to AmazonHelp responses using Twitter response IDs instead of simply filtering by author, because brand and customer tweets have different directions.

## 4. Intent Taxonomy
Defined 11 practical intents: delivery_issue, order_issue, return_refund, wrong_damaged_item, payment_billing, prime_membership, account_access, technical_issue, cancellation_change, customer_service, and other.

## 5. Keyword-Based Silver Labels
Used keyword rules only for development/intent discovery. They were not treated as ground truth because keyword overlap creates false positives.

## 6. Golden Evaluation Set
Created a manually labelled 200-example golden set to provide an independent evaluation of the classifier.

## 7. Majority Baseline
Included a majority-class classifier to establish a trivial baseline and expose the effect of class imbalance.

## 8. TF-IDF + Logistic Regression
Used TF-IDF with Logistic Regression as the simple ML baseline because it is lightweight, interpretable, and fast.

## 9. Hybrid Rule + ML Classifier
Combined high-confidence rules with ML because certain support cases, such as account compromise and delivery issues, have recognizable patterns while ML provides broader coverage.

## 10. Historical Evidence Retrieval
Retrieved similar historical AmazonHelp conversations so generated replies could be grounded in how the brand previously handled similar cases.

## 11. Top-3 Evidence
Used the top three historical examples to provide enough brand-specific context while keeping the generation prompt compact.

## 12. Risk-Aware Escalation
Escalated security/account-compromise and selected sensitive financial/refund cases because confidence alone should not determine whether automation is appropriate.

## 13. Macro-F1
Reported macro-F1 alongside accuracy and weighted F1 because accuracy can hide poor performance on smaller intent classes.

## 14. LLM-as-a-Judge
Designed a separate reply-quality evaluation using relevance, groundedness, helpfulness, hallucination, and overall quality because intent accuracy does not measure response quality.

## 15. Human Calibration
Planned comparison between LLM judge scores and human ratings because an automated judge should not automatically be treated as ground truth.

