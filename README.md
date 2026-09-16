# AmazonHelp AI Support Agent

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## Hiver SDE Intern — Take-Home Assignment

An AI-powered customer-support agent built using real customer-support conversations from Twitter. The system classifies incoming customer messages, retrieves historically similar AmazonHelp interactions, generates a brand-grounded response, and decides whether the request should be automatically handled or escalated to a human.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 1. Problem

Customer-support messages are noisy, short, multilingual, and often ambiguous. A useful support agent should do more than classify an issue.

For each incoming customer message, this project performs four tasks:

1. Identify the customer's intent.
2. Retrieve similar historical support conversations.
3. Generate a response grounded in historical AmazonHelp responses.
4. Decide whether to auto-handle the request or escalate it to a human.

The goal is not to replace human support completely. Sensitive cases such as account compromise and selected financial/refund situations are deliberately routed to human support.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 2. Architecture

```text
Customer Message
       |
       v
Text Cleaning
       |
       v
Rule-Based Intent Detection
       |
       +--------------------+
       |                    |
       v                    v
High-confidence Rule    ML Classifier
       |                    |
       +---------+----------+
                 |
                 v
          Final Intent
                 |
                 v
      Historical Retrieval
                 |
                 v
           Top-3 Evidence
                 |
                 v
       Risk / Confidence
          Evaluation
          /         \
         /           \
      AUTO           HUMAN
        |              |
        v              v
    AI Reply       Escalation
```

The pipeline is designed so that classification, evidence retrieval, response generation, and escalation are separate stages.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 3. Dataset

The primary dataset is the **Customer Support on Twitter** dataset.

It contains real customer-support conversations between customers and brand support accounts.

The selected brand is **AmazonHelp**.

The raw dataset is not included in the repository because it is large and may contain sensitive customer information.

After processing, the development pipeline contained:

**151,745 usable AmazonHelp customer conversations.**

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 4. Why AmazonHelp?

AmazonHelp was selected after comparing major support accounts in the dataset.

AmazonHelp provided the largest number of customer messages linked to support responses among the candidate brands examined.

This provided a sufficiently large historical corpus for:

* intent discovery
* classifier development
* historical evidence retrieval
* response generation

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 5. Intent Taxonomy

The system uses 11 practical support intents:

| Intent                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `delivery_issue`      | Late, missing, delayed or tracking-related delivery problems |
| `order_issue`         | General order placement or order management issues           |
| `return_refund`       | Returns and refund requests                                  |
| `wrong_damaged_item`  | Wrong, damaged or broken items                               |
| `payment_billing`     | Payments, charges and billing problems                       |
| `prime_membership`    | Amazon Prime membership-related issues                       |
| `account_access`      | Login, password and account-access problems                  |
| `technical_issue`     | Amazon device, application or technical problems             |
| `cancellation_change` | Cancelling or changing an existing purchase                  |
| `customer_service`    | Requests to contact support or general support complaints    |
| `other`               | Unclear, unrelated or non-actionable requests                |

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 6. Classification Approach

### Baseline 1 — Majority Class

The majority baseline predicts the most frequent intent for every example.

This provides a trivial reference point and helps show the effect of class imbalance.

### Baseline 2 — TF-IDF + Logistic Regression

Customer messages are converted into TF-IDF features and classified using Logistic Regression.

This was selected as a lightweight classical machine-learning baseline because it is:

* fast to train
* easy to reproduce
* interpretable
* suitable for short text

### Hybrid AI Agent

The final classifier combines:

* high-confidence rule-based intent detection
* TF-IDF + Logistic Regression
* confidence and risk-based routing

Rules are used for recognizable cases such as account compromise and specific delivery patterns, while the ML classifier provides broader language coverage.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 7. Historical Evidence Retrieval

After intent classification, the system retrieves similar historical AmazonHelp customer conversations using TF-IDF similarity.

The top three historical interactions are selected as evidence.

The retrieved evidence contains:

* previous customer message
* historical AmazonHelp response
* similarity score
* ranking score

This provides brand-specific context for response generation rather than relying only on general LLM knowledge.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 8. Response Generation

The system uses an LLM through the Groq API when available.

The generation context contains:

* customer message
* predicted intent
* historical examples
* historical AmazonHelp responses

The prompt instructs the model to generate a concise support response grounded in the retrieved evidence and avoid unsupported claims.

The system also includes safe intent-specific fallback responses.

Fallback responses are used when:

* the API is unavailable
* the API request fails
* the generated response is too short

The system tracks the source of the final response:

```text
llm_generated
fallback_short_response
fallback_api_error
fallback_no_client
```

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 9. Human Escalation

The system does not rely only on classifier confidence.

A request can be escalated when it involves:

* account compromise or security concerns
* sensitive refund or financial situations
* low classification confidence
* insufficient historical evidence

For example:

```text
Customer:
Someone hacked my Amazon account.

Intent:
account_access

Confidence:
0.99

Decision:
HUMAN

Reason:
Security or account-compromise risk.
```

This demonstrates that a high-confidence prediction does not automatically mean the request should be handled without human review.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 10. Golden Evaluation Set

A manually labelled evaluation set containing **200 examples** was created.

Keyword-based labels were initially used only for candidate discovery and development.

The final `gold_intent` labels were manually reviewed.

Some customer messages are inherently ambiguous or context-dependent. These cases were retained rather than changing labels simply to improve the reported score.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 11. Evaluation Results

The final evaluation was performed on the 200-example golden set.

| System                       |   Accuracy |   Macro F1 | Weighted F1 |
| ---------------------------- | ---------: | ---------: | ----------: |
| Majority Class               |     35.50% |     0.0476 |      0.1860 |
| TF-IDF + Logistic Regression |     55.50% |     0.5564 |      0.5749 |
| **Hybrid AI Agent**          | **57.50%** | **0.5591** |  **0.5965** |

The hybrid agent improves accuracy by **2 percentage points** over the TF-IDF + Logistic Regression baseline.

The improvement is reported as modest rather than being presented as a large performance gain.

Macro-F1 is reported because accuracy alone can hide poor performance on smaller intent classes.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 12. Automation and Escalation Results

On the 200-example evaluation set:

* **Auto-handle:** 98 examples (49%)
* **Human escalation:** 102 examples (51%)
* **Total classification errors:** 85
* **High-confidence incorrect predictions:** 31

The escalation layer therefore acts as a separate risk-control mechanism rather than simply trusting classifier confidence.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 13. Failure Analysis

### Failure Mode 1 — Delivery vs Return/Refund Confusion

Customers frequently combine delivery problems with refund requests.

For example, a customer may report that an order has not arrived and immediately request their money back.

These messages can contain evidence for multiple intents, while the current system assigns one primary intent.

### Failure Mode 2 — Prime Multi-Intent Messages

Prime-related conversations frequently contain additional billing, cancellation, delivery or technical issues.

This creates ambiguity between:

```text
prime_membership
payment_billing
cancellation_change
delivery_issue
technical_issue
```

### Failure Mode 3 — Customer-Service Complaints

Some customers mainly express frustration with support while also mentioning another underlying problem.

This makes `customer_service` difficult to separate from the actual operational issue.

### Failure Mode 4 — Difficult `other` Cases

The `other` class contains praise, unclear messages, unrelated requests and context-dependent messages.

Because these messages do not share a consistent linguistic pattern, they are difficult for both rules and the ML classifier.

### Failure Mode 5 — High-Confidence Rule Errors

Explicit rules can produce very high confidence even when a keyword appears in a different context.

This was visible in the evaluation where some rule-based predictions were incorrect despite confidence values around 0.95–0.98.

This is an important limitation of deterministic rules.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 14. What Is Misleading About My Headline Number?

The headline classification accuracy is:

**57.50%**

However, this number should **not** be interpreted as saying that the complete support agent successfully solves 57.5% of real customer-support problems.

There are several reasons:

1. The evaluation contains only 200 manually labelled examples.
2. Some customer messages have genuinely ambiguous intent.
3. The original dataset is noisy and multilingual.
4. Intent classification and reply quality are separate tasks.
5. Accuracy can hide differences between frequent and less frequent intents.
6. The golden set is a sampled evaluation set and is not a statistically representative estimate of all Amazon customer-support traffic.
7. The system currently forces multi-intent messages into one primary intent.
8. Historical retrieval quality can vary depending on the wording of the incoming message.

For these reasons, the evaluation reports:

* accuracy
* macro-F1
* weighted F1
* failure analysis
* retrieval evidence
* escalation behaviour

rather than relying on accuracy alone.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 15. Reply Quality Evaluation

Reply quality is evaluated separately from intent classification.

The planned LLM-as-a-judge rubric evaluates:

* relevance
* groundedness
* helpfulness
* hallucination
* overall quality

Human ratings are used as a calibration reference because an automated judge should not automatically be considered ground truth.

During development, the Groq API rate limit affected large-scale LLM reply generation and judging.

Therefore, fallback responses are **not presented as independent evidence of LLM generation quality**.

This distinction is important because a reply produced by a deterministic fallback template should not be counted as an LLM-generated response.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 16. Limitations

The current system has several limitations:

* TF-IDF retrieval is lexical rather than deeply semantic.
* Multilingual conversations can be difficult for the current classifier.
* Multi-intent messages are currently assigned one primary intent.
* Rule-based classification can make confident mistakes.
* The golden evaluation set is relatively small.
* The escalation policy is heuristic.
* LLM generation depends on API availability and token limits.
* Historical responses may contain outdated support language.
* The current intent taxonomy may not cover every possible customer-support scenario.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 17. What I Would Do With One More Week

### 1. Better Semantic Retrieval

Replace TF-IDF retrieval with multilingual sentence embeddings and FAISS/vector search.

### 2. Multi-Intent Classification

Allow the system to detect multiple intents when a customer message contains more than one request.

### 3. Better Escalation Calibration

Use a validation set to calibrate confidence and escalation thresholds instead of relying mainly on heuristics.

### 4. Larger Golden Set

Expand the manually labelled evaluation set and stratify it by intent and difficulty.

### 5. Better Reply Evaluation

Collect a larger human-rated reply-quality dataset and measure agreement between human evaluators and the automated judge.

### 6. Multilingual Evaluation

Measure performance separately across languages and use multilingual embeddings for retrieval.

### 7. Better Privacy Protection

Add stronger PII detection and anonymization before storing or displaying evaluation examples.

### 8. Temporal Retrieval

Prefer more recent historical responses so outdated support instructions are less likely to influence generated responses.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 18. Reproducibility

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the support agent:

```bash
python src/support_agent.py
```

Run the evaluation:

```bash
python evaluation/run_evaluation.py
```

Run baseline comparison:

```bash
python evaluation/compare_baselines.py
```

Run failure analysis:

```bash
python src/failure_analysis.py
```

The raw dataset is not included in the repository.

For local reproduction, place the dataset at:

```text
data/twcs.csv
```

The Groq API key should be stored locally in:

```text
.env
```

and must never be committed to the repository.

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 19. Repository Structure

```text
hiver-sde-assignment/
│
├── data/
│   ├── golden/
│   └── processed/
│
├── evaluation/
│   ├── run_evaluation.py
│   ├── compare_baselines.py
│   ├── llm_judge.py
│   └── generate_reply_evaluation.py
│
├── src/
│   ├── support_agent.py
│   ├── intent_rules.py
│   ├── retrieval.py
│   ├── intent_aware_retrieval.py
│   ├── failure_analysis.py
│   └── ...
│
├── decision_log.md
├── requirements.txt
└── README.md
```

---

### Groq API Setup

The support agent uses the Groq API for LLM-based response generation.

Create a `.env` file in the project root and add:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Use your own Groq API key. Do not commit the `.env` file or expose the API key publicly. The `.env` file is excluded through `.gitignore`.

The application also includes fallback responses when LLM generation is unavailable.


## 20. Summary

This project demonstrates an end-to-end AI customer-support pipeline using noisy real-world customer-support data:

```text
Real Support Data
       |
       v
Intent Discovery
       |
       v
Manual Golden Set
       |
       v
Baseline Models
       |
       v
Hybrid Classification
       |
       v
Historical Evidence Retrieval
       |
       v
Grounded AI Response
       |
       v
Risk-Aware Escalation
       |
       v
Evaluation + Failure Analysis
```

The project focuses on measurable behaviour, reproducibility, historical grounding, risk-aware automation, and honest evaluation rather than maximizing a single headline metric.