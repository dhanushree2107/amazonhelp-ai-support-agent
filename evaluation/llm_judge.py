import os
import sys
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = Groq(api_key=GROQ_API_KEY)

JUDGE_MODEL = "openai/gpt-oss-120b"


# ============================================================
# JUDGE RUBRIC
# ============================================================

JUDGE_PROMPT = """
You are an evaluation judge for an AI customer-support agent.

Evaluate the candidate support reply using ONLY:
1. The customer's message
2. The historical support evidence
3. The candidate reply

Do not assume facts that are not present in the evidence.

Score the reply on five dimensions.

1. RELEVANCE
5 = Directly addresses the customer's actual problem.
4 = Mostly addresses the problem with only minor omissions.
3 = Partially addresses the problem.
2 = Mostly misses the actual problem.
1 = Completely irrelevant.

2. GROUNDEDNESS
5 = Every important claim/action is supported by the historical evidence.
4 = Mostly grounded with only minor unsupported wording.
3 = Some unsupported claims, but generally grounded.
2 = Several unsupported claims.
1 = Clearly invents facts, actions, policies, or outcomes.

3. HELPFULNESS
5 = Gives a clear, useful next step appropriate for the customer.
4 = Helpful but could be more actionable.
3 = Somewhat useful but incomplete.
2 = Very limited help.
1 = Provides no meaningful help.

4. HALLUCINATION
5 = No unsupported factual claims.
4 = Very minor unsupported wording.
3 = Some questionable unsupported claims.
2 = Significant unsupported claims.
1 = Major fabricated claims or promises.

5. OVERALL
5 = Excellent support response.
4 = Good response with minor issues.
3 = Acceptable but incomplete.
2 = Poor response.
1 = Unacceptable response.

IMPORTANT:
- Do not reward a reply simply because it sounds polite.
- Do not assume that a generic customer-service response is correct.
- Historical evidence is the grounding source.
- If the evidence does not support a specific action, do not give full groundedness.
- Return ONLY the JSON object.
- Do not use markdown.
- Do not include any text before or after the JSON.

Required JSON schema:

{
  "relevance": 1,
  "groundedness": 1,
  "helpfulness": 1,
  "hallucination": 1,
  "overall": 1,
  "reason": "short explanation"
}
"""


# ============================================================
# JSON PARSER
# ============================================================

REQUIRED_FIELDS = [
    "relevance",
    "groundedness",
    "helpfulness",
    "hallucination",
    "overall",
    "reason"
]


def parse_judge_json(content):

    content = content.strip()

    # Remove markdown fences if present
    if content.startswith("```"):

        lines = content.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Try direct JSON parsing
    try:

        result = json.loads(content)

    except json.JSONDecodeError:

        # Try extracting the first JSON object
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                "No valid JSON object found in judge response."
            )

        candidate = content[start:end + 1]

        try:

            result = json.loads(candidate)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"Malformed JSON returned by judge: {e}"
            )

    # Validate required fields
    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in result
    ]

    if missing:

        raise ValueError(
            f"Judge JSON missing fields: {missing}"
        )

    # Validate score ranges
    score_fields = [
        "relevance",
        "groundedness",
        "helpfulness",
        "hallucination",
        "overall"
    ]

    for field in score_fields:

        value = result[field]

        if not isinstance(value, (int, float)):

            raise ValueError(
                f"{field} must be numeric."
            )

        if not 1 <= value <= 5:

            raise ValueError(
                f"{field} must be between 1 and 5."
            )

    return result


# ============================================================
# JUDGE ONE RESPONSE
# ============================================================

def judge_response(customer_message, evidence, reply):

    evidence_text = ""

    for i, item in enumerate(evidence[:3], start=1):

        evidence_text += f"""
Historical Example {i}:
Customer: {item.get("customer_text", "")}
AmazonHelp Reply: {item.get("brand_text", "")}
Similarity: {item.get("similarity", 0):.3f}
"""

    # If no evidence is available, explicitly tell the judge.
    if not evidence_text:

        evidence_text = (
            "No historical support evidence was supplied "
            "for this calibration run."
        )

    prompt = f"""
{JUDGE_PROMPT}

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EVIDENCE:
{evidence_text}

CANDIDATE AI REPLY:
{reply}
"""

    last_error = None

    # Try up to two times
    for attempt in range(2):

        try:

            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
                max_completion_tokens=500,
                response_format={
                    "type": "json_object"
                }
            )

            content = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

            return parse_judge_json(content)

        except Exception as e:

            last_error = e

            if attempt == 0:
                continue

    raise last_error


# ============================================================
# MAIN
# ============================================================

def main():

    # Use the 20-example human calibration set.
    test_path = "evaluation/human_calibration_fixed.csv"

    test_df = pd.read_csv(
        test_path,
        dtype=str,
        keep_default_na=False
    )

    results = []

    print("=" * 70)
    print("LLM-AS-A-JUDGE TEST")
    print("=" * 70)

    print(
        f"Testing examples: {len(test_df)}"
    )

    print()

    for index, row in test_df.iterrows():

        print("-" * 70)

        print(
            f"Example {index + 1}/{len(test_df)}"
        )

        print(
            f"Customer: {row['customer_text'][:200]}"
        )

        try:

            # IMPORTANT:
            # Use the EXISTING reply from the calibration CSV.
            #
            # We do NOT call run_agent() here.
            # This prevents another support-agent generation
            # request and avoids wasting Groq generation tokens.
            reply = row["reply"]

            # Historical evidence is not stored in the
            # calibration CSV.
            #
            # Therefore, this calibration run evaluates
            # the stored candidate reply without regenerating
            # the response.
            evidence = []

            judge = judge_response(
                row["customer_text"],
                evidence,
                reply
            )

            result = {

                "customer_tweet_id":
                    row["customer_tweet_id"],

                "gold_intent":
                    row["gold_intent"],

                "predicted_intent":
                    row["predicted_intent"],

                "reply":
                    reply,

                "relevance":
                    judge["relevance"],

                "groundedness":
                    judge["groundedness"],

                "helpfulness":
                    judge["helpfulness"],

                "hallucination":
                    judge["hallucination"],

                "overall":
                    judge["overall"],

                "judge_reason":
                    judge["reason"]
            }

            results.append(result)

            print(
                f"Intent: {row['predicted_intent']}"
            )

            print(
                f"Reply: {reply[:200]}"
            )

            print(
                f"Relevance:     "
                f"{judge['relevance']}/5"
            )

            print(
                f"Groundedness:  "
                f"{judge['groundedness']}/5"
            )

            print(
                f"Helpfulness:   "
                f"{judge['helpfulness']}/5"
            )

            print(
                f"Hallucination: "
                f"{judge['hallucination']}/5"
            )

            print(
                f"Overall:       "
                f"{judge['overall']}/5"
            )

            print(
                f"Judge reason:  "
                f"{judge['reason']}"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = (
        "evaluation/llm_judge_test.csv"
    )

    pd.DataFrame(results).to_csv(
        output_path,
        index=False
    )

    print()

    print("=" * 70)
    print("JUDGE TEST COMPLETE")
    print("=" * 70)

    print(
        f"Successful evaluations: "
        f"{len(results)}"
    )

    print(
        f"Failed evaluations: "
        f"{len(test_df) - len(results)}"
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()