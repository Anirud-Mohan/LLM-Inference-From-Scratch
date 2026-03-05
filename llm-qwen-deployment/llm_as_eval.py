"""LLM-as-a-judge evaluation for the Qwen moral story endpoint.

Sends each prompt from eval_data.json to the deployed vLLM endpoint on
a RunPod Pod (OpenAI-compatible), then asks an external judge LLM
to score the response on four dimensions tailored to children's moral stories.

Usage:
    # Set in .env:
    #   POD_1_URL=https://<pod-id>-8000.proxy.runpod.net
    #   MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
    #   JUDGE_API_KEY=...
    #   JUDGE_API_BASE=https://api.openai.com/v1
    #   JUDGE_MODEL=gpt-4o-mini

    python llm_as_eval.py
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of children's stories. Given a story prompt, a \
reference story, and a candidate story produced by a language model, rate the \
candidate on four dimensions. Each dimension is scored from 0.0 to 1.0.

Dimensions:
1. **Moral clarity** — Does the story convey a clear, identifiable moral lesson?
   1.0 = crystal-clear lesson; 0.0 = no discernible moral.
2. **Age-appropriateness** — Is the language, tone, and content suitable for \
children aged 4-10?
   1.0 = perfectly child-friendly; 0.0 = inappropriate vocabulary or themes.
3. **Narrative coherence** — Does the story have a beginning, middle, and end? \
Is it logically consistent?
   1.0 = complete, well-structured story; 0.0 = incoherent fragments.
4. **Relevance** — Does the story match the requested character, moral value, \
and setting from the prompt?
   1.0 = fully matches all elements; 0.0 = ignores the prompt entirely.

Respond with ONLY a JSON object in this exact format:
{
  "moral_clarity": <float>,
  "age_appropriateness": <float>,
  "narrative_coherence": <float>,
  "relevance": <float>,
  "reason": "<one sentence summary>"
}"""

DIMENSIONS = ["moral_clarity", "age_appropriateness", "narrative_coherence", "relevance"]


def load_eval_data(path: str = "eval_data.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def query_deployed_model(
    client: OpenAI, model: str, prompt: str, max_tokens: int = 150
) -> tuple[str, float]:
    """Send prompt to the deployed vLLM endpoint via OpenAI-compatible API."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency


def judge_story(
    client: OpenAI,
    model: str,
    prompt: str,
    reference: str,
    candidate: str,
) -> dict:
    """Ask the judge LLM to score a candidate story."""
    user_msg = (
        f"Prompt: {prompt}\n\n"
        f"Reference story: {reference}\n\n"
        f"Candidate story: {candidate}"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=300,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            d: 0.0 for d in DIMENSIONS
        } | {"reason": f"Judge returned unparseable response: {raw[:200]}"}


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge story evaluation")
    parser.add_argument("--eval-data", default="eval_data.json")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--model-name", default=None,
                        help="Deployed model name (default: from MODEL_NAME env var)")
    args = parser.parse_args()

    # Deployed model client (RunPod Pod vLLM endpoint)
    pod_url = os.environ["POD_1_URL"].rstrip("/")
    model_name = args.model_name or os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    model_client = OpenAI(
        api_key="none",
        base_url=f"{pod_url}/v1",
    )

    # Judge LLM client
    judge_client = OpenAI(
        api_key=args.judge_api_key or os.getenv("JUDGE_API_KEY"),
        base_url=args.judge_api_base or os.getenv("JUDGE_API_BASE"),
    )
    judge_model = args.judge_model or os.getenv("JUDGE_MODEL", "deepseek-chat")

    eval_data = load_eval_data(args.eval_data)
    print(f"Loaded {len(eval_data)} evaluation prompts")
    print(f"Model: {model_name} via RunPod Pod {pod_url}")
    print(f"Judge: {judge_model}\n")

    Path("results").mkdir(exist_ok=True)
    out_csv = "results/eval_scores.csv"
    csv_file = open(out_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["prompt", "moral_clarity", "age_appropriateness", "narrative_coherence", "relevance", "overall_avg", "latency_s", "reason"])

    all_scores: list[dict] = []
    latencies: list[float] = []

    for i, item in enumerate(eval_data):
        prompt = item["prompt"]
        reference = item["reference"]
        label = prompt[:70]

        print(f"[{i + 1}/{len(eval_data)}] {label}...")

        try:
            candidate, latency = query_deployed_model(model_client, model_name, prompt)
            latencies.append(latency)
        except Exception as e:
            print(f"  FAILED to query model: {e}\n")
            all_scores.append({d: 0.0 for d in DIMENSIONS})
            continue

        try:
            result = judge_story(judge_client, judge_model, prompt, reference, candidate)
            scores = {d: result.get(d, 0.0) for d in DIMENSIONS}
            avg = sum(scores.values()) / len(scores)
            all_scores.append(scores)
            reason = result.get("reason", "")
            print(f"  avg={avg:.2f}  latency={latency:.2f}s  — {reason}\n")
            csv_writer.writerow([prompt[:80], scores["moral_clarity"], scores["age_appropriateness"], scores["narrative_coherence"], scores["relevance"], round(avg, 4), round(latency, 3), reason])
            csv_file.flush()
        except Exception as e:
            print(f"  JUDGE ERROR: {e}\n")
            all_scores.append({d: 0.0 for d in DIMENSIONS})

    # -- Summary ------------------------------------------------------------
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Stories evaluated: {len(eval_data)}")

    if all_scores:
        for dim in DIMENSIONS:
            vals = [s[dim] for s in all_scores]
            print(f"  {dim:>25s}: {sum(vals) / len(vals):.4f}")
        overall = [sum(s.values()) / len(DIMENSIONS) for s in all_scores]
        print(f"  {'overall_avg':>25s}: {sum(overall) / len(overall):.4f}")

    if latencies:
        latencies.sort()
        print(f"\n  {'avg_latency':>25s}: {sum(latencies) / len(latencies):.4f}s")
        print(f"  {'p90_latency':>25s}: {latencies[int(len(latencies) * 0.9)]:.4f}s")

    print("=" * 70)
    csv_file.close()
    print(f"\nPer-story scores saved to {out_csv}")


if __name__ == "__main__":
    main()
