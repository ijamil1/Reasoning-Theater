"""Select 3 hardest distractors per MMLU-Pro question via OpenRouter LLM."""

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

CATEGORIES = {"math", "physics", "chemistry", "law", "biology"}

SYSTEM_PROMPT = (
    "You are an adversarial question designer. Your goal is to maximally confuse a highly "
    "capable language model answering multiple-choice questions. You do this by selecting the "
    "most deceptive, plausible-sounding incorrect answer choices — ones that a smart model is "
    "most likely to mistake for the correct answer."
)


def build_user_message(question: str, correct_text: str, incorrect_options: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(incorrect_options))
    return (
        "You will be given a multiple-choice question, its correct answer, and a numbered list "
        "of incorrect options.\n\n"
        "Your task: select exactly 3 incorrect options that, when combined with the correct "
        "answer into a 4-choice question, would most likely cause a capable language model to "
        "answer incorrectly. Prefer distractors that are:\n"
        "- Superficially similar to the correct answer (same domain, similar wording, plausible magnitude)\n"
        "- Likely to exploit common misconceptions or reasoning shortcuts\n"
        "- Difficult to rule out without deep subject knowledge\n\n"
        f"Question: {question}\n"
        f"Correct answer: {correct_text}\n"
        f"Incorrect options:\n{numbered}\n\n"
        "Return ONLY a JSON array of exactly 3 distinct integers — the numbers of 3 different "
        "incorrect options you selected. Do not repeat the same number.\n"
        "Example: [2, 5, 7]"
    )


def request_completion(*, api_key: str, model: str, user_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 20,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }
    response = requests.post(API_URL, headers=headers, json=body, timeout=60)
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    return content if isinstance(content, str) else ""


def parse_indices(response: str, n_incorrect: int) -> list[int]:
    """Parse JSON array of integers from model response. Returns 0-based valid indices."""
    match = re.search(r"\[.*?\]", response, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    valid = []
    seen = set()
    for val in raw:
        if isinstance(val, int) and 1 <= val <= n_incorrect and val not in seen:
            valid.append(val - 1)  # convert to 0-based
            seen.add(val)
    return valid


def select_distractors(
    question: str,
    correct_text: str,
    incorrect_options: list[str],
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> list[str]:
    """Call OpenRouter to pick 3 distractor indices, with fallback to random sampling."""
    user_message = build_user_message(question, correct_text, incorrect_options)
    selected_indices = []

    for attempt in range(max_retries):
        try:
            response = request_completion(api_key=api_key, model=model, user_message=user_message)
            selected_indices = parse_indices(response, len(incorrect_options))
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    # Fill any missing slots with random samples from unchosen options
    if len(selected_indices) < 3:
        chosen_set = set(selected_indices)
        remaining = [i for i in range(len(incorrect_options)) if i not in chosen_set]
        random.shuffle(remaining)
        needed = 3 - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    return [incorrect_options[i] for i in selected_indices[:3]]


def main():
    parser = argparse.ArgumentParser(description="Select hardest MMLU-Pro distractors via OpenRouter")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "dataset_prep" / "mmlu_pro_distractors.json",
    )
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if args.output.exists():
        with args.output.open() as f:
            existing = json.load(f)
        logger.info(f"Loaded {len(existing)} existing results from {args.output}")

    logger.info("Loading MMLU-Pro test split...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = [item for item in dataset if item["category"].lower() in CATEGORIES]
    logger.info(f"Filtered to {len(rows)} questions in categories: {CATEGORIES}")

    if args.limit is not None:
        rows = rows[: args.limit]
        logger.info(f"Limited to {len(rows)} questions")

    results = dict(existing)

    for item in tqdm(rows, desc="Selecting distractors"):
        question_index = str(item["question_index"])
        if question_index in results:
            continue

        options = list(item["options"])          # 10 choices
        answer_index = item["answer_index"]       # 0-based index of correct answer
        correct_text = options[answer_index]
        incorrect_options = [opt for i, opt in enumerate(options) if i != answer_index]

        distractors = select_distractors(
            question=item["question"],
            correct_text=correct_text,
            incorrect_options=incorrect_options,
            api_key=api_key,
            model=args.model,
        )
        results[question_index] = distractors

        # Save incrementally after each question
        with args.output.open("w") as f:
            json.dump(results, f, indent=2)

    logger.info(f"Done. {len(results)} questions written to {args.output}")


if __name__ == "__main__":
    main()
