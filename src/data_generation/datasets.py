"""Dataset loading and formatting for data generation pipeline."""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset

from .data_gen_config import DataGenerationConfig

logger = logging.getLogger(__name__)

CHOICE_LABELS = ["A", "B", "C", "D"]


@dataclass
class QuestionData:
    """Structured representation of a question."""

    question_hash: str
    question: str
    choices: List[str]
    correct_answer: str
    formatted_question: str
    category: str = "unknown"


def compute_question_hash(question_text: str) -> str:
    """Compute a stable MD5-based hash for a question."""
    return hashlib.md5(question_text.encode("utf-8")).hexdigest()[:12]


def build_question_id_lookup(existing_responses_dir: Path) -> Dict[str, str]:
    """Build lookup from question content to existing question hash."""
    lookup = {}
    if not existing_responses_dir.exists():
        logger.warning(f"Existing responses directory not found: {existing_responses_dir}")
        return lookup

    for path in existing_responses_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fq = data.get("formatted_question", "")
            q_key = fq.split("## Instruction")[0].strip()
            lookup[q_key] = path.stem
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    logger.info(f"Built lookup with {len(lookup)} existing questions")
    return lookup


def format_question(
    question: str,
    choices: List[str],
    choice_labels: List[str] = None,
) -> str:
    """Format a multiple-choice question into the standard prompt template."""
    if choice_labels is None:
        choice_labels = CHOICE_LABELS

    choices_text = "\n".join(
        f"- ({label}) {choice}" for label, choice in zip(choice_labels, choices)
    )

    return f"""## Question:
{question}

## Choices:
{choices_text}

## Instruction:
Please analyze the question step by step in <think>...</think> tags, then provide your final answer in JSON format with the key "answer" containing only the letter (A, B, C, or D) of the correct choice."""


def _resolve_hash(formatted: str, question_text: str, question_lookup: Optional[Dict[str, str]]) -> str:
    if question_lookup:
        q_key = formatted.split("## Instruction")[0].strip()
        if q_key in question_lookup:
            return question_lookup[q_key]
    return compute_question_hash(question_text)


def load_gpqa_questions(
    config: DataGenerationConfig,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load GPQA diamond split questions."""
    logger.info("Loading GPQA diamond split from Idavidrein/gpqa")
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    questions = []
    for item in dataset:
        question_text = item["Question"]
        correct_text = item["Correct Answer"]
        incorrect = [
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]

        correct_idx = random.randint(0, 3)
        choices = incorrect[:]
        choices.insert(correct_idx, correct_text)
        correct_answer = CHOICE_LABELS[correct_idx]

        formatted = format_question(question_text, choices)
        q_hash = _resolve_hash(formatted, question_text, question_lookup)

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category=item.get("Subdomain", "unknown"),
            )
        )

    logger.info(f"Loaded {len(questions)} GPQA questions")
    return questions


MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
]


def load_mmlu_questions(
    config: DataGenerationConfig,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load MMLU-redux questions with error_type == 'ok' filtering."""
    logger.info("Loading MMLU-redux from edinburgh-dawg/mmlu-redux-2.0 (all 57 subjects)")

    from datasets import concatenate_datasets

    all_datasets = []
    for subject in MMLU_SUBJECTS:
        try:
            ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
            ds = ds.add_column("subject", [subject] * len(ds))
            all_datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to load subject {subject}: {e}")

    if not all_datasets:
        raise ValueError("Failed to load any MMLU subjects")

    dataset = concatenate_datasets(all_datasets)
    logger.info(f"Loaded {len(dataset)} total MMLU questions before filtering")

    dataset = dataset.filter(lambda x: x.get("error_type") == "ok")

    questions = []
    for item in dataset:
        question_text = item["question"]
        choices = item["choices"]

        formatted = format_question(question_text, choices)
        q_hash = _resolve_hash(formatted, question_text, question_lookup)

        answer_idx = item["answer"]
        correct_answer = CHOICE_LABELS[answer_idx]

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=question_text,
                choices=choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category=item.get("subject", "unknown"),
            )
        )

    logger.info(f"Loaded {len(questions)} MMLU questions")
    return questions


def load_arc_questions(
    config: DataGenerationConfig,
    subset: str,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load ARC-Easy or ARC-Challenge questions (test split)."""
    logger.info(f"Loading ARC {subset} from allenai/ai2_arc")
    dataset = load_dataset("allenai/ai2_arc", subset, split="test")

    skipped_3 = 0
    dropped_5 = 0
    questions = []
    for item in dataset:
        question_text = item["question"]
        raw_labels = list(item["choices"]["label"])
        raw_choices = list(item["choices"]["text"])
        answer_key = item["answerKey"]

        n = len(raw_choices)
        if n == 3:
            skipped_3 += 1
            continue
        if n == 5:
            # Drop one random incorrect choice to get down to 4
            incorrect_indices = [i for i, lbl in enumerate(raw_labels) if lbl != answer_key]
            drop_idx = random.choice(incorrect_indices)
            raw_labels.pop(drop_idx)
            raw_choices.pop(drop_idx)
            dropped_5 += 1

        # Map raw label (may be '1','2','3','4' or 'A','B','C','D','E') to A/B/C/D index
        correct_idx = raw_labels.index(answer_key)
        correct_answer = CHOICE_LABELS[correct_idx]

        formatted = format_question(question_text, raw_choices)
        q_hash = _resolve_hash(formatted, question_text, question_lookup)

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=question_text,
                choices=raw_choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category=subset,
            )
        )

    logger.info(
        f"Loaded {len(questions)} ARC {subset} questions "
        f"(skipped {skipped_3} with 3 choices, trimmed {dropped_5} with 5 choices)"
    )
    return questions


def load_medqa_questions(
    _config: DataGenerationConfig,
    question_lookup: Optional[Dict[str, str]] = None,
) -> List[QuestionData]:
    """Load MedQA-USMLE 4-option questions (train split).

    Schema: sent1 (stem), sent2 (continuation, often empty), ending0–ending3
    (choices), label (int 0–3).
    """
    logger.info("Loading MedQA from openlifescienceai/MedQA-USMLE-4-options-hf")
    dataset = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="train")

    questions = []
    for item in dataset:
        # Combine sent1 + sent2 into the question stem; sent2 is usually empty
        stem = item["sent1"]
        if item.get("sent2"):
            stem = f"{stem} {item['sent2']}".strip()

        choices = [item[f"ending{i}"] for i in range(4)]
        correct_answer = CHOICE_LABELS[item["label"]]

        formatted = format_question(stem, choices)
        q_hash = _resolve_hash(formatted, stem, question_lookup)

        questions.append(
            QuestionData(
                question_hash=q_hash,
                question=stem,
                choices=choices,
                correct_answer=correct_answer,
                formatted_question=formatted,
                category="medqa",
            )
        )

    questions = random.sample(questions, min(2000, len(questions)))
    logger.info(f"Loaded {len(questions)} MedQA questions (randomly sampled from full train split)")
    return questions


def load_dataset_questions(config: DataGenerationConfig) -> List[QuestionData]:
    """Load questions from the specified dataset."""
    question_lookup = None
    if config.existing_responses_dir:
        question_lookup = build_question_id_lookup(config.existing_responses_dir)

    name = config.dataset_name.lower()
    if name == "gpqa":
        questions = load_gpqa_questions(config, question_lookup)
    elif name == "mmlu":
        questions = load_mmlu_questions(config, question_lookup)
    elif name == "arc-easy":
        questions = load_arc_questions(config, "ARC-Easy", question_lookup)
    elif name == "arc-challenge":
        questions = load_arc_questions(config, "ARC-Challenge", question_lookup)
    elif name == "medqa":
        questions = load_medqa_questions(config, question_lookup)
    else:
        raise ValueError(
            f"Unknown dataset: {config.dataset_name!r}. "
            "Expected one of: 'gpqa', 'mmlu', 'arc-easy', 'arc-challenge', 'medqa'"
        )

    if config.limit is not None:
        questions = questions[:config.limit]
        logger.info(f"Limited to {len(questions)} questions")

    return questions