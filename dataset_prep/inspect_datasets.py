"""
Inspect all four datasets from HuggingFace in the difficulty ladder.
Run with: uv run --no-project python dataset_prep/inspect_datasets.py
"""

from datasets import load_dataset, concatenate_datasets


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def inspect_arc(name: str, dataset) -> None:
    inspect(name, dataset)
    all_labels = sorted({label for row in dataset for label in row["choices"]["label"]})
    all_answer_keys = sorted({row["answerKey"] for row in dataset})
    unique_lengths = sorted({len(row["choices"]["text"]) for row in dataset})
    print(f"  Distinct choice labels : {all_labels}")
    print(f"  Distinct answerKeys    : {all_answer_keys}")
    print(f"  Distinct choices count : {unique_lengths}")


def inspect(name: str, dataset) -> None:
    print(f"\n  Rows   : {len(dataset)}")
    print(f"  Columns: {dataset.column_names}")
    print(f"  Schema :")
    for col, dtype in dataset.features.items():
        print(f"    {col!r:<35} {dtype}")
    print(f"\n  Sample row:")
    row = dataset[0]
    for k, v in row.items():
        preview = repr(v)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        print(f"    {k!r:<35} {preview}")


# ---------------------------------------------------------------------------
# MMLU-Redux 2.0  (existing pipeline: edinburgh-dawg/mmlu-redux-2.0, all 57
# subjects, test split, filtered to error_type == 'ok')
# ---------------------------------------------------------------------------
print_section("MMLU-Redux 2.0  —  edinburgh-dawg/mmlu-redux-2.0  [test, error_type=='ok']")

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

mmlu_parts = []
for subject in MMLU_SUBJECTS:
    try:
        ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", subject, split="test")
        ds = ds.add_column("subject", [subject] * len(ds))
        mmlu_parts.append(ds)
    except Exception as e:
        print(f"  [warn] failed to load subject {subject!r}: {e}")

mmlu_raw = concatenate_datasets(mmlu_parts)
mmlu = mmlu_raw.filter(lambda x: x.get("error_type") == "ok")
print(f"\n  Pre-filter rows : {len(mmlu_raw)}")
print(f"  Post-filter rows: {len(mmlu)}")
inspect("MMLU-Redux", mmlu)

# ---------------------------------------------------------------------------
# ARC-Challenge  (test split)
# ---------------------------------------------------------------------------
print_section("ARC-Challenge  —  allenai/ai2_arc  [ARC-Challenge, test]")
arc_challenge = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
inspect_arc("ARC-Challenge", arc_challenge)

# ---------------------------------------------------------------------------
# MedQA-USMLE  (train split)
# ---------------------------------------------------------------------------
print_section("MedQA-USMLE  —  openlifescienceai/MedQA-USMLE-4-options-hf  [train]")
medqa = load_dataset("openlifescienceai/MedQA-USMLE-4-options-hf", split="train")
inspect("MedQA-USMLE", medqa)

# ---------------------------------------------------------------------------
# GPQA-Diamond  (existing pipeline: Idavidrein/gpqa, gpqa_diamond, train split
# — the dataset has no canonical test split; full pool used for our splits)
# ---------------------------------------------------------------------------
print_section("GPQA-Diamond  —  Idavidrein/gpqa  [gpqa_diamond, train]")
gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
inspect("GPQA-Diamond", gpqa)

print("\n")