"""UMLS candidate reranking with knowledge-enhanced generative LLMs.

This script reranks dense retrieval candidates for biomedical concept
normalization by enriching each candidate with structured UMLS knowledge
(definitions, synonyms, hierarchies, semantic groups) and prompting a
generative LLM to select the best match.

Usage:
    python rerank.py --input-dir <path> --output-dir <path> [options]

See README.md for full documentation.
"""

import argparse
import ast
import glob
import json
import os
import re
from datetime import datetime

import pandas as pd
from vllm import LLM, SamplingParams

import config


# =============================================================================
# Data loading utilities
# =============================================================================


def load_json(file_path: str) -> dict:
    """Load a JSON file, merging list-of-dicts into a single dict if needed."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle list-of-dicts format (e.g., definition files)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        merged = {}
        for item in data:
            merged.update(item)
        return merged

    return data


def load_data_dicts() -> dict:
    """Load UMLS knowledge dictionaries based on enabled feature flags.

    Only loads files that are needed for the current feature combination
    to minimize memory usage.

    Returns:
        Dictionary with keys: preferred_term, synonym, semantic,
        definition, hierarchy.
    """
    data_dicts = {}

    # Preferred terms are also needed to resolve hierarchy CUIs
    if config.USE_PREFERRED_TERM or config.USE_HIERARCHY:
        data_dicts["preferred_term"] = load_json(config.PREFERRED_TERM_FILE)
    else:
        data_dicts["preferred_term"] = {}

    data_dicts["synonym"] = (
        load_json(config.SYNONYM_FILE) if config.USE_SYNONYMS else {}
    )

    data_dicts["semantic"] = (
        load_json(config.SEMANTIC_FILE) if config.USE_SEMANTIC_GROUPS else {}
    )

    data_dicts["definition"] = (
        load_json(config.DEFINITION_FILE) if config.USE_DEFINITION else {}
    )

    data_dicts["hierarchy"] = (
        load_json(config.HIERARCHY_FILE) if config.USE_HIERARCHY else {}
    )

    return data_dicts


# =============================================================================
# Data cleaning utilities
# =============================================================================


def clean_list(items: list) -> list[str]:
    """Filter out non-string values (NaN, None, floats) from a list."""
    if not items:
        return []
    cleaned = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, float) and pd.isna(item):
            continue
        cleaned.append(str(item))
    return cleaned


def get_preferred_terms_for_cuis(
    cui_list: list[str], preferred_term_dict: dict, max_terms: int = None
) -> list[str]:
    """Resolve a list of CUIs to their preferred term strings."""
    if max_terms is None:
        max_terms = config.MAX_HIERARCHY
    terms = []
    for cui in cui_list:
        term = preferred_term_dict.get(cui)
        if term is not None and not (isinstance(term, float) and pd.isna(term)):
            terms.append(str(term))
    return terms[:max_terms]


def get_synonyms_for_cui(
    cui: str,
    synonym_dict: dict,
) -> list[str]:
    """Get English synonyms for a CUI."""
    if cui not in synonym_dict:
        return []

    synonyms = []
    cui_data = synonym_dict[cui]
    if "ENG" in cui_data:
        synonyms.extend(cui_data["ENG"])

    return clean_list(synonyms)[: config.MAX_SYNONYMS]


# =============================================================================
# Candidate context building
# =============================================================================


def build_candidate_context(cui: str, data_dicts: dict) -> dict:
    """Build UMLS knowledge context for a single candidate CUI.

    Assembles only the features that are enabled in config.

    Args:
        cui: UMLS Concept Unique Identifier.
        data_dicts: Pre-loaded UMLS knowledge dictionaries.

    Returns:
        Dictionary with 'cui' and enabled feature values.
    """
    context = {"cui": cui}

    if config.USE_PREFERRED_TERM:
        pref_term = data_dicts["preferred_term"].get(cui, "N/A")
        if pref_term is None or (isinstance(pref_term, float) and pd.isna(pref_term)):
            pref_term = "N/A"
        context["preferred_term"] = str(pref_term)

    if config.USE_SYNONYMS:
        context["synonyms"] = get_synonyms_for_cui(cui, data_dicts["synonym"])

    if config.USE_SEMANTIC_GROUPS:
        semantic_info = data_dicts["semantic"].get(cui, {})
        context["semantic_groups"] = clean_list(
            semantic_info.get("sem_group", [])
        )

    if config.USE_DEFINITION:
        definition_data = data_dicts["definition"].get(cui, {})
        final_definition = "No definition available"
        if definition_data:
            first_key = next(iter(definition_data))
            final_definition = definition_data[first_key]
        if final_definition is None or (
            isinstance(final_definition, float) and pd.isna(final_definition)
        ):
            final_definition = "No definition available"
        context["definition"] = str(final_definition)

    if config.USE_HIERARCHY:
        hierarchy = data_dicts["hierarchy"].get(cui, {})
        hypernym_cuis = clean_list(hierarchy.get("hypernyms", []))
        context["hypernyms"] = get_preferred_terms_for_cuis(
            hypernym_cuis, data_dicts["preferred_term"]
        )
        hyponym_cuis = clean_list(hierarchy.get("hyponyms", []))
        context["hyponyms"] = get_preferred_terms_for_cuis(
            hyponym_cuis, data_dicts["preferred_term"]
        )

    return context


# =============================================================================
# Prompt construction
# =============================================================================


def format_candidate_description(
    index: int,
    retrieved_term: str,
    ctx: dict,
) -> str:
    """Format a single candidate's description block for the prompt."""
    if retrieved_term is None or (
        isinstance(retrieved_term, float) and pd.isna(retrieved_term)
    ):
        retrieved_term = "N/A"

    lines = [f"Candidate {index} -- CUI: {ctx['cui']}, Top retrieved term: {retrieved_term}"]

    if config.USE_PREFERRED_TERM:
        lines.append(f"- Preferred term: {ctx.get('preferred_term', 'N/A')}")

    if config.USE_SYNONYMS:
        synonyms = ctx.get("synonyms", [])
        lines.append(f"- Synonyms: {'; '.join(synonyms) if synonyms else 'None'}")

    if config.USE_SEMANTIC_GROUPS:
        sem_groups = ctx.get("semantic_groups", [])
        lines.append(f"- Semantic groups: {'; '.join(sem_groups) if sem_groups else 'None'}")

    if config.USE_DEFINITION:
        lines.append(f"- Definition: {ctx.get('definition', 'N/A')}")

    if config.USE_HIERARCHY:
        hypernyms = ctx.get("hypernyms", [])
        lines.append(f"- Hypernyms (parent concepts): {'; '.join(hypernyms) if hypernyms else 'None'}")
        hyponyms = ctx.get("hyponyms", [])
        lines.append(f"- Hyponyms (child concepts): {'; '.join(hyponyms) if hyponyms else 'None'}")

    return "\n".join(lines)


def create_prompt(
    query: str,
    target_ontology: str,
    semantic_group: str,
    retrieved_candidate_terms: list[str],
    candidates_context: list[dict],
) -> str:
    """Create a reranking prompt for the generative LLM.

    Constructs a structured prompt following the normalization algorithm:
    the query term, target ontology, semantic group, and all candidate
    descriptions with enabled UMLS features.

    Args:
        query: The input biomedical term to normalize.
        target_ontology: Target ontology name (e.g., SNOMED CT, MedDRA).
        semantic_group: Semantic group of the query term.
        retrieved_candidate_terms: Top retrieved term strings per candidate.
        candidates_context: UMLS knowledge context per candidate.

    Returns:
        Formatted prompt string.
    """
    n_candidates = len(candidates_context)

    # Format all candidate descriptions
    candidate_descriptions = []
    for i, ctx in enumerate(candidates_context):
        desc = format_candidate_description(
            i + 1, retrieved_candidate_terms[i], ctx
        )
        candidate_descriptions.append(desc)

    candidates_block = "\n\n".join(candidate_descriptions)

    prompt = f"""### Objective: Biomedical concept normalization via candidate re-ranking.

You are a domain expert in biomedical terminology and ontology systems (e.g., UMLS, SNOMED CT, MedDRA). Your task is to perform concept normalization: given a biomedical query term and a ranked list of candidate concepts from the target ontology, identify the single candidate that most accurately represents the query.

### Important note: The query term may be in a language other than English. However, the provided attributes for each candidate are in English.

### Re-ranking methodology:
1. Parse the query term and determine its intended biomedical meaning, accounting for lexical variation, abbreviation, and contextual ambiguity.
2. Use the query's semantic group to constrain the expected domain of the correct candidate.
3. For each candidate, evaluate all provided attributes (preferred term, synonyms, semantic groups, definition, hypernyms, hyponyms).
4. Assess semantic equivalence between the query and each candidate.
5. Select the single best-matching candidate.

### Output format: Respond with only the integer index of the best candidate (between 1 and {n_candidates}). No explanation.

### Target ontology: {target_ontology}

### Query semantic group: {semantic_group}

### Query term: {query}

### Candidates:

{candidates_block}

### Answer:"""

    return prompt


# =============================================================================
# Feature flag management
# =============================================================================

FEATURE_FLAG_MAP = {
    "preferred_term": "USE_PREFERRED_TERM",
    "synonyms": "USE_SYNONYMS",
    "semantic_groups": "USE_SEMANTIC_GROUPS",
    "definition": "USE_DEFINITION",
    "hierarchy": "USE_HIERARCHY",
}


def set_feature_flags(enabled_features: list[str]) -> None:
    """Set config feature flags based on a list of feature names."""
    # Reset all to False
    for attr in FEATURE_FLAG_MAP.values():
        setattr(config, attr, False)

    # Enable requested features
    for feature in enabled_features:
        attr = FEATURE_FLAG_MAP.get(feature)
        if attr:
            setattr(config, attr, True)


def get_enabled_features() -> list[str]:
    """Return list of currently enabled feature names."""
    return [
        name
        for name, attr in FEATURE_FLAG_MAP.items()
        if getattr(config, attr, False)
    ]


# =============================================================================
# LLM inference
# =============================================================================


def extract_answer_after_think(text: str) -> str | None:
    """Extract the first number appearing after </think> tag.

    For models with chain-of-thought (thinking mode), the final answer
    appears after the closing </think> tag.

    Returns:
        The extracted number as a string, or None if not found.
    """
    if "</think>" in text:
        after_think = text.split("</think>", 1)[1]
        match = re.search(r"\d+\.?\d*", after_think)
        if match:
            return match.group()
    return None


def process_batch(
    llm: LLM, prompts: list[str], sampling_params: SamplingParams
) -> tuple[list[str | None], list[str]]:
    """Run batch inference and extract predictions.

    Args:
        llm: Loaded vLLM model instance.
        prompts: List of prompt strings.
        sampling_params: vLLM sampling configuration.

    Returns:
        Tuple of (predicted_answers, raw_thinking_outputs).
    """
    batch_messages = [[{"role": "user", "content": p}] for p in prompts]
    batch_outputs = llm.chat(
        batch_messages,
        sampling_params,
        chat_template_kwargs={"enable_thinking": config.ENABLE_THINKING},
    )

    answers = [extract_answer_after_think(o.outputs[0].text) for o in batch_outputs]

    return answers


# =============================================================================
# Instance and file processing
# =============================================================================


def process_single_instance(idx: int, df: pd.DataFrame, data_dicts: dict) -> str:
    """Build a reranking prompt for a single DataFrame row."""
    query = df["term"].iloc[idx]
    retrieved_codes = ast.literal_eval(df["retrieved_codes"].iloc[idx])[
        : config.TOP_CANDIDATES2RANK
    ]
    retrieved_terms = ast.literal_eval(df["retrieved_terms"].iloc[idx])[
        : config.TOP_CANDIDATES2RANK
    ]

    # Extract target ontology and semantic group from the row
    target_ontology = str(df["target_ontologies"].iloc[idx]) if "target_ontologies" in df.columns else "UMLS"
    semantic_group = str(df["semantic_group"].iloc[idx]) if "semantic_group" in df.columns else "N/A"

    candidates_context = [
        build_candidate_context(cui, data_dicts) for cui in retrieved_codes
    ]
    return create_prompt(query, target_ontology, semantic_group, retrieved_terms, candidates_context)


def get_predicted_cui(row: pd.Series) -> list[str] | None:
    """Map a predicted candidate index back to its CUI."""
    try:
        prediction_idx = int(row["prediction"]) - 1
        retrieved_codes = row["retrieved_codes"]
        if isinstance(retrieved_codes, str):
            retrieved_codes = ast.literal_eval(retrieved_codes)
        if 0 <= prediction_idx < len(retrieved_codes):
            return [retrieved_codes[prediction_idx]]
    except (ValueError, TypeError, SyntaxError, IndexError):
        pass
    return None


def process_csv_file(
    csv_file: str,
    llm: LLM,
    sampling_params: SamplingParams,
    data_dicts: dict,
    timestamp_str: str,
    output_dir: str,
    split: str,
) -> str:
    """Process a single CSV file: generate prompts, run inference, save results.

    Args:
        csv_file: Path to input CSV with dense retrieval candidates.
        llm: Loaded vLLM model instance.
        sampling_params: vLLM sampling configuration.
        data_dicts: Pre-loaded UMLS knowledge dictionaries.
        timestamp_str: Timestamp for output filenames.
        output_dir: Directory for saving result CSVs.
        split: Dataset split name (e.g., 'test').

    Returns:
        Path to the saved output CSV file.
    """
    print(f'\n{"=" * 80}')
    print(f"Processing file: {csv_file}")
    print(f'{"=" * 80}')

    df = pd.read_csv(csv_file)
    df = df.drop(columns=["embedding"], errors="ignore")

    # For test split, only process non-exact-match instances
    if split == "test":
        df = df[df["exact_match"] == 0].reset_index(drop=True)

    print(f"Loaded {len(df)} instances")

    # Process in batches
    all_predictions = []
    for batch_start in range(0, len(df), config.BATCH_SIZE):
        batch_end = min(batch_start + config.BATCH_SIZE, len(df))
        batch_num = batch_start // config.BATCH_SIZE + 1
        total_batches = (len(df) - 1) // config.BATCH_SIZE + 1
        print(
            f"Processing batch {batch_num}/{total_batches} "
            f"(rows {batch_start}-{batch_end - 1})"
        )

        batch_prompts = [
            process_single_instance(idx, df, data_dicts)
            for idx in range(batch_start, batch_end)
        ]

        preds = process_batch(llm, batch_prompts, sampling_params)
        all_predictions.extend(preds)

    # Save results
    df["predicted_cui"] = df.apply(get_predicted_cui, axis=1)

    # Reorder columns
    preferred_order = [
        "term", "code", "language","semantic_group",
        "target_ontologies", "predicted_cui", "retrieved_codes", "retrieved_terms",
    ]
    available = [c for c in preferred_order if c in df.columns]
    remaining = [c for c in df.columns if c not in available]
    df = df[available + remaining]

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    if config.ENABLE_THINKING:
        base_name += "_thinking_"
    output_file = os.path.join(output_dir, f"{base_name}_results_{timestamp_str}.csv")
    df.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    return output_file


# =============================================================================
# CLI argument parsing
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, overriding config defaults."""
    parser = argparse.ArgumentParser(
        description="UMLS candidate reranking with knowledge-enhanced generative LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Path arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=f"Directory containing dense retrieval output CSVs (default: {config.INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for reranking results (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--umls-dir",
        type=str,
        default=None,
        help="Directory containing UMLS JSON files. Overrides individual file paths.",
    )

    # Individual UMLS file overrides
    parser.add_argument("--synonym-file", type=str, default=None)
    parser.add_argument("--preferred-term-file", type=str, default=None)
    parser.add_argument("--hierarchy-file", type=str, default=None)
    parser.add_argument("--definition-file", type=str, default=None)
    parser.add_argument("--semantic-file", type=str, default=None)

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"HuggingFace model name (default: {config.MODEL_NAME})",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking/chain-of-thought mode",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)

    # Knowledge feature limits
    parser.add_argument(
        "--max-synonyms",
        type=int,
        default=None,
        help=f"Max synonyms per candidate (default: {config.MAX_SYNONYMS})",
    )
    parser.add_argument(
        "--max-hierarchy",
        type=int,
        default=None,
        help=f"Max hypernyms/hyponyms each per candidate (default: {config.MAX_HIERARCHY})",
    )

    # Experiment arguments
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=None,
        help="Top-K candidate values to experiment with (default: 10)",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        default=None,
        help='Feature combinations as comma-separated strings, '
        'e.g., "synonyms,hierarchy" "definition,synonyms"',
    )
    parser.add_argument(
        "--dr-llms",
        type=str,
        nargs="+",
        default=None,
        help='Dense retrieval models to process (default: config.DR_LLMS)',
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="Dataset splits to process (default: test)",
    )

    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace) -> None:
    """Override config values with CLI arguments where provided."""
    if args.input_dir:
        config.INPUT_DIR = args.input_dir
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir

    # UMLS directory shortcut
    if args.umls_dir:
        umls_dir = args.umls_dir
        config.SYNONYM_FILE = os.path.join(umls_dir, "cui_synonyms.json")
        config.PREFERRED_TERM_FILE = os.path.join(umls_dir, "umls_preferred_terms.json")
        config.HIERARCHY_FILE = os.path.join(umls_dir, "umls_hierarchies.json")
        config.DEFINITION_FILE = os.path.join(umls_dir, "umls_definitions.json")
        config.SEMANTIC_FILE = os.path.join(umls_dir, "cui_semantic_mapping.json")

    # Individual file overrides (take priority over --umls-dir)
    if args.synonym_file:
        config.SYNONYM_FILE = args.synonym_file
    if args.preferred_term_file:
        config.PREFERRED_TERM_FILE = args.preferred_term_file
    if args.hierarchy_file:
        config.HIERARCHY_FILE = args.hierarchy_file
    if args.definition_file:
        config.DEFINITION_FILE = args.definition_file
    if args.semantic_file:
        config.SEMANTIC_FILE = args.semantic_file

    # Model settings
    if args.model:
        config.MODEL_NAME = args.model
    if args.no_thinking:
        config.ENABLE_THINKING = False
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.tensor_parallel_size:
        config.TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
    if args.gpu_memory_utilization:
        config.GPU_MEMORY_UTILIZATION = args.gpu_memory_utilization

    # Knowledge feature limits
    if args.max_synonyms is not None:
        config.MAX_SYNONYMS = args.max_synonyms
    if args.max_hierarchy is not None:
        config.MAX_HIERARCHY = args.max_hierarchy

    # Experiment settings
    if args.top_k:
        config.TOP_CANDIDATES_LIST = args.top_k
    if args.features:
        config.FEATURES_FOR_EXP = [f.split(",") for f in args.features]
    if args.dr_llms:
        config.DR_LLMS = args.dr_llms
    if args.splits:
        config.SPLITS = args.splits


# =============================================================================
# Main entry point
# =============================================================================


def main():
    args = parse_args()
    apply_args_to_config(args)

    timestamp_str = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    print(f"Started: {timestamp_str}")

    # Load model once
    print("\nLoading vLLM model...")
    llm = LLM(
        model=config.MODEL_NAME,
        max_model_len=config.MAX_MODEL_LEN,
        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=config.GPU_MEMORY_UTILIZATION,
    )
    sampling_params = SamplingParams(
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        min_p=config.MIN_P,
        seed=config.SEED,
        max_tokens=config.MAX_TOKENS,
    )
    print("Model loaded successfully")

    model_save_name = config.MODEL_NAME.replace("/", "").replace("_", "-")
    if config.ENABLE_THINKING:
        model_save_name += "_thinking"

    # Resolve DR LLM directories
    parent_dir = config.INPUT_DIR
    if config.DR_LLMS == ["all"]:
        dr_llms = [
            d
            for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]
    else:
        dr_llms = config.DR_LLMS

    # Experiment loops
    for top_k_idx, top_k_value in enumerate(config.TOP_CANDIDATES_LIST, 1):
        print(f'\n{"@" * 80}')
        print(
            f"TOP_K {top_k_idx}/{len(config.TOP_CANDIDATES_LIST)}: {top_k_value}"
        )
        print(f'{"@" * 80}\n')
        config.TOP_CANDIDATES2RANK = top_k_value

        for feat_idx, feature_combination in enumerate(config.FEATURES_FOR_EXP, 1):
            print(f'\n{"#" * 80}')
            print(
                f"FEATURES {feat_idx}/{len(config.FEATURES_FOR_EXP)}: "
                f"{feature_combination}"
            )
            print(f'{"#" * 80}\n')

            set_feature_flags(feature_combination)
            enabled_features = get_enabled_features()
            print(f"Enabled features: {', '.join(enabled_features) or 'No_umls_knowledge'}")
            print(f"Model: {config.MODEL_NAME}")
            print(f"Top-K candidates: {config.TOP_CANDIDATES2RANK}")
            print(f"Max synonyms: {config.MAX_SYNONYMS}")
            print(f"Max hierarchy (each): {config.MAX_HIERARCHY}")

            features_folder = "_".join(enabled_features) or "No_umls_knowledge"

            print("\nLoading UMLS knowledge...")
            data_dicts = load_data_dicts()
            print("UMLS knowledge loaded")

            for dr_llm in dr_llms:
                for split in config.SPLITS:
                    split_path = os.path.join(parent_dir, dr_llm, split)
                    csv_files = glob.glob(os.path.join(split_path, "*.csv"))

                    if not csv_files:
                        print(f"\nNo CSV files found in {split_path}")
                        continue

                    print(f"\nFound {len(csv_files)} CSV file(s):")
                    for f in csv_files:
                        print(f"  - {f}")

                    # Build output directory
                    final_output_dir = os.path.join(
                        config.OUTPUT_DIR,
                        str(config.TOP_CANDIDATES2RANK),
                        model_save_name,
                        dr_llm,
                        features_folder,
                        split,
                    )
                    os.makedirs(final_output_dir, exist_ok=True)
                    print(f"Output directory: {final_output_dir}")

                    output_files = []
                    for i, csv_file in enumerate(csv_files, 1):
                        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file}")
                        output_file = process_csv_file(
                            csv_file,
                            llm,
                            sampling_params,
                            data_dicts,
                            timestamp_str,
                            final_output_dir,
                            split,
                        )
                        output_files.append(output_file)

                    print(f'\n{"=" * 80}')
                    print(f"Completed: {dr_llm}/{split}")
                    print(f"Processed {len(csv_files)} file(s)")
                    for f in output_files:
                        print(f"  - {f}")
                    print(f'{"=" * 80}')

    # Summary
    end_time = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    n_experiments = len(config.TOP_CANDIDATES_LIST) * len(config.FEATURES_FOR_EXP)
    print(f'\n{"@" * 80}')
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Finished: {end_time}")
    print(f"Total experiments: {n_experiments}")
    print(f'{"@" * 80}')


if __name__ == "__main__":
    main()