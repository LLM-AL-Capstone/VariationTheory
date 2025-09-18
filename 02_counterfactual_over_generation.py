# 02_counterfactual_over_generation.py
# Use candidate phrases to generate counterfactuals with OpenAI (if key present).
import os
import sys
import ast
import re
import pandas as pd
import configparser

def read_config():
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#',';'))
    cfg.read('config.ini')
    get = lambda k, fb='': cfg.get('settings', k, fallback=fb).strip()
    api_key = get('openai_api')
    data_file = get('data_file')
    seed = get('seed', '42')
    return api_key, data_file, seed

API_KEY, DATA_FILE, SEED = read_config()
if not DATA_FILE:
    print("ERROR: No data_file provided in config.ini")
    sys.exit(1)

seed = int(SEED)
stem = DATA_FILE[:-4] if DATA_FILE.lower().endswith('.csv') else DATA_FILE
candidate_file = f"output_data/[{SEED}]{stem}_candidate_phrases_annotated_data.csv"
out_file = f"output_data/[{seed}]counterfactuals_{DATA_FILE}"

# Detect whether a *real* key is present
def has_real_key(key: str) -> bool:
    if not key:
        return False
    k = key.strip()
    if k in ("<<API_KEY>>", "SK-PLACEHOLDER"):
        return False
    if k.startswith("sk-"):  # typical OpenAI key prefix
        return True
    return False

def load_candidates(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: Candidate phrases file not found: {path}")
        sys.exit(0)  # exit cleanly so the pipeline can continue
    df = pd.read_csv(path)
    if df.empty:
        print(f"INFO: Candidate phrases file is empty: {path}")
        sys.exit(0)
    return df

def parse_candidate_phrases(raw):
    """candidate_phrases may be a list-like string; make it a clean comma-separated string for the prompt."""
    if isinstance(raw, list):
        return ", ".join([str(x) for x in raw])
    if isinstance(raw, str):
        # Try to parse list string like "['a','b']"
        try:
            val = ast.literal_eval(raw)
            if isinstance(val, (list, tuple)):
                return ", ".join([str(x) for x in val])
        except Exception:
            pass
        return raw
    return str(raw)

def main():
    print("LOG: Starting Step 02 (counterfactual generation)")
    print(f"LOG: Using candidate file: {candidate_file}")

    df = load_candidates(candidate_file)
    print("LOG: Data loaded successfully. Generating counterfactuals...")

    # If no real key, do a dry run and exit cleanly
    if not has_real_key(API_KEY):
        print("INFO: OpenAI key not found or placeholder — skipping generation (dry run).")
        print(f"INFO: Script ran OK. Found {len(df)} candidate rows in {candidate_file}.")
        return

    # We have a real key — proceed with OpenAI generation
    try:
        from openai import OpenAI
    except Exception as e:
        print("ERROR: openai package not installed. pip install openai")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    col_names = [
        "id", "ori_text", "ori_label", "pattern",
        "highlight", "candidate_phrases", "target_label", "counterfactual"
    ]
    data_collector = []

    for index, row in df.iterrows():
        print(f"Processing {index}...")
        text = row.get("ori_text", "")
        label = row.get("ori_label", "")
        target_label = row.get("target_label", "")
        generated_phrases_raw = row.get("candidate_phrases", "")
        highlight = row.get("highlight", "")
        pattern = row.get("pattern", "")

        generated_phrases = parse_candidate_phrases(generated_phrases_raw)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "The assistant will generate a counterfactual example close to the original sentence that contains one of the given phrases."},
                    {"role": "user",
                     "content": (
                         f"Your task is to change the given sentence from the current label to the target.\n"
                         f"For example: 'Find me a train ticket next monday to new york city' with original label: transport "
                         f"would be turned to 'Play me a song called New York City by Taylor Swift' with a label audio.\n"
                         f"You can use the following phrases to help you generate the counterfactuals.\n"
                         f"Please make the sentence about {target_label}. Make sure that the new sentence is not {label}.\n"
                         f"You must use one of the following phrases without rewording it in the new sentence based on the following three criteria: {generated_phrases}\n"
                         f"criteria 1: the phrase should change the label from {label} to {target_label} to the highest degree.\n"
                         f"criteria 2: the modified sentence cannot also be about {label} and make sure the word {target_label} is not part of the modified sentence.\n"
                         f"criteria 3: the modified sentence should be grammatically correct.\n"
                     )},
                    {"role": "user",
                     "content": (
                         f"original text: {text}\n"
                         f"original label: {label}\n"
                         f"target label: {target_label}\n"
                         f"generated phrases: {generated_phrases}\n"
                         f"modified text:"
                     )},
                ],
                temperature=0,
                max_tokens=256,
                stop=["\n"]
            )
            counterfactual = response.choices[0].message.content
        except Exception as e:
            print(f"ERROR: OpenAI call failed at row {index}: {e}")
            counterfactual = ""

        data_collector.append([
            row.get("id", index), text, label, pattern,
            highlight, generated_phrases, target_label, counterfactual
        ])

    out_df = pd.DataFrame(data_collector, columns=col_names)
    os.makedirs("output_data", exist_ok=True)
    out_df.to_csv(out_file, index=False)
    print(f"LOG: Wrote {len(out_df)} rows → {out_file}")

if __name__ == "__main__":
    main()

    