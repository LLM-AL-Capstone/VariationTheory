import sys
import os
import pandas as pd
from pandas.api.types import is_integer_dtype
#import PaTAT things
from PaTAT_piped.api_helper import APIHelper
from PaTAT_piped.helpers import get_similarity_dict
import re
import ast
import configparser


from openai import OpenAI
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")


config = configparser.ConfigParser()

# Read the config.ini file
config.read('config.ini')

API_KEY = config.get("settings", "openai_api").split('#')[0].strip()

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()

SEED = config.get("settings", "seed").split('#')[0].strip()

client = OpenAI(
    api_key=API_KEY
)
seed=int(SEED)

total_annotations = 150
# total_annotations = 50


def get_patat_patterns():
    print("INFO: Executing data formatting with Python interpreter path:", sys.executable)

    # --- load dataset ---
    if len(DATA_FILE) < 1:
        print("ERROR: No Data file provided.")
        sys.exit(1)

    file_path = f"input_data/{DATA_FILE}"
    try:
        df = pd.read_csv(file_path)
        print("INFO: Finished reading data")
    except Exception:
        print(f"ERROR: can not read file {file_path}")
        sys.exit(1)

    # --- normalize columns PaTAT expects ---
    # example text column
    if 'example' not in df.columns:
        if 'Text' in df.columns:
            df['example'] = df['Text']
        else:
            print("ERROR: Dataset needs an 'example' or 'Text' column.")
            sys.exit(1)
    df['example'] = df['example'].astype(str).str.strip()

    # numeric id (keep original alphanumeric in 'orig_id' for output)
    needs_remap = (
        'id' not in df.columns
        or not is_integer_dtype(df['id'])
        or df['id'].astype(str).str.contains(r'\D').any()
    )
    if needs_remap:
        if 'id' in df.columns:
            df['orig_id'] = df['id']
        df = df.reset_index(drop=True)
        df['id'] = df.index
    if 'orig_id' not in df.columns:
        df['orig_id'] = df['id']
    id_to_orig = dict(zip(df['id'], df['orig_id']))

    # small-set friendly shuffle (optional)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    unique_labels = df["Label"].unique()

    # --- build PaTAT once for this dataset ---
    df_patat = df[['id', 'example', 'Label']].copy()
    patat = APIHelper(user=DATA_FILE[:-4])               # __init__ is patched to allow empty data
    # if you added set_data in APIHelper, you can use: patat.set_data(df_patat, theme=DATA_FILE[:-4])
    patat.theme = DATA_FILE[:-4]
    patat.data = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data['example'].values,
        soft_threshold=patat.soft_threshold,
        soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk,
        file_name=patat.theme
    )

    # --- seed: 1 example per label so each theme exists ---
    annotated_ids = []
    for lab in unique_labels:
        cand = df[df['Label'] == lab]
        if not cand.empty:
            sid = int(cand.iloc[0]['id'])
            if sid not in annotated_ids:
                annotated_ids.append(sid)

    # --- collector for output rows ---
    col_names = ["id", "ori_text", "ori_label", "pattern", "highlight"]
    data_collector = []

    # --- batching (works for tiny datasets) ---
    BATCH_SIZE = 5
    total_cap = min(total_annotations, len(df))

    for i in range(0, len(df), BATCH_SIZE):
        # take 5 at a time, avoid duplicates
        batch_ids = [int(x) for x in df.iloc[i:i+BATCH_SIZE]['id'].tolist() if x not in annotated_ids]
        if not batch_ids:
            # nothing new this round
            pass
        else:
            annotated_ids.extend(batch_ids)

        if len(annotated_ids) >= total_cap:
            print(f"INFO: Reached annotation cap ({len(annotated_ids)}/{total_cap})")
            # still label before breaking so the last batch is included
        print(f"INFO: Length of annotated ids {len(annotated_ids)} of {total_cap}")

        # label the growing set with PaTAT (note: ids are now numeric)
        for sid in annotated_ids:
            lbl = df.loc[df['id'] == sid, 'Label'].values[0]
            patat.label_element(sid, lbl)

        # only train on themes that actually have annotations so far
        labels_with_ann = sorted(df.loc[df['id'].isin(annotated_ids), 'Label'].unique())

        for label in labels_with_ann:
            try:
                patat.set_theme(label)
                results = patat.get_linear_model_results()  # may return {'message': 'Annotate Some More', ...}
            except KeyError:
                print(f"SKIP: no synthesizer for theme '{label}' yet")
                continue

            if "explanation" not in results:
                print(f"ERROR: Not enough annotations ({len(annotated_ids)}) for {label}, {results}")
                continue

            explanations = results['explanation']  # {pattern: {sentence_id: highlight, ...}, ...}
            for pattern_explanation, sent_map in explanations.items():
                for sentence_id, highlight in sent_map.items():
                    # sentence_id is numeric now
                    orig_label = df.loc[df['id'] == sentence_id, 'Label'].values[0]
                    if label != orig_label:
                        continue
                    ori_text = df.loc[df['id'] == sentence_id, 'example'].values[0]
                    # write original id form for output (if you had alphanumeric ids)
                    out_id = id_to_orig.get(sentence_id, sentence_id)
                    data_collector.append([out_id, ori_text, orig_label, pattern_explanation, highlight])

        if len(annotated_ids) >= total_cap:
            break

    # --- write output ---
    out_df = pd.DataFrame(data_collector, columns=col_names)
    os.makedirs("output_data", exist_ok=True)
    out_df.to_csv(f"output_data/annotated_data_with_pattern_{DATA_FILE[:-4]}.csv", index=False)



def get_candidate_phrases():
    file_path = f"input_data/{DATA_FILE}"
    try:
        df = pd.read_csv(file_path)
        print("INFO: Finished reading data")
    except Exception:
        print(f"ERROR: can not read file {file_path}")
        sys.exit(1)

    # --- Normalize columns PaTAT expects ---

    # Ensure 'example' exists
    if 'example' not in df.columns:
        if 'Text' in df.columns:
            df['example'] = df['Text']
        else:
            print("ERROR: Dataset needs an 'example' or 'Text' column.")
            sys.exit(1)

    # Ensure 'id' is integer (keep original if alphanumeric)
    from pandas.api.types import is_integer_dtype
    needs_remap = (
        'id' not in df.columns
        or not is_integer_dtype(df['id'])
        or df['id'].astype(str).str.contains(r'\D').any()
    )
    if needs_remap:
        if 'id' in df.columns:
            df['orig_id'] = df['id']  # keep original alphanumeric ids
        df = df.reset_index(drop=True)
        df['id'] = df.index          # 0..N-1

    # Clean text
    df['example'] = df['example'].astype(str).str.strip()

    unique_labels = df["Label"].unique()

    # --- Build PaTAT once for this dataset (new wiring) ---
    df_patat = df[['id', 'example', 'Label']].copy()
    patat = APIHelper(user=DATA_FILE[:-4])
    patat.theme = DATA_FILE[:-4]
    patat.data = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data['example'].values,
        soft_threshold=patat.soft_threshold,
        soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk,
        file_name=patat.theme
    )
    # ------------------------------------------------------

    # Load annotated patterns produced by get_patat_patterns()
    try:
        ann = pd.read_csv(f"output_data/annotated_data_with_pattern_{DATA_FILE[:-4]}.csv")
    except Exception:
        print("ERROR: can not read annotated data file")
        sys.exit(1)

    # Collect rows for output
    col_names_2 = ["id", "ori_text", "ori_label", "pattern", "highlight", "target_label", "candidate_phrases"]
    data_collector_2 = []
    num_tokens = 0

    # Generate candidate phrases for each annotated row
    for i, row in ann.iterrows():
        if num_tokens > 10_000_000:
            print("INFO: Skipping the request due to token limit")
            break

        sent_id = row['id']
        sentence = row['ori_text']
        pattern = row['pattern']
        label = row['ori_label']
        highlight_str = row['highlight']

        # Parse highlight spans → list of phrases we actually modify
        try:
            hl = ast.literal_eval(highlight_str)  # e.g. [[['too','many'], start, end], ...]
            marked_phrases = [" ".join(h[0]) for h in hl if isinstance(h, (list, tuple)) and h]
            if not marked_phrases:
                continue
            print(f"INFO: Marked phrases for id {sent_id}: {marked_phrases}")
        except Exception:
            continue

        for matched_phrase in marked_phrases:
            if num_tokens > 10_000_000:
                print("INFO: Skipping the request due to token limit")
                break

            print(f"INFO: Generating candidate phrases for id {sent_id} with matched phrase '{matched_phrase}'")

            # For each non-original label, ask the LLM for candidate phrases
            for target_label in unique_labels:
                if target_label == label or target_label == 'none':
                    continue

                # Few-shot + task description (kept as in your original)
                messages = [
                    {"role": "system", "content": [{"text":
                        "The assistant will create a list of candidate phrases that match the given symbolic domain specific pattern. The domain specific pattern definition is given below.\n\n"
                        "The domain specific pattern symbols includes the following patterns:\n"
                        "- Part-of-speech (POS) tags are capital: VERB, PROPN, NOUN, ADJ, ADV, AUX, PRON, NUM\n"
                        "- Word stemming are surrounded in [] and should have an exact match: [WORD]\n"
                        "- Soft match are surrounded by () and will match words with their synonyms. The list of synonms for each soft match in a pattern are given in the user instruction: (word)\n"
                        "- Entity type start with $ sign: $ENT-TYPE (e.g., $LOCATION, $DATE)\n"
                        "- Wildcard is * and can match anything: *\n\n"
                        "Patterns can use + for AND and | for OR. Soft matches may only be replaced with the provided allowed words.\n"
                        "For the following text and pattern, generate diverse example phrases that match the pattern and can belong to the target label. Separate by commas.",
                        "type": "text"}]},
                    {"role": "user", "content": [{"text":
                        "sentence:'Too many other places to shop with better prices .', phrase to modify: 'prices .', pattern: '(price)+*', current label: price,  softmatch:[price:[purchase, pricey, cheap, cost, pricing]], target label: service,",
                        "type": "text"}]},
                    {"role": "assistant", "content": [{"text":
                        "purchase options, pricey service, cheap help, pricing plans, cost breakdown",
                        "type": "text"}]},
                    {"role": "user", "content": [{"text":
                        "sentence:' they have great produce and seafood', phrase to modify: 'seafood', pattern: '[seafood]|NOUN', current label: products, target label: service",
                        "type": "text"}]},
                    {"role": "assistant", "content": [{"text":
                        "hospitality, seafood, help, management, staff",
                        "type": "text"}]},
                    {"role": "user", "content": [{"text":
                        "sentence:' the wings were delicious', phrase to modify: 'delicious', pattern: '(delicious)|$DATE', current label: products,  softmatch:[delicious:['taste','flavor','deliciousness','yummy','tasty','flavour','delicious']], target label: price",
                        "type": "text"}]},
                    {"role": "assistant", "content": [{"text":
                        "affordably delicious, on sale today, priced well for their flavour, deliciousness",
                        "type": "text"}]},
                    {"role": "user", "content": [{"text":
                        "sentence:' they should be shut down for terrible service .', phrase to modify: 'service', pattern: '(service)+(manager)+*', current label: service,  softmatch:[service:['customer','service'], manager:['management','manage','manager']], target label: price",
                        "type": "text"}]},
                    {"role": "assistant", "content": [{"text":
                        "[ service charge, service fee, customer cost, manage pricing, management price]",
                        "type": "text"}]},
                ]

                # Build softmatch dictionary from the pattern using PaTAT similarity_dict
                softmatch_collector = {}
                matches = re.findall(r'\(([^)]+)\)', pattern)
                if matches:
                    for m in matches:
                        if m in patat.similarity_dict:
                            words = list(patat.similarity_dict[m].keys()) + [m]
                            softmatch_collector[m] = words

                # Final task request (use the specific matched phrase, not the raw highlight blob)
                req_message = (
                    f"sentence:' {sentence}', "
                    f"phrase to modify: '{matched_phrase}', "
                    f"pattern: '{pattern}', "
                    f"current label: {label}, "
                    f"{'softmatch:' + str(softmatch_collector) + ', ' if softmatch_collector else ''}"
                    f"target label: {target_label}"
                )
                messages.append({"role": "system", "content": [{"text": req_message, "type": "text"}]})

                # Token accounting (rough)
                for message in messages:
                    num_tokens += len(encoding.encode(message["content"][0]["text"]))
                print(f"INFO: Number of tokens so far {num_tokens}")

                # Call the model (kept as in your code)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0,
                    max_tokens=256,
                    stop=["\n"]
                )

                data = response.choices[0].message.content
                generated_phrases = [x.strip().replace('"', '').replace("'", "") for x in data.split(",") if x.strip()]
                data_collector_2.append([sent_id, sentence, label, pattern, matched_phrase, target_label, generated_phrases])

    df2 = pd.DataFrame(data_collector_2, columns=col_names_2)
    df2.to_csv(f"output_data/[{seed}]{DATA_FILE[:-4]}_candidate_phrases_annotated_data.csv", index=False)

if __name__ == "__main__":
    get_patat_patterns()
    get_candidate_phrases()











