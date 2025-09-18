# 05_AL_testing.py  — minimal-change version; only Counterfactual runs by default
import os
import logging
import random
import pandas as pd
import sys
import copy

from sklearn.metrics import f1_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
import torch
import tiktoken

from openai import OpenAI
import configparser


# ======================
# TOGGLES (leave code, skip runs)
# ======================
RUN_RANDOM = False
RUN_CLUSTER = False
RUN_UNCERTAINTY = False
RUN_COUNTERFACTUAL = True
RUN_NON_VT = False   # keep code; skip run unless you enable and provide file


# ======================
# Config & globals
# ======================
config = configparser.ConfigParser()
config.read('config.ini')

DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
TEST_FILE = (
    config.get("settings", "testing_file").split('#')[0].strip()
    if config.has_option("settings", "testing_file")
    else config.get("settings", "test_file").split('#')[0].strip()
)
SEED = config.get("settings", "seed").split('#')[0].strip()
API_KEY = config.get("settings", "openai_api").split('#')[0].strip()

seed = int(SEED)
random.seed(seed)
np.random.seed(seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
client = OpenAI(api_key=API_KEY)


# ======================
# Helpers
# ======================
def num_tokens_from_string(string: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(enc.encode(string))


def get_initial_message(label_list):
    return [
        {"role": "system",
         "content": "You will predict one label for a text. All the labels have been masked, and you need to learn the pattern from given examples"},
        {"role": "system",
         "content": f"You must choose label from the following list: {label_list}"},
        {"role": "user",
         "content": "the doctor does not seem to be very knowledgeable, their treatment did not work."},
        {"role": "assistant",
         "content": "concept B"}
    ]


def get_initial_message_with_confidence(label_list):
    return [
        {"role": "system",
         "content": "You will predict one label for a text. All the labels have been masked, and you need to learn the pattern from given examples"},
        {"role": "system",
         "content": f"You must choose label from the following list: {label_list}"},
        {"role": "user",
         "content": "the doctor does not seem to be very knowledgeable, their treatment did not work."},
        {"role": "assistant",
         "content": "concept B, confidence: 0.806"}
    ]


def get_response(messages, query, model="gpt-4o-mini"):
    query_messages = copy.deepcopy(messages)
    query_messages.append({"role": "user", "content": query})
    resp = client.chat.completions.create(
        model=model, messages=query_messages, temperature=0, max_tokens=256, stop=["\n"]
    )
    return resp.choices[0].message.content


def get_response_with_confidence(messages, query, model="gpt-4o-mini"):
    query_messages = copy.deepcopy(messages)
    query_messages.append({"role": "user", "content": "return the confidence of the prediction along with the prediction"})
    query_messages.append({"role": "user", "content": query})
    resp = client.chat.completions.create(
        model=model, messages=query_messages, temperature=0, max_tokens=256, stop=["\n"]
    )
    return resp.choices[0].message.content


def update_example(messages, text, label):
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": label})


def update_example_with_confidence(messages, text, label):
    # same behavior as original (random conf)
    rc = round(random.uniform(0.1, 1), 3)
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": f"{label}, confidence: {rc}"})


# ======================
# Clustering helper (kept, with tiny guardrails)
# ======================
def get_clusters(shuffled_df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    all_text = shuffled_df['ori_text'].tolist()
    embeddings = model.encode(all_text, convert_to_tensor=True, show_progress_bar=True, device=device)
    embeddings = embeddings.cpu().numpy()

    # guard: PCA components cannot exceed samples/features
    n_components = min(50, embeddings.shape[0], embeddings.shape[1])
    if n_components < 2:
        n_components = max(1, n_components)
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # guard: k cannot exceed number of samples
    k = min(5, len(shuffled_df)) if len(shuffled_df) > 0 else 1
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(reduced_embeddings)
    clusters = kmeans.labels_
    cluster_indices = defaultdict(list)
    for idx, cid in enumerate(clusters):
        cluster_indices[cid].append(idx)
    return cluster_indices


# ======================
# Strategies (kept)
# ======================
def random_shots(df, df_test, unique_ids, label_map, N, test_num, shuffleSeed):
    selections = [10]  # original used 10
    results = []

    df_unique = df.drop_duplicates(subset='id')
    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

    for selection in selections:
        messages = get_initial_message(list(label_map.values()))

        for i in range(selection):
            if i >= len(shuffled_df):
                print("End of data ", i)
                break
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

        correct_count = 0
        valid_count = 0
        all_count = 0
        y_true, y_pred = [], []

        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

        for j in range(0, test_num):
            print("Getting  ", j, " of  ", test_num)
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            if df_test_shuffled['Label'][j] == "none":
                continue

            response = get_response(messages, df_test_shuffled['example'][j])
            if df_test_shuffled['Label'][j] not in label_map:
                continue

            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

            if response == label_map[df_test_shuffled['Label'][j]]:
                correct_count += 1
            if response in list(label_map.values()):
                valid_count += 1
            all_count += 1

        fscore = f1_score(y_true, y_pred, average='macro')
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])
        logger.info(f"Random, {selection}, {test_num}, {correct_count}, {valid_count}, {all_count}, {fscore}")
        print("Random: ", selection, test_num, correct_count, valid_count, all_count, fscore)

    out = pd.DataFrame(results, columns=['shots', 'precision', 'recall', 'fscore'])
    out.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]random_{DATA_FILE[:-4]}.csv", index=False)


def uncertainty_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]
    selected_ids = []
    results = []

    df_unique = df.drop_duplicates(subset='id').sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
    remaining_df = df_unique
    uncertainty_indices = []

    for idx_sel, selection in enumerate(selections):
        messages = get_initial_message_with_confidence(list(label_map.values()))
        if idx_sel == 0:
            # pick first 'selection' examples
            picked = min(selection, len(df_unique))
            selected_ids.extend(df_unique['id'][:picked].tolist())
            for idx in selected_ids:
                row = df_unique[df_unique['id'] == idx].iloc[0]
                update_example_with_confidence(messages, row['ori_text'], label_map[row['ori_label']])
        else:
            # from uncertainty list
            annotation_count = selection - len(selected_ids)
            if annotation_count > 0 and len(uncertainty_indices) > 0 and len(remaining_df) > 0:
                top_uncertainty = uncertainty_indices[:annotation_count]
                for u_idx in top_uncertainty:
                    if u_idx < 0 or u_idx >= len(remaining_df):
                        continue
                    selected_ids.append(remaining_df['id'].iloc[u_idx])
                    text = remaining_df['ori_text'].iloc[u_idx]
                    label = remaining_df['ori_label'].iloc[u_idx]
                    update_example_with_confidence(messages, text, label_map[label])

        # recompute remaining
        remaining_df = remaining_df[~remaining_df['id'].isin(selected_ids)].reset_index(drop=True)
        probabilities = []
        if len(remaining_df) > 0:
            for i in range(len(remaining_df)):
                text = remaining_df['ori_text'].iloc[i]
                response = get_response_with_confidence(messages, text)
                try:
                    response_label = response.split(",")[0]
                    response_confidence = float(response.split(",")[1].split(":")[1])
                except Exception:
                    response_label = response
                    response_confidence = 0.5
                probabilities.append(float(response_confidence))

            probabilities = np.array(probabilities)
            uncertainties = 1 - probabilities
            uncertainty_indices = list(np.argsort(uncertainties)[::-1])
        else:
            uncertainty_indices = []

        # evaluate
        y_true, y_pred = [], []
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
        for j in range(0, test_num):
            print("Getting  ", j, " of  ", test_num)
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            if df_test_shuffled['Label'][j] == "none":
                continue

            response = get_response_with_confidence(messages, df_test_shuffled['example'][j])
            try:
                response_label = response.split(",")[0]
            except Exception:
                response_label = response

            if df_test_shuffled['Label'][j] not in label_map:
                continue
            y_true.append(label_map[df_test_shuffled['Label'][j]])
            # small nudge (kept from original spirit)
            response_label = (
                label_map[df_test_shuffled['Label'][j]]
                if label_map[df_test_shuffled['Label'][j]] in response_label
                else response_label
            )
            y_pred.append(response_label)

        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

    out = pd.DataFrame(results, columns=['shots', 'precision', 'recall', 'fscore'])
    out.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]_uncertainty_{DATA_FILE[:-4]}_prf.csv", index=False)


def clustered_shots(df, df_test, unique_ids, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]
    results = []

    df_unique = df.drop_duplicates(subset='id').sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
    cluster_indices = get_clusters(df_unique)

    for selection in selections:
        messages = get_initial_message(list(label_map.values()))
        selected_indices = []  # reset per selection

        # sample 1 per cluster
        for _, indices in cluster_indices.items():
            if len(indices) > 0:
                selected_indices.extend(random.sample(indices, 1))
        # top-up until 'selection'
        all_candidates = [ix for _, indices in cluster_indices.items() for ix in indices]
        while len(selected_indices) < min(selection, len(df_unique)) and len(selected_indices) < len(all_candidates):
            choice_pool = [ix for ix in all_candidates if ix not in selected_indices]
            if not choice_pool:
                break
            selected_indices.append(random.choice(choice_pool))

        for i in selected_indices:
            update_example(messages, df_unique['ori_text'][i], label_map[df_unique['ori_label'][i]])

        y_true, y_pred = [], []
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
        for j in range(0, test_num):
            print("Getting  ", j, " of  ", test_num)
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            if df_test_shuffled['Label'][j] == "none":
                continue

            response = get_response(messages, df_test_shuffled['example'][j])
            if df_test_shuffled['Label'][j] not in label_map:
                continue

            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

    out = pd.DataFrame(results, columns=['shots', 'precision', 'recall', 'fscore'])
    out.to_csv(f"output_data/archive/gpt/[{shuffleSeed}][GPT]_cluster_{DATA_FILE[:-4]}_prf.csv", index=False)


def counterfactual_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    df_unique = df.drop_duplicates(subset='id')
    selections = [min(10, len(df_unique))]  # keep small for smoke tests
    results = []

    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

    for selection in selections:
        messages = get_initial_message(list(label_map.values()))

        for i in range(selection):
            if i >= len(shuffled_df):
                print("End of data ", i)
                continue
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

            current_id = shuffled_df['id'][i]
            counters = df[(df['id'] == current_id)
                          & (df['matched_pattern'])
                          & (df['heuristic_filtered'])
                          & (~df['is_ori'])
                          & (df['is_target'])]
            if counters.shape[0] > 4:
                counters = counters.sample(n=4, random_state=seed).reset_index(drop=True)
            for _, row in counters.iterrows():
                if row['target_label'] not in label_map:
                    continue
                update_example(messages, row['counterfactual'], label_map[row['target_label']])

        y_true, y_pred = [], []
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
        for j in range(0, test_num):
            print("Getting  ", j, " of  ", test_num)
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            if df_test_shuffled['Label'][j] == "none":
                continue

            # token guard kept (logging only)
            tokens = num_tokens_from_string(str(messages))
            if tokens > 16385:
                print(f"End of tokens {tokens}, examples {j}")

            try:
                response = get_response(messages, df_test_shuffled['example'][j])
            except Exception as e:
                print("ERROR: ", j, e)
                continue

            if df_test_shuffled['Label'][j] not in label_map:
                print("Label not in label map ", df_test_shuffled['Label'][j])
                continue

            true_label = label_map[df_test_shuffled['Label'][j]]
            y_true.append(true_label)
            y_pred.append(response)

        # log PRF
        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

    out = pd.DataFrame(results, columns=['shots', 'precision', 'recall', 'fscore'])
    out_path = f"output_data/archive/gpt/[{shuffleSeed}][GPT]_counter_{DATA_FILE[:-4]}_prf.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved PRF to {out_path}")
    return prf


def non_VTcounterfactual_shots(df, df_test, label_map, N, test_num, shuffleSeed):
    selections = [10, 15, 30, 50, 70, 90, 120]
    results = []

    df_unique = df.drop_duplicates(subset='id')
    shuffled_df = df_unique.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)

    for selection in selections:
        messages = get_initial_message(list(label_map.values()))
        for i in range(selection):
            if i >= len(shuffled_df):
                print("End of data ", i)
                continue
            update_example(messages, shuffled_df['ori_text'][i], label_map[shuffled_df['ori_label'][i]])

            current_id = shuffled_df['id'][i]
            counters = df[(df['id'] == current_id)]
            if counters.shape[0] > 4:
                counters = counters.sample(n=4, random_state=seed).reset_index(drop=True)
            for _, row in counters.iterrows():
                if row['target_label'] not in label_map:
                    continue
                update_example(messages, row['counterfactual'], label_map[row['target_label']])

        y_true, y_pred = [], []
        df_test_shuffled = df_test.sample(frac=1, random_state=shuffleSeed).reset_index(drop=True)
        for j in range(0, test_num):
            print("Getting  ", j, " of  ", test_num)
            if j >= len(df_test_shuffled):
                print("End of data ", j)
                break
            if df_test_shuffled['Label'][j] == "none":
                continue

            try:
                response = get_response(messages, df_test_shuffled['example'][j])
            except Exception as e:
                print("ERROR: ", j, e)
                continue

            if df_test_shuffled['Label'][j] not in label_map:
                continue

            y_true.append(label_map[df_test_shuffled['Label'][j]])
            y_pred.append(response)

        prf = precision_recall_fscore_support(y_true, y_pred, average='macro')
        results.append([selection, prf[0], prf[1], prf[2]])

    out = pd.DataFrame(results, columns=['shots', 'precision', 'recall', 'fscore'])
    out.to_csv(f"output_data/archive/gpt/nonVT[{shuffleSeed}][GPT]_counter_{DATA_FILE[:-4]}_prf.csv", index=False)


# ======================
# Main
# ======================
if __name__ == "__main__":
    os.makedirs("output_data/archive/gpt", exist_ok=True)

    # training file: prefer non-finetuned; fallback to [finetuned]
    training_primary = f"output_data/[{SEED}]filtered_{DATA_FILE}"
    training_finetuned = f"output_data/[{SEED}][finetuned]filtered_{DATA_FILE}"
    if os.path.exists(training_primary):
        training_path = training_primary
    elif os.path.exists(training_finetuned):
        training_path = training_finetuned
    else:
        print(f"ERROR: cannot find training file at either:\n  {training_primary}\n  {training_finetuned}")
        sys.exit(1)
    print(f"\nUsing training file: {training_path}")

    test_path = f"input_data/{TEST_FILE}"
    if not os.path.exists(test_path):
        print(f"ERROR: can not read file {test_path}")
        sys.exit(1)

    # optional non-VT path (only read if RUN_NON_VT True and it exists)
    nonVT_path = f"output_data/non_VT_counter/[{SEED}]{DATA_FILE[:-4]}_non_VT_counterfactuals.csv"

    # load
    df = pd.read_csv(training_path)
    df_test = pd.read_csv(test_path)

    if RUN_NON_VT and os.path.exists(nonVT_path):
        non_VT_df = pd.read_csv(nonVT_path)
    elif RUN_NON_VT:
        print(f"Skipping non-VT counters (not found): {nonVT_path}")
        non_VT_df = None
    else:
        non_VT_df = None

    # label mapping
    label_list = df["ori_label"].unique().tolist()
    ids = df["id"].unique().tolist()
    label_map = {label: f'concept {chr(65 + i)}' for i, label in enumerate(label_list)}

    # logger (kept)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f'AL_results_random_{DATA_FILE[:-4]}.log',
                        encoding='utf-8', level=logging.INFO)

    # evaluate on 20 test cases now (auto-scales down if fewer available)
    test_num = min(20, len(df_test))

    # single seed for smoke test; add more later if desired
    seedss = [42]

    for shuffleseed in seedss:
        if RUN_CLUSTER:
            clustered_shots(df, df_test, ids, label_map, 30, test_num, shuffleSeed=shuffleseed)
        if RUN_RANDOM:
            random_shots(df, df_test, ids, label_map, 30, test_num, shuffleSeed=shuffleseed)
        if RUN_COUNTERFACTUAL:
            counterfactual_shots(df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
        if RUN_UNCERTAINTY:
            uncertainty_shots(df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
        if RUN_NON_VT and non_VT_df is not None:
            non_VTcounterfactual_shots(non_VT_df, df_test, label_map, 30, test_num, shuffleSeed=shuffleseed)
