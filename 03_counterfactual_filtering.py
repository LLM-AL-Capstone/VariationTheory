import os
import sys
import re
import configparser
import pandas as pd

from PaTAT_piped.helpers import expand_working_list, match_positives, get_similarity_dict
from PaTAT_piped.api_helper import APIHelper

from openai import OpenAI
import spacy
from spacy.matcher import Matcher

config = configparser.ConfigParser()
config.read('config.ini')

API_KEY = config.get("settings", "openai_api").split('#')[0].strip()
DATA_FILE = config.get("settings", "data_file").split('#')[0].strip()
SEED = config.get("settings", "seed").split('#')[0].strip()
seed = int(SEED)

client = OpenAI(api_key=API_KEY)

nlp = spacy.load("en_core_web_sm")

# ensure output dirs exist
os.makedirs("output_data/interim_output", exist_ok=True)

# heuristic filtering: check if the counterfactual is a valid sentence
def heuristic_filtering(df, finetuned=False):
    herustic_filtered = []
    for _, row in df.iterrows():
        invalid = False
        invalid = bool(re.search("given the constraints", str(row['counterfactual']), re.IGNORECASE))
        invalid = invalid and bool(re.search("the assistant", str(row['counterfactual']), re.IGNORECASE))
        herustic_filtered.append(not invalid)

    df['heuristic_filtered'] = herustic_filtered

    if not finetuned:
        df.to_csv(f'output_data/interim_output/[{seed}]heuristic_filtered_{DATA_FILE}', index=False)
    else:
        df.to_csv(f'output_data/interim_output/[{seed}][finetuned]heuristic_filtered_{DATA_FILE}', index=False)
    return df

# Symbolic filtering: check if the pattern matches the counterfactual
def symbolic_filtering(df, similarity_dict, finetuned=False):
    matched_pattern_list = []
    total_true = 0
    total_false = 0

    for _, row in df.iterrows():
        pattern = row['pattern']
        counterfactual = row['counterfactual']

        pattern_working_list = expand_working_list(pattern, soft_match_on=True, similarity_dict=similarity_dict)
        matched_test = check_matching(counterfactual, pattern_working_list)

        matched_pattern_list.append(matched_test)
        if matched_test:
            total_true += 1
        else:
            total_false += 1

    df['matched_pattern'] = matched_pattern_list

    denom = (total_true + total_false) if (total_true + total_false) else 1
    print("The pattern keeping rate is: ", total_true / denom)

    if finetuned:
        df.to_csv(f'output_data/interim_output/[{seed}][finetuned]symbolic_filtered_{DATA_FILE}', index=False)
    else:
        df.to_csv(f'output_data/interim_output/[{seed}]symbolic_filtered_{DATA_FILE}', index=False)

    return df

def check_matching(sent, working_list, explain=False):
    matcher = Matcher(nlp.vocab)
    for index, patterns in enumerate(working_list):
        matcher.add(f"rule{index}", [patterns])
    doc = nlp(str(sent))

    matches = matcher(doc)
    if matches is not None and len(matches) > 0:
        for _, start, end in matches:
            if (str(doc[start:end]).strip() != ""):
                return True
    return False

# GPT discriminator filtering: check if the counterfactual changes to the target label
def gpt_discriminator_filtering(df, finetuned=False):
    label_list = df['ori_label'].unique()
    print('The number of generations to be evaluated: ', len(df))

    is_ori = []
    is_target = []
    is_ori_count = 0
    ori_invalid_count = 0
    is_counter_count = 0
    counter_invalid_count = 0
    finetune_data = []
    finetune_rows = []
    filter_out_rows = []
    flagg = False

    for i in range(len(df)):
        chosen_flag = 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "The assistant will judge whether a text belongs to a certain topic, the text is from real Amazon shopping reviews"},
                {"role": "system", "content": f"There are {len(label_list)} different topics, they are: {label_list}"},
                {"role": "user", "content": "Two labels will be given, seperated by comma. You need to make two independent judgements. Return 'YES' if the text belongs to a label, otherwise return 'NO'. Your two responses should also be seperated by comma. Follow the examples below:"},
                {"role": "user", "content": "great customer service , great food that is fresh and cooked to order\nLabel: customerservice, product"},
                {"role": "assistant", "content": "YES, YES"},
                {"role": "user", "content": "i bought a living social deal $ 30 for $ 60 awhile back.\nLabel: price, environment"},
                {"role": "assistant", "content": "YES, NO"},
                {"role": "user", "content": "I visited a living social event, a 30 people gathering in a 60 square meter area awhile back.\nLabel: price, product"},
                {"role": "assistant", "content": "NO, NO"},
                {"role": "user", "content": "All the latest news about the world of  Movies, Theater,  TV, Popular Music and Books.  Well written and illustrated.\nLabel: magazine subscriptions, artcrafts and sewing"},
                {"role": "assistant", "content": "YES, NO"},
                {"role": "user", "content": f"{df['counterfactual'][i]}\nLabel: {df['ori_label'][i], df['target_label'][i]}"},
            ],
            max_tokens=10,
            temperature=0,
        ).choices[0].message.content

        try:
            response_ori, response_counter = response.split(', ')
        except Exception:
            print("BAD RESPONSE ", response)
            response_ori = 'no'
            response_counter = 'no'
            flagg = True

        # is not original label
        if response_ori.lower() == 'yes':
            is_ori_count += 1
            chosen_flag = 0
            is_ori.append(True)
        elif response_ori.lower() == 'no':
            is_ori.append(False)
        else:
            ori_invalid_count += 1
            is_ori.append(False)

        # is counter label
        if response_counter.lower() == 'yes':
            is_counter_count += 1
            is_target.append(True)
        elif response_counter.lower() == 'no':
            is_target.append(False)
        else:
            chosen_flag = 0
            if response_counter.lower() != 'no':
                counter_invalid_count += 1
                is_target.append(False)

        if chosen_flag:
            valid_message = {
                "messages": [
                    {"role": "system", "content": "The assistant will make small changes to the given text to change the topic label, but the pattern contained in the text content cannot be changed"},
                    {"role": "user", "content": f'''{df['ori_text'][i]}
Original label: {df['ori_label'][i]}
Pattern: {df['pattern'][i]}
Target label: {df['target_label'][i]}'''}, 
                    {"role": "assistant", "content": df['counterfactual'][i]},
                ]
            }
            finetune_data.append(valid_message)
            finetune_rows.append([df['id'][i], df['ori_text'][i], df['ori_label'][i], df['pattern'][i], df['counterfactual'][i], df['target_label'][i]])
        else:
            filter_out_rows.append([df['id'][i], df['ori_text'][i], df['ori_label'][i], df['pattern'][i], df['counterfactual'][i], df['target_label'][i]])

        if i % 30 == 0:
            print(f"{i}/{len(df)}")

    counter_column = ['id', 'ori_text', 'ori_label', 'pattern', 'counter_text', 'counter_label']
    df_finetune = pd.DataFrame(finetune_rows, columns=counter_column)
    df_out = pd.DataFrame(filter_out_rows, columns=counter_column)
    print("The percentage of is not original label: ", 1 - is_ori_count / (len(df) if len(df) else 1))
    print("The percentage of is the target counterfactual label: ", is_counter_count / (len(df) if len(df) else 1))
    print("Invalid percentage in ori test: ", ori_invalid_count / (len(df) if len(df) else 1))
    print("Invalid percentage in counter test: ", counter_invalid_count / (len(df) if len(df) else 1))
    print("The number of finetune dataset is: ", len(df_finetune))

    df['is_ori'] = is_ori
    df['is_target'] = is_target

    if finetuned:
        df.to_csv(f'output_data/[{seed}][finetuned]filtered_{DATA_FILE}', index=False)
    else:
        df.to_csv(f'output_data/[{seed}]filtered_{DATA_FILE}', index=False)
        df_finetune.to_csv(f'output_data/[{seed}]fine_tuneset_{DATA_FILE}', index=False)

if __name__ == "__main__":
    file = f"output_data/[{seed}]counterfactuals_{DATA_FILE}"
    finetuned = False

    if len(sys.argv) > 1:
        file = f"output_data/{sys.argv[1]}"
        finetuned = True

    print(f"Reading file {file}")
    try:
        df = pd.read_csv(file)
    except Exception:
        print(f"ERROR: can not read file {file}")
        sys.exit(1)

    # Load the original dataset to build soft-match similarity dict
    file_path = f"input_data/{DATA_FILE}"
    try:
        data = pd.read_csv(file_path)
        print("INFO: Finished reading data")
    except Exception:
        print(f"ERROR: can not read file {file_path}")
        sys.exit(1)

    # ---- PaTAT wiring (no 'dataset=' kwarg) ----
    STEM = DATA_FILE[:-4] if DATA_FILE.lower().endswith(".csv") else DATA_FILE

    # Ensure a text column named 'example' exists for similarity building
    if 'example' not in data.columns:
        if 'Text' in data.columns:
            data['example'] = data['Text']
        else:
            # fallback: use first text-like column
            first_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
            data['example'] = data[first_col].astype(str)
    data['example'] = data['example'].astype(str)

    df_patat = pd.DataFrame({
        "id": range(len(data)),
        "example": data['example'],
        "Label": data['Label'] if 'Label' in data.columns else ['none'] * len(data),
    })

    patat = APIHelper(user=STEM)
    patat.theme = STEM
    patat.data = df_patat
    patat.words_dict, patat.similarity_dict = get_similarity_dict(
        patat.data["example"].values,
        soft_threshold=patat.soft_threshold,
        soft_topk_on=patat.soft_topk_on,
        soft_topk=patat.topk,
        file_name=patat.theme,
    )
    similarity_dict = patat.similarity_dict
    # --------------------------------------------

    # 1) heuristic filtering
    herustic_filtered_df = heuristic_filtering(df, finetuned=finetuned)

    # 2) symbolic filtering
    symbolic_filtered_df = symbolic_filtering(herustic_filtered_df, similarity_dict, finetuned=finetuned)

    # 3) GPT discriminator filtering
    gpt_discriminator_filtering(symbolic_filtered_df, finetuned=finetuned)
