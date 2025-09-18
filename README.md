# Variation-Theory-in-Counterfactual-Data-Augmentation

## Quick Start

### 1. Clone the Repository
```bash
git clone <repo-url>
cd VariationTheory
```

### 2. Create and Activate a Virtual Environment

**Recommended:** Python 3.10

Create your virtual environment with:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

If you don't have Python 3.10 installed, you can install it on macOS with:
```bash
brew install python@3.10
/opt/homebrew/bin/python3.10 -m venv venv
```

> **Note:**  
> Avoid Python 3.13, as some dependencies may not be compatible.

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Create and Configure `config.ini`
Copy the example config file and edit it:
```bash
cp config.ini.example config.ini
```
Then edit `config.ini` and set these values:
```
[settings]
environment= #leave blank if using venv
openai_api=sk-...        # your OpenAI API key
data_file=input_data.csv # your data file
testing_file=test_data.csv # your test file
seed=42
```

### 6. Run the Pipeline
You can run all scripts in sequence using:
```bash
bash run_scripts.sh
```

Or run individual scripts, e.g.:
```bash
python 05_AL_testing.py
```

### 7. Output Files
Results are saved in:
```
output_data/archive/gpt/
```
Each run may overwrite previous output files with the same name.

---

**Note:**  
- Make sure your Python version matches the one specified in the project.
- If you use conda, set up the environment name in `config.ini` under `environment=...`

## Files

### 01_data_formatting
This file iteratively trains the symbolic model and generates patterns using the data's ground truth. The initial run may take several minutes due to caching, but subsequent runs should be quicker.

### 02_counterfactual_over_generation
This file uses the candidate phrases generated in the previous file to generate the counterfactual examples that will be used for finetuning a GPT-3.5.

### 03_counterfactual_filtering
This file uses the three level filters -- heuristic, symbolic, and GPT-discriminator -- to assess the quality of previously generated counterfactuals.

### 04_fine_tuning (Not using)
This file fine-tunes a GPT-3.5 model to generate counterfactual data.

### 05_AL_testing_BERT(Not using), 05_AL_testing
These files iteratively train BERT and GPT models using the different cases and measure the performance of the trained models.