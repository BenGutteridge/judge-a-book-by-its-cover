# Judge a Book by Its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription

Code for reproducing experiments in the paper "Judge a Book by its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription", submitted to ICLR 2026. 

### Installation
For Unix:
```
conda create --name judge python=3.10
conda activate judge
pip install -r requirements.txt
pip install -e .
cp .example.env .env
mkdir results
```
Then set up API keys in .env.

### Data

Some experiments are based on [IAM Handwriting DB](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), which should be downloaded and unzipped inside `/data/IAM Handwriting DB` before running `notebooks/00_iam_preprocessing.py`. 

(Registration to access the dataset can be faulty, I found [these instructions](https://www.reddit.com/r/datasets/comments/l2agom/comment/ksww8co/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) to be convoluted but effective.)

The **Bentham** dataset can be downloaded [here](https://zenodo.org/records/44519).

The **Malvern-Hills** dataset can be downloaded [here](https://judgeocr.s3.eu-north-1.amazonaws.com/malvern_hills_trust.zip).

All cached LLM results can be downloaded from [here](https://judgeocr.s3.eu-north-1.amazonaws.com/gpt_cache_dbs.zip) as SQLite DBs. Unzip the .db files into `data/` and all LLM calls in notebooks/01_experiments.py should use cached results. If you start to see costs and API calls being logged to the cache, you aren't using the caches; check your filepaths. By default, the `only_load_from_cache` argument of  is set to True; change it 


### Running Experiments
Experimental results can be reproduced with the notebooks below as follows:
- `notebooks/00_{iam, malvern_hills, bentham}_preprocessing.py`: run OCR engines on the datasets, save to pandas dataframes, produce multi-page datasets and save dataframes for downstream notebooks to .pkl files
- `notebooks/01_experiments.py`: make OpenAI API calls with various prompting strategies to get improved OCR transcriptions
  - Can be run as a notebook or using command line arguments with .yaml config files in `configs/`

### Commands to run all exps
```
# Pre-processing
python notebooks/00_iam_preprocessing.py
python notebooks/00_malvern_hills_preprocessing.py
python notebooks/00_bentham_preprocessing.py

# Experiments
## IAM
python notebooks/01_experiments.py task iam_multipage_minpages=02 model gemma-3-27b-it
python notebooks/01_experiments.py task iam_multipage_minpages=02 model gpt-4o
python notebooks/01_experiments.py task iam_multipage_minpages=02 model gemini-2.5-pro
## MHills
python notebooks/01_experiments.py task malvern_hills_multipage model gemma-3-27b-it
python notebooks/01_experiments.py task malvern_hills_multipage model gpt-4o
python notebooks/01_experiments.py task malvern_hills_multipage model gemini-2.5-pro
## Bentham
python notebooks/01_experiments.py task bentham_multipage_consecutive model gemma-3-27b-it
python notebooks/01_experiments.py task bentham_multipage_consecutive model gpt-4o
python notebooks/01_experiments.py task bentham_multipage_consecutive model gemini-2.5-pro

# Evaluation
python notebooks/02_experiments_eval.py
```

**N.B.** 
- Though we use a seed, OpenAI and Gemini API calls are [not strictly reproductible](https://platform.openai.com/docs/advanced-usage#reproducible-outputs), so slight differences in output are possible
- For a few API calls the model failed to return valid JSON, so nothing was cached. In these cases, an 'API CALL FAILED' error will show, but this is expected; the returned output is empty, the transcription is evaluated as failed.
- All prompts used for experiments can be found in `configs/prompts.json`