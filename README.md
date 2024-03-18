# Installation
In a new virtual environment (python version 3.10) install the packages in `requirements.txt`

# Configuration
To change the input and output data locations update the file `data_config.yaml`.

# Steps
## 1. Download Data
The `get_data.py` script downloads the following data:
 * `public.synonyms` table from drugcentral.org
 * The Study brief summary from [clinicaltrials.gov](https://clinicaltrials.gov)'s API

Studies are defined by their NCT IDs `data/nctids.csv`.
  
To download the data, get the username and password from https://drugcentral.org/download and run the following script:
```
python get_data.py --username myusername --password mypassword
```

This will store a csv file with synonyms and parquet file with the summaries under the data folder.

## 2. Process the synonyms
Run the notebook `process_synonyms.ipynb`.

## 3. Train and evaluate non-LLM drug recognizer
Run the notebook `extract_entities_nltk.ipynb`
