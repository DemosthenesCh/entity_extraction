entity_extraction
==============================

Entity recognition for drug names in clinical trial short description text.

# Installation
In a new virtual environment (Python version 3.10) install the packages in `requirements.txt`

# Configuration
Create the `.env` file by copying `.env_template` and fill in the `postgresql_username`, `postgresql_password`,
`gemini_api_key`, and `base_dir` variables. note that `base_dir` needs to be the full path. you can optionally change the input and output data locations.
```bash
cp .env_template .env
```

Get the postgresql username and password from https://drugcentral.org/download.

To get the api key for gemini follow these instructions: https://ai.google.dev/tutorials/setup.

# Steps
## 1. Download data
The `get_data.py` script downloads the following data:
 * `public.synonyms` table from drugcentral.org
 * the study brief summary from [clinicaltrials.gov](https://clinicaltrials.gov)'s api

Studies are defined by their nct ids `data/nctids.csv`.
It also builds the ground truth data which contains terms that are copied verbatim from the shorDescription field. It contains duplicates.

  
To download the data, you need to have prepared the `.env` file (see the [configuration](#configuration) section above) and run the following script:
```bash
python -m src.data.get_data .env
```

This will store a csv file with synonyms and parquet file with the summaries under the data folder.

TODO: Update
## 2. Process the data to generate features.
Run the `src/features/build_features.py script`. This script:
 * Processes the ground truth.
 * Processes the synonyms data.
 * Prepares features for the NLTK extraction model.

```bash
python -m src.features.build_features .env
```

TODO: Update
## 3. Train and evaluate non-LLM drug recognizer
Tun the notebook `extract_entities_nltk.ipynb`

TODO: Update
## 4. Score and evaluate LLM drug recognizer
Tun the notebook `extract_entities_nltk.ipynb`


# Third Party Code

The `third_party` package contains code from other sources. More specifically:
 * `third_party/nltk_book` contains modified snippets from the [nltk book](https://www.nltk.org/book/). its licence [permits use of code examples without permission](https://www.nltk.org/book/ch00.html#using-code-examples).
 * 




TODO: Update

Project Organization
------------

    ├── license
    ├── makefile           <- makefile with commands like `make data` or `make train`
    ├── readme.md          <- the top-level readme for developers using this project.
    ├── data
    │   ├── external       <- data from third party sources.
    │   ├── interim        <- intermediate data that has been transformed.
    │   ├── processed      <- the final, canonical data sets for modeling.
    │   └── raw            <- the original, immutable data dump.
    │
    ├── docs               <- a default sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- jupyter notebooks. naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- generated analysis as html, pdf, latex, etc.
    │   └── figures        <- generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- the requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- source code for use in this project.
    │   ├── __init__.py    <- makes src a python module
    │   │
    │   ├── data           <- scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
