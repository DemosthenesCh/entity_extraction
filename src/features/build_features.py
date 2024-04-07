import click
import dotenv
import json
import logging
import nltk
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from third_party.nltk_book import chunkers


def get_feature_generator(feature_generator: str = chunkers.DEFAULT_FEATURE_GENERATOR) -> chunkers.FeatureGenerator:
    feature_generators: dict[str, chunkers.FeatureGenerator] = {
        chunkers.DEFAULT_FEATURE_GENERATOR: chunkers.npchunk_features_bigram_pos,
        'chunk_features_unigram_word': chunkers.npchunk_features_unigram_word,
        'chunk_features_unigram_pos': chunkers.npchunk_features_unigram_pos,
    }
    if feature_generator not in feature_generators:
        raise ValueError('Invalid feature generator')
    return feature_generators[feature_generator]


def clean_up_synonym_term(synonym: str) -> str:
    return synonym.replace(' ', '').lower()


def get_synonym_maps(synonyms_df: pd.DataFrame, clean_up_terms: bool = True) -> dict[str, dict[str, str]]:
    """
    Creates index and reverse index dictionaries from synonyms dataframe.

    param: synonyms_df: synonyms dataframe.
    param: clean_up_terms: whether to apply synonym cleaning for the terms only, not the synonym preferred name.

    """
    preferred_name_by_term = {}
    terms_by_preferred_name = {}
    for _, row in synonyms_df.iterrows():
        preferred_name = row['preferred_name']
        synonyms = row['synonyms']
        if clean_up_terms:
            synonyms_cleaned = [clean_up_synonym_term(synonym) for synonym in synonyms]
        else:
            synonyms_cleaned = synonyms
        terms_by_preferred_name[preferred_name] = synonyms_cleaned
        for synonym in synonyms_cleaned:
            preferred_name_by_term[synonym] = preferred_name

    return {
        'preferred_name_by_term': preferred_name_by_term,
        'terms_by_preferred_name': terms_by_preferred_name,
    }


def clean_up_text(text: str) -> str:
    return text.lower()


def tokenize_text(text: str) -> list[str]:
    return nltk.word_tokenize(text)


def prerpocess_text(text: str) -> list[str]:
    cleaned_text = clean_up_text(text)
    tokens = tokenize_text(cleaned_text)
    return tokens


def process_ground_truth(config):
    with open(config['ground_truth_raw_file_name'], 'r') as fin:
        ground_truth = json.loads(fin.read())

    synonyms_df = pd.read_parquet(config['processed_synonyms_file_name'])

    synonym_maps = get_synonym_maps(synonyms_df)
    preferred_name_by_term = synonym_maps['preferred_name_by_term']

    ground_truth_cleaned = {}
    for nct_id, terms in ground_truth.items():
        terms_cleaned = []
        for term in terms:
            cleaned = clean_up_synonym_term(term)
            if cleaned not in preferred_name_by_term:
                continue
            terms_cleaned.append(preferred_name_by_term[cleaned])
        ground_truth_cleaned[nct_id] = list(set(terms_cleaned))

    with open(config['ground_truth_cleaned_file_name'], 'w') as fout:
        fout.write(json.dumps(ground_truth_cleaned))


def process_synonyms(config):
    df = pd.read_csv(config['raw_synonyms_file_name'])
    logging.info(df.describe(include='all'))
    profile_raw = ProfileReport(
        df,
        title='Raw Synonyms Profiling Report',
        correlations=None,
    )

    profile_raw.to_file(config['raw_synonyms_profile_file_name'])

    logging.info(f'Deleting synonym groups that do not have a preferred name.')
    check_df = df.groupby('id').agg(count_preferred_name=('preferred_name', 'sum'))
    df = df.set_index(['id']).drop(labels=check_df[check_df.count_preferred_name == 0].index)
    df = df.reset_index()
    del check_df
    assert df.syn_id.unique().shape[0] == df.shape[0], 'syn_id must be unique'
    assert df.name.unique().shape[0] == df.shape[0], 'name must be unique'
    # assert df.lname.unique().shape[0] == df.shape[0]

    logging.info('Grouping synonyms.')
    df_preferred_names = df[df.preferred_name == 1][['id', 'name']].copy()
    df_preferred_names.rename(columns={'name': 'preferred_name'}, inplace=True)
    df_processed = df[['id', 'name', 'lname']].copy().merge(df_preferred_names[['id', 'preferred_name']].copy(), on='id')

    synonyms = df_processed.groupby('id').agg({
        # 'name': np.unique,
        # 'lname': np.unique,
        'name': lambda names: np.unique([clean_up_text(name) for name in names]),
        'lname': lambda names: np.unique([clean_up_text(name) for name in names]),
        'preferred_name': lambda g: g.iloc[0],
    })
    synonyms['synonyms'] = synonyms.apply(
        lambda row: np.unique(np.concatenate([row['name'], row['lname']], axis=0)),
        axis=1
    )
    del synonyms['name'], synonyms['lname']
    del df_processed, df_preferred_names
    synonyms.to_parquet(path=config['processed_synonyms_file_name'])

    profile = ProfileReport(
        synonyms,
        title='Synonyms Profiling Report',
        correlations=None,
    )
    profile.to_file(config['synonyms_profile_file_name'])


@click.command()
@click.argument('config_path', type=click.Path(exists=True), default=dotenv.find_dotenv())
def main(config_path):
    logger = logging.getLogger(__name__)
    config = dotenv.dotenv_values(config_path)

    logger.info('Preparing Ground truth.')
    process_ground_truth(config=config)

    logger.info(f'Processing Synonyms.')
    process_synonyms(config=config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
