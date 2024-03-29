import nltk
import pandas as pd
import yaml

from third_party.nltk_book import chunkers


def get_data_config(config_path: str = 'data_config.yaml') -> dict[str, str]:
    with open(config_path) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    return data_config


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
