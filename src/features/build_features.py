import click
from collections import defaultdict
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


def preprocess_text(text: str) -> list[str]:
    cleaned_text = clean_up_text(text)
    tokens = tokenize_text(cleaned_text)
    return tokens


def process_ground_truth(config):
    """Processes ground truth.

     It applies cleaning rules for synonyms uses the preferred name per synonym.
     """
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
    """Processes synonyms.

    Output dataframe contains:
     * id: the synonym id.
     * preferred_name: as they appear in the inputs.
     * synonyms: list of unique and lower-case synonyms.
    """
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

    synonyms_df = df_processed.groupby('id').agg({
        # 'name': np.unique,
        # 'lname': np.unique,
        'name': lambda names: np.unique([clean_up_text(name) for name in names]),
        'lname': lambda names: np.unique([clean_up_text(name) for name in names]),
        'preferred_name': lambda g: g.iloc[0],
    })
    synonyms_df['synonyms'] = synonyms_df.apply(
        lambda row: np.unique(np.concatenate([row['name'], row['lname']], axis=0)),
        axis=1
    )
    del synonyms_df['name'], synonyms_df['lname']
    del df_processed, df_preferred_names
    synonyms_df.to_parquet(path=config['processed_synonyms_file_name'])

    profile = ProfileReport(
        synonyms_df,
        title='Synonyms Profiling Report',
        correlations=None,
    )
    profile.to_file(config['synonyms_profile_file_name'])


def preprocess_summaries(config):
    """Runs text cleaning and tokenization on brief summaries."""
    summaries_df = pd.read_parquet(config['summaries_file_name'])
    # summaries['brief_summary_raw'] = summaries['brief_summary'].copy()
    summaries_df['brief_summary_preprocessed'] = summaries_df['brief_summary'].map(preprocess_text)
    summaries_df.to_parquet(config['preprocessed_summaries_file_name'])


def get_synonym_tags(tokens: list[str]) -> list[tuple[str, str]]:
    """Creates tags for a single synonym word, the word must be tokenized already."""
    synonym_tags = []
    if not tokens:
        return synonym_tags

    for token in tokens[:-1]:
        synonym_tags.append((token, f'C{len(tokens)}S'))
    synonym_tags.append((tokens[-1], f'C{len(tokens)}E'))
    return synonym_tags


def create_tagged_synonyms(config):
    """Tags synonyms and groups them according to n-gram size."""
    synonyms_df = pd.read_parquet(config['processed_synonyms_file_name'])
    dataset_by_ngram = defaultdict(list)
    for synonyms in synonyms_df['synonyms']:
        for synonym in synonyms:
            tokens = preprocess_text(synonym)
            tags = get_synonym_tags(tokens)
            dataset_by_ngram[len(tags)].append(tags)
    with open(config['tagged_synonyms_file_name'], 'w') as fout:
        json.dump(dataset_by_ngram, fout)


def get_random_filler_trees(words: list[str], size=3, tag='O') -> list[nltk.Tree]:
    trees = []
    selection = np.random.choice(words, size).tolist()
    for word in selection:
        # trees.append(nltk.Tree(word, [tag]))
        trees.append((word, tag))
    return trees


def split_summary_by_keywords(brief_summary: str, keywords: list[str]) -> list[str]:
    tokens = [brief_summary]
    for keyword in keywords:
        new_tokens = []
        for token in tokens:
            split_tokens = token.split(keyword)
            for index, split_token in enumerate(split_tokens):
                new_tokens.append(split_token)
                if index < len(split_tokens) - 1:
                    new_tokens.append(keyword)
        tokens = new_tokens
    return tokens


def create_chunk_tree(nodes: list[tuple[str, str]], chunk_tag: str = 'NP') -> nltk.Tree:
    return nltk.Tree(chunk_tag, nodes)
# def create_chunk_tree(nodes: list[tuple[str, str]], chunk_tag: str = 'NP') -> tuple[str, list]:
#     return chunk_tag, nodes


def get_synonym_subtree(synonym):
    tokens = nltk.word_tokenize(synonym)
    synonym_tags = get_synonym_tags(tokens)

    tree = create_chunk_tree(synonym_tags)
    return tree


def get_gt_tree(brief_summary: str, keywords: list[str]) -> nltk.Tree:
    tokens = split_summary_by_keywords(brief_summary, keywords)
    tree = []
    for part in tokens:
        if part in keywords:
            tree.append(get_synonym_subtree(part))
        else:
            tree.extend([(t, 'O') for t in nltk.word_tokenize(part)])
    return nltk.Tree('S', tree)


def create_synonym_ground_truth(config):
    synonyms_df = pd.read_parquet(config['processed_synonyms_file_name'])
    records = []
    for synonyms in synonyms_df['synonyms']:
        for synonym in synonyms:
            records.append({
                'synonym': synonym,
                'synonym_tag_gt': get_synonym_tags(preprocess_text(synonym)),
                'synonym_gt_trees': str(nltk.Tree('S', [get_synonym_subtree(synonym)])),
            })
    pd.DataFrame.from_records(records).to_parquet(config['synonyms_ground_truth_file_name'])


def parse_gt_tree(tree_str: str) -> nltk.Tree:
    tree = nltk.tree.Tree.fromstring(
        tree_str,
        read_leaf=lambda leaf_str: tuple(leaf_str.rsplit('/', 1))
    )
    return tree


@click.command()
@click.argument('config_path', type=click.Path(exists=True), default=dotenv.find_dotenv())
def main(config_path):
    logger = logging.getLogger(__name__)
    config = dotenv.dotenv_values(config_path)

    logger.info('Preparing Ground truth.')
    process_ground_truth(config=config)

    logger.info(f'Processing Synonyms.')
    process_synonyms(config=config)

    logger.info(f'Processing Summaries.')
    preprocess_summaries(config=config)

    logger.info(f'Creating Tagged Synonyms.')
    create_tagged_synonyms(config=config)

    logger.info(f'Creating Ground Truth for Synonyms.')
    create_synonym_ground_truth(config=config)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
