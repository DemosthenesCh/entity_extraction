import yaml

from third_party.nltk_book import chunker


def get_data_config(config_path: str = 'data_config.yaml') -> dict[str, str]:
    with open(config_path) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    return data_config


def get_feature_generator(feature_generator: str = chunker.DEFAULT_FEATURE_GENERATOR) -> chunker.FeatureGenerator:
    feature_generators: dict[str, chunker.FeatureGenerator] = {
        chunker.DEFAULT_FEATURE_GENERATOR: chunker.npchunk_features_bigram_pos,
        'chunk_features_unigram_word': chunker.npchunk_features_unigram_word,
        'chunk_features_unigram_pos': chunker.npchunk_features_unigram_pos,
    }
    if feature_generator not in feature_generators:
        raise ValueError('Invalid feature generator')
    return feature_generators[feature_generator]
