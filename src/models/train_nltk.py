
import nltk

from src.features import build_features


def parse(text, tagger, chunker):
    """Applies text preprocessing, tagger, and chunker to raw text."""
    tagged = tagger.tag(build_features.preprocess_text(text))
    tree = chunker.parse(tagged)
    return tree


def extract_chunked_terms(tree: nltk.Tree, chunk_tag: str = 'NP') -> list[str]:
    terms = []
    for subtree in tree:
        if type(subtree) != nltk.Tree:
            continue
        if subtree.label() != chunk_tag:
            continue
        term = ''.join([leaf[0] for leaf in subtree.leaves()])
        terms.append(term)
    return terms


class WordNgramTagger(nltk.NgramTagger):
    """
    My override of the NLTK NgramTagger class that considers previous
    tokens rather than previous tags for context.
    """
    def __init__(self, n, train=None, model=None,
                 backoff=None, cutoff=0, verbose=False):
        nltk.NgramTagger.__init__(self, n, train, model, backoff, cutoff, verbose)

    def context(self, tokens, index, history):
        tag_context = tuple(tokens[max(0,index-self._n+1):index])
        return tag_context, tokens[index]
