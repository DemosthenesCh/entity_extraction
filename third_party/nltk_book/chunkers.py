from typing import Callable, Any
import nltk

TaggedSentence = list[tuple[str, str]]
FeatureGenerator = Callable[[TaggedSentence, int, Any], dict[str, str]]


DEFAULT_FEATURE_GENERATOR = 'npchunk_features_bigram_pos'


def npchunk_features_bigram_pos(sentence: TaggedSentence, i: int, history: Any) -> dict[str, str]:
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = '<START>', '<START>'
    else:
        prevword, prevpos = sentence[i-1]
    return {'pos': pos, 'prevpos': prevpos}


def npchunk_features_unigram_pos(sentence: TaggedSentence, i: int, history: Any) -> dict[str, str]:
    word, pos = sentence[i]
    return {"pos": pos}


def npchunk_features_unigram_word(sentence: TaggedSentence, i: int, history: Any) -> dict[str, str]:
    word, pos = sentence[i]
    return {"word": word}


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents, feature_generator_fn: FeatureGenerator):
        self.feature_generator_fn = feature_generator_fn
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = self.feature_generator_fn(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = self.feature_generator_fn(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents, feature_generator_fn: FeatureGenerator):
        self.feature_generator_fn = feature_generator_fn
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents, feature_generator_fn)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)
