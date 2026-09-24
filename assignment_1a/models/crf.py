"""Course-provided skeleton for Part A: CRF slot filling.

The constructor, `fit()`, and `predict()` are fully implemented — they
wire the model together and call into `sklearn-crfsuite`. The one thing
left to implement is `extract_features()`: turning a tokenized sentence
into a per-token feature dict. Per the README, feature engineering *is*
the Part A assignment — look for the `# TODO: implement` block below.
"""

from __future__ import annotations
from typing import Any, cast
import sklearn_crfsuite


class CRFModel:
    def __init__(self) -> None:
        """Initialize underlying `sklearn-crfsuite` CRF estimator.

        Reasonable defaults are set here; override them by editing this
        constructor or by reconstructing `self.model` with different
        hyperparameters as you experiment.
        
        (some of) CRF class params:
        algorithm - lbfgs | l2sgd | ap | pa | arow
        c1 - coefficient for l1 regularization
        c2 - coefficient for l2 regularization
        max_iterations - the default value depends on the training algorithm
        all_possible_transitions - specifies whether CRFsuite generates
                                   transition features that associate all
                                   possible label pairs.
    
        """
        self.model = sklearn_crfsuite.CRF(
            algorithm = "lbfgs",
            c1 = 0.1,
            c2 = 0.1,
            max_iterations = 100,
            all_possible_transitions = True,
            model_filename = None, 
        )
        self.model_type = 'crf'

    def extract_features(self, sentence: list[str]) -> list[dict[str, Any]]:
        """Convert a tokenized sentence into one feature dict per token.

        This is the part you implement.

        Args:
            sentence: A list of token strings, e.g. ["show", "me", "flights", "to", "boston"].

        Returns:
            A list the same length as `sentence`, where each element is a
            dict of features for the token at that position. Typical
            features to consider: word identity, casing, prefixes/suffixes,
            surrounding-token context window, gazetteers if you build them.
        """
        # ------------------------------------------------------------------
        # feature engineering (v1)
        #
        # this was stolen from sklean-crfsuite's api docs (word2feature function)
        # intent:
        # i think we should build up the features by increasing complexity. let's first start with cases and _fixes.
        # then let's build in surrounding-token context windows.
        # 
        # i will design this method assuming it will be invoked in a loop.
        # per sentence, append a dictionary per token.
        #
        all_features = []
        for t, word in enumerate(sentence):
            features = {
                'bias': 1.0,
                'word.lower()': word.lower(),
                'word[-3:]': word[-3:],   # suffix features (here, i am making an assumption 
                                          # that the last three letters is a suitable suffix for most words in my corpus)
                'word[-2:]': word[-2:],
                'word.isdigit()': word.isdigit(),
            }

            if t == 0:
                features['BOS'] = True
            else:
                prev = sentence[t - 1]
                features.update({
                    '-1:word.lower()': prev.lower(),
                })

            if t == len(sentence) - 1:
                features['EOS'] = True
            else:
                nxt = sentence[t + 1]
                features.update({
                    '+1:word.lower()': nxt.lower(),
                })
            all_features.append(features)

        # ------------------------------------------------------------------


        # this is the context window!
        # ------------------------------------------------------------------
        # feature engineering (v2) 
        # toggle to True.
        if False:
            # the following conditionals only activate for intermediate tokens.
            if t > 0:
                word1 = sentence[t - 1][0] # this is the first character in every preceding token
                #postag1 = sent[t-1][1]
                features.update({
                    '-1:word.lower()': word1.lower(),
                    '-1:word.istitle()': word1.istitle(),
                    '-1:word.isupper()': word1.isupper(),
                    #'-1:postag': postag1,
                    #'-1:postag[:2]': postag1[:2],
                })
            else:
                features['BOS'] = True

            if t < len(sentence) - 1:
                word1 = sentence[t + 1][0] # this is the first character in every subsequent token
                #postag1 = sent[t+1][1]
                features.update({
                    '+1:word.lower()': word1.lower(),
                    '+1:word.istitle()': word1.istitle(),
                    '+1:word.isupper()': word1.isupper(),
                    #'+1:postag': postag1,
                    #'+1:postag[:2]': postag1[:2],
                })

        return all_features # if implemented, the function returns the dict - the error is not invoked.
        # ------------------------------------------------------------------

        raise NotImplementedError("extract_features() is left for you to implement.")

    def fit(self, sentences: list[list[str]], labels: list[list[str]]) -> None:
        """Train CRF on tokenized sentences and their BIO label sequences.

        Args:
            sentences: A list of tokenized sentences.
            labels: A list of BIO label sequences, one per sentence, aligned
                token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        self.model.fit(features, labels)

    def predict(self, sentences: list[list[str]], vocab: Any = None) -> list[list[str]]:
        """Predict BIO label sequences for tokenized sentences.

        Args:
            sentences: A list of tokenized sentences.
            vocab: Unused. Accepted only so `CRFModel.predict()` and
                `BiLSTMModel.predict()` share the exact same call signature —
                sklearn-crfsuite works on raw token features, not vocab ids,
                so there's nothing for CRF to look up here. This is
                intentional, not a bug.

        Returns:
            A list of predicted BIO label sequences, one per sentence,
            aligned token-for-token with `sentences`.
        """
        features = [self.extract_features(sentence) for sentence in sentences]
        return cast(list[list[str]], self.model.predict(features))
