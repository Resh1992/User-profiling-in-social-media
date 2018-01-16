import sys
import os

current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../../'))  # add reference of parent directory

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import models.model_manipulation as mp


personality_traits = ["ope", "con", "ext", "agr", "neu"]

def extract_features(data, data_type, feature):
    if feature == "liwc":
        return _extract_liwc_features(data)
    elif feature == "text":
        return _extract_text_features(data, data_type)
    elif feature == "bigrams":
        return _extract_ngram_feature(data, data_type, 2)
    elif feature == "trigrams":
        return _extract_ngram_feature(data, data_type, 3)
    elif feature == "4grams":
        return _extract_ngram_feature(data, data_type, 4)
    else:
        raise ValueError("Unknown feature: " + feature)

def _extract_ngram_feature(data, data_type, n):
    count_vect_pickle_file_name = "count_vector_" + str(n) + "gram.pickle"
    if data_type == "train":
        count_vect = CountVectorizer(ngram_range=(1,n))

        # data
        x_ngrams = count_vect.fit_transform(data["text"])

        # save count_vect
        mp.save_as_pickle(count_vect_pickle_file_name, count_vect)

        return x_ngrams.toarray()
    else:
        count_vect = mp.load_from_pickle(count_vect_pickle_file_name)
        x_ngrams = count_vect.transform(data["text"])
        return x_ngrams.toarray()


def _extract_text_features(data, data_type):
    count_vect_pickle_file_name = "count_vector.pickle"
    if data_type == "train":
        # count_vect = TfidfVectorizer(stop_words = "english", strip_accents = "unicode", max_df = 0.3 * len(data), min_df = 100)
        count_vect = CountVectorizer()
        # count_vect = CountVectorizer(stop_words = "english", strip_accents = "unicode", max_df = 0.3 * len(data), min_df = 100)

        # data
        x_text = count_vect.fit_transform(data["text"])

        # save count_vect
        mp.save_as_pickle(count_vect_pickle_file_name, count_vect)

        return x_text.toarray()
    else:
        count_vect = mp.load_from_pickle(count_vect_pickle_file_name)
        x_text = count_vect.transform(data["text"])
        return x_text.toarray()


def _extract_liwc_features(data):
    # get liwc feature column names
    liwc_feature_columns = _get_liwc_feature_column_names(data)
    return  data[liwc_feature_columns]

def _get_liwc_feature_column_names(data):
    non_feature_columns = list(personality_traits)
    non_feature_columns.extend(["text", "userid", "Unnamed: 0", "age", "gender"])
    liwc_feature_columns = [x for x in data.columns.tolist()[:] if x not in non_feature_columns]
    return liwc_feature_columns

