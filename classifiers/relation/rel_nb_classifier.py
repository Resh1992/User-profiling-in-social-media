import sys
sys.path.append('./')
import pickle
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score
import os
import models.model_manipulation as mp
import utils

# train model on give test/train data for given attribute, attribute can be age/gender etc.
def train_model(data_test, data_train, attribute_to_train):
    count_vect = CountVectorizer()

    # train data
    x_train = count_vect.fit_transform(data_train['like_id'])
    y_train = data_train[attribute_to_train]

    # test data
    x_test = count_vect.transform(data_test['like_id'])
    y_test = data_test[attribute_to_train]  # transforming test data also in matrix

    # train model
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    # Save model and count_vect
    mp.save_as_pickle("rel_nb_classifier" + attribute_to_train + ".pickle", classifier)
    _save_count_vect(count_vect,attribute_to_train)

    # predict on test, note this is still part of training data
    y_predicted = classifier.predict(x_test)
    print("Accuracy: %.2f" % accuracy_score(_transform(y_test, attribute_to_train), _transform(y_predicted, attribute_to_train)))

    return classifier

def make_predictions(data_prediction, attribute_to_predict):
    # Load model and count vec
    classifier = _load_model(attribute_to_predict)
    count_vect = _load_count_vect(attribute_to_predict)

    # prediction data
    # print (data_prediction)
    x_prediction = count_vect.transform(data_prediction['like_id'])
    y_predicted = classifier.predict(x_prediction)

    return _transform(y_predicted, attribute_to_predict)

def _transform(y_data, attribute):
    if (attribute == "age"):
        # transform age into groups
        return utils._transform_ages_to_group(y_data)
    else:
        # transform, genders into string
        return utils._transform_genders_into_string(y_data)


def _save_count_vect(count_vect,attribute_to_train):
    count_vect_file_path = _get_count_vec_location(attribute_to_train)
    count_vect_file = open(count_vect_file_path, "wb")
    pickle.dump(count_vect,count_vect_file)
    count_vect_file.close()

def _load_count_vect(attribute_to_predict):
    count_vect_file_path = _get_count_vec_location(attribute_to_predict)
    count_vect_file = open(count_vect_file_path, "rb")
    count_vect = pickle.load(count_vect_file)
    count_vect_file.close()
    return  count_vect

def _load_model(attribute_to_predict):
    model_file_path = mp._get_model_location("rel_nb_classifier" + attribute_to_predict + ".pickle")
    model_file = open(model_file_path, "rb")
    classifier = pickle.load(model_file)
    model_file.close()
    return  classifier


def _get_count_vec_location(attribute_to_predict):
    current_file_location = os.path.dirname(os.path.realpath(__file__))
    count_vect_file_path = os.path.join(current_file_location, "../../models/count_vector_"+attribute_to_predict+".pickle")
    return  count_vect_file_path
