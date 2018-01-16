import sys
import os

current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../../'))  # add reference of parent directory

import numpy as np
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten

import models.model_manipulation as mp
import classifiers.text.text_feature_extractor as tfe
import utils

def train_model(data_test, data_train, classifier_name, attribute_to_train, feature):
    # train data
    x_train = tfe.extract_features(data_train, "train", feature)
    y_train = data_train[attribute_to_train]

    # test data
    x_test = tfe.extract_features(data_test, "test", feature)
    y_test = data_test[attribute_to_train]

    # train model
    classifier = _get_trained_classifier(classifier_name, attribute_to_train, feature, x_train, y_train)

    # predict on test, note this is still part of training data
    y_predicted = _predict(classifier_name, attribute_to_train, feature, x_test)

    # print accuracy
    _print_accuracy(attribute_to_train, _transform(y_test, attribute_to_train), _transform(y_predicted, attribute_to_train))

    return classifier

def make_predictions(data_prediction, classifier_name, attribute_to_predict, feature):
    # prediction data
    x_prediction = tfe.extract_features(data_prediction, "predictions", feature)
    y_predicted = _predict(classifier_name, attribute_to_predict, feature, x_prediction)

    return _transform(y_predicted, attribute_to_predict)

def _predict(classifier_name, attribute_to_predict, feature, x):
    if classifier_name == "neural":
        # load classifier
        classifier = mp._get_keras_model(_get_classifier_keras_file_name(feature, classifier_name, attribute_to_predict))

        return classifier.predict(_reshape_input(x, len(x[0])))
    else:
        # load classifier
        classifier = mp.load_from_pickle(_get_classifier_pickle_file_name(feature, classifier_name, attribute_to_predict))

        return classifier.predict(x)

def _get_trained_classifier(classifier_name, attribute_to_train, feature, x_train, y_train):
    if classifier_name == "neural":
        return _get_trained_neural_classifier(attribute_to_train, feature, x_train, y_train)
    else:
        # get scikit classifier
        classifier = _get_scikit_classifier(classifier_name)

        # train
        classifier.fit(x_train, y_train)

        # save model
        mp.save_as_pickle(_get_classifier_pickle_file_name(feature, classifier_name, attribute_to_train), classifier)

        return classifier

def _get_trained_neural_classifier(attribute_to_train, feature, x_train, y_train):
    input_dim = len(x_train[0])

    # training data
    x_train = _reshape_input(x_train, input_dim)
    y_train = _reshape_input(y_train, 1)

    print (y_train)

    # model
    model = Sequential()
    model.add(LSTM(64, input_shape = (1, input_dim), activation = "relu", return_sequences = True))
    model.add(LSTM(64, activation = "relu", return_sequences = True))
    model.add(LSTM(64, activation = "relu", return_sequences = True))
    model.add(LSTM(64, activation = "relu", return_sequences = True))
    model.add(Dense(1, activation="relu"))

    # compile
    if attribute_to_train == "gender":
        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    else:
        model.compile(optimizer = "adam", loss = "mse", metrics=["mse"])

    # train
    model.fit(x_train, y_train)

    # save
    mp._save_keras_model(_get_classifier_keras_file_name(feature, "neural", attribute_to_train), model)

    return model


def _reshape_input(data, dim):
    data = np.array(data)
    data = data.reshape(len(data), 1, dim)
    return data

def _transform(y_data, attribute):
    if attribute == "age":
        # transform age into groups
        return utils._transform_ages_to_group(y_data)
    elif attribute == "gender":
        # transform, genders into string
        return utils._transform_genders_into_string(y_data)
    else:
        # personality train. no transform
        return y_data

def _print_accuracy(attribute, transformed_y_test, transformed_y_predict):
    if attribute == "age" or attribute == "gender":
        # for age/geder, print accuracy
        print("Accuracy: %.2f" % accuracy_score(transformed_y_test, transformed_y_predict))
    else:
        # for personality traits, print root mean squared error
        print('RMSE: ', np.sqrt(metrics.mean_squared_error(transformed_y_test, transformed_y_predict)))

def _get_scikit_classifier(classifier_name):
    # classifiers used in ensemble
    ensemble_classifiers = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(),
        "svc": svm.LinearSVC()
    }

    if classifier_name in ensemble_classifiers:
        return ensemble_classifiers[classifier_name]
    elif classifier_name == "linear_regression":
        return LinearRegression()
    elif classifier_name == "random_forest":
        return RandomForestClassifier(n_estimators=20)
    elif classifier_name == "k_neighbor":
        return KNeighborsClassifier(n_neighbors = 5)
    elif classifier_name == "ensemble":
        return VotingClassifier(estimators = list(ensemble_classifiers.items()))
    else:
        raise ValueError("Unknown classifier name: " + classifier_name)

def _get_classifier_pickle_file_name(feature, classifier_name, attribute):
    # example: liwc_naive_bayes_gender.pickle
    return _get_classifier_file_name(feature, classifier_name, attribute, ".pickle")

def _get_classifier_keras_file_name(feature, classifier_name, attribute):
    # example: liwc_naive_bayes_gender.pickle
    return _get_classifier_file_name(feature, classifier_name, attribute, ".h5")

def _get_classifier_file_name(feature, classifier_name, attribute, extension):
    # example: liwc_naive_bayes_gender.pickle
    return feature + "_" + classifier_name + "_" + attribute + extension