
import sys
sys.path.append('./')
import pickle
from sklearn.naive_bayes import MultinomialNB # for images
from sklearn.metrics import accuracy_score
import os
import models.model_manipulation as mp

# train model on give test/train data for given attribute, attribute can be age/gender etc.
def train_model(data_test, data_train, attribute_to_train):
   

    # train data
    X_train = data_train.drop(attribute_to_train, axis=1)
    X_train = data_train.drop('userid', axis =1)
    y_train = data_train[attribute_to_train]

    # test data
    X_test = data_test.drop(attribute_to_train, axis=1)
    X_test = data_test.drop('userid', axis =1)
    y_test = data_test[attribute_to_train]  # transforming test data also in matrix

    # train model
    classifier = MultinomialNB()#for using Naive Bayes
    classifier.fit(X_train, y_train)# fit a naive bayes model # fit method is used to train and predict is used to predict

    # Save model 
    mp._save_model("relation_nb_classifier_" + attribute_to_train + ".pickle", classifier)
    
    print("I'm in train_model() and saving the dataframe now!")
    # Save dataframe
    mp._save_df_("rnb_X_prediction_"+attribute_to_train + ".pickle",X_train)
   
    print("Saved df!")
    
    # predict on test, note this is still part of training data
    y_predicted = classifier.predict(X_test)
    print("Accuracy: %.2f" % accuracy_score(_transform(y_test, attribute_to_train), _transform(y_predicted, attribute_to_train)))

    return classifier

def make_predictions(data_prediction, attribute_to_predict):
    # Load model 
    classifier = _load_model(attribute_to_predict)
    X_reference = _load_df_(attribute_to_predict)
    print("I'm in make_predictions() and this is the X_Reference")
    #print(X_reference)
    #print(classifier)
    
    # prediction data
    #print("In relation -> make_prediction-> prediction data")
    #print (data_prediction)
    
    print("data_prediction before")
    print(len(data_prediction.columns))
    
    print("X_refernce column length")
    print(len(X_reference.columns))
    ##### Testing Prediction using my technique of adding dummy columns
    print("Testing my method")
    for i in range(0, len(X_reference.columns) - len(data_prediction.columns) + 2):
        data_prediction['dummy' + str(i)] = 0
    
    X_prediction = data_prediction.drop(attribute_to_predict, axis=1)
    X_prediction = X_prediction.drop('userid', axis=1)
    print(X_prediction.info())
    print("In relation -> make_prediction-> X_prediction")
   # print(X_prediction)
    y_predicted = classifier.predict(X_prediction)

    return _transform(y_predicted, attribute_to_predict)

def _transform(y_data, attribute):
    if (attribute == "age"):
        # transform age into groups
        return _transform_ages_to_group(y_data)
    else:
        # transform, genders into string
        return _transform_genders_into_string(y_data)

def _transform_genders_into_string(genders):
    genders_as_string = []
    for gender in genders:
        gender_as_string = 'female' if gender == 1 else 'male'
        genders_as_string.append(gender_as_string)

    return genders_as_string

def _transform_ages_to_group(ages):
    age_groups = []
    for age in ages:
        age_groups.append(_get_age_group(age))
    return age_groups

def _get_age_group(age):
    if age <= 24:
        return 'xx-24'
    elif age <= 34:
        return '25-34'
    elif age <= 49:
        return '35-49'
    else:
        return '50-xx'

def _load_model(attribute_to_predict):
    model_file_path = mp._get_model_location("relation_nb_classifier_" + attribute_to_predict + ".pickle")
    model_file = open(model_file_path, "rb")
    classifier = pickle.load(model_file)
    model_file.close()
    return  classifier

def _load_df_(attribute_to_predict):
    df_file_path = mp._get_model_location("rnb_X_prediction_"+attribute_to_predict + ".pickle")
    model_file = open(df_file_path, "rb")
    X_reference = pickle.load(model_file)
    model_file.close()
    return X_reference