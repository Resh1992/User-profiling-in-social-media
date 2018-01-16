import numpy as np
import random
import sys
import os

current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location))  # add reference of parent directory


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


def encode_age(age):
    '''
    Encode age buckets into integer values to train the CNN model
    :param age:
    :return: encoded integer value of the age
    '''
    if age <= 24:
        return 0
    elif age <= 34:
        return 1
    elif age <= 49:
        return 2
    else:
        return 3


def partitionData(dataInput, testPercentage, validationPercentage):
    '''
    slices a numpy array based on the percentage size of each bucket

    :param dataInput: numpy array
    :param testPercentage: 0-1, to range between 0% to 100%
    :param validationPercentage: 0,1, to range between 0% to 100%
    :return: sliced nunmpy array based on the training and validation percentage
    '''

    totalLen = len(dataInput)
    testSlice = slice(0, int(totalLen * testPercentage))
    validationSlice = slice(int(totalLen * testPercentage), int(totalLen * (validationPercentage + testPercentage)))
    trainSlice = slice(int(totalLen * (validationPercentage + testPercentage)), totalLen)
    return (dataInput[testSlice], dataInput[validationSlice], dataInput[trainSlice])


def partition_data(input, testPercentage, validationPercentage):
    '''
    splits input dataframe based on the percentage size of each bucket

    :param input: dataframe
    :param testPercentage: 0-1, to range between 0% to 100%
    :param validationPercentage: 0-1, to range between 0% to 100%
    :return:
    '''

    # arranging data in indices to split
    all_Ids = np.arange(input.shape[0])

    # test and train ids
    random.shuffle(all_Ids)
    (test_Ids, validate_Ids, train_Ids) = partitionData(all_Ids, testPercentage, validationPercentage)

    data_test = input.loc[test_Ids, :]
    data_validate = input.loc[validate_Ids, :]
    data_train = input.loc[train_Ids, :]

    if validationPercentage == 0:
        return (data_test, data_train)
    else:
        return (data_test, data_validate, data_train)
