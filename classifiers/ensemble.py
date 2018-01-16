def ensemble_predictions(multiple_classifier_predictions):
    ensembled_predictions = []

    # each row contains predictions from multiple classifier
    # go through each row and ensemble it
    for i in range(0, len(multiple_classifier_predictions[0])):
        predictions = []

        # for ith row, get predictions from all classifiers
        for j in range(0, len(multiple_classifier_predictions)):
            predictions.append(multiple_classifier_predictions[j][i])

        # add majority
        ensembled_predictions.append(_get_majority_prediction(predictions))
    return ensembled_predictions

def _get_majority_prediction(predictions):
    frequency = {}

    # get frequency of each prediction
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        if prediction in frequency:
            frequency[prediction] += 1
        else:
            frequency[prediction] = 1

    max_frequency = 0
    max_frequency_prediction = None
    for prediction in frequency:
        if frequency[prediction] > max_frequency:
            max_frequency = frequency[prediction]
            max_frequency_prediction = prediction

    return max_frequency_prediction