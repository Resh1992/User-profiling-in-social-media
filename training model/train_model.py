import sys
import os

current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../')) # add reference of parent directory
import read_input as rp
import numpy as np
import random
import classifiers.text.text_classifier as tcf
import classifiers.text.text_feature_extractor as tfe
import classifiers.image.image_cnn_classifier as icnn
import classifiers.relation.rel_nb_classifier as rnb

# check and store the input output path
if len(sys.argv) < 2:
    exit()
input_path = sys.argv[1]

def partition_data_into_test_and_train(input, number_of_test_ids):
    # arranging data in indices to split
    all_Ids = np.arange(len(input.userid))

    # test and train ids
    random.shuffle(all_Ids)
    test_Ids = all_Ids[0:number_of_test_ids]
    train_Ids = all_Ids[number_of_test_ids:]

    # test and train data
    data_test = input.loc[test_Ids, :]
    data_train = input.loc[train_Ids, :]

    return (data_test, data_train)


# TEXT CLASSIFICATION
print("Reading text input")
input = rp.read_text(input_path)

# test and train data
(data_test, data_train) = partition_data_into_test_and_train(input, 1900)

print("Making text model for age")
text_age_model = tcf.train_model(data_test, data_train, "neural", "age", "text")
exit(0)
print("Making text model for gender")
text_gender_model = tcf.train_model(data_test, data_train, "naive_bayes", "gender", "text")

# personality models, work better with liwc as feature instead of text
text_personality_models = {}
for personality_trait in tfe.personality_traits:
    print("Making liwc model for", personality_trait)
    text_personality_models[personality_trait] = tcf.train_model(data_test, data_train, "linear_regression", personality_trait, "liwc")





# IMAGE CLASSIFICATION
print("\nReading image input")
input_image_dataframe = rp.read_images(input_path, ['userid', 'gender','age'])

print("Train image model for gender")
image_gender_model = icnn.train_model(input_path, input_image_dataframe, 'gender')
image_age_model = icnn.train_model(input_path, input_image_dataframe, 'age')





# RELATION CLASSIFICATION
print("Making relation model for age")
# Training relation model
relation_merged_dataframe = rp.new_read_rel(input_path)

# test and train data
(data_test, data_train) = partition_data_into_test_and_train(relation_merged_dataframe, 1500)         

relation_gender_model = rnb.train_model(data_test, data_train, 'age')