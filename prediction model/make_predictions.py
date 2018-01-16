import sys
import os
current_file_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_file_location, '../')) # add reference of parent directory

import read_input as rp
import classifiers.text.text_classifier as tcf
import classifiers.text.text_feature_extractor as tfe
import classifiers.relation.rel_nb_classifier as rnb
import classifiers.image.image_cnn_classifier as icc
import classifiers.ensemble as ens


#check and store the input output path
if len(sys.argv) < 3:
    exit()
input_path = sys.argv[1]
output_path = sys.argv[2]

#input_path = "C:\\Users\\bhagatsanchya\\Desktop\\UWash\\Machine_Learning\\Project\\python_code\\data\\public-test-data"
#output_path = "C:\\Users\\bhagatsanchya\\Desktop\\Myoutput"
#maj_gender_predic_str="female"
##majority_age="xx-24"
#extrovert="3.48685789474"
#neurotic="2.73242421053"
#agreeable="3.58390421053"
#conscientious="3.44561684211"
#openness="3.90869052632"

"""
    Predictions using text
"""

# read text input
print("Reading input")
input = rp.read_text(input_path)

# age predictions using text
print("Making age prediction using text")
age_predictions_text = tcf.make_predictions(input, "naive_bayes", "age", "text")

# personality trait predictions
personality_predictions = {}
for personality_trait in tfe.personality_traits:
    print("Making", personality_trait, "prediction using liwc")
    personality_predictions[personality_trait] = tcf.make_predictions(input, "linear_regression", personality_trait, "liwc")

# gender predictions
print("Making gender prediction using text")
gender_predictions_text = tcf.make_predictions(input, "naive_bayes", "gender", "text")

'''
    Predictions using image
'''

print("Making gender prediction using image")
print(input_path)
gender_predictions_image = icc.make_predictions(input_path, "gender")

"""
    Predictions using relations
"""

# Prediction using relations
print("Reading input for age using relation")
relation_merged_dataframe = rp.read_rel_test(input_path)

# age predictions using RELATION
print("Making age prediction with relation")
age_predictions_rel = rnb.make_predictions(relation_merged_dataframe, 'age')

print("Reading input for gender using relation")
relation_merged_dataframe = rp.read_rel_test_gender(input_path)

#gender predictions using RELATION
print("Making gender prediction with relation")
gender_predictions_rel = rnb.make_predictions(relation_merged_dataframe, 'gender')


"""
    Ensemble predictions
"""

# ensebmle gender predictions
print("Ensembling gender predictions")
gender_predictions = ens.ensemble_predictions([gender_predictions_text, gender_predictions_image, gender_predictions_rel])
#gender_predictions = ens.ensemble_predictions([gender_predictions_text, gender_predictions_rel])

# ensemble age predictions
print("Ensembling age predictions")
age_predictions = ens.ensemble_predictions([age_predictions_text, age_predictions_rel])

def _get_xml_string(userid, age, gender, ext, neu, agr, con, ope):
    id_attr = '\n\tid=\"' + str(userid) + '"'
    age_attr = '\n\tage_group=\"' + str(age) + '"'
    gender_attr = '\n\tgender=\"' + str(gender) + '"'

    extrovert_attr = '\n\textrovert=\"' + str(ext) + '"'
    neurotic_attr = '\n\tneurotic=\"' + str(neu) + '"'
    agreeable_attr = '\n\tagreeable=\"' + str(agr) + '"'
    conscientious_attr = '\n\tconscientious=\"' + str(con) + '"'
    openness_attr = '\n\topenness=\"' + str(ope) + '"'

    personality_attr = extrovert_attr + neurotic_attr + agreeable_attr + conscientious_attr + openness_attr

    # creating the xml content by combining userid and predictions
    return '<' + 'user' + id_attr + age_attr + gender_attr  + personality_attr + '\n/>'


for i in range(0, len(input.userid)):
    userid = input.userid[i]

    # create a string of the format userid.xml
    filename = str(userid) + ".xml"
    #print(filename)

    ext_predictions = personality_predictions["ext"]
    neu_predictions = personality_predictions["neu"]
    agr_predictions = personality_predictions["agr"]
    con_predictions = personality_predictions["con"]
    ope_predictions = personality_predictions["ope"]

    # creating the complete xml file by combining userid and fixvalue into xml format
    file_content = _get_xml_string(userid, age_predictions[i], gender_predictions[i], ext_predictions[i], neu_predictions[i], agr_predictions[i], con_predictions[i], ope_predictions[i])

    # create a file at the specified directory. We use the os library to avoid any cross platform(windows/linux) inconsistencies
    f_path = os.path.join(output_path, filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #open the file in append mode. If the filename doesn't exist, it creates one
    f = open(f_path, "w+")
    f.write(file_content)
    f.close()
