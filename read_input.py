import numpy as np
import pandas as pd
import os
from pathlib import Path
import utils

#image processing libraries
from PIL import Image
from keras.preprocessing.image import img_to_array

'''
Data taken from : https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}

@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year = {2015},
  month = {December},
}

'''

def read_wiki_images(directory):
    # read wiki metadata file
    images = np.zeros([0, 1, 28, 28])
    label = np.zeros([1,])
    image_numpy_file = Path(directory+"wiki/wiki_images.npy")
    label_numpy_file = Path(directory+"wiki/wiki_image_labels.npy")
    if image_numpy_file.is_file() and label_numpy_file.is_file():
        images = np.load(directory+"wiki/wiki_images.npy")
        label = np.load(directory+"wiki/wiki_image_labels.npy")
        return (images,label)

    '''
    print("Retrieving new data")
    meta_info = sio.loadmat(directory+'wiki/wiki.mat')
    image_content = meta_info['wiki'].item(0)
    required_data = np.column_stack((image_content[2][0], image_content[3][0]))
    required_data = required_data[~pd.isnull(required_data[:, 1])]
    label = required_data[:, 1]
    label = 1-label
    image_path_list = required_data[:, 0]
    images = np.zeros([0, 1, 28, 28])
    image_size_28 = (28, 28)
    counter = 0;
    for item in image_path_list:
        print(counter)
        image_file_name = item[0]
        image_file_path = os.path.join(directory, "wiki", image_file_name)
        im = Image.open(image_file_path)
        im = im.convert('L')  # makes it greyscale
        im = im.resize(image_size_28, Image.ANTIALIAS)  # opening the image and resizing it to 28*28
        image_to_array = img_to_array(im)
        images = np.concatenate((images, image_to_array[np.newaxis]))
        counter+=1
    images = images.round(decimals=0)
    np.save(directory+"wiki/wiki_images.npy", images)
    np.save(directory + "wiki/wiki_image_labels.npy", label)
    '''
    return (images, label)

def getImageFromInput(input, directory):
    images = np.zeros([0, 1, 28,28])
    image_size_28 = (28,28)
    # TODO loop on the basis of file name and not userid
    for id in input.userid:
        #Get the file
        image_file_name = id + ".jpg"
        image_file_path = os.path.join(directory, "image", image_file_name)
        im = Image.open(image_file_path)
        im = im.convert('L')
        im = im.resize(image_size_28, Image.ANTIALIAS)  # resizing it to 28*28
        image_to_array = img_to_array(im)
        images = np.concatenate((images, image_to_array[np.newaxis]))
    images = images.round(decimals=0)
    # normalize inputs from 0-255 to 0-1
    images = images / 255
    return (images)

def get_label_for_input(input, attribute):
    label = np.asarray(input[attribute])
    return (label)

def read_images(directory, attribute):
    #load profile information into a dataframe
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    input = pd.read_csv(profile_file_path, usecols=attribute)
    if attribute is 'age':
        input['age'] = input['age'].apply(utils.encode_age)
    return(input)

def read_text(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    input = pd.read_csv(profile_file_path)

    # read text
    texts = []
    for id in input.userid:
        text_file_name = id + ".txt"
        text_file_path = os.path.join(directory, "text", text_file_name)
        text_file = open(text_file_path, encoding = "ISO-8859-1")
        content = text_file.read()
        texts.append(content)
        text_file.close()
    input["text"] = texts

    # read liwc
    liwc_file_path = os.path.join(directory, "LIWC/LIWC.csv")
    input_liwc = pd.read_csv(liwc_file_path)
    input = input.join(input_liwc.set_index("userId"), on = "userid")

    return input

def read_relation(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    df2 = pd.read_csv(profile_file_path)
    df2 = df2.loc[:,['userid', 'age']]
    
    
    relation_file_path = os.path.join(directory,"relation/relation.csv")
    df1 =  pd.read_csv(relation_file_path)
    df1 =  df1.loc[:200000,['userid', 'like_id']]
    df1['value'] = 1
    
    # using pivot_table to get the userid-likeid table
    data = df1.pivot_table(index = 'userid', columns='like_id',values='value', aggfunc = 'count')
    
    #replace naN with 0's
    data = data.fillna(0)
    
    #set index as userid to begin merging
    df2.set_index('userid',inplace = True)
    
    #merging
    
    df_merge = pd.merge(df2,data,left_index = True, right_index = True)
    
    # reset index again to be ready for the classifier
    
    df_merge.reset_index(inplace = True)
    
    print("in read input.py")
    print(df_merge)
    
    return df_merge
    
def read_relation_test(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    df2 = pd.read_csv(profile_file_path)
    df2 = df2.loc[:,['userid', 'age']]
    
    
    relation_file_path = os.path.join(directory,"relation/relation.csv")
    df1 =  pd.read_csv(relation_file_path)
    df1 =  df1.loc[:,['userid', 'like_id']]
    df1['value'] = 1
    
    # using pivot_table to get the userid-likeid table
    data = df1.pivot_table(index = 'userid', columns='like_id',values='value', aggfunc = 'count')
    
    #replace naN with 0's
    data = data.fillna(0)
    
    #set index as userid to begin merging
    df2.set_index('userid',inplace = True)
    
    #merging
    
    df_merge = pd.merge(df2,data,left_index = True, right_index = True)
    
    # reset index again to be ready for the classifier
    
    df_merge.reset_index(inplace = True)
    
#    print("in read input.py")
#    print(df_merge)
    
    return df_merge
    
def new_read_rel(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    df2 = pd.read_csv(profile_file_path)
    #df_pro = df2.loc[:,['userid', 'gender']]
    df_pro = df2.loc[:,['userid', 'age']]
    
    
    relation_file_path = os.path.join(directory,"relation/relation.csv")
    df1 =  pd.read_csv(relation_file_path)
    df_rel =  df1.loc[:,['userid', 'like_id']]


    #learning about the counts of each like_id

    #print(df_rel['like_id'].value_counts())

    counts = df_rel['like_id'].value_counts()
    #print(counts)

    df_rel = df_rel[df_rel['like_id'].isin(counts[counts > 4].index)] #4
    df_rel = df_rel[df_rel['like_id'].isin(counts[counts < 850].index)] #850(0.76),800(0.75) gender
    #print(df_rel['like_id'].value_counts())
    #print(df_rel.info())
    #converting like_id column to string 
    df_rel['like_id'] = df_rel['like_id'].astype(str)
    #print(df_rel.info())

    # getting a userid: like_id as s long string, note that this will give us a panda series as output

    df_rel = df_rel.groupby('userid')['like_id'].apply(lambda x: "%s" % ' '.join(x))
    #print(df_rel)

    # converting the series into A DATAFRAME
    df_final = pd.DataFrame({'userid':df_rel.index, 'like_id':df_rel.values})

    #merging the profile and rel dataframes to get final df ready for training model
    df_merge = pd.merge(df_pro,df_final, on='userid')    
    
    return df_merge
    
def read_rel_test(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    df2 = pd.read_csv(profile_file_path)
    df_pro = df2.loc[:,['userid', 'age']]
    
    
    relation_file_path = os.path.join(directory,"relation/relation.csv")
    df1 =  pd.read_csv(relation_file_path)
    df_rel =  df1.loc[:,['userid', 'like_id']]


    #converting like_id column to string 
    df_rel['like_id'] = df_rel['like_id'].astype(str)
    #print(df_rel.info())

    # getting a userid: like_id as s long string, note that this will give us a panda series as output

    df_rel = df_rel.groupby('userid')['like_id'].apply(lambda x: "%s" % ' '.join(x))
    #print(df_rel)

    # converting the series into A DATAFRAME
    df_final = pd.DataFrame({'userid':df_rel.index, 'like_id':df_rel.values})

    #merging the profile and rel dataframes to get final df ready for training model
    df_merge = pd.merge(df_pro,df_final, on='userid')    
    
    return df_merge
    
def read_rel_test_gender(directory):
    # read profiles
    profile_file_path = os.path.join(directory, "profile/profile.csv")
    df2 = pd.read_csv(profile_file_path)
    df_pro = df2.loc[:,['userid', 'gender']]
    
    
    relation_file_path = os.path.join(directory,"relation/relation.csv")
    df1 =  pd.read_csv(relation_file_path)
    df_rel =  df1.loc[:,['userid', 'like_id']]


    #converting like_id column to string 
    df_rel['like_id'] = df_rel['like_id'].astype(str)
    #print(df_rel.info())

    # getting a userid: like_id as s long string, note that this will give us a panda series as output

    df_rel = df_rel.groupby('userid')['like_id'].apply(lambda x: "%s" % ' '.join(x))
    #print(df_rel)

    # converting the series into A DATAFRAME
    df_final = pd.DataFrame({'userid':df_rel.index, 'like_id':df_rel.values})

    #merging the profile and rel dataframes to get final df ready for training model
    df_merge = pd.merge(df_pro,df_final, on='userid')    
    
    return df_merge
        
    
    