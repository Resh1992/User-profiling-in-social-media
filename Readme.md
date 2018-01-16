## User Profiling in Social Media

We intend to predict age, gender and the big five personality traits of facebook users, given their profile picture, status updates and page likes. The project works in two steps:
1.	Training the classifiers on the data set of 9500 users 
2.	Predicting the age, gender and personality traits of new users. 


## Getting Started
Before profiling a new user, there are a couple of initial steps required to setup the system and train the models.


## Prerequisites
Other than python3.X, the project uses keras with tensorflow. Below are a few installation steps for a linux machine. 
pip3 install -U scikit-learn
pip3 install tensorflow
sudo pip3 install keras
sudo pip3 install h5py

## Training the models
The project input takes user data in the hierarchy as below:
1.	LIWC – contains the LIWC.csv file
2.	Image – contains text files in the form <userid.jpg>
3.	Profile – contains the profile.csv file
4.	Relation – contains the relation.csv file
5.	Text – contains text files in the form <userid.txt>
6.	Wiki – contains numpy wiki image files.

To train the model, train_model.py file is called. This file take the folder path as input as described above. 
python3 training\ model/train_model.py <path to input directory>
 
The training process can take a few minutes to a couple of hours depending upon the machine configuration and the number of images, on which the model is trained.

## Running the prediction script
The project can be run using:
1.	The make_prediction.py file
2.	The tcss555 script file
The program takes two files as input: the folder path for new instances (as explained above) and the folder path where the xml outputs are to be saved. 
./tcss555 -i <Path to input directory> -o <path to output directory>

Built With
1.	PyCharm - The IDE used for developement
2.	Keras – Neural networks API

