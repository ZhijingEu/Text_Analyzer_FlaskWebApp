# Text_Analyzer_FlaskWebApp
By Zhijing Eu , 27 Sep 2020

Description:

This is a Python Flask Web App that takes user provided text or URL addresses and returns text analysis metrics (such as word counts, word freq. etc) and also predicts personality profiles and Medium Clap Counts 

Note:

The code for personality profiling for MyersBriggsTypes is adapted from code and pre-trained models from :  https://towardsdatascience.com/text-analytics-what-does-your-linkedin-profile-summary-say-about-your-personality-f80df46875d1

Similarly for OCEAN Big 5 Personality Traits , the app uses code snippets and pre-trained models from : https://github.com/jcl132/personality-prediction-from-text 

Instructions:

To run the Flask App:

-Please clone the entire repo to your local machine and UNZIP any .zip model files in the folders "BigFiveModels" and "ClapPredictionModels" (I had to zip it as they Github had a size limit per file)

-Use the requirements.txt to install the dependencies on your local virtual environment

-Rename the 04_flask_app.py file as main.py and run using either 'python main.py' OR 'set FLASK_APP=main.py' and 'flask run'

Description Of Files/Folders:

01_Text_Analysis_JupyterNotebook.ipynb contains the original working notebook I used to develop the flask app

02_Exploratory_Data_Analysis.ipynb shows the initial data exploration for the 200 article dataset that was web-scraped AND the simple linear regression model used for the Clap Prediction

03_Training_A_Doc2Vec_Model.ipynb shows how I trained the Doc2Vec model on the 200 article dataset and used it to build a simple classifier to prediction Clap Count category for any given article

04_flask_app.py is the actual Flask App itself. Remember to rename it before running

'Static' and 'Templates' contains the HTML and images for the flask app




