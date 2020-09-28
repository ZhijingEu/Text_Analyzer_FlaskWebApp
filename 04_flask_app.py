#For data cleaning and visualization
import pandas as pd
import numpy as np
from numpy import save
from numpy import load
import matplotlib.pyplot as plt
from collections import Counter

#For web scraping
from bs4 import BeautifulSoup
import urllib.request
from urllib.request import Request, urlopen
from htmldate import find_date

#For Text Analysis
import textstat
textstat.set_lang("en")

#For NLP
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *

from textblob import TextBlob
from textblob import Word

#For Wordclouds
from wordcloud import WordCloud, STOPWORDS 
wcstopwords = set(STOPWORDS)

#For Random Forest Regressions For Personality Prediction
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#For Similarity Between Docs
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec

from scipy import spatial

#Other Misc
import random
import pickle
from tqdm.notebook import tqdm
import openpyxl
from math import pi
import datetime

#Courtesy of Tristan Brown at https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

#Load Trained Models for MTBI Classification
#Credit to authors of the pre-trained classifiers here
#https://github.com/wiredtoserve/datascience/tree/master/PersonalityDetection

dummy_fn = lambda x:x

cv = CustomUnpickler(open('./MTBIModels/cv.pickle', 'rb')).load()

idf_transformer = CustomUnpickler(open('./MTBIModels/idf_transformer.pickle', 'rb')).load()

lr_ie = CustomUnpickler(open('./MTBIModels/LR_clf_IE_kaggle.pickle', 'rb')).load()
lr_jp = CustomUnpickler(open('./MTBIModels/LR_clf_JP_kaggle.pickle', 'rb')).load()
lr_ns = CustomUnpickler(open('./MTBIModels/LR_clf_NS_kaggle.pickle', 'rb')).load()
lr_tf = CustomUnpickler(open('./MTBIModels/LR_clf_TF_kaggle.pickle', 'rb')).load()

# Load Pre-Trained Models for Big5 OCEAN Personality Classification
# Credit to authors of the original pre-trained classifiers here https://github.com/jcl132/personality-prediction-from-text
# Note - the full project above does more than classify as it also extracts and classifies the FB profiles of your friends 
# and plots it on a web app- check it out yourself!

class Model(): # Note "Model" represents the Big 5 OCEAN Model. I can't rename as it seems to affect the pickled files
    def __init__(self):
        self.rfr = RandomForestRegressor(bootstrap=True,
         max_features='sqrt',
         min_samples_leaf=1,
         min_samples_split=2,
         n_estimators= 200)
        self.rfc = RandomForestClassifier(max_features='sqrt', n_estimators=110)
        self.tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii')

    def fit(self, X, y, regression=True):
        X = self.tfidf.fit_transform(X)
        if regression:
            self.rfr = self.rfr.fit(X, y)
        else:
            self.rfc = self.rfc.fit(X, y)

    def predict(self, X, regression=True):
        X = self.tfidf.transform(X)
        if regression:
            return self.rfr.predict(X)
        else:
            return self.rfc.predict(X)

    def predict_proba(self, X, regression=False):
        X = self.tfidf.transform(X)
        if regression:
            raise ValueError('Cannot predict probabilites of a regression!')
        else:
            return self.rfc.predict_proba(X)

traits = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
#### (O) Openess
#### (C) Conscientiousness
#### (E) Extraversion
#### (A) Agreeableness
#### (N) Neuroticism

model = Model()
models = {}    
models['OPN'] = CustomUnpickler(open('BigFiveModels/OPN_model.pkl', 'rb')).load()
models['CON'] = CustomUnpickler(open('BigFiveModels/CON_model.pkl', 'rb')).load()
models['EXT'] = CustomUnpickler(open('BigFiveModels/CON_model.pkl', 'rb')).load()
models['AGR'] = CustomUnpickler(open('BigFiveModels/AGR_model.pkl', 'rb')).load()
models['NEU'] = CustomUnpickler(open('BigFiveModels/NEU_model.pkl', 'rb')).load()

#Load the Clap Prediction Model with the 'simple' Lasso Regression model - CAUTION - It is not very accurate ! 

with open("./ClapPredictionModels/clap_prediction_model_lasso.pkl", 'rb') as file:
    clap_prediction_model_lasso = pickle.load(file)

column_for_regression=["sentence_count","title_word_count","average_word_count_per_sentence",
                      "text_word_count","vocab_count_excl_commonwords","imgs_per_1000words",
                      "FS_GradeScore","vids_per_1000words","polarity","subjectivity"]

#Load the pre-trained Doc2Vec Model trained on 200 sample Medium Data Science articles with 300 vec dimensions

Doc2VecModel= Doc2Vec.load("./ClapPredictionModels/Doc2Vec.model")

#Load the average document vector for the 37 out of the 200 reference articles that have > 5k Claps  

VH_Vec=load('./ClapPredictionModels/VH_Claps_Vector.npy')
H_Vec=load('./ClapPredictionModels/H_Claps_Vector.npy')
M_Vec=load('./ClapPredictionModels/M_Claps_Vector.npy')
L_Vec=load('./ClapPredictionModels/L_Claps_Vector.npy')
VL_Vec=load('./ClapPredictionModels/VL_Claps_Vector.npy')

def get_html(url):
    user_agent_list = ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0',
                            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0',
                            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
                        ]
    
    for i in range(1,4):
        #Pick a random user agent
        user_agent = random.choice(user_agent_list)
        #Set the headers 
        headers = {'User-Agent': user_agent}
        req = Request(url, headers=headers)

    return urlopen(req).read()
    

def cleaned_text(htmlinput):
    cleaned_text=BeautifulSoup(htmlinput, "html.parser").get_text(" ").replace("\r", " ").replace("\t", " ").replace("\n", " ").replace(u'\xa0', u' ')
    return cleaned_text
    

def tokens_alpha(htmlinput):
    raw = BeautifulSoup(htmlinput, 'html.parser').get_text(strip=True)
    words = nltk.word_tokenize(raw)
    tokens_alpha= [word for word in words if word.isalpha()] # or use option of := if word.isalnum()
    return tokens_alpha
    
def extract_title(htmlinput):
    return BeautifulSoup(htmlinput, 'html.parser').title.string
        
def get_date(urlinput):
    return find_date(urlinput)
            
def extract_tags(htmlinput):
    tags = []
    tags = [x.text for x in BeautifulSoup(htmlinput,'html.parser').find_all("a", href=re.compile(".*tag.*"))]
    if len(tags) == 0:
        return 'None'
    else:
        return tags

def image_count(htmlinput):
    try:
        article_soup = BeautifulSoup(htmlinput,'html.parser').find('article')
        figures = article_soup.find_all('figure')
        if len(figures)>0:
            return len(figures)
        else:
            return len(BeautifulSoup(htmlinput,'html.parser').find_all('img'))
    except:
        return len(BeautifulSoup(htmlinput,'html.parser').find_all('img'))    

def embedded_vids_count(htmlinput):
    return str(htmlinput).count("embed&display_name=YouTube")
               
def other_embedded_items_count(htmlinput,embedded_vids_count):
    return int(str(htmlinput).count("iframeSrc"))-int(embedded_vids_count)
    
def word_count(cleanedtext):
    return textstat.lexicon_count(cleanedtext, removepunct=True)
    
def tokenize_excl_stopwords(cleanedtext): #a bit of repetition but this version unlike the other tokenize text removes stop words
    lst= re.findall(r'\b\w+', cleanedtext)
    lst = [x.lower() for x in lst]
    counter = Counter(lst)
    occs = [(word,count) for word,count in counter.items() if count > 1]
    occs.sort(key=lambda x:x[1])
    WordListWithCommonWordsRemoved=[(occs[i][0],occs[i][1]) for i in range(len(occs)) if not occs[i][0] in stopwords.words()]
    return WordListWithCommonWordsRemoved
        
def vocab_count_excl_commonwords(tokenize_excl_stopwords):
    return len(tokenize_excl_stopwords)
    
def most_common_words(tokenize_excl_stopwords):
    return tokenize_excl_stopwords()[-5:]
    
def sentence_count(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text=blob.sentences
    No_Of_Sentences=len(split_text)
    #Initially used this but split_text tends to overcount
    return textstat.sentence_count(cleanedtext)
         
def content_summary(cleanedtext):
    
    try:
        blob = TextBlob(cleanedtext)
        split_text=blob.sentences

        nouns = list()
        for word, tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
        text_summary=["This text is about..."]

        for item in random.sample(nouns, 5):
            word = Word(item)
            if len(word)>3:
                text_summary.append(word) # Can also use word.pluralize()
    except:
        text_summary=[]
        
    return text_summary

def sentiment(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text=blob.sentences
    df=pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))),columns=['Sentences'])
    df[["TextBlob_Polarity","TextBlob_Subjectivity"]]=pd.DataFrame((split_text[i].sentiment for i in range(len(split_text))))
    df=df[df['Sentences'].map(len) > 15] #Remove all short sentences
    #Avoid counting any sentences with Polarity 0 or Subjectivity 0 
    TextBlob_Overall_Polarity=df[df["TextBlob_Polarity"] != 0]['TextBlob_Polarity'].median()
    TextBlob_Overall_Subjectivity=df[df["TextBlob_Subjectivity"] != 0]['TextBlob_Subjectivity'].median()
    return TextBlob_Overall_Polarity,TextBlob_Overall_Subjectivity
       
def polarity(sentiment):
    return sentiment()[0]*100
    
def subjectivity(self):
    return sentiment()[1]*100
       
def common_trigrams(tokensalpha):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(tokensalpha)
    finder.apply_freq_filter(2)
    finder.nbest(trigram_measures.pmi, 5)  # doctest: +NORMALIZE_WHITESPACE
    FiveMostCommonlyOccuringTrigrams=finder.nbest(trigram_measures.pmi, 5)
    return FiveMostCommonlyOccuringTrigrams
        
def FS_ReadingEaseScore(cleanedtext):
    FS_GradeScore=textstat.flesch_reading_ease(cleanedtext)
    return FS_GradeScore
    
def FS_ReadingEaseLevel(FS_GradeScore):
    if FS_GradeScore >= 90:
        FS_Grade="Very easy to read"
    elif FS_GradeScore <90 and FS_GradeScore >=80:
        FS_Grade="Easy"
    elif FS_GradeScore <80 and FS_GradeScore >=70:
        FS_Grade="Fairly easy"
    elif FS_GradeScore <70 and FS_GradeScore >=60:
        FS_Grade="Standard"
    elif FS_GradeScore <60 and FS_GradeScore >=50:
        FS_Grade="Fairly Difficult"
    elif FS_GradeScore <50 and FS_GradeScore >=30:
        FS_Grade="Difficult"
    else :
        FS_Grade="Very Confusing"
    return FS_Grade    

def MTBI_Analysis(tokensalpha):           
    c = cv.transform([tokensalpha])
    x = idf_transformer.transform(c)

    ie = lr_ie.predict_proba(x).flatten()
    ns = lr_ns.predict_proba(x).flatten()
    tf = lr_tf.predict_proba(x).flatten()
    jp = lr_jp.predict_proba(x).flatten()

    probs = np.vstack([ie, ns, tf, jp])

#         names = ["Introversion - Extroversion", "Intuiting - Sensing", "Thinking - Feeling", "Judging - Perceiving"]        
        
#         for i, dim in enumerate(names):
#             print(f"{dim:28s}: {probs[i,1]:.3f} - {probs[i, 0]:.3f}")

    Extraversion=probs[0][0]
    Introversion=probs[0][1]
    Sensing=probs[1][0]
    Intuiting=probs[1][1]
    Feeling=probs[2][0]
    Thinking=probs[2][1]
    Perceiving=probs[3][0]
    Judging=probs[3][1]

    if Introversion>=0.5:
        IE="I"
    else:
        IE="E"
    if Intuiting>=0.5:
        NS="N"
    else:
        NS="S"
    if Thinking>=0.5:
        TF="T"
    else:
        TF="F"
    if Judging>=0.5:
        JP="J"
    else:
        JP="P"
            
    MTBI_Results = [IE+NS+TF+JP, Introversion, Intuiting, Thinking, Judging]

    return MTBI_Results
        
def plot_MTBI(tokensalpha):
    c = cv.transform([tokensalpha])
    x = idf_transformer.transform(c)

    ie = lr_ie.predict_proba(x).flatten()
    ns = lr_ns.predict_proba(x).flatten()
    tf = lr_tf.predict_proba(x).flatten()
    jp = lr_jp.predict_proba(x).flatten()
        
    probs = np.vstack([ie, ns, tf, jp])
        
    Extraversion=probs[0][0]
    Introversion=probs[0][1]
    Sensing=probs[1][0]
    Intuiting=probs[1][1]
    Feeling=probs[2][0]
    Thinking=probs[2][1]
    Perceiving=probs[3][0]
    Judging=probs[3][1]
        
    if Introversion>=0.5:
        IE="I"
    else:
        IE="E"
    if Intuiting>=0.5:
        NS="N"
    else:
        NS="S"
    if Thinking>=0.5:
        TF="T"
    else:
        TF="F"
    if Judging>=0.5:
        JP="J"
    else:
        JP="P"
        
    names = ["Introversion - Extroversion", "Intuiting - Sensing", "Thinking - Feeling", "Judging - Perceiving"]        

    print("Myers-Briggs Type Indicator: "+IE+NS+TF+JP)
    print("")    
    for i, dim in enumerate(names):
        print(f"{dim:28s}: {probs[i,1]:.3f} - {probs[i, 0]:.3f}")
                
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca() 
    xlabels = ["Judging (J)","Thinking (T)", "Intuiting (N)","Introversion (I)"]
    xlabels2 =["Perceiving (P)","Feeling (F)","Sensing (S)","Extraversion(E)"]
    ax.barh(xlabels, [1, 1, 1, 1])   
    ax.barh(xlabels, [Perceiving, Feeling, Sensing, Extraversion]) 
    ax.set_xlim([0, 1])
    ax.set_xlabel("Propensity")
    ax2 = ax.twinx()
    ax2.barh(xlabels2,[1, 1, 1, 1])
    ax2.barh(xlabels2, [Judging, Thinking, Intuiting, Introversion]) 
    return fig #plt.show(fig)

def OCEAN_Analysis(cleanedtext):
    predictions = {}
    trait_list = ['OPN', 'CON', 'EXT', 'AGR', 'NEU']
    X=[cleanedtext]

    for trait in trait_list:
        pkl_model = models[trait]

        trait_scores = pkl_model.predict(X, regression=True).reshape(1, -1)

        predictions['pred_s'+trait] = trait_scores.flatten()[0]

        trait_categories = pkl_model.predict(X, regression=False)
        predictions['pred_c'+trait] = str(trait_categories[0])

        trait_categories_probs = pkl_model.predict_proba(X)
        predictions['pred_prob_c'+trait] = trait_categories_probs[:, 1][0]
        OCEAN_Analysis = predictions
      
    return OCEAN_Analysis      
                  

def plot_OCEAN_Radar(OCEANAnalysis):        
    # Set data

    listofval=["Sample",OCEANAnalysis.get('pred_sOPN'),OCEANAnalysis.get('pred_sCON'),OCEANAnalysis.get('pred_sEXT'),OCEANAnalysis.get('pred_sAGR'),OCEANAnalysis.get('pred_sNEU')]

    xdf = pd.DataFrame([listofval],columns=["Label","Openess","Conscientiousness","Extraversion","Agreeableness","Neuroticism"])

    # number of variable
    categories=list(xdf)[1:]
    N = len(categories)

        # We are going to plot the first line of the data frame.
        # But we need to repeat the first value to close the circular graph:
    values=xdf.loc[0].drop('Label').values.flatten().tolist()
    values += values[:1]
    values

            # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

            # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

            # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)

            # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([1,2,3,4,5], ["1","2","3","4","5"], color="grey", size=7)
    plt.ylim(0,5)

            # Plot data
    ax.plot(angles, values, linewidth=1, linestyle='solid')

            # Fill area
    ax.fill(angles, values, 'b', alpha=0.1)

    return plt #plt.show()
              
def sentence_by_sentence_analysis(cleanedtext):
    blob = TextBlob(cleanedtext)
    split_text=blob.sentences
    df=pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))),columns=['Sentences'])
    df["Sentence Word Count"]= pd.DataFrame(len(df["Sentences"][i].split()) for i in range(len(df)))
    df["FS_GradeScore"] = pd.DataFrame((textstat.flesch_reading_ease(df["Sentences"][i]) for i in range(len(df))))
    df[["TextBlob_Polarity","TextBlob_Subjectivity"]]=round(pd.DataFrame((split_text[i].sentiment for i in range(len(split_text))))*100,1)
    return df
    
def plot_sentence_by_sentence(cleanedtext,type="polarity"):
    blob = TextBlob(cleanedtext)
    split_text=blob.sentences
    df=pd.DataFrame((''.join(split_text[i]) for i in range(len(split_text))),columns=['Sentences'])
    df["Sentence Word Count"]= pd.DataFrame(len(df["Sentences"][i].split()) for i in range(len(df)))
    df["FS_GradeScore"] = pd.DataFrame((textstat.flesch_reading_ease(df["Sentences"][i]) for i in range(len(df))))
    df[["TextBlob_Polarity","TextBlob_Subjectivity"]]=round(pd.DataFrame((split_text[i].sentiment for i in range(len(split_text))))*100,1)

    # gca stands for 'get current axis'
    figure = plt.figure()
    ax = plt.gca()

    if type=="polarity":
        plotvar='TextBlob_Polarity'
    elif type=="subjectivity":
        plotvar='TextBlob_Subjectivity'
    elif type=="readability":
        plotvar="FS_GradeScore"
    elif type=="wordcount":
        plotvar="Sentence Word Count"
    else:
        plotvar="polarity"
        
    print("Plot Of ",plotvar," By Sentence")
    df.plot(kind='line',y=plotvar,ax=ax)
    # set the x-spine
    #ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    #ax.spines['right'].set_color('none')
    #ax.yaxis.tick_left()

    # set the y-spine
    #ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    #ax.spines['top'].set_color('none')
    #ax.xaxis.tick_bottom()

    print("The X-Axis represents sentences within the document")
    return plt #plt.show()
    
def wordcloud(tokensalpha):
    comment_words = " ".join(tokensalpha)+" "

    wordcloud = WordCloud(width = 800, height = 500, 
                background_color ='white', 
                stopwords = wcstopwords, 
                min_font_size = 8).generate(comment_words) 
                       
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
  
    return plt # plt.show()        

def predict_claps(ArticleExtractSummary,timing=0):    
    if ArticleExtractSummary.get('Analysis Type')=='Extracted Web-Page':
        article_pub_date = datetime.datetime.strptime(ArticleExtractSummary.get('pub_date'), '%Y-%m-%d')
        ref_date = datetime.datetime(2020, 9, 19)
        Age_19Sep20= (ref_date-article_pub_date).days+timing
        X_vals_for_regression= [Age_19Sep20]
        for column in column_for_regression:
            X_vals_for_regression.append(ArticleExtractSummary.get(column))
        log_predicted_claps = clap_prediction_model_lasso.predict(np.array([X_vals_for_regression]))
        predicted_claps=round(np.exp(log_predicted_claps[0]),1)
    else:
        predicted_claps="NA - Can Only Predict Claps For Web Pages, Not Free Form Text"
    return predicted_claps

def predict_clap_category(cleanedtext):
    vec=Doc2VecModel.infer_vector(cleanedtext.split())
    sim_VH=(1-spatial.distance.cosine(VH_Vec,vec))
    sim_H=(1-spatial.distance.cosine(H_Vec,vec))
    sim_M=(1-spatial.distance.cosine(M_Vec,vec))
    sim_L=(1-spatial.distance.cosine(L_Vec,vec))
    sim_VL=(1-spatial.distance.cosine(VL_Vec,vec))             
    if (sim_VH > sim_H) and (sim_VH > sim_M) and (sim_VH > sim_L) and (sim_VH > sim_VL) :
        Predicted="VH: >10,000 Claps"
    elif (sim_H > sim_VH) and (sim_H > sim_M) and (sim_H > sim_L) and (sim_H > sim_VL) :
        Predicted="H: 5,000-10,000 Claps"
    elif (sim_M > sim_VH) and (sim_M > sim_H) and (sim_M > sim_L) and (sim_M > sim_VL) :
        Predicted="M: 1,000-5,000 Claps"
    elif (sim_L > sim_VH) and (sim_L > sim_H) and (sim_L > sim_M) and (sim_L > sim_VL) :
        Predicted="L: 100-1,000 Claps"
    elif (sim_VL > sim_VH) and (sim_VL > sim_H) and (sim_VL > sim_M) and (sim_VL > sim_L) :
        Predicted="VL: < 100 Claps"
    similarity_list=[["VH:>10,000 Claps:",sim_VH],["H:5,000-10,000 Claps:",sim_H],["M:1,000-5,000 Claps:",sim_M],["L:100-1,000 Claps: ",sim_L],["VL:< 100 Claps: ",sim_VL],["Semantic Similarity Is Highest With:",Predicted]]
    return similarity_list

#Loads the reference dataset of ~40 articles with >5,000 Claps and also list of key metrics used in Lasso Regression model
refdataset = pd.read_excel('./ClapPredictionModels/Dataset.xlsx')
metrics=['title_word_count', 'text_word_count',
        'vocab_count_excl_commonwords', 'sentence_count',
       'average_word_count_per_sentence','FS_GradeScore',
       'imgs_per_1000words', 'vids_per_1000words', 'polarity',
       'subjectivity']

class imgcounterclass():
    """Class container for processing stuff."""

    _counter = 0

    def addcounter(self):
        self._counter += 1

image_counter=imgcounterclass()

class cleanedtext_store:
    def __init__(self):
        self.text = None
    
    def addtext(self,newtext):
        self.text = newtext
    
    def cleartext(self):
        self.text = None
        
cleanedtextstore=cleanedtext_store()

from flask import Flask, render_template, url_for, request, session
import os

app = Flask(__name__)
app.secret_key = "randomstring"
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/about', methods = ['POST'])
def about():
    return render_template('about.html')

@app.route('/single_page_text', methods = ['POST'])
def freeform_text_results():
    image_counter.addcounter()
    freeform_text_returned =request.form['Freeform_Text']

    cleanedtext=cleaned_text(freeform_text_returned)
    tokensalpha=tokens_alpha(freeform_text_returned)
    tokenizeexclstopwords=tokenize_excl_stopwords(cleanedtext)
    sentiments=sentiment(cleanedtext)
    MTBIAnalysises=MTBI_Analysis(tokensalpha)
    OCEANAnalysises=OCEAN_Analysis(cleanedtext)
    
    cleanedtextstore.cleartext()
    cleanedtextstore.addtext(cleanedtext)
 
    summary= {
            "Analysis Type":"Freeform Text Entry",

            #Semantic Content
            "content_summary":content_summary(cleanedtext),
            "most_common_trigrams":common_trigrams(tokensalpha),

            #Lexical Content
            "text_word_count": word_count(cleanedtext),
            "vocab_count_excl_commonwords": len(tokenizeexclstopwords),
            "most frequent words":tokenizeexclstopwords[-5:],
            "sentence_count":sentence_count(cleanedtext),
            "average_word_count_per_sentence":word_count(cleanedtext)/sentence_count(cleanedtext),

            #Readability
            "FS_GradeLevel":FS_ReadingEaseLevel(FS_ReadingEaseScore(cleanedtext)),
            "FS_GradeScore":FS_ReadingEaseScore(cleanedtext),

            #Metadata Metrics

            #Sentiment Analysis
            "polarity":sentiments[0]*100,
            "subjectivity":sentiments[1]*100,

            #Personality Analysis
            "MTBI_Type": MTBIAnalysises[0],
            "MTBI_Introversion":MTBIAnalysises[1],
            "MTBI_Extraversion":1-MTBIAnalysises[1],
            "MTBI_iNtuiting":MTBIAnalysises[2],
            "MTBI_Sensing":1-MTBIAnalysises[2], 
            "MTBI_Thinking":MTBIAnalysises[3],
            "MTBI_Feeling":1-MTBIAnalysises[3],
            "MTBI_Judging":MTBIAnalysises[4],
            "MTBI_Perceiving":1-MTBIAnalysises[4],

            "Big5_Openess":OCEANAnalysises.get('pred_sOPN'),
            "Big5_Conscientiousness":OCEANAnalysises.get('pred_sCON'),
            "Big5_Extraversion":OCEANAnalysises.get('pred_sEXT'),
            "Big5_Agreeableness":OCEANAnalysises.get('pred_sAGR') , 
            "Big5_Neuroticism": OCEANAnalysises.get('pred_sNEU')}
    
    text_df=pd.DataFrame.from_dict(summary,orient='index')
    pd.set_option('display.max_colwidth', 800)
    
    wordcloud(tokensalpha).savefig(f'static/wordcloud{image_counter._counter}.png',bbox_inches="tight")
    plot_MTBI(tokensalpha).savefig(f'static/MTBI{image_counter._counter}.png',bbox_inches="tight")
    plot_OCEAN_Radar(OCEANAnalysises).savefig(f'static/OCEAN{image_counter._counter}.png',bbox_inches="tight")
   
    return render_template('single_page_text.html',image_counter=str(image_counter._counter),freeform_text = freeform_text_returned, text_tables=[text_df.to_html(classes='data', header="false")])

@app.route('/single_page_URL', methods = ['POST'])
def single_page_URL_results():
    image_counter.addcounter()
    url_returned =request.form['Single_URL']
    
    htmlinput=get_html(url_returned)
    cleanedtext=cleaned_text(htmlinput)
    tokensalpha=tokens_alpha(htmlinput)
    tokenizeexclstopwords=tokenize_excl_stopwords(cleanedtext)
    sentiments=sentiment(cleanedtext)
    MTBIAnalysises=MTBI_Analysis(tokensalpha)
    OCEANAnalysises=OCEAN_Analysis(cleanedtext)
    
    cleanedtextstore.cleartext()
    cleanedtextstore.addtext(cleanedtext)
    
    summary= {
            "Analysis Type":"Extracted Web-Page",
            "title": extract_title(htmlinput),
            "pub_date":get_date(url_returned),

            "url":url_returned,
            #Semantic Content
            "tags": extract_tags(htmlinput),
            "content_summary":content_summary(cleanedtext),
            "most_common_trigrams":common_trigrams(tokensalpha),

            #Lexical Content
            "title_word_count": (len(extract_title(htmlinput).replace(":","").replace("'","").replace("/","").split())),
            "text_word_count": word_count(cleanedtext),
            "vocab_count_excl_commonwords": len(tokenizeexclstopwords),
            "most frequent words":tokenizeexclstopwords[-5:],

            "sentence_count":sentence_count(cleanedtext),
            "average_word_count_per_sentence":word_count(cleanedtext)/sentence_count(cleanedtext),

            #Readability
            "FS_GradeLevel":FS_ReadingEaseLevel(FS_ReadingEaseScore(cleanedtext)),
            "FS_GradeScore":FS_ReadingEaseScore(cleanedtext),

            #Metadata Metrics
            "image_count": image_count(htmlinput),
            "embedded_vids_count": embedded_vids_count(htmlinput),
            "other_embedded_items_count": other_embedded_items_count(htmlinput,embedded_vids_count(htmlinput)),
            "imgs_per_1000words": ((image_count(htmlinput))/(word_count(cleanedtext)))*1000,
            "vids_per_1000words": ((embedded_vids_count(htmlinput))/(word_count(cleanedtext)))*1000,

            #Sentiment Analysis

            "polarity":sentiments[0]*100,
            "subjectivity":sentiments[1]*100,

            #Personality Analysis
            "MTBI_Type": MTBIAnalysises[0],
            "MTBI_Introversion":MTBIAnalysises[1],
            "MTBI_Extraversion":1-MTBIAnalysises[1],
            "MTBI_iNtuiting":MTBIAnalysises[2],
            "MTBI_Sensing":1-MTBIAnalysises[2], 
            "MTBI_Thinking":MTBIAnalysises[3],
            "MTBI_Feeling":1-MTBIAnalysises[3],
            "MTBI_Judging":MTBIAnalysises[4],
            "MTBI_Perceiving":1-MTBIAnalysises[4],

            "Big5_Openess":OCEANAnalysises.get('pred_sOPN'),
            "Big5_Conscientiousness":OCEANAnalysises.get('pred_sCON'),
            "Big5_Extraversion":OCEANAnalysises.get('pred_sEXT'),
            "Big5_Agreeableness":OCEANAnalysises.get('pred_sAGR') , 
            "Big5_Neuroticism": OCEANAnalysises.get('pred_sNEU')}
    
    session.pop('summary',None)
    session['summary']=summary
    
    url_df=pd.DataFrame.from_dict(summary,orient='index')
    pd.set_option('display.max_colwidth', 800)
    wordcloud(tokensalpha).savefig(f'static/wordcloud{image_counter._counter}.png',bbox_inches="tight")
    plot_MTBI(tokensalpha).savefig(f'static/MTBI{image_counter._counter}.png',bbox_inches="tight")
    plot_OCEAN_Radar(OCEANAnalysises).savefig(f'static/OCEAN{image_counter._counter}.png',bbox_inches="tight")

    return render_template('single_page_URL.html',image_counter=str(image_counter._counter),url_text= url_returned,urldetails_tables=[url_df.to_html(classes='data', header="false")])

@app.route('/detailed_analysis_text', methods = ['POST'])
def detailed_analysis_text():
    cleanedtext=cleanedtextstore.text
    image_counter.addcounter()
    freeform_text_returned =request.form['Freeform_Text']
    text_df=sentence_by_sentence_analysis(cleanedtext)

    pd.set_option('display.max_colwidth', 800)

    plot_sentence_by_sentence(cleanedtext,type='wordcount').savefig(f'static/wordcount{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='readability').savefig(f'static/readability{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='subjectivity').savefig(f'static/subjectivity{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='polarity').savefig(f'static/polarity{image_counter._counter}.png',bbox_inches="tight")
 
    return render_template('detailed_analysis_text.html',image_counter=str(image_counter._counter),text_tables=[text_df.to_html(classes='data', header="false")])

@app.route('/detailed_analysis_URL', methods = ['POST'])
def detailed_analysis_URL():
    # remove the xxxx from the session if it is there
    #session.pop('xxxx', None)
    cleanedtext=cleanedtextstore.text
    image_counter.addcounter()
    freeform_text_returned =request.form['URL_Text']
    url_df=sentence_by_sentence_analysis(cleanedtext)
    
    pd.set_option('display.max_colwidth', 800)

    plot_sentence_by_sentence(cleanedtext,type='wordcount').savefig(f'static/wordcount{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='readability').savefig(f'static/readability{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='subjectivity').savefig(f'static/subjectivity{image_counter._counter}.png',bbox_inches="tight")

    plot_sentence_by_sentence(cleanedtext,type='polarity').savefig(f'static/polarity{image_counter._counter}.png',bbox_inches="tight")
 
    return render_template('detailed_analysis_URL.html',image_counter=str(image_counter._counter),url_text= freeform_text_returned,text_tables=[url_df.to_html(classes='data', header="false")])

@app.route('/clap_prediction', methods = ['POST'])
def clap_prediction():
    image_counter.addcounter()
    freeform_text_returned =request.form['URL_TextCP']
    summary=session['summary']
    cleanedtext=cleanedtextstore.text
    PredictionNow=round(predict_claps(summary),0)
    PredictionInSixMonths=round(predict_claps(summary,timing=180),0)
    CategoryPrediction=predict_clap_category(cleanedtext)
    
    for metric in metrics:
        sample_metric=summary.get(metric)
        sample_percentile=stats.percentileofscore(refdataset.loc[refdataset['popularity'] == 'More Than 5,000 Claps'][metric], sample_metric, kind='strict')

        if sample_percentile >= 100:
            annotation="Sample "+metric+" Exceeds The Max Of Reference Hi Claps Dataset !"
        elif sample_percentile <= 0 :
            annotation="Sample "+metric+" is Below The Min Of Reference Hi Claps Dataset !"
        elif sample_percentile <100 and sample_percentile>0 :
            annotation="Sample "+metric+" is "+str(round(sample_percentile,1))+"% Percentile Of Reference Hi Claps Dataset"

        fig, (ax1,ax2) = plt.subplots(1, 2, sharey='row')
        ax1.boxplot(refdataset.loc[refdataset['popularity'] == 'More Than 5,000 Claps'][metric])
        ax2.scatter(1,sample_metric,500,marker="X")
        fig.suptitle(metric)
        ax1.set_title('Articles > 5k Claps')
        ax2.set_title('vs Sample')
        ax2.get_xaxis().set_ticks([])
        plt.figtext(0.5, 0.01, annotation, ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        plt.savefig(f'static/{metric}{image_counter._counter}.png',bbox_inches="tight") # plt.show()

    return render_template('clap_prediction.html',image_counter=str(image_counter._counter),url_text= freeform_text_returned,PredictionNow=PredictionNow,PredictionInSixMonths=PredictionInSixMonths,CategoryPrediction=CategoryPrediction[5][1],VeryHigh=CategoryPrediction[0],High=CategoryPrediction[1],Medium=CategoryPrediction[2],Low=CategoryPrediction[3],VeryLow=CategoryPrediction[4])

@app.route('/side_by_side', methods = ['POST'])
def side_by_side():
    image_counter.addcounter()

    url_returned =request.form['Double_URL1']
    htmlinput=get_html(url_returned)
    cleanedtext=cleaned_text(htmlinput)
    tokensalpha=tokens_alpha(htmlinput)
    tokenizeexclstopwords=tokenize_excl_stopwords(cleanedtext)
    sentiments=sentiment(cleanedtext)
    MTBIAnalysises=MTBI_Analysis(tokensalpha)
    OCEANAnalysises=OCEAN_Analysis(cleanedtext)
    
    cleanedtext1=cleanedtext
    
    summary1= {
            "Analysis Type":"Extracted Web-Page",
            "title": extract_title(htmlinput),
            "pub_date":get_date(url_returned),

            "url":url_returned,
            #Semantic Content
            "tags": extract_tags(htmlinput),
            "content_summary":content_summary(cleanedtext),
            "most_common_trigrams":common_trigrams(tokensalpha),

            #Lexical Content
            "title_word_count": (len(extract_title(htmlinput).replace(":","").replace("'","").replace("/","").split())),
            "text_word_count": word_count(cleanedtext),
            "vocab_count_excl_commonwords": len(tokenizeexclstopwords),
            "most frequent words":tokenizeexclstopwords[-5:],

            "sentence_count":sentence_count(cleanedtext),
            "average_word_count_per_sentence":word_count(cleanedtext)/sentence_count(cleanedtext),

            #Readability
            "FS_GradeLevel":FS_ReadingEaseLevel(FS_ReadingEaseScore(cleanedtext)),
            "FS_GradeScore":FS_ReadingEaseScore(cleanedtext),

            #Metadata Metrics
            "image_count": image_count(htmlinput),
            "embedded_vids_count": embedded_vids_count(htmlinput),
            "other_embedded_items_count": other_embedded_items_count(htmlinput,embedded_vids_count(htmlinput)),
            "imgs_per_1000words": ((image_count(htmlinput))/(word_count(cleanedtext)))*1000,
            "vids_per_1000words": ((embedded_vids_count(htmlinput))/(word_count(cleanedtext)))*1000,

            #Sentiment Analysis

            "polarity":sentiments[0]*100,
            "subjectivity":sentiments[1]*100,

            #Personality Analysis
            "MTBI_Type": MTBIAnalysises[0],
            "MTBI_Introversion":MTBIAnalysises[1],
            "MTBI_Extraversion":1-MTBIAnalysises[1],
            "MTBI_iNtuiting":MTBIAnalysises[2],
            "MTBI_Sensing":1-MTBIAnalysises[2], 
            "MTBI_Thinking":MTBIAnalysises[3],
            "MTBI_Feeling":1-MTBIAnalysises[3],
            "MTBI_Judging":MTBIAnalysises[4],
            "MTBI_Perceiving":1-MTBIAnalysises[4],

            "Big5_Openess":OCEANAnalysises.get('pred_sOPN'),
            "Big5_Conscientiousness":OCEANAnalysises.get('pred_sCON'),
            "Big5_Extraversion":OCEANAnalysises.get('pred_sEXT'),
            "Big5_Agreeableness":OCEANAnalysises.get('pred_sAGR') , 
            "Big5_Neuroticism": OCEANAnalysises.get('pred_sNEU')}    
    
    
    url1_df=pd.DataFrame.from_dict(summary1,orient='index')
    plot_MTBI(tokensalpha).savefig(f'static/MTBI{image_counter._counter}.png',bbox_inches="tight")
    plot_OCEAN_Radar(OCEANAnalysises).savefig(f'static/OCEAN{image_counter._counter}.png',bbox_inches="tight")
    wordcloud(tokensalpha).savefig(f'static/wordcloud{image_counter._counter}.png',bbox_inches="tight")


    image_counter.addcounter()
    
    url_returned =request.form['Double_URL2']

    htmlinput=get_html(url_returned)
    cleanedtext=cleaned_text(htmlinput)
    tokensalpha=tokens_alpha(htmlinput)
    tokenizeexclstopwords=tokenize_excl_stopwords(cleanedtext)
    sentiments=sentiment(cleanedtext)
    MTBIAnalysises=MTBI_Analysis(tokensalpha)
    OCEANAnalysises=OCEAN_Analysis(cleanedtext)
    
    cleanedtext2=cleanedtext
    
    summary1= {
            "Analysis Type":"Extracted Web-Page",
            "title": extract_title(htmlinput),
            "pub_date":get_date(url_returned),

            "url":url_returned,
            #Semantic Content
            "tags": extract_tags(htmlinput),
            "content_summary":content_summary(cleanedtext),
            "most_common_trigrams":common_trigrams(tokensalpha),

            #Lexical Content
            "title_word_count": (len(extract_title(htmlinput).replace(":","").replace("'","").replace("/","").split())),
            "text_word_count": word_count(cleanedtext),
            "vocab_count_excl_commonwords": len(tokenizeexclstopwords),
            "most frequent words":tokenizeexclstopwords[-5:],

            "sentence_count":sentence_count(cleanedtext),
            "average_word_count_per_sentence":word_count(cleanedtext)/sentence_count(cleanedtext),

            #Readability
            "FS_GradeLevel":FS_ReadingEaseLevel(FS_ReadingEaseScore(cleanedtext)),
            "FS_GradeScore":FS_ReadingEaseScore(cleanedtext),

            #Metadata Metrics
            "image_count": image_count(htmlinput),
            "embedded_vids_count": embedded_vids_count(htmlinput),
            "other_embedded_items_count": other_embedded_items_count(htmlinput,embedded_vids_count(htmlinput)),
            "imgs_per_1000words": ((image_count(htmlinput))/(word_count(cleanedtext)))*1000,
            "vids_per_1000words": ((embedded_vids_count(htmlinput))/(word_count(cleanedtext)))*1000,

            #Sentiment Analysis

            "polarity":sentiments[0]*100,
            "subjectivity":sentiments[1]*100,

            #Personality Analysis
            "MTBI_Type": MTBIAnalysises[0],
            "MTBI_Introversion":MTBIAnalysises[1],
            "MTBI_Extraversion":1-MTBIAnalysises[1],
            "MTBI_iNtuiting":MTBIAnalysises[2],
            "MTBI_Sensing":1-MTBIAnalysises[2], 
            "MTBI_Thinking":MTBIAnalysises[3],
            "MTBI_Feeling":1-MTBIAnalysises[3],
            "MTBI_Judging":MTBIAnalysises[4],
            "MTBI_Perceiving":1-MTBIAnalysises[4],

            "Big5_Openess":OCEANAnalysises.get('pred_sOPN'),
            "Big5_Conscientiousness":OCEANAnalysises.get('pred_sCON'),
            "Big5_Extraversion":OCEANAnalysises.get('pred_sEXT'),
            "Big5_Agreeableness":OCEANAnalysises.get('pred_sAGR') , 
            "Big5_Neuroticism": OCEANAnalysises.get('pred_sNEU')}    
    
    
    url2_df=pd.DataFrame.from_dict(summary1,orient='index')
    plot_MTBI(tokensalpha).savefig(f'static/MTBI{image_counter._counter}.png',bbox_inches="tight")
    plot_OCEAN_Radar(OCEANAnalysises).savefig(f'static/OCEAN{image_counter._counter}.png',bbox_inches="tight")
    wordcloud(tokensalpha).savefig(f'static/wordcloud{image_counter._counter}.png',bbox_inches="tight")
      
    SideBySideComparison=pd.concat([url1_df, url2_df], axis=1)
    SideBySideComparison.columns=["",""]
    
    pd.set_option('display.max_colwidth', 800)
    
    vecSampleOne = Doc2VecModel.infer_vector(cleanedtext1.split())
    vecSampleTwo = Doc2VecModel.infer_vector(cleanedtext2.split())
    
    similarity = round((1-spatial.distance.cosine(vecSampleOne, vecSampleTwo))*100,3)
    
    return render_template('side_by_side.html',similarity=similarity,image_counter1=str(image_counter._counter-1),image_counter2=str(image_counter._counter),urldetails_tables=[SideBySideComparison.to_html(classes='data', header="false")])

# @app.after_request
# def add_header(response):
#     # response.cache_control.no_store = True
#     response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
#     response.headers['Pragma'] = 'no-cache'
#     response.headers['Expires'] = '-1'
#     return response

print("Hey, I Loaded Up...")

if __name__ == '__main__':
    app.run(debug = True)


