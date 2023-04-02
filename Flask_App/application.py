from flask import Flask, render_template, request, url_for, flash, redirect
import numpy as np
import logging
# Packages for loading models and sparse matrix
from scipy import sparse
# Natural Language Toolkit 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
# sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, WordPunctTokenizer
from nltk.stem import PorterStemmer
# 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


application = Flask(__name__)

question = ""
categories = [[[], []], ['', '', '', ['', '', '']]]
tossup_list = []

f = open('static/models/nb.pickle', 'rb')
nb = pickle.load(f)
f.close()

f = open('static/models/nb_tfidf.pickle', 'rb')
nb_tfidf = pickle.load(f)
f.close()

tfs_vecs = sparse.load_npz("static/models/tfs_vecs.npz")

f = open('static/models/cs_tfidf.pickle', 'rb')
cs_tfidf = pickle.load(f)
f.close()


tokenizer      = nltk.tokenize.word_tokenize
stop_words     = set(nltk.corpus.stopwords.words('english'))
wnl            = nltk.WordNetLemmatizer()
# stemmer        = nltk.stem.PorterStemmer()

sub_class_index = [ '', 'Literature: European', 'Fine Arts: Visual', 'Mythology', 'Literature: American',
                   'Science: Chemistry', 'Geography', 'Philosophy', 'Fine Arts: Audio', 'Social Science',
                  '', '', 'Literature: World', 'History: American', 'Science: Biology',
                  '', 'History: Classical', '', 'Science: Physics', 'Religion',
                  'History: World', '', 'Literature: British', '', 'History: European',
                  'Fine Arts: Other', 'Science: Math']

cosine_similarity_labels = ['Fine Arts: Audio', 'Fine Arts: Other','Fine Arts: Visual', 'Geography', 'History: American',
                            'History: Classica', 'History: European', 'History: World','Literature: American',
                            'Literature: British', 'Literature: European', 'Literature: World','Mythology',
                            'Philosophy', 'Religion', 'Science: Biology', 'Science: Chemistry', 'Science: Math',
                            'Science: Physics', 'Social Science']

def cleanText(raw_text): #tokenize, lowercase, remove stopwords, removePunctuation, lemmatize
    tokens         = tokenizer(raw_text)                                #step 1  
    tokens         = [ word.lower() for word in tokens ]                #step 2
    tokens         = [ w for w in tokens if not w in stop_words ]       #step 3
    tokens         = [ w for w in tokens if w.isalpha() ]               #step 4
    tokens         = [ wnl.lemmatize ( t ) for t in tokens ]            #step 5
    #text           = ' '.join(tokens)
    return tokens

def classifyTossup(raw_text,tfidf,clf): #tfidf is our vectorizer, clf is our classifier
    clean_text = ' '.join(cleanText(raw_text)) #important to prepare data the same way
    tfs_vec   = tfidf.transform([clean_text]) #must just be transform, don't re-fit
    tfidf_data = tfs_vec.toarray()
    y_prob     = clf.predict_proba(tfidf_data)
    topPredIndices = np.argsort(y_prob, axis=1)[:,:-4:-1]
    print(topPredIndices)
    classes = clf.classes_[topPredIndices[0]].astype(int)
    categories = np.array(sub_class_index)[classes]
    topProbs = np.sort(y_prob, axis=1)[:,:-4:-1][0]
    topProbs = [round(p, 4) for p in topProbs]
    return [categories, topProbs]

# Print cosine similarity of example tossups
def cosine_similarity_ranking():
    for tossup in tossup_list:
        query_tokens = cleanText(tossup)
        query_matrix = cs_tfidf.transform([query_tokens])     # we need the [] to make it a list
        cosine_result = cosine_similarity(query_matrix, tfs_vecs)
    #     print(cosine_result)
        # https://stackoverflow.com/questions/6193498/pythonic-way-to-find-maximum-value-and-its-index-in-a-list
        maxSimilarity = max(range(len(cosine_result[0])), key = lambda x: cosine_result[0][x])
        maxSimilarityP = cosine_result[0][maxSimilarity]
        cosine_result[0][maxSimilarity] = -1
        secondSimilarity = max(range(len(cosine_result[0])), key = lambda x: cosine_result[0][x])
        secondSimilarityP = cosine_result[0][secondSimilarity]
        cosine_result[0][secondSimilarity] = -1
        thirdSimilarity = max(range(len(cosine_result[0])), key = lambda x: cosine_result[0][x])
        thirdSimilarityP = cosine_result[0][thirdSimilarity]
        topProbs = [maxSimilarityP,secondSimilarityP,thirdSimilarityP]
        topProbs = [round(p, 4) for p in topProbs]
        return [cosine_similarity_labels[maxSimilarity], cosine_similarity_labels[secondSimilarity], cosine_similarity_labels[thirdSimilarity], topProbs]


@application.route('/')
def index():
    # application.logger.info(categories)
    return render_template('index.html', categories=categories)

@application.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        global question
        question = request.form['question']
        if len(tossup_list) != 0:
            tossup_list.pop()
        tossup_list.append(question)
        nb_categories = classifyTossup(tossup_list[0], nb_tfidf, nb)
        cs_categories = cosine_similarity_ranking()
        while categories:
            categories.pop()
        categories.append(nb_categories)
        categories.append(cs_categories)

        return redirect(url_for('index'))