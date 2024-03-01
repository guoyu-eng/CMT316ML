from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import nltk
import sklearn
import operator
import requests
import pandas as pd
from sklearn.svm import SVR
import os
from sklearn.model_selection import train_test_split
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import chardet
import sklearn.svm
from sklearn.metrics import classification_report
import pickle
from sklearn.ensemble import RandomForestClassifier


# pip install -U spacy
# pip install scikit-learn
# pip install nltk
# python -m spacy download en_core_web_sm ： download
# python -m spacy download en_core_web_sm
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz


nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.update(["'s", "the", "of", "in", "on", "at", "to", "from", "by", "for", "with", "as", "such", "a", "an", "the"])
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("'s")


folder_path_techtes = 'datasets_coursework1/bbc/tech'
folder_path_po = 'datasets_coursework1/bbc/politics'
folder_path_en = 'datasets_coursework1/bbc/entertainment'
folder_path_bu = 'datasets_coursework1/bbc/business'
folder_path_sp = 'datasets_coursework1/bbc/sport'



# input .txt , out put string without captitgal empty...

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        rawdata = file.read()
    return chardet.detect(rawdata)['encoding']

def preProcessing(folder_path):
    all_data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            encoding = detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as file:
                file_data = file.readlines()
                # lowwer
                file_data_cleaned = [line.strip().lower() for line in file_data if line.strip()]
                # morphological restoration
                file_data_processed = []
                for line in file_data_cleaned:
                    doc = nlp(line)
                    sentence_tokens = []
                    for token in doc:
                        if not token.is_punct:
                            lemma = token.lemma_.lower()
                            if lemma not in stopwords:
                                sentence_tokens.append(lemma)
                    file_data_processed.append(sentence_tokens)
                all_data.extend(file_data_processed)
    return all_data






#  if u can not use the spacy model , please use this function
# def preProcessing(folder_path):
#     all_data = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(folder_path, file_name)
#             encoding = detect_encoding(file_path)
#             with open(file_path, 'r', encoding=encoding) as file:
#                 file_data = file.readlines()
#                 # lowwer
#                 file_data_cleaned = [line.strip().lower() for line in file_data if line.strip()]
#                 # morphological restoration
#                 file_data_processed = []
#                 for line in file_data_cleaned:
#                     sentence_tokens = []
#                     for token in line.split():
#                         if token.isalpha() and token.lower() not in stopwords:
#                             lemma = token.lower()
#                             sentence_tokens.append(lemma)
#                     file_data_processed.append(sentence_tokens)
#                 all_data.extend(file_data_processed)
#     return all_data




# TF methods ， and get the frequncy high
def get_vocabulary(input_data, n=None):
    word_frequency = {}
    for sentence in input_data:
        for word in sentence:
            if word not in word_frequency:
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1

    sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
    if n is not None:
        sorted_word_frequency = sorted_word_frequency[:n]
    # return the word and frequency
    return sorted_word_frequency





# with lable
def get_vector_text(list_vocab, input_data):
    X = []
    for instance in input_data:
        # sentence = instance[0]
        vector_instance = [instance.count(word) for word in list_vocab]
        X.append(vector_instance)
    return X


# change to the eigenvector (math.)
def get_vector_text_3(vocabulary1, vocabulary2, vocabulary3, input_data):
    X1 = []
    X2 = []
    X3 = []

    for instance in input_data:
        # base on vocabulary1 eigenvector (math.)
        vector_instance_1 = [instance.count(word) for word in vocabulary1]
        X1.append(vector_instance_1)

        vector_instance_2 = [instance.count(word) for word in vocabulary2]
        X2.append(vector_instance_2)

        vector_instance_3 = [instance.count(word) for word in vocabulary3]
        X3.append(vector_instance_3)

    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X3 = np.asarray(X3)

    X = np.concatenate((X1, X2, X3), axis=1)

    return X








# get the train, dev, test set  70% 15% 15%
def prepare_data(all_data, label):
    add_label = [(review, label) for review in all_data]

    x = [example[0] for example in add_label]
    y = [example[1] for example in add_label]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    train_data = list(zip(x_train, y_train))
    pre_test_set = list(zip(x_test, y_test))

    original_size_test = len(pre_test_set)
    size_dev = int(round(original_size_test * 0.5, 0))
    list_dev_indices = random.sample(range(original_size_test), size_dev)

    new_dev_set = []
    new_test_set = []

    for i, example in enumerate(pre_test_set):
        if i in list_dev_indices:
            new_dev_set.append(example)
        else:
            new_test_set.append(example)

    return train_data, new_dev_set, new_test_set




# TF_IDF methods
def get_tfidf_vectors(input_data):
    # Concatenate the list of words for each instance into a single string
    corpus = [' '.join(instance[0]) for instance in input_data]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)
    vocabulary = tfidf_vectorizer.get_feature_names_out()
    word_index_mapping = {word: index for index, word in enumerate(vocabulary)}
    # return tfidf_vectors, vocabulary, word_index_mapping

    return tfidf_vectors, vocabulary,word_index_mapping



# The method is used to extract the top n words with the highest TF-IDF value and the corresponding TF-IDF value
def extract_top_n_word_tfidf(tfidf_vectors, vocabulary, n=None):
    tfidf_array = tfidf_vectors.toarray()
    total_tfidf_values = np.sum(tfidf_array, axis=0)
    if n is None:
        n = len(vocabulary)
    top_n_indices = np.argsort(total_tfidf_values)[-n:]
    top_n_word_tfidf_tuples = [(vocabulary[index], total_tfidf_values[index]) for index in top_n_indices]
    return top_n_word_tfidf_tuples


def extract_top_n_word_tfidf_senten(tfidf_vector, vocabulary, n=None):
    tfidf_array = tfidf_vector.toarray()
    tfidf_values = tfidf_array[0]
    if n is None:
        n = len(vocabulary)
    top_n_indices = np.argsort(tfidf_values)[-n:]

    top_n_word_tfidf_tuples = [(vocabulary[index], tfidf_values[index]) for index in top_n_indices]

    return top_n_word_tfidf_tuples




def tfidf_value_return(input_data):
    # Call the method to get the TF-IDF vectors, word lists and word-to-index mappings
    tfidf_vectors, vocabulary, word_index_mapping = get_tfidf_vectors(input_data)
    top_n_word_tfidf_tuples = extract_top_n_word_tfidf(tfidf_vectors, vocabulary)
    # Create a list of words and their corresponding TF-IDF values
    word_tfidf_list = [(word, tfidf) for word, tfidf in top_n_word_tfidf_tuples]

    return word_tfidf_list



#  IDF methods
def compute_idf(documents, threshold=None):
    N = len(documents)
    word_doc_freq = {}

    # Count the frequency of documents for each word
    for document in documents:
        unique_words = set(document)
        for word in unique_words:
            if word not in word_doc_freq:
                word_doc_freq[word] = 1
            else:
                word_doc_freq[word] += 1
    # Filter out words with a frequency greater than or equal to a threshold value
    if threshold is not None:
        word_doc_freq = {word: freq for word, freq in word_doc_freq.items() if freq >= threshold}

    idf_values = {word: np.log(N / freq) for word, freq in word_doc_freq.items()}
    sorted_idf_values = sorted(idf_values.items(), key=lambda item: item[1], reverse=True)

    # If less than 50 words are selected, the first 50 words are returned
    if len(sorted_idf_values) < 50:
        selected_idf_values = sorted_idf_values[:50]
    else:
        selected_idf_values = sorted_idf_values

    return selected_idf_values



def variance_thresholding(word_frequency_info, threshold):
    # Calculate word frequency list
    frequencies = [frequency for word, frequency in word_frequency_info]
    # Calculating the variance of word frequencies
    variance = np.var(frequencies)
    selected_features = [word for word, frequency in word_frequency_info if frequency >= threshold * variance]
    if len(selected_features) < 50:
        selected_features = [word for word, _ in word_frequency_info[:50]]
        # print( ' feature less than 50 , select head 50')
    return selected_features

def split_data(feature_data):
    quarter_length = len(feature_data) // 4
    half_length = len(feature_data) // 2
    three_quarter_length = (3 * len(feature_data)) // 4

    part1_data = feature_data[:quarter_length]
    part2_data = feature_data[:half_length]
    part3_data = feature_data[:three_quarter_length]
    part4_data = feature_data

    return part1_data, part2_data, part3_data, part4_data



#  Separate data sets ready for dictionary making
techtes_data = preProcessing(folder_path_techtes)
train_data_tech, dev_data_tech, test_data_tech = prepare_data(techtes_data, 1.0)

po_data = preProcessing(folder_path_po)
train_data_po, dev_data_po, test_data_po = prepare_data(po_data, 2.0)

en_data = preProcessing(folder_path_en)
train_data_en, dev_data_en, test_data_en = prepare_data(en_data, 3.0)

bu_data = preProcessing(folder_path_bu)
train_data_bu, dev_data_bu, test_data_bu = prepare_data(bu_data, 4.0)

sp_data = preProcessing(folder_path_sp)
train_data_sp, dev_data_sp, test_data_sp = prepare_data(sp_data,5.0)

print(' the train set, develop set and test set is : 70%, 15%, 15%')


train_set = train_data_po + train_data_tech + train_data_en + train_data_bu + train_data_sp
dev_set = dev_data_tech + dev_data_po + dev_data_en + dev_data_bu + dev_data_sp
testing = test_data_tech + test_data_po +test_data_en +test_data_bu + test_data_sp
print('train_set number : ')
print(len(train_set))
print('dev_set number : ')
print(len(dev_set))
print('testing number : ')
print(len(testing))







# IDF dictory
label_datasets = [train_data_po, train_data_tech, train_data_en, train_data_bu, train_data_sp]
threshold = 6.3
total_val_idf = []

for label_data in label_datasets:
    word_list = [[word for word in item[0]] for item in label_data]
    idf_values = compute_idf(word_list, threshold=threshold)
    # total_val_idf1.extend([word for word, _ in idf_values])
    total_val_idf.extend([word for word, _ in idf_values][:150])
print("IDF features count:", len(total_val_idf))






# TF-IDF dictory
val_po_ti = tfidf_value_return(train_data_po)
val_tech_ti = tfidf_value_return(train_data_tech)
val_en_ti = tfidf_value_return(train_data_en)
val_bu_ti = tfidf_value_return(train_data_bu)
val_sp_ti = tfidf_value_return(train_data_sp)


# Setting thresholds for variance selection
thresholdtfidf = 1
selc_po = variance_thresholding(val_po_ti, thresholdtfidf)
selc_tech = variance_thresholding(val_tech_ti, thresholdtfidf)
selc_en = variance_thresholding(val_en_ti, thresholdtfidf)
selc_bu = variance_thresholding(val_bu_ti, thresholdtfidf)
selc_sp = variance_thresholding(val_sp_ti, thresholdtfidf)
total_word_ti = val_po_ti + val_tech_ti + val_en_ti + val_bu_ti + val_sp_ti

top_n_features = 150
selc_po = selc_po[:top_n_features]
selc_tech = selc_tech[:top_n_features]
selc_en = selc_en[:top_n_features]
selc_bu = selc_bu[:top_n_features]
selc_sp = selc_sp[:top_n_features]

total_val_ti = selc_po + selc_tech + selc_en + selc_bu + selc_sp

print( ' the TF-IDF feature original number is ' , len(total_word_ti ))
print( ' the TF-IDF feature number is ' , len(total_val_ti ))






# TF dictory
label_datasets = [train_data_po, train_data_tech, train_data_en, train_data_bu, train_data_sp]
thresholdtf = 0.05
total_val_tf = []

for label_data in label_datasets:
    sentences = [sentence for sentence, _ in label_data]
    # Calculate word frequency
    word_frequency = get_vocabulary(sentences)
    # Perform variance selection
    selected_features = variance_thresholding(word_frequency, thresholdtf)
    total_val_tf.extend(selected_features[:150])
    # total_val_tf1.extend(selected_features)

print("TF features count:", len(total_val_tf))






# Split the data
tf_quater, tf_half, tf_3quater, tf_all = split_data(total_val_tf)
idf_quater, idf_half, idf_3quater, idf_all = split_data(total_val_idf)
ti_quater, ti_half, ti_3quater, ti_all = split_data(total_val_ti)




def train_rf_classifier(training_set, vocabulary1, vocabulary2, vocabulary3):
    # Extract features and labels from training_set
    X_train1 = get_vector_text(vocabulary1, [instance[0] for instance in training_set])
    X_train2 = get_vector_text(vocabulary2, [instance[0] for instance in training_set])
    X_train3 = get_vector_text(vocabulary3, [instance[0] for instance in training_set])
    Y_train = [instance[1] for instance in training_set]

    # Convert lists to numpy arrays
    X_train1 = np.asarray(X_train1)
    X_train2 = np.asarray(X_train2)
    X_train3 = np.asarray(X_train3)
    Y_train = np.asarray(Y_train)

    # Concatenate the feature matrices horizontally
    X_train = np.concatenate((X_train1, X_train2, X_train3), axis=1)

    # Initialize and train the Random Forest classifier with OOB evaluation
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, oob_score=True)
    rf_clf.fit(X_train, Y_train)

    return rf_clf

# Train the Random Forest classifiers
rf_classifier_all = train_rf_classifier(train_set, total_val_tf, total_val_idf, total_val_ti)
rf_classifier_quater = train_rf_classifier(train_set, tf_quater, idf_quater, ti_quater)
rf_classifier_half = train_rf_classifier(train_set, tf_half, idf_half, ti_half)
rf_classifier_3quater = train_rf_classifier(train_set, tf_3quater, idf_3quater, ti_3quater)

# Print Out-of-Bag scores
print("Out-of-Bag score for all features:", rf_classifier_all.oob_score_)
print("Out-of-Bag score for quarter features:", rf_classifier_quater.oob_score_)
print("Out-of-Bag score for half features:", rf_classifier_half.oob_score_)
print("Out-of-Bag score for three quarters features:", rf_classifier_3quater.oob_score_)


test_data_vectors_all = get_vector_text_3(total_val_tf, total_val_idf, total_val_ti, [instance[0] for instance in dev_set])
predictions_all = rf_classifier_all.predict(test_data_vectors_all)

test_data_vectors_quater = get_vector_text_3(tf_quater, idf_quater, ti_quater, [instance[0] for instance in dev_set])
predictions_quater = rf_classifier_quater.predict(test_data_vectors_quater)


test_data_vectors_half = get_vector_text_3(tf_half, idf_half, ti_half, [instance[0] for instance in dev_set])
predictions_half = rf_classifier_half.predict(test_data_vectors_half)


test_data_vectors_3quater = get_vector_text_3(tf_3quater, idf_3quater, ti_3quater, [instance[0] for instance in dev_set])
predictions_3quater = rf_classifier_3quater.predict(test_data_vectors_3quater)

print(" Accuracy of all using random forests ", accuracy_score([instance[1] for instance in dev_set],predictions_all))
print(" Accuracy of quater using random forests ", accuracy_score([instance[1] for instance in dev_set],predictions_quater))
print(" Accuracy of half using random forests ", accuracy_score([instance[1] for instance in dev_set],predictions_half))
print(" Accuracy of 3quater using random forests ", accuracy_score([instance[1] for instance in dev_set],predictions_3quater))




