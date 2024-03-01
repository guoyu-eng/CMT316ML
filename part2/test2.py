# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# import joblib
# import numpy as np
# import nltk
# import sklearn
# import operator
# import requests
# import pandas as pd
# from sklearn.svm import SVR
# import joblib
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# import random
# import spacy
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import random
# from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# import random
# from sklearn.model_selection import train_test_split

# import numpy as np
# import sklearn.svm
# # python -m spacy download en_core_web_sm ： download


# lemmatizer = nltk.stem.WordNetLemmatizer()
# stopwords=set(nltk.corpus.stopwords.words('english'))
# # stopwords.add("'s")
# # stopwords.add("the")
# stopwords.update(["'s", "the", "of", "in", "on", "at", "to", "from", "by", "for", "with", "as", "such", "a", "an", "the"])

# # stopwords = {word for word in stopwords if not word.isdigit()}
# # stopwords.add(",")
# # stopwords.add("--")
# # stopwords.add("``")

# folder_path = 'datasets_coursework1/bbc/tech'
# folder_path2 = 'datasets_coursework1/bbc/tech2'
# folder_path_po = 'datasets_coursework1/bbc/po'
# nlp = spacy.load("en_core_web_sm")

# # input .txt , out put string without captitgal empty...
# def preProcessing(folder_path):
#     all_data = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.txt'):
#             file_path = os.path.join(folder_path, file_name)
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 file_data = file.readlines()
#                 # lowwer
#                 file_data_cleaned = [line.strip().lower() for line in file_data if line.strip()]
#                 # morphological restoration
#                 file_data_processed = []
#                 for line in file_data_cleaned:
#                     doc = nlp(line)
#                     sentence_tokens = []
#                     for token in doc:
#                         if not token.is_punct:
#                             lemma = token.lemma_.lower()
#                             if lemma not in stopwords:  #
#                                 sentence_tokens.append(lemma)
#                     file_data_processed.append(sentence_tokens)
#                 all_data.extend(file_data_processed)
#     return all_data








# # ['dd','ddaa'] -> wordlist( remove the commod word)

# lemmatizer = nltk.stem.WordNetLemmatizer()
# stopwords=set(nltk.corpus.stopwords.words('english'))
# stopwords.add(".")
# stopwords.add(",")
# stopwords.add("--")
# stopwords.add("``")
# stopwords.add("'s")


# # TF Dictionary， and get the frequncy high
# def get_vocabulary(input_data, n):
#     word_frequency = {}
#     for sentence in input_data:
#         for word in sentence:
#             if word not in stopwords:  # 添加此行以检查单词是否不在停用词中
#                 if word not in word_frequency:
#                     word_frequency[word] = 1
#                 else:
#                     word_frequency[word] += 1

#     sorted_word_frequency = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
#     sorted_word_without_frequency = [word[0] for word in sorted_word_frequency[:n]]

#     return sorted_word_without_frequency







# # with lable  形成特征向量
# def get_vector_text(list_vocab, input_data):
#     X = []
#     for instance in input_data:
#         # sentence = instance[0]
#         vector_instance = [instance.count(word) for word in list_vocab]
#         X.append(vector_instance)
#     return X





# all_data = preProcessing(folder_path2)
# # print(" procesing : ")
# # print(all_data[:3])
# aa = get_vocabulary(all_data,30)
# # print(" get_vocabulary : ")
# # print(aa[:10])
# bb = get_vector_text(aa,all_data)


# import numpy as np

# import numpy as np

# import numpy as np
# def compute_idf(documents):
#     N = len(documents)
#     word_doc_freq = {}  # 用于存储每个词语的文档频率
#     idf_values = {}     # 用于存储每个词语的逆文档频率

#     # 统计每个词语的文档频率
#     for document in documents:
#         unique_words = set(document)
#         for word in unique_words:
#             if word not in word_doc_freq:
#                 word_doc_freq[word] = 1
#             else:
#                 word_doc_freq[word] += 1

#     # 计算每个词语的逆文档频率
#     for word, freq in word_doc_freq.items():
#         idf_values[word] = np.log(N / freq)

#     # 按照 IDF 值从大到小排序
#     sorted_idf_values = {k: v for k, v in sorted(idf_values.items(), key=lambda item: item[1], reverse=True)}

#     return list(sorted_idf_values.keys())  # 仅返回单词列表

# # 示例输入集合
# idf_values = compute_idf(all_data)
# print(idf_values)  # 打印单词列表




# def get_vector_text(list_vocab, input_data):
#     X = []
#     for instance in input_data:
#         sentence = instance[0]  # 获取句子部分
#         vector_instance = [sentence.count(word) for word in list_vocab]  # 生成特征向量
#         X.append(vector_instance)
#     return X

# # 示例用法
# input_data = [(['the', 'ofbe', 'carry', 'out', 'eive', 'to', 'be', 'pro', 'government','use','use'], 1)]

# vocabulary = ['ink', 'net', 'use', 'cafe']

# # 获取特征向量
# X = get_vector_text(vocabulary, input_data)

# # 打印结果
# for i, instance in enumerate(input_data):
#     print("Sentence:", instance[0])
#     print("Label:", instance[1])
#     print("Feature vector:", X[i])
#     print()
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import nltk
import sklearn
import operator
import requests
import pandas as pd
from sklearn.svm import SVR
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.model_selection import train_test_split
import chardet
import numpy as np
import sklearn.svm
from sklearn.metrics import classification_report
import sklearn.svm
import pickle


# python -m spacy download en_core_web_sm ： download
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0.tar.gz
# pip install -U spacy
# pip install chardet
# pip install --upgrade spacy pydantic
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
            # 检测文件编码
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





# ['dd','ddaa'] -> wordlist( remove the commod word)



# TF Dictionary， and get the frequncy high
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

    return sorted_word_frequency  # 返回单词和对应的词频信息，而不仅仅是单词列表





# with lable  形成特征向量
def get_vector_text(list_vocab, input_data):
    X = []
    for instance in input_data:
        # sentence = instance[0]
        vector_instance = [instance.count(word) for word in list_vocab]
        X.append(vector_instance)
    return X



def get_vector_text_3(vocabulary1, vocabulary2, vocabulary3, input_data):
    X1 = []  # 存储基于vocabulary1的特征向量
    X2 = []  # 存储基于vocabulary2的特征向量
    X3 = []  # 存储基于vocabulary3的特征向量

    for instance in input_data:
        # 计算基于vocabulary1的特征向量
        vector_instance_1 = [instance.count(word) for word in vocabulary1]
        X1.append(vector_instance_1)

        # 计算基于vocabulary2的特征向量
        vector_instance_2 = [instance.count(word) for word in vocabulary2]
        X2.append(vector_instance_2)

        # 计算基于vocabulary3的特征向量
        vector_instance_3 = [instance.count(word) for word in vocabulary3]
        X3.append(vector_instance_3)

    # 转换成 numpy 数组
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    X3 = np.asarray(X3)

    # 合并三个特征矩阵
    X = np.concatenate((X1, X2, X3), axis=1)

    return X








# 拆分集合 70 15 15
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





# tf-idf get for whole doc
# TF_idf Dictionary
def get_tfidf_vectors(input_data):
    # 将每个实例的词语列表连接成单个字符串
    corpus = [' '.join(instance[0]) for instance in input_data]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)
    vocabulary = tfidf_vectorizer.get_feature_names_out()
    word_index_mapping = {word: index for index, word in enumerate(vocabulary)}

    return tfidf_vectors, vocabulary,word_index_mapping
        # return tfidf_vectors, vocabulary, word_index_mapping



# 方法用于提取 TF-IDF 值最高的前 n 个单词和对应的 TF-IDF 值
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
    # 调用方法获取 TF-IDF 向量、单词列表和单词到索引的映射关系
    tfidf_vectors, vocabulary, word_index_mapping = get_tfidf_vectors(input_data)

    # 提取 TF-IDF 值最高的前十个单词和对应的 TF-IDF 值
    top_n_word_tfidf_tuples = extract_top_n_word_tfidf(tfidf_vectors, vocabulary)

    # 创建一个包含单词和对应的 TF-IDF 值的列表
    word_tfidf_list = [(word, tfidf) for word, tfidf in top_n_word_tfidf_tuples]

    return word_tfidf_list




# 计算idf 集合 输入要是 [ xxx,xx ] 格式  包含特征选择

def compute_idf(documents, threshold=None):
    N = len(documents)
    word_doc_freq = {}  # 用于存储每个词语的文档频率

    # 统计每个词语的文档频率
    for document in documents:
        unique_words = set(document)
        for word in unique_words:
            if word not in word_doc_freq:
                word_doc_freq[word] = 1
            else:
                word_doc_freq[word] += 1

    # 过滤出频率大于等于阈值的单词
    if threshold is not None:
        word_doc_freq = {word: freq for word, freq in word_doc_freq.items() if freq >= threshold}

    # 计算每个词语的逆文档频率
    idf_values = {word: np.log(N / freq) for word, freq in word_doc_freq.items()}

    # 按照 IDF 值从大到小排序
    sorted_idf_values = sorted(idf_values.items(), key=lambda item: item[1], reverse=True)

    # 如果选择的单词不足50个，则返回前50个单词
    if len(sorted_idf_values) < 50:
        selected_idf_values = sorted_idf_values[:50]
    else:
        selected_idf_values = sorted_idf_values

    return selected_idf_values



def variance_thresholding(word_frequency_info, threshold):
    # 计算词频列表
    frequencies = [frequency for word, frequency in word_frequency_info]
    # 计算词频的方差
    variance = np.var(frequencies)

    # 选择方差大于阈值的词频特征
    selected_features = [word for word, frequency in word_frequency_info if frequency >= threshold * variance]

    # 如果特征数量不足100，直接选择前100个特征
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


#  分离数据集合准备进行字典制造
techtes_data = preProcessing(folder_path_techtes)
train_data_tech, dev_data_tech, test_data_tech = prepare_data(techtes_data, 1.0)
# print('techtes_data')
# print(len(techtes_data))
# print(len(train_data_tech))
po_data = preProcessing(folder_path_po)
train_data_po, dev_data_po, test_data_po = prepare_data(po_data, 2.0)
# print('po_data')
# print(len(po_data))
# print(len(train_data_po))
en_data = preProcessing(folder_path_en)
train_data_en, dev_data_en, test_data_en = prepare_data(en_data, 3.0)
# print('en_data')
# print(len(en_data))
# print(len(train_data_en))
bu_data = preProcessing(folder_path_bu)
train_data_bu, dev_data_bu, test_data_bu = prepare_data(bu_data, 4.0)
# print('bu_data')
# print(len(bu_data))
# print(len(train_data_bu))
sp_data = preProcessing(folder_path_sp)
train_data_sp, dev_data_sp, test_data_sp = prepare_data(sp_data,5.0)
# print('sp_data')
# print(len(sp_data))
# print(len(train_data_sp))
print(' the train set, develop set and test set is : 70%, 15%, 15%')
# tess= train_data[:3] + train_data2[:3]
# print(train_data_po)
# train set 合并 训练集合
train_set = train_data_po + train_data_tech + train_data_en + train_data_bu + train_data_sp
dev_set = dev_data_tech + dev_data_po + dev_data_en + dev_data_bu + dev_data_sp
testing = test_data_tech + test_data_po +test_data_en +test_data_bu + test_data_sp


# print('train_set number : ')
# print(len(train_set))
# print('dev_set number : ')
# print(len(dev_set))
# print('testing number : ')
# print(len(testing))

# # IDF
# label_datasets = [train_data_po, train_data_tech, train_data_en, train_data_bu, train_data_sp]

# threshold = 6.3
# total_val_idf = []

# # 对每个标签数据集进行处理
# for label_data in label_datasets:
#     # 提取单词列表部分
#     word_list = [[word for word in item[0]] for item in label_data]
#     # 计算 IDF
#     idf_values = compute_idf(word_list, threshold=threshold)
#     # 将选择的单词添加到总特征列表中
#     # total_val_idf1.extend([word for word, _ in idf_values])
#     total_val_idf.extend([word for word, _ in idf_values][:100])
# print("IDF features count:", len(total_val_idf))




# tfidf dictory
# 1. 对每个标签的数据集分别调用 tfidf_value_return 方法
val_po_ti = tfidf_value_return(train_data_po)
val_tech_ti = tfidf_value_return(train_data_tech)
val_en_ti = tfidf_value_return(train_data_en)
val_bu_ti = tfidf_value_return(train_data_bu)
val_sp_ti = tfidf_value_return(train_data_sp)



# 设置方差选择的阈值
thresholdtfidf = 1
selc_po = variance_thresholding(val_po_ti, thresholdtfidf)
selc_tech = variance_thresholding(val_tech_ti, thresholdtfidf)
selc_en = variance_thresholding(val_en_ti, thresholdtfidf)
selc_bu = variance_thresholding(val_bu_ti, thresholdtfidf)
selc_sp = variance_thresholding(val_sp_ti, thresholdtfidf)
total_word_ti = val_po_ti + val_tech_ti + val_en_ti + val_bu_ti + val_sp_ti

top_n_features = 100
selc_po = selc_po[:top_n_features]
selc_tech = selc_tech[:top_n_features]
selc_en = selc_en[:top_n_features]
selc_bu = selc_bu[:top_n_features]
selc_sp = selc_sp[:top_n_features]
print("selc_po中的元素数量:", len(selc_po))
print("selc_tech中的元素数量:", len(selc_tech))
print("selc_en中的元素数量:", len(selc_en))
print("selc_bu中的元素数量:", len(selc_bu))
print("selc_sp中的元素数量:", len(selc_sp))

# 2. 将每个标签的 TF-IDF 特征值存储在一个列表中
total_val_ti = selc_po + selc_tech + selc_en + selc_bu + selc_sp

print( ' the TF-IDF feature original number is ' , len(total_word_ti ))
print( ' the TF-IDF feature number is ' , len(total_val_ti ))





# TF

label_datasets = [train_data_po, train_data_tech, train_data_en, train_data_bu, train_data_sp]
thresholdtf = 0.05
total_val_tf = []

# 对每个标签数据集进行处理
for i, label_data in enumerate(label_datasets):
    # 获取标签数据集中的句子部分
    sentences = [sentence for sentence, _ in label_data]
    # 计算词频
    word_frequency = get_vocabulary(sentences)
    # 进行方差选择
    selected_features = variance_thresholding(word_frequency, thresholdtf)
    # 将选择的特征添加到总特征列表中，并保留前 100 个特征
    total_val_tf.extend(selected_features[:100])

    # 打印每个标签数据集中选择的特征数量
    print(f"Selected features for label {i+1}: {len(selected_features)}")

# 打印被选择的特征数量
print("Total TF features count:", len(total_val_tf))



