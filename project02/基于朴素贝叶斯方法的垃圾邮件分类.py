# 导入各种包
import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

# 从数据集读入数据，正常右键label为0，垃圾邮件label为1
data = pd.read_csv('spam_ham_dataset.csv')
data = data.iloc[:, 1:]
print(data.head())
print(data.info())

print('这份数据包含{}条邮件'.format(data.shape[0]))
print('正常邮件一共有{}条'.format(data['label_num'].value_counts()[0]))
print('垃圾邮件一共有{}条'.format(data['label_num'].value_counts()[1]))
plt.style.use('seaborn')
plt.figure(figsize=(6, 4), dpi=100)
data['label'].value_counts().plot(kind='bar')
plt.show()

# 只需要text与label_num
new_data = data.iloc[:, 1:]
length = len(new_data)
print('邮件数量 length =', length)
print(new_data.head())

# 将所有单词替换为小写
new_data['text'] = new_data['text'].str.lower()
print(new_data.head())

# 设置停用词
stop_words = set(stopwords.words('english'))
stop_words.add('subject')


# 提取一长串句子中的每个单词，并且过滤掉各种符号，以及词形换源
def text_process(text):
    tokenizer = RegexpTokenizer('[a-z]+')  # 只匹配单词
    lemmatizer = WordNetLemmatizer()
    token = tokenizer.tokenize(text)  # 分词
    token = [lemmatizer.lemmatize(w) for w in token
             if lemmatizer.lemmatize(w) not in stop_words]  # 停用词+词形还原
    return token


new_data['text'] = new_data['text'].apply(text_process)

# 此时可以得到一个比较干净的数据集
print(new_data.head())

# 将处理后的数据及分为训练集与测试集，比例为3:1
seed = 20200401  # 设定种子值让实验具有重复性
X = new_data['text']
y = new_data['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)  # 75%作为训练集与25%作为测试集

train = pd.concat([X_train, y_train], axis=1)  # 训练集
test = pd.concat([X_test, y_test], axis=1)  # 测试集

train.reset_index(drop=True, inplace=True)  # 重设下标
test.reset_index(drop=True, inplace=True)  # 重设下标

print('训练集含有{}封邮件，测试集含有{}封邮件'.format(train.shape[0], test.shape[0]))

# 训练集中垃圾邮件与正常邮件的数量
print(train['label_num'].value_counts())
plt.figure(figsize=(6, 4), dpi=100)
train['label_num'].value_counts().plot(kind='bar')
plt.show()

# 测试集中垃圾邮件与正常邮件的数量
print(test['label_num'].value_counts())
plt.figure(figsize=(6, 4), dpi=100)
test['label_num'].value_counts().plot(kind='bar')
plt.show()

# 随机抽取正常右键与垃圾邮件各10封内的单词作为单词表
ham_train = train[train['label_num'] == 0]  # 正常邮件
spam_train = train[train['label_num'] == 1]  # 垃圾邮件

ham_train_part = ham_train['text'].sample(10, random_state=seed)  # 随机抽取的10封正常邮件
spam_train_part = spam_train['text'].sample(10, random_state=seed)  # 随机抽取的10封垃圾邮件

part_words = []  # 部分的单词

for text in pd.concat([ham_train_part, spam_train_part]):
    part_words += text

part_words_set = set(part_words)
print('单词表一共有{}个单词'.format(len(part_words_set)))

# 把单词整理成句子，然后统计每个单词出现的次数并计算TF-IDF

# 将正常邮件与垃圾邮件的单词都整理为句子，单词间以空格相隔，
train_part_texts = [
    ' '.join(text)
    for text in np.concatenate((spam_train_part.values, ham_train_part.values))
]
# 训练集所有的单词整理成句子
train_all_texts = [' '.join(text) for text in train['text']]
# 测试集所有的单词整理成句子
test_all_texts = [' '.join(text) for text in test['text']]

cv = CountVectorizer()
part_fit = cv.fit(train_part_texts)  # 以部分句子为参考
train_all_count = cv.transform(train_all_texts)  # 对训练集所有邮件统计单词个数
test_all_count = cv.transform(test_all_texts)  # 对测试集所有邮件统计单词个数
tfidf = TfidfTransformer()
train_tfidf_matrix = tfidf.fit_transform(train_all_count)
test_tfidf_matrix = tfidf.fit_transform(test_all_count)

print('训练集', train_tfidf_matrix.shape)
print('测试集', test_tfidf_matrix.shape)

# 建立模型
mnb = MultinomialNB()
mnb.fit(train_tfidf_matrix, y_train)

# 模型在测试集上的正确率
mnb.score(test_tfidf_matrix, y_test)

y_pred = mnb.predict_proba(test_tfidf_matrix)
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
auc = auc(fpr, tpr)

# roc 曲线
plt.figure(figsize=(6, 4), dpi=100)
plt.plot(fpr, tpr)
plt.title('roc = {:.4f}'.format(auc))
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()
