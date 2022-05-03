# 中文文本分类
import os
import re

import jieba
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import xlrd
import xlwt
import openpyxl

warnings.filterwarnings('ignore')

def cut_words(file_path):
    """
    对文本进行切词
    :param file_path: txt文本路径
    :return: 用空格分词的字符串
    """
    text_with_spaces = ''
    text = open(file_path, 'r', encoding='UTF-8').read()
    textcut = jieba.cut(text)
    for word in textcut:
        text_with_spaces += word + ' '
    return text_with_spaces

def clean_text(text):
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub("\d+", '', text)  # 删除数字
    text = re.sub('[a-zA-Z]', '', text)  # 删除字母
    text = re.sub('[\s]', '', text)  # 删除空格
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text

stop_words = open('stopwords.txt', 'r', encoding='utf-8').read()
stop_words = stop_words.encode('utf-8').decode('utf-8-sig') # 列表头部\ufeff处理
stop_words = stop_words.split('\n') # 根据分隔符分隔

otherwords = ['茂名', '茂名市', '年', '月', '日']
stop_words.extend(otherwords)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print('样本预测数据加载中...')
data1 = pd.read_excel('2018-2019茂名（含自媒体）.xlsx', sheet_name='微信公众号新闻', header = 0)
data2 = pd.read_excel('2020-2021茂名（含自媒体）.xlsx', sheet_name='微信公众号新闻', header = 0)
data_all = data1.append(data2, ignore_index=True)
print('加载完成')

print('样本预测数据转换中...')
data_all['文章'] = data_all['公众号标题'] + data_all['正文']
data_all = data_all[['文章ID', '文章']].dropna()
data_last = data_all['文章'].apply(lambda x: clean_text(x)).apply(
    lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stop_words])
)
#将表格做完处理后保存
data_last.to_excel('fenci.xlsx')
print('转换完成')

print('载入训练数据....')
data = pd.read_csv('baked_data.csv', encoding='gb18030')  # 载入训练数据
labels = data['labels']  # 设置标签
features = data['massages'].astype(str).apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stop_words]))  # 设置样本特征
# 20%作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.20, stratify=labels, random_state=1)

print('计算tfidf特征...')
# 计算单词权重
tf = TfidfVectorizer(stop_words=stop_words, max_df=0.5)

train_features = tf.fit_transform(train_x)
# 上面fit过了，这里transform
test_features = tf.transform(test_x)
predict_features = tf.transform(data_last)
print('计算完成')

# 多项式贝叶斯分类器
print('训练分类器...')
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.01).fit(train_features, train_y)
predicted_labels = clf.predict(test_features)
print('训练完成')

# 计算准确率
print('准确率为：', metrics.accuracy_score(test_y, predicted_labels))

print('预测样本数据...')
predict = clf.predict(predict_features)
print(predict)

print('预测完成')
data_ = data_all.values

count = 0
for i in data_:
    i[1] = predict[count]
    count += 1

print(data_)

columes = ['文章ID', '分类标签']
df = pd.DataFrame(data_, columns=columes)
df.to_csv('result1.csv', encoding='utf-8', index=False)
print('done')
