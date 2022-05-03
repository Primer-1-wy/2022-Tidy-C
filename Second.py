from tqdm import tqdm
import pandas as pd

tqdm.pandas()



Hotel_Info1 = pd.read_excel(
    './data/2018-2019茂名（含自媒体）.xlsx', sheet_name=0)   # 酒店评论
Scenic_Info1 = pd.read_excel(
    './data/2018-2019茂名（含自媒体）.xlsx', sheet_name=1)  # 景区评论
Travel_Info1 = pd.read_excel(
    './data/2018-2019茂名（含自媒体）.xlsx', sheet_name=2)     # 游记攻略
Dining_Info1 = pd.read_excel(
    './data/2018-2019茂名（含自媒体）.xlsx', sheet_name=3)  # 餐饮评论
Wechat_Info1 = pd.read_excel(
    './data/2018-2019茂名（含自媒体）.xlsx', sheet_name=4)  # 微信公众号文章


Hotel_Info2 = pd.read_excel(
    './data/2020-2021茂名（含自媒体）.xlsx', sheet_name=0)   # 酒店评论
Scenic_Info2 = pd.read_excel(
    './data/2020-2021茂名（含自媒体）.xlsx', sheet_name=1)  # 景区评论
Travel_Info2 = pd.read_excel(
    './data/2020-2021茂名（含自媒体）.xlsx', sheet_name=2)     # 游记攻略
Dining_Info2 = pd.read_excel(
    './data/2020-2021茂名（含自媒体）.xlsx', sheet_name=3)  # 餐饮评论
Wechat_Info2 = pd.read_excel(
    './data/2020-2021茂名（含自媒体）.xlsx', sheet_name=4)  # 微信公众号文章

Hotel_Infos = pd.concat([Hotel_Info1, Hotel_Info2],axis=0)  # 酒店评论
Scenic_Infos = pd.concat([Scenic_Info1, Scenic_Info2], axis=0)  # 景区评论
Travel_Infos = pd.concat([Travel_Info1, Travel_Info2], axis=0)  # 游记攻略
Dining_Infos = pd.concat([Dining_Info1, Dining_Info2], axis=0)  # 餐饮评论
Wechat_Infos = pd.concat([Wechat_Info1, Wechat_Info2], axis=0)  # 微信公众号文章
'''
旅游产品，亦称旅游服务产品。是指由实物和服务构成。包括旅行商集合景点、交通、食宿、娱乐等设施设备、
项目及相应服务出售给旅游者的旅游线路类产品，旅游景区、旅游饭店等单个企业提供给旅游者的活动项目类产品
'''


Scenic_Infos.head(10)

def addstr(s):
    return '景区评论-'+str(s)

Scenic_Infos['语料ID'] = Scenic_Infos['景区评论ID'].progress_apply(addstr)
Scenic_Infos['文本'] = Scenic_Infos['评论内容']
Scenic_Infos['产品名称'] = Scenic_Infos['景区名称']
Scenic_Infos['年份'] = pd.to_datetime(Scenic_Infos['评论日期']).dt.year

Hotel_Infos.head(10)


def addstr(s):
    return '酒店评论-'+str(s)

Hotel_Infos['语料ID'] = Hotel_Infos['酒店评论ID'].progress_apply(addstr)
Hotel_Infos['文本'] = Hotel_Infos['评论内容']
Hotel_Infos['产品名称'] = Hotel_Infos['酒店名称']
Hotel_Infos['年份'] = pd.to_datetime(Hotel_Infos['评论日期']).dt.year


def addstr(s):
    return '餐饮评论-'+str(s)


Dining_Infos['语料ID'] = Dining_Infos['餐饮评论ID'].progress_apply(addstr)
Dining_Infos['文本'] = Dining_Infos['评论内容'] + '\n'+Dining_Infos['标题']
Dining_Infos['产品名称'] = Dining_Infos['餐饮名称']
Dining_Infos['年份'] = pd.to_datetime(Dining_Infos['评论日期']).dt.year

# 采用Textrank提取关键词组算法
# 这部分待改进
from textrank4zh import TextRank4Keyword  # 导入textrank4zh模块
import numpy as np


def get_keyphrase(s):
    tr4w = TextRank4Keyword(
        allow_speech_tags=['n', 'nr', 'nr1', 'nr2', 'nrf', 'ns', 'nsf', 'nt', 'nz', 'nz', 'nl', 'ng'])
    tr4w.analyze(text=str(s), lower=True, window=5)  # 文本分析，文本小写，窗口为2
    # 最多5个关键词组，有可能一个也没有。词组在原文中出现次数最少为1。
    phase_list = tr4w.get_keyphrases(keywords_num=5, min_occur_num=1)
    if len(phase_list) == 0:
        return np.nan
    else:
        return phase_list[0]


# 游记攻略
Travel_Infos = pd.concat([Travel_Info1, Travel_Info2], axis=0)  # 游记攻略


def addstr(s):
    return '旅游攻略-' + str(s)


Travel_Infos['语料ID'] = Travel_Infos['游记ID'].progress_apply(addstr)
Travel_Infos['文本'] = Travel_Infos['游记标题'] + '\n' + Travel_Infos['正文']
Travel_Infos['年份'] = pd.to_datetime(Travel_Infos['发布时间']).dt.year
Travel_Infos['产品名称'] = Travel_Infos['文本'].progress_apply(get_keyphrase)

# 微信公众号文章
Wechat_Infos = pd.concat([Wechat_Info1, Wechat_Info2], axis=0)  # 微信公众号文章


def addstr(s):
    return '微信公共号文章-' + str(s)


Wechat_Infos['语料ID'] = Wechat_Infos['文章ID'].progress_apply(addstr)
Wechat_Infos['文本'] = Wechat_Infos['公众号标题'] + '\n' + Wechat_Infos['正文']
Wechat_Infos['年份'] = pd.to_datetime(Wechat_Infos['发布时间']).dt.year
Wechat_Infos['产品名称'] = Wechat_Infos['文本'].progress_apply(get_keyphrase)



# 删除没有产品名称的行
Travel_Infos = Travel_Infos.dropna(subset=["产品名称"])
Wechat_Infos = Wechat_Infos.dropna(subset=["产品名称"])



all_df = pd.DataFrame(columns=['语料ID', '文本', '产品名称'])
all_df['语料ID'] = pd.concat([Dining_Infos['语料ID'], Hotel_Infos['语料ID'],
                           Scenic_Infos['语料ID'], Travel_Infos['语料ID']], axis=0)
all_df['产品名称'] = pd.concat([Dining_Infos['产品名称'],Hotel_Infos['产品名称'],
                           Scenic_Infos['产品名称'], Travel_Infos['产品名称']], axis=0)
all_df['文本'] = pd.concat([Dining_Infos['文本'], Hotel_Infos['文本'],
                         Scenic_Infos['文本'], Travel_Infos['文本']], axis=0)
all_df['年份'] = pd.concat([Dining_Infos['年份'], Hotel_Infos['年份'],
                         Scenic_Infos['年份'], Travel_Infos['年份']], axis=0)
all_df


product_id = ['ID'+str(i+1) for i in range(len(all_df))]
all_df['产品ID'] = product_id
result2 = all_df[['语料ID','产品ID','产品名称']]
result2


result2.to_csv('./data/result2-1.csv', index=False)
all_df.to_csv('./data/问题二所有数据汇总.csv', index=False)

import warnings
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
warnings.filterwarnings('ignore')

all_Info = pd.read_csv('./data/问题二所有数据汇总.csv')
all_Info


from cnsenti import Sentiment
senti = Sentiment(pos='./data/pos.txt',  #正面词典txt文件相对路径
                  neg='./data/neg.txt',  #负面词典txt文件相对路径
                  merge=True,             #融合cnsenti自带词典和用户导入的自定义词典
                  encoding='utf-8')      #两txt均为utf-8编码
def emotion_score(s):

    r = senti.sentiment_count(str(s))
    if r['pos']>r['neg']:
        point = (r['pos']-r['neg'])/r['words']
    elif r['pos'] < r['neg']:
        point = (r['pos']-r['neg'])/r['words']
    else :
        point = 0
    return point

all_Info['情感得分'] = all_Info['文本'].progress_apply(emotion_score)

#senti = Sentiment()      #两txt均为utf-8编码
#for str1 in all_df['情感得分']:
  #  result=senti.sentiment_count(str(str1))
    #result=1
   # print(result)

year_2018_count = all_df[all_df['年份']==2018]
year_2019_count = all_df[all_df['年份'] == 2019]
year_2020_count = all_df[all_df['年份'] == 2020]
year_2021_count = all_df[all_df['年份'] == 2021]

dict_2018 = dict(year_2018_count['产品名称'].value_counts())
def get_frequency(s):
    fre = dict_2018[s]
    return fre
year_2018_count['出现频次'] = year_2018_count['产品名称'].progress_apply(get_frequency)


dict_2019 = dict(year_2019_count['产品名称'].value_counts())
def get_frequency(s):
    fre = dict_2019[s]
    return fre
year_2019_count['出现频次'] = year_2019_count['产品名称'].progress_apply(get_frequency)


dict_2020 = dict(year_2020_count['产品名称'].value_counts())
def get_frequency(s):
    fre = dict_2020[s]
    return fre
year_2020_count['出现频次'] = year_2020_count['产品名称'].progress_apply(get_frequency)


dict_2021 = dict(year_2021_count['产品名称'].value_counts())
def get_frequency(s):
    fre = dict_2021[s]
    return fre
year_2021_count['出现频次'] = year_2021_count['产品名称'].progress_apply(get_frequency)


# 计算综合得分
year_2018_count['产品热度总分'] = 0.8*year_2018_count['出现频次']+200*year_2018_count['情感得分']+0
year_2019_count['产品热度总分'] = 0.8*year_2019_count['出现频次']+200*year_2019_count['情感得分']+1*10
year_2020_count['产品热度总分'] = 0.8*year_2020_count['出现频次']+200*year_2020_count['情感得分']+2*15
year_2021_count['产品热度总分'] = 0.8*year_2021_count['出现频次']+200*year_2021_count['情感得分']+3*20

year_2018_count['产品热度'] = year_2018_count['产品热度总分'].div(np.sum(year_2018_count['产品热度总分']), axis=0)#化成一个小数 加起来为1
year_2019_count['产品热度'] = year_2019_count['产品热度总分'].div(np.sum(year_2019_count['产品热度总分']), axis=0)
year_2020_count['产品热度'] = year_2020_count['产品热度总分'].div(np.sum(year_2020_count['产品热度总分']), axis=0)
year_2021_count['产品热度'] = year_2021_count['产品热度总分'].div(np.sum(year_2021_count['产品热度总分']), axis=0)

year_2018 = year_2018_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
year_2019 = year_2019_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
year_2020 = year_2020_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
year_2021 = year_2021_count.sort_values(by="产品热度", ascending=False).reset_index(drop=True)

product_hot_score = pd.concat([year_2018_count, year_2019_count, year_2020_count, year_2021_count], axis=0)
product_hot_score


# 分词
import re
import jieba
stopword_list = [k.strip() for k in open(
    '.\stop\cn_stopwords.txt', encoding='utf8').readlines() if k.strip() != '']
def clearTxt(line):
    if line != '':
        line = str(line).strip()
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        #只保留中文、大小写字母
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        line = re.sub(reg, '', line)
        #分词
        segList = jieba.cut(line, cut_all=False)
        segSentence = ''
        for word in segList:
            if word != '\t':
                segSentence += word + " "
    # 去停用词
    wordList = segSentence.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopword_list:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()


product_hot_score['文本'] = product_hot_score['文本'].progress_apply(clearTxt)
product_hot_score

# 景区、酒店、网红景点、民宿、特色餐饮、乡村旅游、文创
def get_product_type(s):
    if '景区' in s:
        return '景区'
    elif '酒店' in s:
        return '酒店'
    elif '餐饮' in s:
        return '特色餐饮'
    elif '景点' in s:
        return '景点'
    elif '民宿' in s:
        return '民宿'
    elif '乡村' in s:
        return '乡村旅游'
    elif '文创' in s:
        return '文创'
    else:
        return '景点'

product_hot_score['产品类型判断文本'] = product_hot_score['语料ID'] +' '+product_hot_score['文本']

product_hot_score['产品类型'] = product_hot_score['产品类型判断文本'].progress_apply(get_product_type)


# 去除重复的产品
product_hot_score2 = product_hot_score.drop_duplicates(['产品名称'])
product_hot_score2


# 产品 ID 产品类型 产品名称 产品热度 年份

result2_2 = product_hot_score2[['产品ID','产品类型','产品名称','产品热度','年份']]
result2_2['产品ID'] = ['ID'+str(i+1) for i in range(len(result2_2))]
result2_2


result2_2.to_csv('./data/result2-2.csv',index=False)

# 计算产品热度
pre_data = all_df[all_df['年份']<2020]
after_data = all_df[all_df['年份']>2019]
dict_pre = dict(pre_data['产品名称'].value_counts())
dict_after = dict(after_data['产品名称'].value_counts())
def get_pre_frequency(s):
    fre = dict_pre[s]
    return fre


def get_after_frequency(s):
    fre = dict_after[s]
    return fre

pre_data['出现频次'] = pre_data['产品名称'].progress_apply(get_pre_frequency)
after_data['出现频次'] = after_data['产品名称'].progress_apply(get_after_frequency)
# 计算综合得分
pre_data['产品热度总分'] = 3*pre_data['出现频次']+2*pre_data['情感得分']
after_data['产品热度总分'] = 3*after_data['出现频次']+2*after_data['情感得分']


pre_data['产品热度'] = pre_data['产品热度总分'].div(np.sum(pre_data['产品热度总分']), axis=0)
after_data['产品热度'] = after_data['产品热度总分'].div(np.sum(after_data['产品热度总分']), axis=0)

pre_data_sort = pre_data.sort_values(by="产品热度", ascending=False).reset_index(drop=True)
after_data_sort = after_data.sort_values(
    by="产品热度", ascending=False).reset_index(drop=True)


# 判断产品类型
import re
import jieba
stopword_list = [k.strip() for k in open(
    '.\stop\cn_stopwords.txt', encoding='utf8').readlines() if k.strip() != '']
def clearTxt(line):
    if line != '':
        line = str(line).strip()
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]", "", line)
        #只保留中文、大小写字母
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        line = re.sub(reg, '', line)
        #分词
        segList = jieba.cut(line, cut_all=False)
        segSentence = ''
        for word in segList:
           # global segSentence
            if word != '\t':
                segSentence += word + " "
    # 去停用词
    wordList = segSentence.split(' ')
    sentence = ''
    for word in wordList:
        word = word.strip()
        if word not in stopword_list:
            if word != '\t':
                sentence += word + " "
    return sentence.strip()

# 景区、酒店、网红景点、民宿、特色餐饮、乡村旅游、文创
def get_product_type(s):
    if '景区' in s:
        return '景区'
    elif '酒店' in s:
        return '酒店'
    elif '餐饮' in s:
        return '特色餐饮'
    elif '景点' in s:
        return '景点'
    elif '民宿' in s:
        return '民宿'
    elif '乡村' in s:
        return '乡村旅游'
    elif '文创' in s:
        return '文创'
    else:
        return '景点'

pre_data_sort['文本'] = pre_data_sort['文本'].progress_apply(clearTxt)
pre_data_sort['产品类型判断文本'] = pre_data_sort['语料ID'] + ' ' + pre_data_sort['文本']

pre_data_sort['产品类型'] = pre_data_sort['产品类型判断文本'].progress_apply(get_product_type)

after_data_sort['文本'] = after_data_sort['文本'].progress_apply(clearTxt)
after_data_sort['产品类型判断文本'] = after_data_sort['语料ID'] + \
                                  ' ' + after_data_sort['文本']

after_data_sort['产品类型'] = after_data_sort['产品类型判断文本'].progress_apply(
    get_product_type)


pre_data_sort.to_csv('./data/疫情前产品热度.csv', index=False)
after_data_sort.to_csv('./data/疫情后产品热度.csv', index=False)




