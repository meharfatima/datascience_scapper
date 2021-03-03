# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 16:53:43 2021

@author: FatimMe
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('salary_data_cleaned.csv')
df.head()

def title_simplifier(title):
    if 'scientist' in title.lower():
        return 'data scientist'
    elif 'analyst' in title.lower():
        return 'data analyst'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'director' in title.lower():
        return 'director'
    else:
        return 'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'sr'
    elif 'jr' in title.lower() or 'junior' in title.lower():
        return 'jr'
    else:
        return 'na'
    
df['job_simpl']=df['Job Title'].apply(title_simplifier)
df['job_simpl'].value_counts()
df['seniority'] = df['Job Title'].apply(seniority)  
df['seniority'].value_counts()

df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
df['state'] = df['state'].apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')
df.state.value_counts()

df['num_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != '-1' else 0)
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.per_hour == 1 else x.min_salary, axis = 1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.per_hour == 1 else x.max_salary, axis = 1)
df['avg_salary'] = df.apply(lambda x: x.avg_salary*2 if x.per_hour == 1 else x.avg_salary, axis = 1)

df['company'] = df['company'].apply(lambda x: x.replace('\n',''))
df['company'] = df['company'].apply(lambda x: x.replace('\r',''))

df.describe()

"save"
df.to_csv('salary_data_eda.csv', index=False)

df.min_salary.hist()
df.max_salary.hist()
df.avg_salary.hist()
df.Rating.hist()
df.Size.hist()
df.Revenue.hist()
df.age_of_company.hist()

df.boxplot(column = ['Rating', 'num_comp'])
df.boxplot(column = ['age_of_company', 'avg_salary'])
df[['age_of_company', 'Rating', 'avg_salary', 'desc_len']].corr()
sns.heatmap(df[['age_of_company', 'Rating', 'avg_salary', 'desc_len']])
sns.heatmap(df[['Rating', 'num_comp', 'R', 'python']])

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(df[['age_of_company', 'Rating', 'avg_salary', 'desc_len', 'num_comp']].corr(), cmap=cmap, vmax = .3, center=0, square=True, linewidth=.5, cbar_kws={"shrink": .5})

df_cat = df[['Location', 'Headquarters', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'company', 'state', 'python', 'R', 'spark', 'sql', 'aws', 'excel', 'job_simpl', 'seniority']]
for i in df_cat.columns:
    cat = df_cat[i].value_counts()
    sns.barplot(x = cat.index, y = cat)
    plt.xticks(rotation=90)
    plt.show()

pd.options.display.max_rows
pd.set_option('display.max_rows', None)  
pd.pivot_table(df, index = 'job_simpl', values = 'avg_salary')
pd.pivot_table(df, index = 'seniority', values = 'avg_salary')
pd.pivot_table(df, index = 'state', values = 'avg_salary')
pd.pivot_table(df, index = ['job_simpl', 'seniority'], values = 'avg_salary')
pd.pivot_table(df, index = ['job_simpl', 'seniority'], values = 'avg_salary').sort_values('avg_salary', ascending=False)
pd.pivot_table(df, index = ['state', 'job_simpl'], values = 'avg_salary', aggfunc = 'count').sort_values('state')
pd.pivot_table(df[df.job_simpl=='data scientist'], index = ['state'], values = 'avg_salary').sort_values('state')

df_pivots = df[['Rating', 'Industry', 'Sector', 'Revenue', 'num_comp', 'per_hour', 'employer_provided_salary', 'python', 'R', 'spark', 'aws', 'excel', 'sql', 'Type of ownership','avg_salary']]
for i in df_pivots.columns:
    print(i)
    if i != 'avg_salary':
        print(pd.pivot_table(df_pivots, index = i, values = 'avg_salary').sort_values('avg_salary', ascending = False))
        
pd.pivot_table(df_pivots, index = 'Revenue', columns = 'python', values = 'avg_salary', aggfunc = 'count')

from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

words = " ".join(df['Job Description'])

def punctuation_stop(text):
    """remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered


words_filtered = punctuation_stop(words)

text = " ".join([ele for ele in words_filtered])

wc= WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =800, height = 1500)
wc.generate(text)
plt.figure(figsize=[10,10])
plt.imshow(interpolation="bilinear", X = wc.generate(text))
plt.axis('off')
plt.show()