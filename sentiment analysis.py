#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install textblob


# In[2]:


pip install wordcloud


# In[3]:


pip install cufflinks


# In[4]:


import numpy as np
import pandas as pd
import nltk 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from wordcloud import WordCloud
import seaborn as sns   
import matplotlib.pyplot as plt
import cufflinks as cf
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
cf.go_offline();
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.max_columns',None)


# In[5]:


df=pd.read_csv("amazon.csv")


# In[6]:


df.head()


# In[7]:


df=df.sort_values("wilson_lower_bound",ascending=False)
df.drop('Unnamed: 0',inplace=True,axis=1)
df.head()


# In[8]:


#function for missing values

def missing_values_analysis(df):
    na_columns_=[col for col in df.columns if df [col].isnull().sum() > 0]

    n_miss=df[na_columns_].isnull().sum().sort_values(ascending=True)
    
    ratio_=(df[na_columns_].isnull().sum() /df.shape[0]*100).sort_values(ascending=True)

    missing_df=pd.concat([n_miss, np.round(ratio_, 2)], axis =1, keys=['Missing Values', 'Ratio'])

    missing_df=pd.DataFrame(missing_df)

    return missing_df

def check_dataframe(df, head=5, tail=5):
    
    print("SHAPE".center(82,'~'))
          
    print('ROWS:{}'.format(df.shape[0]))
    
    print('columns:{}'.format(df.shape[1]))
 
    print('TYPES'.center(82,'~'))
    
    print(df.dtypes)
          
    print("".center(82,'~'))
          
    print(missing_values_analysis(df))
          
    print('DUPLICATED VALUES'.center(83,'~'))
          
    print(df.duplicated().sum())
          
    print('QUANTILES'.center(82,'~'))
          
    print(df.quantile([0,0.05,0.50,0.95,0.99,1]).T)
   


check_dataframe(df)


# In[9]:


def check_class(dataframe):
    nunique_df=pd.DataFrame({'Varaiable':dataframe.columns,
                            'Classes':[dataframe[i].nunique()\
                                      for i in dataframe.columns]})
    nunique_df = nunique_df.sort_values('Classes',ascending=False)
    nunique_df = nunique_df.reset_index(drop=True)
    return nunique_df
    
check_class(df)    


# In[10]:


pip install pandas plotly


# In[39]:


constraints = ['#B34D22', '#EBE00C', '#1FEB0C', '#0C92EB', '#EB0CD5']
def categorical_variable_summary(df, column_name):

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Percentage'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])

    fig.add_trace(go.Bar(y=df[column_name].value_counts().values.tolist(),
                         x=[str(i) for i in df[column_name].value_counts().index],
                         text=df[column_name].value_counts().values.tolist(),
                         textfont=dict(size=15),
                         name=column_name,
                         textposition='auto',
                         showlegend=False,
                         marker=dict(color=constraints,
                                     line=dict(color='#DBE6EC',
                                               width=1))),
                  row=1, col=1)

   
    fig.add_trace(go.Pie(
        labels=df[column_name].value_counts().keys(),
        values=df[column_name].value_counts().values,
        textinfo='label+percent',
        showlegend=False,
        name=column_name,
        marker=dict(colors=constraints)
    ), row=1, col=2)
    fig.update_layout(title={'text': column_name,
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    iplot(fig)
    




# In[12]:


data_frame = df
column_name = 'overall'
categorical_variable_summary(data_frame, column_name)


# In[13]:


df.reviewText.head()


# In[14]:


review_example=df.reviewText[2031]
review_example


# In[15]:


review_example=re.sub("[^a-zA-Z]",'',review_example)
review_example


# In[16]:


review_example = review_example.lower().split()
review_example


# In[17]:


rt=lambda x: re.sub("[^a-zA]",' ',str(x))
df["reviewText"]=df["reviewText"].map(rt)
df["reviewText"]=df["reviewText"].str.lower()
df.head()


# In[18]:


pip install vaderSentiment


# In[20]:


pip install nltk


# In[23]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[35]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
df[['polarity','subjectivity']]=df['reviewText'].apply(lambda Text:pd.Series(TextBlob(Text).sentiment))

for index,row in df['reviewText'].iteritems():
    
    score=SentimentIntensityAnalyzer().polarity_scores(row)
    
    neg=score['neg']
    neu=score['neu']
    pos=score['pos']
    
    if neg>pos:
        df.loc[index,'sentiment']="Negative"
    elif pos>neg:
        df.loc[index,'sentiment']="positive"
    else:
        df.loc[index,'sentiment']="Neutral"
        
        
 #categorial_variable_summary(df,'sentiment')       
#df[df["sentiment"]=="Positive"].sort_values("wilson_lover_bound",ascending=False).head(5)        


# In[37]:


df[df["sentiment"]=="Positive"].sort_values("wilson_lower_bound",ascending=False).head(5)    


# In[40]:


categorical_variable_summary(df,'sentiment') 


# In[ ]:




