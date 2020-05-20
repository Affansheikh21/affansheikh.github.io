---
title: "SMS Spam Detector using NLTK"
date: 2020-05-20
tags: [machine learning, spam detector, python]
categories: machinelearning
header:
  image: "/images/spam/spam_img.jpg"
excerpt: "SMS Spam Detector in Python"
mathjax: "true"
---













### Building an SMS Spam Detector on Python using NLTK Library

#### Imports and Configuration


```python
import nltk
import pandas as pd
```


```python
nltk.download_shell()
```

#### Reading the File

[Here is the Link to Data](https://github.com/Affansheikh21/Datasets/tree/master/smsspamcollection)

```python
messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
```


```python
print(len(messages))
```

    5574
    


```python
messages[0]
```




    'ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'



#### Exploring the messages


```python
for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')
```

#### Creating  a Dataframe


```python
messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',
                      names=['label','message'])
```


```python
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5572</td>
      <td>5572</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>5169</td>
    </tr>
    <tr>
      <th>top</th>
      <td>ham</td>
      <td>Sorry, I'll call later</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4825</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages.groupby('label').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">message</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4825</td>
      <td>4516</td>
      <td>Sorry, I'll call later</td>
      <td>30</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
      <td>653</td>
      <td>Please call our customer service representativ...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
messages['length'] = messages['message'].apply(len)
```


```python
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>message</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
messages['length'].plot.hist(bins=150)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2a1b99d6ac8>




![png](Spam_Detector_using_NLTK_files/Spam_Detector_using_NLTK_18_1.png)



```python
messages['length'].describe()
```




    count    5572.000000
    mean       80.489950
    std        59.942907
    min         2.000000
    25%        36.000000
    50%        62.000000
    75%       122.000000
    max       910.000000
    Name: length, dtype: float64



#### Outlier


```python
messages[messages['length']==910]['message'].iloc[0]
```




    "For me the love should start with attraction.i should feel that I need her every time around me.she should be the first thing which comes in my thoughts.I would start the day and end it with her.she should be there every time I dream.love will be then when my every breath has her name.my life should happen around her.my life will be named to her.I would cry for her.will give all my happiness and take all her sorrows.I will be ready to fight with anyone for her.I will be in love when I will be doing the craziest things for her.love will be when I don't have to proove anyone that my girl is the most beautiful lady on the whole planet.I will always be singing praises for her.love will be when I start up making chicken curry and end up makiing sambar.life will be the most beautiful then.will get every morning and thank god for the day because she is with me.I would like to say a lot..will tell later.."




```python
messages.hist(column='length',by='label',bins=60,figsize=(12,4))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x000002A1B9ADBE48>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x000002A1B9BC8208>],
          dtype=object)




![png](Spam_Detector_using_NLTK_files/Spam_Detector_using_NLTK_22_1.png)



```python
import string
```


```python
mess = 'Sample message! Notice: it has punctuation'
```


```python
string.punctuation
```




    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'




```python
#remove punctuation from the mess
```


```python
nopunc = [c for c in mess if c not in string.punctuation]
```


```python
from nltk.corpus import stopwords
```


```python
stopwords.words('English')
```


```python
nopunc = ''.join(nopunc) # remove spaces from punc removed str
```


```python
nopunc
```




    'Sample message Notice it has punctuation'




```python
nopunc.split()
```




    ['Sample', 'message', 'Notice', 'it', 'has', 'punctuation']




```python
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('English')]
```


```python
clean_mess
```




    ['Sample', 'message', 'Notice', 'punctuation']




```python
#function

def text_process(mess):
    """
    1- Remove punctuation
    2- Remove stop words
    3- Return list of clean words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('English')]
```


```python
messages['message'].head(5).apply(text_process)
```




    0    [Go, jurong, point, crazy, Available, bugis, n...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
    3        [U, dun, say, early, hor, U, c, already, say]
    4    [Nah, dont, think, goes, usf, lives, around, t...
    Name: message, dtype: object




```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])
```


```python
print(len(bow_transformer.vocabulary_))
```

    11425
    


```python
mess4 = messages['message'][3]
```


```python
print(mess4)
```

    U dun say so early hor... U c already then say...
    


```python
bow4 = bow_transformer.transform([mess4])
```


```python
print(bow4)
```

      (0, 4068)	2
      (0, 4629)	1
      (0, 5261)	1
      (0, 6204)	1
      (0, 6222)	1
      (0, 7186)	1
      (0, 9554)	2
    


```python
bow_transformer.get_feature_names()[9554]
```




    'say'




```python
messages_bow = bow_transformer.transform(messages['message'])
```


```python
print('Shape of Sparse Matrix: ',messages_bow.shape)
```

    Shape of Sparse Matrix:  (5572, 11425)
    


```python
messages_bow.nnz
```




    50548




```python
from sklearn.feature_extraction.text import TfidfTransformer
```


```python
tfidf_transformer = TfidfTransformer().fit(messages_bow)
```


```python
tfidf4 = tfidf_transformer.transform(bow4)
```


```python
print(tfidf4)
```

      (0, 9554)	0.5385626262927564
      (0, 7186)	0.4389365653379857
      (0, 6222)	0.3187216892949149
      (0, 6204)	0.29953799723697416
      (0, 5261)	0.29729957405868723
      (0, 4629)	0.26619801906087187
      (0, 4068)	0.40832589933384067
    


```python
tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
```




    8.527076498901426




```python
messages_tfidf = tfidf_transformer.transform(messages_bow)
```


```python
from sklearn.naive_bayes import MultinomialNB
```


```python
spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])
```


```python
spam_detect_model.predict(tfidf4)[0]
```




    'ham'




```python
messages['label'][3]
```




    'ham'




```python
all_pred = spam_detect_model.predict(messages_tfidf)
```


```python
from sklearn.model_selection import train_test_split
```


```python
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)
```

### Building up a Pipeline for Classification


```python
from sklearn.pipeline import Pipeline
```


```python
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])
```


```python
pipeline.fit(msg_train,label_train)
```




    Pipeline(memory=None,
         steps=[('bow', CountVectorizer(analyzer=<function text_process at 0x000002A1B9DE3950>,
            binary=False, decode_error='strict', dtype=<class 'numpy.int64'>,
            encoding='utf-8', input='content', lowercase=True, max_df=1.0,
            max_features=None, min_df=1, ngram_range=(1, 1), preprocesso...f=False, use_idf=True)), ('classifier', MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True))])



### Predictions and Evaluation


```python
predictions = pipeline.predict(msg_test)
```


```python
from sklearn.metrics import classification_report
```


```python
print(classification_report(label_test,predictions))
```

                  precision    recall  f1-score   support
    
             ham       0.96      1.00      0.98      1462
            spam       1.00      0.70      0.82       210
    
       micro avg       0.96      0.96      0.96      1672
       macro avg       0.98      0.85      0.90      1672
    weighted avg       0.96      0.96      0.96      1672
    
    


```python

```
