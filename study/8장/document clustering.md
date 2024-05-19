문서 군집화
===

## 1. 문서 군집화(Document clustering)
- 비슷한 텍스트 구성의 문서를 군집화하는 것
- 동일한 군집에 속하는 문서를 같은 카테고리 소속으로 분류할 수 있으므로 앞서 소개한 텍스트 분류 기반의 문서 분류와 유사함. 단, 텍스트 분류 기반의 문서 분류가 사전에 결정 카테고리 값을 가진 학습데이터가 필요한 것에 반해, 문서 군집화는 학습데이터가 필요 없는 비지도학습 기반으로 동작함

```python
import pandas as pd
import glob, os
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 700)

path = f'C:\Users\chkwon\Text\OpinosisDataset1/0\OpinosisDataset1.0\topics'
all_files = glob.glob(os.path.join(paht, "*.data"))
filename_list=[]
opinion_text=[]

for file_ in all_files:
    df = pd.read_table(file_, index_col=None, header=0, encoding='latin1')
    filename_ = "file_.split('\\')[-1]
    filename = filename_.split('.')[0]
    
    filename_list.append(filename)
    opinion_text.append(df.to_string())

document_df = pd.DataFrame({'filename':filename_list, 'opinion_text':opinion_text})
document_df.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='englisht', \
                            ngram_range=(1,2), min_df=0.05, max_df=0.85)
feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])

from sklearn.cluster import KMeans

km_cluster = KMeans(n_clusters=5, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_

document_df['cluster_label'] = cluster_label
document_df.head()
document_df[document_df['cluster_label']=0].sort_values(by='filename')

# 군집별 핵심 단어 추출하기

cluster_centers = km_cluster.cluster_centers_
print('cluster_centers shape :', cluster_centers.shape)
```
