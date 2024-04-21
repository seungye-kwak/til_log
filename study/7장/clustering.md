군집화
======

### 1. K-means Clustering
- 군집화에서 가장 일반적으로 사용되는 알고리즘. 군집 중심 점(centroid)이라는 특정한 임의의 지점을 선택해 해당 중심에 가장 가까운 포인트들을 선택하는 군집화 기법
- 군집 중심점은 선택된 포인트의 평균 지점으로 이동하고 이동된 중심점에서 다시 가까운 포인트를 선택, 다시 중심점을 평균 지점으로 이동하는 프로세스를 반복적으로 수행함. 모든 데이터 포인트에서 더이상 중심점의 이동이 없을 경우에 반복을 멈추고 해당 중심점에 속하는 데이터 포인트들을 군집화하는 기법  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/ff2ebf14-bcbf-455f-96f2-0aa1eb1f4f4d)  
  1) 군집화의 기준이 되는 중심을 구성하려는 군집화 개수만큼 임의의 위치에 가져다 놓음 (임의로 두면 이동 수행이 너무 많아져 초기화 알고리즘으로 적합한 위치에 중심점을 가져놔야 함)
  2) 각 데이터는 가장 가까운 곳에 위치한 중심점에 소속됨.
  3) 이렇게 소속이 결정되면 군집 중심점을 소속된 데이터의 평균 중심으로 이동
  4) 중심점을 이동했기 때문에 각 데이터는 기존에 속한 중심점보다 더 가까운 중심점이 잇다면 해당 중심점으로 다시 소속 변경.
  5) 다시 중심을 소속된 데이터의 평균 중심으로 이동함.
  6) 중심점을 이동했는데 데이터의 중심점 소속 변경이 없으면 군집화 종료. 그렇지 않다면 4번 과정 거쳐서 소속을 변경하고 이 과정을 반복함
- 장점 : 일반적으로 군집화에서 가장 많이 활용되는 알고리즘으로, 쉽고 간결함
- 단점 : 거리 기반 알고리즘으로 속성의 개수가 매우 많을 경우 군집화 정확도가 떨어짐(이를 위해 PCA로 차원 감소를 적용해야 할 수 있음). 반복을 수행하는데 횟수가 많으면 수행 시간이 매우 느려지고, 몇 개의 군집(cluster)을 선택해야 할지 기준이 없음


### 1-2. 사이킷런 KMeans 클래스 소개
```python
class sklearn.cluster.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001,
                             precompute_distances='quto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
```
- Kmeans 초기화 파라미터 중 가장 중요한 파라미터는 n_clusters이며, 이는 군집화할 개수, 즉 군집 중심점의 개수를 의미함
- init은 초기에 군집 중심점의 좌표를 설정할 방식을 말하며 보통은 임의로 중심을 설정하지 않고 일반적으로 k-means++ 방식으로 최초 설정함
- max_iter는 최대 반복 횟수이며, 이 횟수 이전에 모든 데이터의 중심점 이동이 없으면 종료함.
- fit 또는 fit_transform 메서드를 이용해 수행. 수행된 KMeans 객체는 군집화 수행이 완료돼 군집화와 관련된 주요 속성을 알 수 있음
  + labels_ : 각 데이터 포인트가 속한 군집 중심점 레이블
  + cluster_centers_: 각 군집 중심점 좌표(Shape는 [군집 개수, 피쳐 개수]). 이를 이용하면 군집 중심점 좌표가 어디인지 시각화할 수 있음
- 군집화 알고리즘 테스트를 위한 데이터 생성
  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.cluster import KMeans
  from sklearn.datasets import make_blobs
  %matplotlib inline

  X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
  unique, counts = np.unique(y, return_counts=True)

  clusterDF = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
  clusterDF['target'] = y

  traget_list = np.unique(y)

  # 각 타깃별 산점도의 마커 값
  markers=['o', 's', '^', 'P', 'D', 'H', 'x']
  for target in target_list:
      target_cluster = clusterDF[dlusterDF['target']=target]
      plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'], edgecolor='k', marker=markers[target])
  plt.show()

  # KMeans 객체를 이용해 X 데이터를 K-Means 클러스터링 수행
  kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, random_state=0)
  cluster_labels = kmans.fit_predict(X)
  clusterDF['kmeans_label'] = cluster_labels

  centers = kmeans.cluster_centers_
  unique_labels=np.unique(cluster_labels)

  # 군집된 label 유형별로 iteration 하면서 marker 별로 scatter plot 수행
  for label in unique_labels:
      label_cluster = clusterDF[clusterDF['kmeans_label']==label]
      center_x_y = cnters[label]
      plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], edgecolor='k', marker=markers[label])

      # 군집별 중심 위치 좌표 시각화
      plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='white',
                  alpha=0.9, edgecolor='k', marker=markers[label])
      plt.scatter(x=center_x_y[0], y=center_x_y[1], x=70, color='k', edgecolor='k', marker='$%d#' % label)

  plt.show()
  ```


### 2-1. 군집 평가(Cluster Evaluation)
- 대부분의 군집화 데이터는 비교할 만한 타깃 레이블을 갖고 있지 않음
- 분류와 유사해 보일 수 있으나 성격이 많이 다름. 데이터 내에 숨어 있는 별도의 그룹을 찾아서 의미를 부여하거나 동일한 분류 값에 속하더라도 그 안에서 더 세분화된 군집화를 추구하거나, 서로 다른 분류 값의 데이터도 더 넓은 군집화 레벨화 등의 영역을 가지고 있음
- 군집화가 효율적으로 잘 됐는지 평가할 수 있는 지표로 __실루엣 분석__ 을 이용함

### 2-2. 실루엣 분석(silhouette analysis)
- 각 군집 간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타냄
- 효율적 분리 : 다른 군집과의 거리는 떨어져 있고 동일 군집끼리의 데이터는 서로 가깝게 잘 뭉쳐있다는 의미. 군집화가 잘 될수록 개별 군집은 비슷한 정도의 여유공간을 가지고 떨어져 있을 것
- 실루엣 분석은 실루엣 계수(silhouette coefficient, 개별 데이터가 가지는 군집화 지표)를 기반으로 함. 개별 데이터가 가지는 실루엣 계수는 해당 데이터가 같은 군집 내의 데이터와 얼마나 가깝게 군집화돼 있고, 다른 군집에 있는 데이터와는 얼마나 분리돼 있는지를 나타내는 지표
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/cee96faf-e650-4f80-813d-38c70c355035)
  + (max(a(i), b(i)) 로 나눠주는 것은 정규화를 하기 위함
  + i는 데이터 포인트를 의미함
  + 1에 가까워질수록 근처의 군집과 더 멀리 떨어져 있다는 것, 0에 가까울수록 근처의 군집과 가까워진다는 것. - 값은 아예 다른 군집에 데이터 포인트가 할당됐음을 의미함
- 좋은 군집화의 조건
  1) 전체 실루엣 계수의 평균값(사이킷런의 silhouette_score() 값)은 0~1 사이의 값을 가지며 1에 가까울수록 좋음
  2) 하지만 전체 실루엣 계수의 평균값과 더불어 개별 군집의 평균값의 편차가 크지 않아야 함. 즉, 개별 군집의 실루엣 계수 평균값이 전체 실루엣 계수의 평균값에서 크게 벗어나지 않는 것이 중요함. 만약 전체 실루엣 계수의 평균값은 높지만, 특정 실루엣 계수 평균값만 유난히 높고 다른 군집들의 실루엣 계수 평균값이 낮으면 좋은 군집화가 아님

```python
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
# 실루엣 분석 평가 지표 값을 구하기 위한 API 추가
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline

iris = load_iris()
feature_names = ['sepal_length','sepal_width','petal_length','petal_width']
irisDF = pd.DataFrame(data=iris.data, columns=feature_names)

# 군집화 수행
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300,random_state=0)
kmeans.fit(irisDF)

# 개별 데이터에 대한 군집 결과를 cluster 컬럼으로 DataFrame에 저장
irisDF['cluster'] = kmeans.labels_

irisDF.head(10)

# iris 의 모든 개별 데이터에 실루엣 계수값을 구함
score_samples = silhouette_samples(iris.data, irisDF['cluster'])
print('silhouette_samples( ) return 값의 shape' , score_samples.shape)

# irisDF에 실루엣 계수 컬럼 추가
irisDF['silhouette_coeff'] = score_samples

# 보통 0.5가 넘으면 꽤 좋은 수치로 봄
print(np.mean(score_samples))
print(silhouette_score(iris.data, irisDF['cluster']))
```

### 2-3. 시각화를 통한 군집 개수 최적화
- 전체 데이터의 평균 실루엣 계수 값이 높다고 해서 반드시 최적의 군집 개수로 군집화가 잘 됐다고 볼 수 없음. 특정 군집 내의 실루엣 계수 값만 너무 높고, 다른 군집은 내부 데이터끼리의 거리가 너무 떨어져 있어 실루엣 계수 값이 낮아져도 평균적으로 높은 값을 가질 수 있음
- 개별 군집별로 적당히 분리된 거리를 유지하면서도 군집 내의 데이터가 서로 뭉쳐 있는 경우에 K-평균의 적절한 군집 개수가 설정됐다고 판단할 수 있음

```python
def visualize_silhouette(cluster_lists, X_features): 
    
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import math
    
    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)
    
    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성 
    fig, axs = plt.subplots(figsize=(4*n_cols, 4), nrows=1, ncols=n_cols)
    
    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, n_cluster in enumerate(cluster_lists):
        
        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산. 
        clusterer = KMeans(n_clusters = n_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)
        
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)
        
        y_lower = 10
        axs[ind].set_title('Number of Cluster : '+ str(n_cluster)+'\n' \
                          'Silhouette Score :' + str(round(sil_avg,3)) )
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (n_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        
        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현. 
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels==i]
            ith_cluster_sil_values.sort()
            
            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = cm.nipy_spectral(float(i) / n_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                                facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
            
        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

# make_blobs 을 통해 clustering 을 위한 4개의 클러스터 중심의 500개 2차원 데이터 셋 생성  
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=500, n_features=2, centers=4, cluster_std=1, \
                  center_box=(-10.0, 10.0), shuffle=True, random_state=1)  

# cluster 개수를 2개, 3개, 4개, 5개 일때의 클러스터별 실루엣 계수 평균값을 시각화 
visualize_silhouette([ 2, 3, 4, 5], X)
```

![image](https://github.com/seungye-kwak/til_log/assets/112370282/2d23e667-3c9a-4226-8038-5e8cee5a3935)  
+ 전체 실루엣 계수만 보면 n=2 클러스터링이 좋아보이지만, 보통 n=4 클러스터링처럼 군집 간 편차가 적은 것이 좋다.
