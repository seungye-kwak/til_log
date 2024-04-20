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
  import matplotlib.pyplot as plt
  from sklearn.cluster import KMeans
  from sklearn.datasets import make_blobs
  %matplotlib inline

  X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)
  ```
