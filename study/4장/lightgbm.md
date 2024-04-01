LightGBM
=========

### LightGBM 등장 배경 및 개요
- Xgboost가 여전히 학습 시간이 오래 걸림 (GridSearchCV로 하이퍼 파라미터 튜닝을 하기 때문에) - GBM보다는 빠르지만 대용량 데이터의 경우 만족할 만한 학습 성능을 기대하려면 많은 CPU 코어를 가진 시스템에서 높은 병렬도로 학습을 진행해야 함
- LightGBM은 적은 데이터 세트(약 10,000건 이하)를 사용할 경우 과적합이 발생하기 쉽다는 단점은 있지만, 그 외에는 XGBoost와 비슷한 성능을 보이면서 훨씬 빠르게 학습을 할 수 있는 장점이 있음
- 원핫인코딩을 하지 않고 Category Type의 변수를 자동 변환해서 학습함 : 트리 모델에서는 범주의 개수가 많은 변수를 원핫인코딩해서 넣을 경우 트리가 언밸런스 해지고 좋은 성능을 내기 위해 더 깊어지면서 훈련 시간도 더 소요되고 과적합할 위험이 높은데 LightGBM은 그렇지 않음
- 일반 GBM 계열의 트리 분할 방법과 다르게 __리프 중심 트리 분할(Leaf Wise)__ 방식을 사용함. 따라서 __최대한 균형 잡힌 트리를 유지하면서 분할하기 때문에 트리의 깊이가 최소화 될 수 있음__
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/0f1a89c2-41dd-4752-b468-cd57b795c124)
  * 리프 중심 트리 분할 : 최대 손실 값을 가지는 리프노드를 지속적으로 분할하면서 트리가 깊어지고 비대칭적으로 생성하며 예측 오류 손실을 최소화함. 트리의 어느 레벨에서 모든 노드를 확장시키는 것이 아닌 최종 노드 하나만 훈할하는 방식을 사용. loss가 가장 큰 부분을 쪼개고 쪼개서 최대한으로 줄이려고 함. 다른 노드들을 분할시키지 않고 오직 residual이 큰 노드만 쪼개서 메모리의 절약과 속도를 향상시킬 수 있음
- 장점 요약 : 빠른 속도, 리소스 절약, GPI 학습 지원, 정확도에 초점을 맞춤

### LightGBM 사용 알고리즘
* GOSS(Gradient-based One-Side Sampling) : 데이터셋의 샘플 (instance) 수를 줄이는 알고리즘.
* EFB(Exclusive Feature Bundling) : 데이터셋의 Feature 수를 줄이는 알고리즘. 고차원 데이터는 sparse 하다는 가정 하에 상호배타적인 변수들을 하나의 buckent으로 묶어서 Feature 수를 줄이는 방식. 여기서 '상호배타적인 변수들'은 동시에 0이 아닌 값을 갖지 않는 변수들.


### LightGBM 하이퍼 파라미터
- 주요 파라미터
  * num_iterations : 반복 수행하려는 트리의 개수를 지정함. 크게 지정할수록 예측 성능이 높아질 수 있으나, 너무 크게 지정하면 오히려 과적합으로 성능이 저하될 수 있음 (default=100, lightgbm scikit-learn 호환 클래스에서는 n_estimators로 이름이 변경됨)
  * learning_rate : 0~1 사이 값을 지정하며 부스팅 스탭을 반복적으로 수행할 때 업데이트 되는 학습률 값. 일반적으로 n_estimators를 크게하고 learning_rate를 작게 해서 예측 성능을 향상시킬 수 있으나 마찬가지로 과적합 이슈와 학습 시간이 길어지는 단점을 고려해야 함 (default=0.1)
  * max_depth : 0보다 작은 값을 지정하면 깊이에 제한이 없음. LightGBM은 Leaf wise 기반이므로 깊이가 상대적으로 더 깊음 (default=-1)
  * min_data_in_leaf : 결정 트리의 min_samples_leaf와 같은 파라미터. 최종 결정 클래스인 리프 노드가 되기 위해서 필요한 레코드 수이며, 과적합을 제어하기 위한 파라미터 (default=20, 사이킷런 래퍼 lightgbmclassifier에서는 min_child_samples 파라미터로 이름 변경)
  * num_leaves : 하나의 트리가 가질 수 있는 최대 리프 개수 (default=31)
  * boosting : 부스팅의 트리를 생성하는 알고리즘 기술 (default=gbdt(일반적인 그래디언트 부스팅 결정 트리), rf(랜덤포레스트)
  * bagging_fraction : 트리가 커져서 과적합되는 것을 제어하기 위해서 데이터를 샘플링하는 비율을 지정 (LightGBMClassifier에서는 subsample로 동일하게 파라미터 이름이 변경됨)
  * feature_fraction : 개별 트리를 학습할 때마다 무작위로 선택하는 피처의 비율 (default=1.0, LightGBMClassifier에서는 동일하게 colsample_bytree로 변경됨)
  * lambda_l2 : L2 regulation 제어를 위한 값. (default=0.0, LightGBMClassifier에서는 reg_lambda로 변경됨)
  * lambda_l1 : L2 regulation 제어를 위한 값 (default=0.0, LightGBMClassifier에서는 reg_alpha로 변경됨)

- Learning Task 파라미터
  * objective: 최소값을 가져야 할 손실함수 정의

- 하이퍼 파라미터 튜닝 방안
  * num_leaves는 개별 트리가 가질 수 있는 최대 리프의 개수로 LightGBM 모델의 복잡도를 제어하는 주요 파라미터. 일반적으로 num_leaves의 개수를 높이면 정확도가 높아지지만, 반대로 트리의 깊이가 깊어지고 모델이 복잡도가 커져 과적합 영향도가 커짐
  * min_data_in_leaf(min_child_samples)는 과적합 개선에 중요한 파라미터로 보통 큰 값으로 설정하면 트리가 깊어지는 것을 방지함
  * max_depth는 명시적으로 깊이의 크기를 제한함
  * learning_rate를 작게 하면서 n_estimators를 크게 하는 것이 부스팅 계열 튜닝에서 가장 기본적인 튜닝 방안. 하지만 n_estimators를 너무 크게 하면 과적합이 될 수 있음
 
### 코드 구현
```Python
from lightgbm import LGBMClassifier

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

lgbm_wrapper = LGBMClassifier(n_estimators=400, learning_rate=0.05)
evals = [(X_tr, y_tr), (X_val, y_val)]
lgbm_wrapper.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric='logloss', eval_set=evals, verbose=True)
preds = lgbm_wrapper.predict(X_test)
pred_proba = lgbm_wrapper.predict_proba(X_test)[:,1]
```


