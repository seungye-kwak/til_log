랜덤 포레스트
=============

### 1. 랜덤포레스트란?
* 배깅(Bagging) : 같은 알고리즘으로 여러 개의 분류기를 만들어서 보팅으로 최종 결정하는 알고리즘으로, 대표적인 알고리즘이 랜덤 포레스트임
* 랜덤포레스트 특징
  - 앙상블 알고리즘 중 비교적 빠른 수행 속도를 가지고 다양한 영역에서 높은 예측 성능을 보임
  - 결정 트리의 쉽고 직관적인 장점을 가지고 있음
  - 여러 개의 결정 트리 분류기가 전체 데이터에서 배깅 방식으로 각자의 데이터를 샘플링해 개별적으로 학습을 수행한 뒤 최종적으로 모든 분류기가 보팅을 통해 예측 결정함
  - 개별 트리가 학습하는 데이터 세트는 전체 데이터에서 일부가 중첩되게 샘플링된 부트스트래핑(bootstrapping) 분할 방식을 사용한 데이터 세트임
    + 부트스트랩 : 통계학에서 여러 개의 작은 데이터 세트를 임의로 만들어 개별 평균의 분포도를 측정하는 목적을 위한 샘플링 방식.
  - 랜덤 포레스트의 서브세트 데이터는 부트스트래핑으로 데이터가 임의로 만들어지는데, 이 때 서브세트의 데이터 건수는 전체 데이터 건수와 동일함. 다만, 개별 데이터가 중첩되어 만들어짐.
    
![image](https://github.com/seungye-kwak/til_log/assets/112370282/3f4ea4b5-7131-4352-99a8-7427fe29a96c)


### 2. 랜덤 포레스트 하이퍼 파라미터 튜닝
* 트리 기반의 자체의 하이퍼 파라미터가 원래 많다 보니 트리 기반 앙상블 알고리즘의 경우 하이퍼 파라미터가 너무 많고 그래서 튜닝 시간이 많이 소모된다는 점이 단점.
* 그나마 랜덤 포레스트는 결정 트리에서 사용되는 하이퍼 파라미터와 같은 파라미터가 대부분이라 그나마 적은 편
* 하이퍼 파라미터 종류
  - n_estimators : 랜덤 포레스트에서 결정 트리의 개수를 지정함. 많이 설정할수록 좋은 성능을 기대할 수 있지만 계속 증가시킨다고 성능이 무조건 향상되는 것은 아님. 늘릴수록 학습 수행 시간이 오래 걸림 (default=10)
  - max_features : 결정 트리에 사용된 max_features 파라미터와 같음. 하지만 RandomForestClassifier의 기본 max_features는 'None'이 아니라 'auto'='sqrt'. 따라서 랜덤 포레스트의 트리를 분할하는 피처를 참조할 때 전체 피처가 아니라 sqrt(전체 피처 개수)만큼 참조함 (ex. 16개 피처 중 4개 참조)
  - max_depth, min_samples_leat, min_samples_split와 같이 결정 트리에서 과적합을 개선하기 위해 사용되는 파라미터가 랜덤포레스트에도 똑같이 적용됨

### 3. 실습 코드
* 랜덤포레스트 수행
```python
from sklearn.model_selection import GridSearchCV

params = {
  'max_depth': [8, 16, 24],
  'min_samples_leaf' : [1, 6, 12],
  'min_samples_split' : [2, 8, 16]
}
# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1) # 멀티 코어 환경에서 RandomForestClassifier와 GridSearchCV 모두에 n_jobs=-1 파라미터를 추가하면 모든 CPU 코어를 이용해 학습할 수 있음
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2, n_jobs=-1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
```

* 시각화
```python
import matplitlib.pyplot as plt
import seaborn as sns
%matplotlib inline

ftr_importances_values = rf_clf.feature_importances_
ftr_importances = pd.Series(ftr_importance_values, index=X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature importances Top 20')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
```
