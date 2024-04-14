로지스틱 회귀
=============

### 1. 로지스틱 회귀 개요
* 개념 : 선형 회귀 방식을 분류에 적용한 알고리즘.
* 선형 회귀 계열. 회귀가 선형인가 비선형인가는 독립변수가 아닌 가중치 변수가 선형인지 아닌지를 따르는데, 로지스틱 회귀와 선형 회귀와 다른 점은 학습을 통해 선형 함수의 회귀 최적선을 찾는 것이 아니라 __시그모이드 함수__ 최적선을 갖고 이 시그모이드 함수의 반환 값을 확률로 간주해 확률에 따라 분류를 결정한다는 것
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/99b8f0b6-0671-4a9c-8f06-c73014123848)

### 2. 코드
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
feature_scaled = scaler.fit_transform(cancer.data)

X_train, X_test, y_train, y_test = train_test_split(feature_scaled, cancer.target, test_size = 0.2, random_state = 12)

from sklearn.metrics import roc_auc_score

lrc = LogisticRegression() # sklearn 1.0.2 기준 solver='lbfgs' 기본값
lrc.fit(X_train, y_train)

pred = lrc.predict(X_test)

score = roc_auc_score(y_test, pred)

print('ROC AUC Score: ', round(score,4))

solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

for solver in solvers:
    lrc = LogisticRegression(solver = solver, max_iter = 500)
    lrc.fit(X_train, y_train)
    
    pred = lrc.predict(X_test)

    score = roc_auc_score(y_test, pred)

    print(solver, 'ROC AUC Score: ', round(score,10))

from sklearn.model_selection import GridSearchCV

params = {'solver': ['liblinear', 'lbfgs'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 5, 10]}

lrc = LogisticRegression()

lrc_grid = GridSearchCV(lrc, param_grid = params, scoring='roc_auc', cv = 5)

lrc_grid.fit(feature_scaled, cancer.target)

print('최적 하이퍼 파라미터:', lrc_grid.best_params_, '\nROC AUC Score:', round(lrc_grid.best_score_, 4))
```
