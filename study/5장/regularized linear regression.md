규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷
========================================

### 1. 규제 선형 모델 개요
* 회귀 모델은 적절히 데이터에 적합하면서도 회귀 계수가 기하급수적으로 커지는 것을 제어할 수 있어야 함. 이전까지 선형 모델의 비용 함수는 RSS를 최소화하는, 즉 실제 값과 예측값의 차이를 최소화하는 것만 고려하다보니 학습데이터에 맞춰지고 회귀 계수가 쉽게 커지는 문제가 있었음 -> 과적합이 발생해서 테스트 데이터 세트에서는 예측 성능이 저하되기 쉬움
* 이를 반영해 비용 함수는 학습 데이터의 잔차 오류 값을 최소로 하는 RSS 최소화 방법과 과적합을 방지하기 위해 회귀 계수 값이 커지지 않도록 하는 방법이 서로 균형을 이뤄야 함


### 2. 릿지 회귀 (Ridge Regression)
![image](https://github.com/seungye-kwak/til_log/assets/112370282/a8d2fd8d-627b-49e8-a74a-9248bae3a020)  
* alpha : 학습데이터 적합 정도와 회귀 계수 값의 크기 제어를 수행하는 튜닝 파라미터. alpah가 0 또는 매우 작은 값이라면 비용 함수 식은 기존과 동일한 Min(RSS(W) + 0) 이 됨. 반대로 alpha가 무한대(또는 매우 큰 값)라면 비용 함수 식은 RSS(W)에 비해 규제식의 값이 너무 커지게 되므로 W 값을 0(또는 매우 작게)으로 만들어야 Cost가 최소화되는 비용 함수 목표를 달성할 수 있음 = alpha 값을 크게 하면 비용함수는 회귀 계수의 W의 값을 작게 해 과적합을 개선할 수 있으며 alpha 값을 작게 하면 회귀 계수 W 값이 커져도 어느 정도 상쇄가 가능하므로 학습데이터의 적합을 더  개선할 수 있음
* __alpha 값으로 페널티를 부여해 회귀 계수 값의 크기를 감소시켜 과적합을 개선하는 방식을 규제(Regularization)__라고 부르며 규제는 크게 L2 방식과 L1 방식으로 구분됨
* L2 규제 방식을 사용한 회귀가 릿지 회귀 (Ridge)
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# alpha=10으로 설정해 릿지 회귀 수행.
ridge=Ridge(alpha=10)
neg_mse_scores = cross_val_score(ridge, X_data, y_target, scoring="neg_mean_sqaured_error", cv=5)
rmse_scores = np.sqrt(-1 * neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 Folds 의 개별 Negative MSE scores: ', np.round(neg_mse_scores, 3))
print(' 5 Folds의 개별 RMSE scores  : ', np.round(rmse_scores, 3))
print(' 5 Folds 의 평균 RMSE : {0:.3f} '.format(avg_rmse))

# alpha 값의 변화에 따른 5 폴드의 RMSE 평균값 반환하는 코드
alpahs = [0, 0.1, 1, 10, 100]

for alpha in slphas :
  ridge = Ridge(alpah=alpha)

  # cross_val_scores를 이용해 5 폴드의 평균 RMSE 를 계산
  neg_mse_scores = cross_val_score(redge, X_data, y_target, scoring="neg_mean_sqared_error", cv=5)
  avg_rmse = np.mean(np.sqrt(-1*neg_mese_scores))
  print('alpha {0} 일 때 5 folds의 평균 RMSE : {1:.3f} '.format(alpha, avg_rmse))
```
