규제 선형 모델 - 릿지, 라쏘, 엘라스틱넷
========================================

### 1. 규제 선형 모델 개요
* 회귀 모델은 적절히 데이터에 적합하면서도 회귀 계수가 기하급수적으로 커지는 것을 제어할 수 있어야 함. 이전까지 선형 모델의 비용 함수는 RSS를 최소화하는, 즉 실제 값과 예측값의 차이를 최소화하는 것만 고려하다보니 학습데이터에 맞춰지고 회귀 계수가 쉽게 커지는 문제가 있었음 -> 과적합이 발생해서 테스트 데이터 세트에서는 예측 성능이 저하되기 쉬움
* 이를 반영해 비용 함수는 학습 데이터의 잔차 오류 값을 최소로 하는 RSS 최소화 방법과 과적합을 방지하기 위해 회귀 계수 값이 커지지 않도록 하는 방법이 서로 균형을 이뤄야 함
* 기존 비용 함수(RSS(W))에 가중치 규제식을 더해주는 것이 규제 선형 모델인데 이때 가중치 규제식의 값에 영향을 주는 튜닝 파라미터가 alpha
* alpha : 학습데이터 적합 정도와 회귀 계수 값의 크기 제어를 수행하는 튜닝 파라미터. alpah가 0 또는 매우 작은 값이라면 비용 함수 식은 기존과 동일한 Min(RSS(W) + 0) 이 됨. 반대로 alpha가 무한대(또는 매우 큰 값)라면 비용 함수 식은 RSS(W)에 비해 규제식의 값이 너무 커지게 되므로 W 값을 0(또는 매우 작게)으로 만들어야 Cost가 최소화되는 비용 함수 목표를 달성할 수 있음 = alpha 값을 크게 하면 비용함수는 회귀 계수의 W의 값을 작게 해 과적합을 개선할 수 있으며 alpha 값을 작게 하면 회귀 계수 W 값이 커져도 어느 정도 상쇄가 가능하므로 학습데이터의 적합을 더  개선할 수 있음
* alpha 값으로 페널티를 부여해 회귀 계수 값의 크기를 감소시켜 과적합을 개선하는 방식을 규제(Regularization)라고 부르며 규제는 크게 L2 Regularization 방식과 L1 Regularization 방식으로 구분됨


### 2. L1 norm, L2 norm, Lp norm
* L1과 L2는 모델의 손실함수에 각각 L1 Loss Function, L2 Loss Function을 추가해 준 것을 말함
* Norm : 유한 차원의 벡터 공간에서 벡터의 절대적인 크기(Magnitude)나 벡터 간 거리를 나타냄
* 측정 가능한 기능의 공간인 LP space 혹은 르베그 공간(Lebesgue Space)에서의 norm을 Lp norm(p-norm)이라고 하는데 Lp norm을 수식화 하면 다음과 같음
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/350c30eb-ec49-47d5-8f9f-f8d85a38f9ba)  
* 위 수식에서 n, p는 실수이며 p는 norm의 차수, n은 벡터의 차원수를 나타냄. p의 차수에 따라 Lp norm 은 아래와 같은 형태를 띄게 됨. p=1 이 L1, p=2가 L2임
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/e19f3aad-4a1e-46ba-a663-b28bcd109663)
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/211c7822-afec-4198-a589-2acf4d88e6e9)
* L1 norm 은 맨하탄(택시) 거리로 특정 방향으로만 움직일 수 있는 조건이 있는 경우, 두 벡터 간의 최단 거리를 찾는데 사용되는 방법.
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/c5e14bdf-dd66-4e2a-bec0-a635b9aebce9)
* L2 norm은 유클리드 거리라고 하며 두 점 사이의 최단 거리를 측정할 때 사용됨.
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/dc87d31b-5353-4919-bba4-57d974372259)
* L1 norm vs. L2 norm
  - L1 norm은 다른 점으로 이동하는데에 다양한 방법이 있는 반면에 L2 norm은 단 한 가지의 방법만 있음
  - L2 norm은 수식에서 오차의 제곱을 하기 때문에 outlier에 더 큰 영향을 받음
  - L1 norm은 이에 비해 다른 점으로 가는 다양한 방법(feature, weight) 중 특정 방법을 0으로 처리하는 것이 가능해 중요한 가중치만 남길 수 있는 feature selcetion이 가능하고, 오차의 절대값을 사용하기 때문에 L2 대비 outlier에 좀 더 robust 함. 하지만 0에서 미분이 불가능해 Gradient-Based Learning 시 사용에 주의가 필요하고, 편미분 시 weight의 부호만 남기 때문에 weight의 크기에 따라 규제의 크기가 변하지 않아 규제 효과가 L2에 비해 떨어짐
  - L2 norm은 오차의 제곱을 사용하기 때문에 outlier에 대해 L1보다 민감하게 작용함. 따라서 weight의 부호 뿐만 아니라 그 크기만큼 페널티를 줄 수 있어 특정 weight가 너무 커지는 것을 방지하는 weight decay가 가능해짐. 이러한 특징 때문에 L2 norm이 weight에 대한 규제에 더 효과적이고 일반적으로 학습 시 더 좋은 결과를 얻을 수 있어서 L1 보다 많이 사용됨
![image](https://github.com/seungye-kwak/til_log/assets/112370282/37c07c94-3c77-44ae-9af2-3e47dda1531b)


### 3. 릿지 회귀 (Ridge Regression)
![image](https://github.com/seungye-kwak/til_log/assets/112370282/a8d2fd8d-627b-49e8-a74a-9248bae3a020)  
* 
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
