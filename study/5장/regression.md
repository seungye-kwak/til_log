회귀
===

### 1. 회귀란?
* 회귀 분석 : 데이터 값이 평균과 같은 일정한 값으로 돌아가려는 경향을 이용한 통계학 기법. 회귀는 통계학에서 __여러 개의 독립변수와 한 개의 종속변수 간의 상관관계를 모델링__ 하는 기법을 통칭함.
* Y = W1*X1 + W2*X2 + ... Wn*Xn, 이 때 Y가 종속변수, X가 독립변수, W1...Wn은 독립변수의 값에 영향을 미치는 회귀 계수(Regression coefficients)를 의미함. 머신러닝 관점에서는 독립변수는 피쳐, 종속변수는 결정 값에 해당됨
* 머신러닝 회귀 예측의 핵심 : 주어진 피처와 결정 값 데이터 기반에서 학습을 통해 __최적의 회귀 계수__ 를 찾아내는 것.
* 독립 변수의 개수에 따라 단일 회귀와 다중 회귀로, 회귀 계수가 선형이냐 아니냐에 따라 선형 회귀와 비선형 회귀로 분류됨. 여러 가지 회귀 중 선형 회귀가 가장 많이 사용됨


### 2. 대표적인 선형 회귀 모델.
* 일반 선형 회귀 : 예측값과 실제 값의 RSS(Residual Sum of Squeres)를 최소화할 수 있도록 회귀 계수를 최적화 하며 규제(Regularization)를 적용하지 않은 모델
* 릿지(Ridge) : 선형 회귀에 L2 규제를 추가한 회귀 모델. L2 규제는 상대적으로 큰 회귀 계수 값의 예측 영향도를 감소시키기 위해서 회귀 계수값을 더 작게 만드는 규제 모델
* 라쏘(Lasso) : 선형 회귀에 L1 규제를 적용한 방식. L2 규제가 회귀 계수 값의 크기를 줄이는 데 반해, L1 규제는 예측 영향력이 작은 피처의 회귀 계수를 0으로 만들어 회귀 예측 시 피처가 선택되지 않게 하는 것.
* 엘라스틱넷(ElasticNet) : L2, L1 규제를 함께 결합한 모델. 주로 피처가 많은 데이터 세트에서 사용되며, L1 규제로 피처의 개수를 줄임과 동시에 L2 규제로 계수 값의 크기를 조정함
* 로지스틱 회귀(Logistic Regression) : 회귀란 이름이지만 사실 분류에 사용되는 선형 모델.

### (추가자료) L1 과 L2 규제 (https://seongyun-dev.tistory.com/52)
* L1 - L2 비교식  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/390900fe-23d1-4d92-88fd-bdff6d464354)

* L1 규제 : 기존 Cost Funtion에 가중치의 절대값의 합을 더하는 형태로 미분 시 weight의 크기에 상관 없이 부호에 따라 일정한 상수값을 더하거나 빼주게 됨. 따라서 특정 weight들을 0으로 만들 수 있어 원하는 weight만 남길 수 있는 feature selection 역할을 함. 아래 그림을 보면, L1 규제를 적용한 Cost Function의 편미분 결과인데 가중치 부호에 따라 가중치 값에 상수만큼 더하거나 뺄 수 있다. 
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/510d8038-f1e9-466f-848f-92839ee0c9e6)

* L2 규제 : 기존 Cost Function에 가중치 제곱의 합을 더하는 형태로, weight 크기에 따라 weight 값이 큰 값을 더 빠르게 감소시키는 weight decay 기법. weight 크기에 따라 가중치 패널티 정도가 달라지기 때문에 가중치가 전반적으로 작아져서 학습 효과가 L1 대비 더 좋게 나옴. 람다 값에 따라 패널티의 정도를 조절할 수 있음  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/0d1cb7a5-8a01-45b3-9d8e-c04cb77e9933)

* L1과 L2 규제에 대한 Loss Function의 최적해 위치. 아래 Loss Function의 Space에서 L1, L2의 규제 영역을 두어 실제 최적값에 대한 bias에 손해를 보더라도 variance를 낮춰 Overfitting 발생을 낮추는 것. L1, L2 Loss에서 람다 값이 커질수록 아래의 규제 영역 크기가 작아지게 되어 bias는 더 커지고 variance는 줄어들게 (underfitting 가능성이 커짐)되며 L1, L2 규제를 추가한 Loss Function의 최적값은 규제 영역 내에서 Global Optimum과 제일 가까운 지점이라 볼 수 있음. 아래 그림에서 L1의 경우 세타1의 값이 0인 것을 볼 수 있는데, 이를 통해 특정 변수(Feature)를 삭제할 수 있다는 것을 알 수 있음  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/3a2755d9-1d5f-4fe1-91f3-75ecfbbf83b6)

* L1 - L2 비교 정리  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/dc367100-cbd2-4926-bc58-b8a50b752ef3)


### 회귀의 이해
* 잔차(Residual) : 실제 값과 회귀 모델의 차이에 따른 오류 값. 최적의 회귀 모델을 만든다는 것은 바로 전체 데이터의 잔차 합이 최소가 되는 모델을 만든다는 의미. 동시에 오류 값 합이 최소가 될 수 있는 최적의 회귀 계수를 찾는다는 의미가 됨
* 오류 값은 +나 -가 될 수 있기 때문에 단순히 합을 해서는 안됨.
* 오류 합을 계산할 때는 보통 절댓값을 취해서 더하거나(MAE) 오류 값의 제곱을 구해서 더하는 방식(RSS, Residual Sum of Sqaure)을 취함. 일반적으로 미분 계산을 편리하게 하기 위해 RSS 방식으로 오류 합을 구함 (Error**2 = RSS)
* 평가 지표 (오류 합산 방법)
  - MAE (Mean Absolute Error) : 실제 값과 예측값의 차이를 절댓값으로 변환해 평균한 것  
    ![image](https://github.com/seungye-kwak/til_log/assets/112370282/ad5fea18-d71e-4a79-b92b-63329d10920e)

  - MSE (Mean Squared Error) : 실제 값과 예측값의 차이를 제곱해 평균한 것
    ![image](https://github.com/seungye-kwak/til_log/assets/112370282/9d184100-7341-44e6-bd1b-bacbb1fa524d)

  - RMSE(Root Mean Sqaured Error) : 오류의 제곱을 구하므로 실제 오류 평균보다 더 커지는 MSE에 루트를 씌운 것
    ![image](https://github.com/seungye-kwak/til_log/assets/112370282/9e20881d-dcfa-49a6-b1b6-264036609414)

  - R**2 : 분산 기반으로 예측 성능을 평가함. 실제 값의 분산 대비 예측값의 분산 비율을 지표로 하며, 1에 가까울수록 예측 정확도가 높음
    ![image](https://github.com/seungye-kwak/til_log/assets/112370282/e137bcac-f2e7-4753-89d0-77e77bdc1593)

  - 이 밖에 MSE나 RMSE에 로그를 적용한 MSLE(Mean Sqaured Log Error)와 RMSLE(Root Mean Sqaured Log Error)도 사용함

* 사이킷런 평가 지표 API
  |평가방법|사이킷런 평가 지표 API|Scoring 함수 적용 값|
  |:--:|:--:|:--:|
  |MAE|metrics.mean_absolute_error|'neg_mean_absolute_error'|
  |MSE|metrics.mean_sqaured_error|'neg_mean_sqared_error'|
  |RMSE|metrics.mean_sqaured_error를 그대로 사용하되 sqaured 파라미터를 False로 설정|'neg_root_mean_sqared_error'|
  |MSLE|metrics.mean_sqared_log_error|'neg_mean_sqaured_log_error|
  |R**2|metrics.r2_score|'r2'|

  + 사이킷런의 Scoring 함수가 score 값이 클수록 좋은 평가 결과로 자동으로 평가하는데, 사실 실제 값과 예측 값의 오류 차이를 기반으로 하는 회귀 평가 지표의 경우 값이 커지면 나쁜 모델이기 때문에 이를 사이킷런의 Scoring 함수에 일반적으로 반영하려면 보정이 필요함 -> -1 을 원래의 평가 지표 값에 곱해서 음수(Negative)를 만들어 작은 오류 값이 더 큰 숫자로 인식하게 만듦. 이를 위해 MAE 등의 Scoring 함수 적용 값 앞에 'neg_'라는 접두어가 붙는 것.




