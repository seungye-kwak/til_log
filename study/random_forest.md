앙상블 학습(Ensemble Learning)
===========================
1. 개념 : 여러 개의 분류기를 생성하고 그 예측을 결합해 보다 정확한 최종 예측을 도출하는 방법
2. 유형 : 보팅(Voting), 배깅(Bagging), 부스팅(Boosting), 스태깅 등
* 보팅 - 배깅 : 여러 개의 분류기가 투표를 통해 최종 예측 결과를 결정
  * 보팅 : 일반적으로 서로 다른 알고리즘을 가진 분류기를 결합하는 것
    * 하드 보팅 : 다수결 원칙으로 다수의 분류기가 예측한 값을 최종 보팅 결과값으로 선정
    * 소프트보팅 : 각 분류기의 레이블 값 예측 확률을 평균 내서 최종 결정함. 일반적으로 하드보팅보다는 소프트보팅이 더 성능이 좋음
  * 배깅 : 각각의 분류기가 모두 같은 유형의 알고리즘 기반이지만 데이터 샘플링을 서로 다르게 가져가면서 학습을 수행해 보팅을 수행하는 것 (ex. 랜덤 포레스트)
  <img width="510" alt="보팅_배깅" src="https://github.com/seungye-kwak/til_log/assets/112370282/dd6d1923-2888-4404-9d7f-9bd167db1ad5">

* * *
GBM(Gradient Boosting Machine)
==============================
1. 부스팅 알고리즘 : 여러 개의 약한 학습기를 **순차적으로** 학습, 예측하면서 잘못 예측한 데이터에 가중치 부여를 통해 오류를 개선하면서 학습하는 방식
2. 에이다부스트(AdaBoost)
  <img width="694" alt="adaboost" src="https://github.com/seungye-kwak/til_log/assets/112370282/bca3ca11-ad54-423b-b49e-c1fede3f6f69">
  
3. GBM(Gradient Boost Machine) : 에이다부스트와 유사하나, 가중치 업데이트를 경사하강법을 이용함. 오류 값은 실제값 - 예측값이고 이 오류값을 최소화하는 방향성을 가지고 반복적으로 가중치를 업데이트하는 것
 
 ```python
 from sklearn.ensemble import GradientBoostingClassifier
 import time
 import warnings
 warnings.filterwarnings('ignore')
 
 X_train, X_test, y_train, y_test = get_human_dataset()
 
 # GBM 수행 시간 측정을 위함. 시작 시간 설정.
 start_time = time.time()
 
 gb_clf = GradientBoostingClassifier(random_state=0)
 gb_clf.fit(X_train, y_train)
 gb_pred = gb_clf.predict(X_test)
 gb_accuracy = accuracy_score(y_test, gb_pred)
 
 print('GBM 정확도 : {0:.4f}'.format(gb_accuracy))
 print('GBM 수행 시간 : {0:.1f} 초 '.format(time.time() - start_time))
 ```
 * GBM 하이퍼 파라미터
   * loss : 경사하강법에서 사용할 비용 함수 지정 (default = 'deviance')
   * learning_rate : GBM이 학습을 진행할 때마다 적용하는 학습률로 0~1 사이의 값. 너무 작은 값을 입력하면 예측 성능은 높아지지만 시간이 너무 오래 걸리거나 모든 반복이 끝나도 최소 오류 값을 찾지 못할 수 있음. 반대로 큰 값을 적용하면 최소 오류 값을 찾지 못하고 그냥 지나쳐 버려 예측 성능이 떨어질 수 있지만 빠른 수행이 가능함. 이런 특성 때문에 learning_rate는 **n_estimators**와 상호 보완적으로 조합해 사용함 (default=0.1)
   * n_estimators : weak learner의 개수로 많을수록 예측 성능이 일정 수준까지는 좋아질 수 있지만 수행 시간이 오래 걸림 (default = 100)
   * subsample : weak learner가 학습에 사용하는 데이터 샘플링 비율. 기본 값은 1로 전체 학습 데이터를 기반으로 학습한다는 의미임. 과적합이 염려되는 경우에는 1보다 낮은 값으로 설정 (defualt=1)

 * GBM 특징 : 과적합에도 강한 예측 성능을 가지지만 수행 시간이 오래 걸림



  

