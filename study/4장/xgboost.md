XGBoost
=======

### XGBoost 장점
1. 뛰어난 예측 성능과 GBM 대비 빠른 수행시간
2. 과적합 규제
3. 나무 가지치기 (긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄임)
4. 자체 내장된 교차 검증
5. 결손값 자체 처리

### XGBoost 하이퍼 파라미터
* 일반 파라미터 : 일반적으로 실행 시 스레드의 개수나 silent 모드 등의 선택을 위한 파라미터로서 디폴트 파라미터 값을 바꾸는 경우는 거의 없음
  - booster : gbtree(tree based model, default) 또는 gblinear(linear model) 선택
  - silent : 0 (default) 출력 메시지를 나타내고 싶지 않을 경우 1로 설정
  - nthread : cpu의 실행 스레드 개수 조정 (default : 전체 스레드 모두 사용). 멀티 코어/스레드 cpu 시스템에서 전체 cpu를 사용하지 않고 일부 cpu만 사용해 ml 애플리케이션을 구동하는 경우에 변경함
    
* 부스터 파라미터 : 트리 최적화, 부스팅, regularization 등과 관련 파라미터 등을 지칭함
  - eta : gpm의 학습률과 같은 파라미터. 0에서 1 사이의 값을 지정하며 부스팅 스텝을 반복적으로 수행할 때 업데이트하는 학습률 값. 보통은 0.01 ~ 0.2 사이의 값을 선호함(default=0.3, alias: learning_rate)
  - num_boost_rounds : GBM의 n_estimators와 같은 파라미터
  - min_child_weight : 트리에서 추가적으로 가지를 나눌지를 결정하기 위해 필요한 데이터들의 weight 총합. 이 값이 클수록 분할을 자제하며, 과적합을 조절하기 위해 사용 (default=1)
  - gamma : 트리의 리프 노드를 추가적으로 나눌지를 결정할 최소 손실 감소 값. 해당 값보다 큰 손실이 감소된 경우에 리프 노드를 분리함. 값이 클수록 과적합 감소 효과 있음 (default=0, alias: min_split_loss)
  - max_depth: 트리 기반 알고리즘의 max_depth와 동일함. 0을 지정하면 깊이에 제한이 없음. 해당 값이 높으면 특정 피처 조건에 특화되어 룰 조건이 만들어지므로 과적합 가능성이 높아지며 보통은 3~10 사이의 값을 적용함 (default=6)
  - sub_sample : gbm의 sumsample과 동일함. 트리가 커져서 과적합되는 것을 제어하기 위해 데이터를 샘플링하는 비율을 지정함. 일반적으로 0.5 ~ 1 사이의 값을 사용함 (default=1)
  - colsample_bytree: gbm의 mas_feautres와 유사. 트리 생성에 필요한 피처(칼럼)를 임의로 샘플링하는 데 사용됨. 매우 많은 피처가 있는 경우 과적합을 조정하는 데 적용함 (default=1)
  - lambda : l2 regularization 적용 값. 피처 개수가 많을 경우 적용을 검토하며 값이 클수록 과적합 감소 효과 있음(defualt=1, alias: reg_lambda)
  - alpha : l1 regularization 적용 값. 피처 개수가 많을수록 적용을 검토하며 값이 클수록 과적합 감소 효과 있음 (default=0, alias: reg_alpha)
  - scale_pos_wiehgt : 특정한 값으로 치우친 비대칭한 클래스로 구성된 데이터 세트의 균형을 유지하기 위한 파라미터 (default=1)
    
* 학습 태스크 파라미터 : 학습 수행 시의 객체 함수, 평가를 위한 지표 등을 설정하는 파라미터
  - objective : 최소값을 가져야 할 손실 함수 정의. 주로 이진 분류인지 다중 분류인지에 따라 달라짐
  - binary:logistic : 이진분류일 때 적용
  - multi:softmax : 다중 분류일 때 적용
  - multi:softprob: multi:softmax와 유사하나 개별 레이블 클래스의 해당되는 예측 확률을 반환함
  - eval_metric : 검증에 사용되는 함수를 정의함. 회귀의 기본은 rmse, 분류일 경우 error
    + rmse, mae, logloss, error, merror, mlogloss, auc
