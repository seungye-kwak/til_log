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


### 3. 릿지 회귀 (Ridge Regression) = L2
![image](https://github.com/seungye-kwak/til_log/assets/112370282/a8d2fd8d-627b-49e8-a74a-9248bae3a020)  
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

# 각 alpha에 따른 회귀 계수 값을 시각화하기 위해 5개의 열로 된 맷플롯립 축 생성  
fig , axs = plt.subplots(figsize=(18,6) , nrows=1 , ncols=5)
# 각 alpha에 따른 회귀 계수 값을 데이터로 저장하기 위한 DataFrame 생성  
coeff_df = pd.DataFrame()

# alphas 리스트 값을 차례로 입력해 회귀 계수 값 시각화 및 데이터 저장. pos는 axis의 위치 지정
for pos , alpha in enumerate(alphas) :
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_data , y_target)
    # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가.  
    coeff = pd.Series(data=ridge.coef_ , index=X_data.columns )
    colname='alpha:'+str(alpha)
    coeff_df[colname] = coeff
    # 막대 그래프로 각 alpha 값에서의 회귀 계수를 시각화. 회귀 계수값이 높은 순으로 표현
    coeff = coeff.sort_values(ascending=False)
    axs[pos].set_title(colname)
    axs[pos].set_xlim(-3,6)
    sns.barplot(x=coeff.values , y=coeff.index, ax=axs[pos])

# for 문 바깥에서 맷플롯립의 show 호출 및 alpha에 따른 피처별 회귀 계수를 DataFrame으로 표시
plt.show()

ridge_alphas = [0 , 0.1 , 1 , 10 , 100]
sort_column = 'alpha:'+str(ridge_alphas[0])
coeff_df.sort_values(by=sort_column, ascending=False)
```

### 4. 라쏘 회귀 (Lasso Regression) = L1
![image](https://github.com/seungye-kwak/til_log/assets/112370282/fca8f5f6-e7d9-4eaa-92cd-10e8ba586b8e)
* L1 규제를 적용해 불필요한 회귀 계수를 급격하게 감소시켜 0으로 만들고 제거해 Feature Selection의 특성을 가짐
  
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# alpha값에 따른 회귀 모델의 폴드 평균 RMSE를 출력하고 회귀 계수값들을 DataFrame으로 반환 
def get_linear_reg_eval(model_name, params=None, X_data_n=None, y_target_n=None, 
                        verbose=True, return_coeff=True):
    coeff_df = pd.DataFrame()
    if verbose : print('####### ', model_name , '#######')
    for param in params:
        if model_name =='Ridge': model = Ridge(alpha=param)
        elif model_name =='Lasso': model = Lasso(alpha=param)
        elif model_name =='ElasticNet': model = ElasticNet(alpha=param, l1_ratio=0.7)
        neg_mse_scores = cross_val_score(model, X_data_n, 
                                             y_target_n, scoring="neg_mean_squared_error", cv = 5)
        avg_rmse = np.mean(np.sqrt(-1 * neg_mse_scores))
        print('alpha {0}일 때 5 폴드 세트의 평균 RMSE: {1:.3f} '.format(param, avg_rmse))
        # cross_val_score는 evaluation metric만 반환하므로 모델을 다시 학습하여 회귀 계수 추출
        
        model.fit(X_data_n , y_target_n)
        if return_coeff:
            # alpha에 따른 피처별 회귀 계수를 Series로 변환하고 이를 DataFrame의 컬럼으로 추가. 
            coeff = pd.Series(data=model.coef_ , index=X_data_n.columns )
            colname='alpha:'+str(param)
            coeff_df[colname] = coeff
    
    return coeff_df

lasso_alphas = [ 0.07, 0.1, 0.5, 1, 3]
coeff_lasso_df =get_linear_reg_eval('Lasso', params=lasso_alphas, X_data_n=X_data, y_target_n=y_target)

sort_column = 'alpha:'+str(lasso_alphas[0])
coeff_lasso_df.sort_values(by=sort_column, ascending=False)
```

### 5. 엘라스틱넷 회귀 (Elastic Net Regression)
![image](https://github.com/seungye-kwak/til_log/assets/112370282/87db4a1b-02f7-4358-99fe-9b80d4f92f6f)
* L2 와 L1 규제를 결합한 회귀. L1 과 마찬가지로 서로 상관관계가 높은 피처들의 경우 이들 중 중요한 Feature만 선택하고 다른 Feature의 회귀 계수를 모두 0으로 만드는 성향이 강함
* 이 때문에 L1에서는 alpha 값에 따라 회귀 계수가 급격히 변동하는데, Elastic Net은 이것을 완화하기 위해 Lasso에 L2 규제를 추가한 것
