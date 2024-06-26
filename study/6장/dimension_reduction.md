차원축소
==============

## 차원축소 개요
### 1. 차원축소 배경
- 수백 개 이상의 피처로 구성된 데이터 세트의 경우 상대적으로 적은 차원에서 학습된 모델보다 예측 신뢰도가 떨어짐
- 피처가 많을 경우 개별 피처 간에 상관관계가 높을 가능성이 큼. 선형 회귀와 같은 선형 모델에서는 입력 변수 간의 상관관계가 높을 경우 이로 인한 다중 공선성 문제로 모델의 예측 성능이 저하됨
- 매우 많은 다차원의 피처를 차원 축소해 피처 수를 줄이면 더 직관적으로 데이터를 해석할 수 있음

### 2. 차원축소 종류 및 특징
- 일반적으로 피처 선택(Feature Selection)과 피처 추출(Feature Extraction)로 나눌 수 있음.
- 피처 선택 : 특정 피처에 종속성이 강한 불필요한 피처는 아예 제거하고 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 것
- 피처 추출 : 기존 피처를 저차원의 중요 피처로 압축해서 추출하는 것. 새롭게 추출된 중요 특성은 기존의 피처가 압축된 것이므로 기존의 피처와는 완전히 다른 값이 됨. 기존 피처를 단순히 압축하는 것이 아닌, __피처를 함축적으로 더 잘 설명할 수 있는 또 다른 공간으로 매핑해 추출하는 것__. 예시로는 PCA, SVD, NMF 등이 있음
- 차원 축소 알고리즘은 매우 많은 픽셀로 이뤄진 이미지 데이터에서 잠재된 특성을 피처로 도출해 함축적 형태의 이미지 변환과 압축을 수행할 수 있음. 변환된 이미지는 원본 이미지보다 훨씬 적은 차원이기 때문에 과적합 영향력이 작아져서 오히려 원본데이터를 사용했을 때보다 예측 성능을 올릴 수 있음.
- 텍스트 문서의 숨겨진 의미를 추출하는 것에도 유용함. 많은 단어로 구성된 문서는 문서를 만드는 사람의 어떤 의도나 의미를 가지고 작성한 문서의 단어들을 차원 축소를 통해 숨겨진 시맨틱 의미나 토믹을 잠재요소로 간주하고 찾아낼 수 있음. SVD와 NMF가 시맨틱 토픽 모델링을 위한 기반 알고리즘으로 사용됨

## 차원축소 알고리즘
### 1-1. PCA (Principal Component Analysis)
- PCA : 여러 변수 간 존재하는 상관관계를 이용해 이를 대표하는 주성분(Principal Component)을 추출해 차원을 축소하는 기법. 기존 데이터의 정보 유실을 최소화하기 위해 __가장 높은 분산을 가지는 데이터의 축을 찾아 이 축으로 차원을 축소하는데, 이것이 PCA의 주성분이 됨(분산이 데이터의 특성을 가장 잘 나타내는 것으로 간주함)__
- 제일 먼저 가장 큰 데이터 변동성(Variance)을 기반으로 첫 번째 벡터 축을 생성하고, 두 번째 축은 이 벡터 축에 직각이 되는 벡터(직교 벡터)를 축으로 함. 세 번째 축은 다시 두 번째 축과 직각이 되는 벡터를 설정하는 방식으로 축을 생성함. 이렇게 생성된 벡터 축에 원본 데이터를 투영하면 벡터 축의 개수만큼 차원으로 원본 데이터가 차원 축소됨
- PCA(주성분 분석)은 이처럼 원본 데이터의 피처 개수에 비해 매우 작은 주성분으로 원본 데이터의 총 변동성을 대부분 설명할 수 있는 분석법
 
### 1-2. PCA 선형대수 관점
- 입력 데이터의 공분산 행렬(Covariance Matrix)을 고유값 분해하고, 이렇게 구한 고유벡터에 입력데이터를 선형 변환하는 것. 이 고유벡터가 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타냄. 고윳값(eigenvalue)은 이 고유벡터의 크기를 나타내며, 동시에 입력 데이터의 분산을 나타냄
- 일반적으로 선형 변환은 특정 벡터에 행렬 A를 곱해 새로운 벡터로 변환하는 것을 의미함. 특정 벡터를 하나의 공간에서 다른 공간으로 투영하는 개념으로도 볼 수 있으며, 이 경우 이 행렬을 바로 공간으로 가정하는 것.
- 보통 분산은 한 개의 특정한 변수의 데이터 변동을 의미하나, __공분산은 두 변수 간 변동을 의미함. 공분산(X,Y) > 0 이라는 말은 X가 증가할 때 Y도 증가한다는 의미__. 공분산 행렬은 여러 변수와 관련된 공분산을 포함하는 정방형 행렬.
- __고유벡터는 행렬 A를 곱하더라도 방향이 변하지 않고 그 크기만 변하는 벡터를 지칭함__. 고유벡터는 여러 개가 존재하며, 정방 행렬은 최대 그 차원 수만큼 고유벡터를 가질 수 있음. 이렇게 고유벡터는 행렬이 작용하는 힘의 방향과 관계가 있어서 행렬을 분해하는데 사용됨.

### 1-3. PCA 과정
1) 입력 데이터 세트의 공분산 행렬을 생성함
2) 공분산 행렬의 고유벡터와 고유값을 계산함
3) 고유값이 가장 큰 순으로 K개(PCA 변환 차수)만큼 고유벡터를 추출함
4) 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환함

 ### 2-1. LDA (Linear Discriminant Analysis)
 - LDA는 선형 판별 분석법으로 불리며 PCA와 매우 흡사함. 입력 데이터 세트를 저차원 공간에 투영해 차원을 축소하는 점에서 PCA와 유사하지만, 중요한 차이는 __LDA는 지도학습의 분류에서 사용하기 쉽도록 개별 클래스를 분별할 수 있는 기준을 최대한 유지하면서 차원을 축소함.__ PCA는 입력 데이터의 변동성이 가장 큰 축을 찾았지만, LDA는 입력 데이터의 결정 값 클래스를 최대한으로 분리할 수 있는 축을 찾음.
 - LDA는 특정 공간상에서 클래스 분리를 최대화하는 축을 찾기 위해 클래스 간 분산(between-class scatter)과 클래스 내부 분산(within-class scatter)의 비율을 최대화 하는 방식으로 차원을 축소함. 즉, 클래스 간 분산은 최대한 크게 가져가고, 클래스 내부의 분산은 최대한 작게 가져가는 방식.

### 2-2. LDA 과정
일반적으로 PCA와 유사하나 가장 큰 차이점은 __공분산 행렬이 아니라 위에 설명한 클래스 간 분산과 클래스 내부 분산 행렬을 생성한 뒤, 이 행렬에 기반해 고유벡터를 구하고 입력데이터를 투영한다는 점__
1) 클래스 내부와 클래스 간 분산 행렬을 구함. 이 두 개의 행렬은 입력 데이터의 결정 값 클래스별로 개별 피처의 평균 벡터(mean vector)를 기반으로 구함
2) 두 행렬을 고유벡터로 분해함
3) 고유값이 가장 큰 순으로 K개(LDA 변환 차수) 추출함
4) 고유값이 가장 큰 순으로 추출된 고유벡터를 이용해 새롭게 입력 데이터를 변환함

### 3. SVD (Singular Value Decomposition)
- SVD 역시 PCA와 유사한 행렬 분해 기법을 사용함. PCA의 경우 정방행렬(행과 열의 크기가 같은 행렬)만을 고유벡터로 분해할 수 있지만, SVD는 정방행렬뿐만 아니라 행과 열의 크기가 다른 행렬에도 적용할 수 있음.
- 일반적으로 SVD는 m x n 크기의 행렬 A를 아래와 같이 분해하는 것을 의미함  
![image](https://github.com/seungye-kwak/til_log/assets/112370282/b11a52b0-99cf-4010-b1a7-63799cb11d54)  
- SVD는 특이값 분해로 불리며, 행렬 U와 V에 속한 벡터는 특이벡터(singular vector)이며, 모든 특이 벡터는 서로 직교하는 성질을 가짐. Σ는 대각행렬이며, 행렬의 대각에 위치한 값만 0이 아니고 나머지 위치의 값은 모두 0임. Σ이 위치한 0이 아닌 값이 바로 행렬 A의 특이값. SVD는 A의 차원이 m x n일 때 U의 차원이 m x m, Σ의 차원이 m x n, VT의 차원이 n x n 으로 분해됨
- Turncated SVD는 Σ의 대각원소 중에 상위 몇 개만 추출해서 여기에 대응하는 U와 V의 원소도 함께 제거해 더욱 차원을 줄인 형태로 분해하는 것

### 4. NMF (Non-Negative Matrix Factorization)
- NMF는 Truncated SVD와 같이 낮은 랭크를 통한 행렬 근사(Low-Rank Approximation) 방식의 변형. NMF는 원본 행렬 내의 모든 원소 값이 모두 양수라는 게 보장되면 다음과 같이 좀 더 간단하게 두 개의 기반 양수 행렬로 분해될 수 있는 기법을 자칭함  
  ![image](https://github.com/seungye-kwak/til_log/assets/112370282/42a6d639-4a10-468b-bedb-73b47b5470c3)
- NMF는 SVD와 유사하게 차원 축소를 통한 잠재 요소 도출로 이미지 변환 및 압축, 텍스트의 토픽 도출 등의 영역에서 사용되고 있음.
