# Recommendation System

추천시스템은 대표적으로
1. 콘텐츠 기반 필터링 (Contents Based Filtering)
2. 협업 필터링 방식 (Collaborative Filterting)

  1) 최근접 이웃 협업 필터링
    * 사용자 기반 필터링
    * 아이템 기반 필터링

  2) 잠재 요인 협업 필터링 -- Netflix 추천 시스템 경연대회에서 우승 알고리즘

## 1. 콘텐츠 기반 필터링

콘텐츠 기반 필터링은 사용자가 특정한 아이템을 매우 선호하는 경우 그 아이템과 비슷한 콘텐츠를 가진 다른 아이템을 추천하는 방식

    e.g) 사용자가 특정 영화에 높은 평점을 줬다면, 그 영화의 장르, 출연 배우, 감독, 영화 키워드 등의 콘텐츠와 유사한 다른 영화를 추천해주는 방식

<img src ="https://drive.google.com/uc?id=11l2dRPCBnMReec1CrkIvyjw5A2ah8SZi" width=600 alt="recommend_system_1" />

## 2. 협업 필터링 방식

### 1) 최근접 이웃 협업 필터링
누군가에게 물어보는 방식과 유사한 방식으로, 사용자가 아이템에 매긴 평점 정보나 상품 구매 이력과 같은 사용자 행동 양식 만으로 추천을 수행하는 방식.

<img src ="https://drive.google.com/uc?id=15xb5bzK9g8ShoHTJGGr3MEp4ncfK9yyy" width=600 alt="recommend_system_2" />


<img src ="https://drive.google.com/uc?id=1-TlCnYJOGnVFk8lbj-lyI739Py0RCYp2" width=600 alt="recommend_system_3" />

협업 필터링의 주된 목표는 축적된 사용자 행동 데이터를 기반으로 사용자가 평가하지 않은 아이템을 예측 평가하는 것이다.

#### **1)-1 사용자 기반 필터링**
사용자 기반 최근접이웃 필터링 방식은 특정 사용자와 유사한 다른 사용자를 Top-N으로 선정해 Top-N 사용자가 좋아하는 아이템을 추천하는 방식이다.

<img src ="https://drive.google.com/uc?id=1Jbrzw2li4uESMt61mlDWO4EzSW2TEtg2" width=600 alt="recommend_system_4" />

#### **1)-2 아이템 기반 필터링**
아이템이 가지는 속성과는 상관 없이 사용자들이 그 아이템을 좋아하는지/싫어하는지의 평가가 유사한 아이템을 추천하는 알고리즘. (**아이템 간의 속성이 얼마나 비슷한지를 기반으로 추천하는 것이 아니다.**)

<img src ="https://drive.google.com/uc?id=1f57dqyfnrMUoJHvKykdJrEmlaxsg1kxk" width=600 alt="recommend_system_5" />

저자 曰, 일반적으로 사용자 기반보다는 아이템 기반 협업 필터링의 정확도가 높다. 이유는 비슷한 영화를 좋아한다고 해서 사람들의 취향이 비슷하다고 판단하기는 어려운 경우가 많기 때문이다. 그리고 매우 유명한 영화는 취향과 관계없이 대부분의 사람이 관람하는 경우가 많고, 사용자들이 평점을 매긴 영화(또는 상품)의 개수가 많지 않은 경우가 일반적인데, 이를 기반으로 다른 사람과의 유사도를 비교하기가 어려운 부분도 있다.

### 2) 잠재 요인 협업 필터링
대규모의 다차원 행렬을 SVD, NMF, ALS, SGD와 같은 차원 감소 기법으로 분해하는 과정에서 잠재요인을 추출하여 협업 필터링을 수행하는 얼고리즘. 말 그대로 '잠재요인'을 끄집어내는 알고리즘. 

잠재요인을 기반으로 다차원 희소행렬 사용자-아이템 행렬 데이터를 저차원 밀집 행렬의 사용자-잠재행렬과 아이템-잠재 요인 행렬의 전치 행렬로 분해할 수 있으며, 이 분해된 두 행렬의 내적을 통해 새로운 예측 사용자-아이템 평점 행렬 데이터를 만들어 평점을 부여하지 않은 아이템에 대한 예측 평점을 생성하는 것

<img src ="https://drive.google.com/uc?id=1EPFCCJ3i1d-wyRIm2L_Y1BE_TZb-B7T4" width=600 alt="recommend_system_6" />


<img src ="https://drive.google.com/uc?id=1eBOaCF8Ie0ghP2LoBOD1bVY1MwNb6LQq" width=600 alt="recommend_system_7" />


<img src ="https://drive.google.com/uc?id=1Bya2nksOFJ1qYBrj6JaN9eFM11jYNnhB" width=600 alt="recommend_system_8" />


<img src ="https://drive.google.com/uc?id=1mSASzdGNx_goEgm5HjjIbyhbjn2vldOc" width=600 alt="recommend_system_9" />


#### **어떻게 행렬을 분해하는 것일까?**

실제로 행렬을 분해 시작시에는 P와 Q의 행렬에 랜덤한 값을 투입한다. 이렇게 얻은 랜덤한 값을 가지고 오차를 계산한 후 값을 계속 최적화 하는 방식을 채택한다.

* 행렬 분해 수행 순서
  1. P와 Q를 임의의 값을 가진 행렬로 설정
  2. P와 Q.T 값을 곱해 예측 R 행렬을 계산하고 예측 R 행렬과 실제 R 행렬에 해당하는 오류값을 계산한다.
  3. 이 오류 값을 최소화 할 수 있도록 P와 Q행렬을 적절한 값으로 각각 업데이트한다.
  4. 만족할 만한 오류 값을 가질때까지 2,3번 작업을 반복하면서 P와 Q 값을 업데이트하여 근사화한다.

* 실제 값과 예측값의 오류 최소화 및 L2 규제를 고려한 비용 함수식
<img src ="https://drive.google.com/uc?id=1U7USnJN7CTNRLiYALu7qHTMdZExAAoSw" width=600 alt="recommend_system_10" />

* 새롭게 업데이트 되는 $\hat{P}u$와 $\hat{Q}i$의 값 계산 식
<img src ="https://drive.google.com/uc?id=1WmuSIjcNtjKp58Y2nq6wjeSDB_iwybLD" width=600 alt="recommend_system_11" />
