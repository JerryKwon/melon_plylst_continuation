# [Kakao Arena 3rd] Melon Playlist Continuation Submission Code

<h2 id="context"> :pushpin: 대회 개요 </h2>
https://arena.kakao.com/c/7

주어진 플레이리스트와 동반된 노래정보(Metadata, Mel-Spectogram)을 활용하여 플레이리스트를 구성하는 #태그와 #노래를 예측하라 

## :clipboard: 목차
<ol>
<li><a href="context">대회개요</a></li>
<li><a href="schedule">진행일정</a></li>
<li><a href="reference">참고자료</a></li>
<li><a href="execution">실행방법</a></li>
</ol>


## :calendar: 진행일정
2020년 4월 27일(월) ~ 2020년 7월 26(일) [90일]
 
※ 실제 참가 시작일: 2020년 5월 2일(토)

* 1주차(5/2~5/10): 추천시스템에 대한 이해
* 2주차(5/11~5/17): 추천시스템 구현 알고리즘에 대한 이해
* 3주차(5/18~5/24): Neural-Net 기반의 MF 알고리즘 구현 & EDA 진행 #1
* 4주차(5/25~5/31): Neural-Net 기반의 MF 알고리즘 구현 & EDA 진행 #2
* 5주차(6/1~6/7): Neural-Net 기반의 MF 알고리즘 구현 & EDA 진행 #3
* 6주차(6/8~6/14): ALS를 통한 MF 알고리즘 구현 & ICBF 알고리즘 구현 
* 7주차(6/15~6/21): ICBF K-Nearest Neighbors 알고리즘 구현
* 8주차(6/22~6/28): Recommendation System Top 20 이해
* 9주차(6/29~7/5): ICBF 알고리즘 구현 #2
* 10주차(7/6~7/12): ICBF 알고리즘 구현 #3, CBF 알고리즘 구현
* 11주차(7/13~7/19): Hybrid(CBF+ICBF) 알고리즘 구현 #1
* 12주차(7/20~7/26): Hybrid(CBF+ICBF) 알고리즘 구현 #2, 제출 github repository 구현

## :books: 참고자료

* 갈아먹는 추천 알고리즘 [1]~[6] <br/>
   https://yeomko.tistory.com/3?category=805638
* 파이썬 머신러닝 완벽가이드 [9장. 추천시스템] <br/>
   http://www.yes24.com/Product/Goods/69752484
* Recommendation System With Implicit Feedback <br/>
   http://sanghyukchun.github.io/95/
* MF Using Deep Neural Networks [NeuMF] <br/>
   1. https://dnddnjs.github.io/recomm/2019/08/15/neural_collaborative_filtering/
   2. https://itkmj.blogspot.com/2019/09/neural-collaborative-filtering.html
   3. https://medium.com/@victorkohler/collaborative-filtering-using-deep-neural-networks-in-tensorflow-96e5d41a39a1
   4. https://towardsdatascience.com/neural-collaborative-filtering-96cef1009401
* Recommendation Top 20 <br/>
    https://www.facebook.com/groups/2611614312273351/permalink/2639027142865401/
* Understading the Mel-Spectogram <br/> 
    https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53
* Introduction to TWO approaches of Content-based Recommendation System <br/>
    1. https://towardsdatascience.com/introduction-to-two-approaches-of-content-based-recommendation-system-fc797460c18c
    2. https://github.com/youonf/recommendation_system/tree/master/content_based_filtering

### 1. 대회에 대한 이해 :grey_question:
 - 대회가 원하는 결과 알고리즘은?
	: 개요, 규칙, 데이터에 대한 개괄적인 이해
	(https://dacon.io/competitions/official/229255/overview/)
	
 - 분류 vs 회귀?
	: 지도학습의 대표적인 두가지 목표에 대한 이해

 - 회귀 측정지표에 대한 이해
	: 회귀 측정지표 종류, 지표별 사용 이유, 직접 만드는 나만의 측정지표
	
### 2. 탐색적 데이터 분석(EDA)
 - 탐색적 데이터 분석?
 
 - EDA에서 고려하는 것들
	1. 자료의 형태 이해
		: 범주형 - (명목형,순서형), 수치형 - (이산형,연속형)
		
	2. null값과 이상치

	3. 데이터 시각화
		* matplotlib.pyplot
		* seaborn
		* **plotly**

 - EDA를 활용한 퇴근시간 승차인원 데이터 셋 분석
 
### 3. Feature	 Engineering
 - 스케일링, 정규화
 - One-Hot Encoding
 - New feature creation
 
### 4. 분석모델 모델링
 - 머신러닝 기법
	1. 선형 모델
		: 릿지, 라쏘, 로지스틱 회귀
	2. 트리 모델
		: 결정 트리, 랜덤 포레스트
	3. Boosting 모델의 이해
	    * Bagging vs Boosting
		* AdaBoost
		* XgBoost
		* LightGBM
		
 - 교차검증(Cross vaildation)
 
### 5. 심화
 - Feature Engineering 심화
	1. 상관관계
	2. 이상치 처리 심화
		* Multiple Imputation
		* Regression Imputation
	3. 차원 축소
	4. high categorical variable 처리 방법
		* Mean Encoding
		* Frequency Encoding
		
 - 순위권 코드 심화
	: 스터디 진행 상황에 따라서 1,2,3등 코드 중 선정 및 클론코딩 진행
	
### 6. 고도화
- Interpretable Machine Learning(lime)
  : 머신러닝/딥러닝 모델의 신뢰도에 대한 해석
  (http://research.sualab.com/introduction/2019/08/30/interpretable-machine-learning-overview-1.html)
- Bayesian optimization
  : Model의 Hyperparameter를 튜닝하는데 사용
- SHAP
  : 결과를 예측하는데 있어 Column의 영향도를 Insight

