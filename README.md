# [Kakao Arena 3rd] Melon Playlist Continuation Submission Code

<h2 id="context"> :pushpin: 대회 개요 </h2>
https://arena.kakao.com/c/7

주어진 **플레이리스트와 동반된 노래정보(Metadata, Mel-Spectogram)을 활용**하여,<br/>
**"플레이리스트를 구성하는 #태그와 #노래를 예측하라"**

## :clipboard: 목차
<ol>
<li><a href="#context">대회개요</a></li>
<li><a href="#schedule">진행일정</a></li>
<li><a href="#reference">참고자료</a></li>
<li><a href="#execution">실행</a></li>
<li><a href="#review">대회후기</a></li>
</ol>


<h2 id="schedule"> :calendar: 진행일정</h2>
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

<h2 id="reference"> :books: 참고자료 </h2>

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

<h2 id="execution"> :exclamation: 실행 </h2>

<h3><b>※주의</b></h3>
해당 프로젝트는 Windows, Linux 환경 모두에서 실행될 수 있도록 만들어졌으나, 시간 상의 이유로 Windows만 구동 테스트를 완료하였습니다.
<b>따라서, Windows 환경에서 실행해야 합니다.</b>

<h3>구현 알고리즘</h3>

1. 아이템 기반 협업 필터링(Item-based Collaborative Filtering - icbf)
2. Hybrid 필터링 방식 (Hybrid Filtering [ICBF+CBF+Reminder] - hybrid)

<h3>사용 패키지</h3>

* Python3 내장 패키지<br/>
argparse, warnings, json, io, platform, os, collections<br/>
* Python3 외장 패키지
    - 데이터 조작 - pandas,numpy,sklearn,scipy <br/>
    - 진행상황 모니터링 - tqdm <br/>
    - plylst_title 불용어 추출 - nltk,selenium <br/>


<h3>실행 전 준비사항</h3>
<ol>
<li>실행을 위한 사용 패키지 설치</li>
<li>/data 디렉터리에 사용되는 input data 적재(train.json, test.json, val.json, song_meta.json, genre_gn_all.json)</li>
<li>/webdriver 디렉터리에 한글 불용어 Crwaling을 위한 실행환경 버전에 맞는 chromedriver 설치 <br/> chrome webdriver 버전 확인 - https://codechacha.com/ko/selenium-chromedriver-version-error/</li>
</ol>

<h3>실행법</h3>

* inference.py <br/>
    1. 설명 <br/>
    주어진 데이터를 활용하여 예측을 수행하는 python 파일
    2. 실행 <br/>
    python inference.py --model_type hybrid --is_valid True
    3. 옵션
        * --model_type - [icbf | hybrid] <br/>
        : 예측을 수행할 모델의 타입을 결정하는 파라미터. <br/>
            * icbf: 아이템 기반 협업 필터링 방식을 통한 Recommendation 진행
            * hybrid: 아이템 기반 협업 필터링 + 컨텐츠기반 필터링 + 예외처리 방식을 통한 Recommendation 진행
        * --is_valid - boolean [True | False] <br/>
        : 예측을 수행할 데이터 타입을 결정하는 파라미터 <br/>
            * True: /data/val.json을 대상으로 하여 Recommendation 진행
            * False: /data/test.json을 대상으로 하여 Recommendation 진행

* train.py <br/>
  **※ 주의** <br/>
    **해당 프로젝트에서는 pytorch, tensorflow, keras등의 학습 모델 생성을 통한 예측을 진행하지 않기 때문에, inference.py 와 동일함.**
   
    1. 설명 <br/>
    주어진 데이터를 활용하여 모델을 학습하는 python 파일 <br/>
    
    2. 실행 <br/>
    python train.py --model_type hybrid --is_valid True
    3. 옵션
        * --model_type - [icbf | hybrid] <br/>
        : 예측을 수행할 모델의 타입을 결정하는 파라미터. <br/>
            * icbf: 아이템 기반 협업 필터링 방식을 통한 Recommendation 진행
            * hybrid: 아이템 기반 협업 필터링 + 컨텐츠기반 필터링 + 예외처리 방식을 통한 Recommendation 진행
        * --is_valid - boolean [True | False] <br/>
        : 예측을 수행할 데이터 타입을 결정하는 파라미터 <br/>
            * True: /data/val.json을 대상으로 하여 Recommendation 진행
            * False: /data/test.json을 대상으로 하여 Recommendation 진행
            
<h3>결과 파일</h3>

**/result 아래, '[valid | test]_[hybrid | icbf]_rcomm_result.json' 형태로 결과값이 반환됨.**

<h3>예측 결과</h3>

* icbf <br/>
  valid LB - song - 0.159576 / tag - 0.340179 = 0.18666645 [61st in leaderboard]

* hybrid <br/>
  valid LB - song - 0.160008 / tag - 0.411810 = 0.197778 [55th in leaderboard]
            
<h2 id="review"> :checkered_flag: 대회후기</h2>

**추후 작성 예정**