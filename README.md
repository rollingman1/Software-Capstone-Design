# 2021-2 Software-Capstone-Design
# 뇌파(EEG)를 이용한 딥러닝 기반 감정 분류 시스템 SW 개발

## 1. 개요
### 1.1. 과제 선정 배경
- (EEG) 뇌파 는 뇌의 전기적인 활동을 머리 표면에 부착한 전극에 의해 측정한 전기신호로써 측정, 대상자의 심신 상태에 따라 다르게 나타나며 뇌의 활동상황을 측정하는 중요한 지표이다 이러한, 뇌파를 활용한 다양한 연구가 진행되고 있다. 루게릭병이나 파킨슨병과 같은 뇌 질환에 관련된 연구, 신체를 물리적으로 움직이지 않고 동작을 단지 상상하는 정신적인 작업인 Motor Imagery 연구, 집중력 측정이나 감정인식과 같은 대상의 상태분석 연구 등이 있다.
- 감정은 논리적인 의사결정, 인지, 지능과 관련돼 있어 일상생활과 건강에 밀접한 관련이 있다. EEG신호를 통해 정확한 감정분석이 가능해지면 불안, 분노와 같이 인간의 면역체계를 방해하고 질병의 위험을 증가시키는 감정들을 탐지해내고 개선해냄으로써 삶의 전반적인 질을 향상시키는데 기여할 수 있다.
- 표정, 음성, 동작 등을 통한 감정 분석과 달리 뇌파는 본인의 의지로 억누르거나 숨길 수 없기 때문에 뇌파를 이용한 감정 분석은 보다 객관적인 분석 결과를 얻을 수 있다.

### 1.2. 과제 주요내용
- EEG 1D signal data를 Continuous wavelet transform(CWT)를 통해 2D scalogram으로 변환 후, 이미지화하여 Time과 Frequency 모두에 걸쳐 표현된 data 획득
- Convolutional Neural Network(CNN)을 통해 감정 분류에 적절한 feature 자동 추출 
    - Parallel 2D CNN을 통한 feature extraction (all-channel) 
    - 일반적인 2D CNN을 통한 feature extraction (single-channel)
- 감정을 Arousal과 Valence를 기준으로 각각 binary classification 수행
    - Arousal Binary Classification
    - Valence Binary Classification
### 1.3. 최종결과물의 목표
- CWT와 CNN을 사용한  70% accuracy 이상의 감정 분류 모델 성능

## 2. 과제 수행방법
### 2.1. AMIGOS dataset으로 부터 EEG signal data 수집
- AMIGOS dataset : http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/readme.html
- AMIGOS dataset에서 16개의 short video를 시청하며 측정된 EEG signal data 수집
- 한 사람당 하나의 video를 시청하며 14개의 채널에 대하여 EEG signal 측정됨
- sampling frequency : 128Hz
- 4-45Hz bandpass filter가 적용되어 해당 주파수 영역 범위에 대하여 필터링 된 signal
- self assessment에 참가자가 스스로 느낀 감정을 총 12개의 label로 기록 [arousal, valence, dominance, liking, familiarity, neural, happiness, sadness, surprise, fear, anger, disgust]

### 2.2. Data cleaning
- EEG signal data에서 결측 치 제거
- 다시 4-45Hz의 signal 필터링 수행
- 흥분의 정도를 나타내는 Arousal과 긍정과 부정의 정도를 나타내는 Valence 두 개의 축으로 대부분의 감정을 설명할 수 있기에 Arousal과 Valence를 제외한 self assessment에 기록된 나머지 감정들 삭제


<img src = "https://user-images.githubusercontent.com/80897270/146937313-7ec2e95e-d928-4172-a33e-ed8d478f2d79.png" width ="300" height = "300"/>


- Arousal과 Valence에 대하여 각각 Binary classification 할 수 있도록 label 정리
    - 1~9사이 값이 할당된 Arousal 값을 5를 기준으로 High Arousal / Low Arousal로 relabeling
    - 1~9사이 값이 할당된 Valence 값을 5를 기준으로 High Valence / Low Valence로 relabeling

### 2.3. Continuous Wavelet Transform(CWT) 수행
- Wavelet Transform
    - cos, sin함수가 아닌 Wavalet 기저함수로 신호분해
    - scale에 따라 wavelet 함수의 주파수와 국소 시간 조정
    - 낮은 주파수에선 넓은 window, 높은 주파수에선 좁은 window

<img src = "https://user-images.githubusercontent.com/80897270/146953714-87270cdb-6451-4a24-be93-b22a94783409.png" width ="500" height = "150"/>


- Morlet Wavelt fucntion을 사용한 Coninuous Wavelt Transform을 통해 기존의 1D EEG signal data를 Time-Frequency의 정보를 모두 갖도록 wavelet coefficient 값을 갖는 2차원 형태의 scalogram으로 변환
    
<img src = "https://user-images.githubusercontent.com/80897270/146952542-c0b49db9-81a7-4c14-ab58-12b5663e658e.png" width ="400" height = "150"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Morelt wavelet]


### 2.4. Scalogram 이미지화
- 2차원 형태의 scalogram을 2D 이미지로 변환
- 국소 부위만 색깔이 드러나 이미지 간 차별적인 특징을 찾기 어려움
- log를 취해 scaling 해줌으로써 큰 wavelet coefficient값에 치우쳐져 색깔이 mapping 되지 않도록 함
- jet colormap으로 이미지화 

<img src = "https://user-images.githubusercontent.com/80897270/146943188-6e2af096-35f8-4155-bf32-d3ad7675c2ec.PNG"/>


### 2.5. Convolutional Neural Network(CNN)을 통한 feature extraction
- Parallel 2D CNN (all-channel)
    - ImageNet으로 pre-trained inceptionV3 model을 평행하게 14개 놓은 구조
    - 14개의 inceptionV3로부터 추출된 각각의 feature를 concatenation후, full-connected layer를 거쳐 classification 수행
    - 한 사람당 하나의 video를 보며 기록된 14개 EEG channel에 대한 14개의 EEG image를 평행하게 동시에 input으로 사용
    - input data수는 40(참가자 수) x 16(비디오 수) - 26(결측 치) = 614개

<img src = "https://user-images.githubusercontent.com/80897270/146942840-39ddf32a-1255-4beb-ad77-3025a306da0d.png" width ="500" height = "300"/>

- 일반적인 2D CNN (single-channel)
    - ImageNet으로 pre-trained 하나의 inceptionV3 사용
    - inceptionV3로부터 추출된 feature는 fully-conneted layer를 거쳐 classification 진행
    - EEG channel 14개에 대한 14개의 이미지들을 하나의 input으로 동시에 사용하지 않고, channel당 생성되는 하나의 image를 하나의 input으로 사용
    - input data수는 614 x 14(channel 수) = 8596개

<img src = "https://user-images.githubusercontent.com/80897270/146942871-e749ac4e-b2c5-465a-bbb3-89d9b11933b1.png" width ="500" height = "300"/>

### 2.6. Classifier
- Fully-connected layer와 CNN으로부터 전달받은 feature를 연결하여 classification 수행
- 다양한 Fully-connected layer 수와 Node 수를 바꿔가며 분류 성능이 좋은 classifier search
- 1층의 Fully-connected layer와 1024개의 Node로 설정




### 2.7. 모델 Training
- batch size, learning rate, optimizer, drop out 등 모델의 accuracy 성능을 높이는 hyper parameter를 찾아가며 training 진행

|Hyperparameter|value|
|-------|-----|
|Fully-connected layer|1층|
|Node|1024개|
|Drop-out|0.5|
|Batch size|16|
|learing rate|0.0001|
|Optimizer|Adam|


## 3. 수행결과
### 3.1.과제수행 결과
- **Arousal Classification**
    - parallel 2D CNN의 validation accuracy는 0.68, 일반적인 2D CNN의 validation accuracy는 0.68로 성능이 유사하지만, test accuracy는 일반적인 2D CNN에서 0.64로 parallel 2D CNN 보다 더 월등한 성능을 보임

###### <Parallel 2D CNN Arousal train & validation accuracy>

<img src = "https://user-images.githubusercontent.com/80897270/146941729-ceeedd35-646b-49c4-9bf3-b24907a327b1.png" width ="300" height = "300"/>


###### <2D CNN Arousal train & validation accuracy>

<img src = "https://user-images.githubusercontent.com/80897270/146941737-7c0189a6-db73-4c51-95d8-1f68e97756db.png" width ="300" height = "300"/>



- **Valence Classification**
    - parallel 2D CNN의 validation accuracy는 0.65, 일반적인 2D CNN은 0.7로 일반적인 2D CNN이 0.05 더 높았으며, test accuracy에서는 2D CNN에서 0.67을 달성하며 parallel 2D CNN보다 더 좋은 성능을 보임

###### <Parallel 2D CNN Valence train & validation accuracy>

<img src = "https://user-images.githubusercontent.com/80897270/146941988-897a04c6-8b16-4cf5-9be7-eda506568a52.png" width ="300" height = "300"/>


###### <2D CNN Valence train & validation accuracy>

<img src = "https://user-images.githubusercontent.com/80897270/146941992-0d7cefc5-8183-4bd7-a553-9a5922cc3fa0.png" width ="300" height = "300"/>





### 3.2. 최종결과물 주요특징 및 설명
- parallel 2D CNN과 일반적인 2D CNN의 Test accuracy와 Validation accuracy 사이의 성능차이를 살펴보면 일반적인 CNN은 parallel 2D CNN에 비해 크지 않음을 확인할 수 있다. 즉, parallel 2D CNN보다 일반적인 2D CNN에서 generalization이 더 잘 이루어졌다고 생각할 수 있다. 이는 parallel 2D CNN에서는 14개의 EEG 2D image를 동시에 하나의 input으로 사용하기 때문에 2D CNN에 비해 데이터 수가 14배 더 적게 되며 이에 대한 영향이 미쳤다고 생각할 수 있다.
- parallel 2D CNN의 성능이 저조한 또 다른 이유는 14개의 EEG channel 별 차이가 크지 않아, 14개의 EEG channel signal을 하나의 input으로 사용하는 것이 오히려 성능을 방해하는 요소로 작용 했을 수 있다. 실제 근접한 channel에 의해 기록된 EEG signal을 사용하여 2D image로 변환해 본 결과, 유사한 형태를 보임을 확인할 수 있었다.


<img src = "https://user-images.githubusercontent.com/80897270/146939747-bfc42b82-c88f-4303-a6e6-c43e306d2fbd.png" width ="300" height = "300"/> <img src = "https://user-images.githubusercontent.com/80897270/146939769-6407e95d-c807-48de-a3ed-08d426c5e7ea.png" width ="300" height = "300"/> <img src = "https://user-images.githubusercontent.com/80897270/146939775-387ae4ca-25f5-4459-a2fb-5c6d5dca0e7f.png" width ="300" height = "300"/>     
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <FC5 channel에 대한 2D image> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  <T7 channel에 대한 2D image> &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; <P7 channel에 대한 2D image>

<p align="center"><img src = "https://user-images.githubusercontent.com/80897270/146939791-21518c45-cc99-4ae6-b9ba-5a7b06773f5f.png" width ="400" height = "400"/></p>
<p align="center"><머리 위에서 바라본 유사한 이미지를 보이는 channel들의 실제 위치></p>


## 4. 기대효과 및 활용방안
- 환자의 정신 상태를 정확히 평가하고 정신 건강 상태를 개선하기 위한 유용한 피드백 제공 가능
- IoT와 센서의 발달로 감정 분류에 기반 한 스마트 헬스 케어 시스템을 통한 삶의 질 개선
- 감정 탐지에 대한 활발한 데이터 수집을 통해 보다 현실적인 Virtual Reality(VR) 구현 가능하며, 이를 치료와 교육 목적으로 활용할 수 있음

## 5. 결론 및 제언
- Continuous Wavelet Transform(CWT)와 Convolutional Neural Network(CNN)을 사용하여 자동으로 적절한 feature를 추출하고 Fully-connected layer를 통해 감정을 분류해보았다. 14개의 channel에 대한 EEG 이미지를 동시에 학습하는 parallel 2D CNN보다 일반적인 2D CNN을 통한 feature extraction을 수행하였을 때 accuracy성능이 더 높았으며, 보다 일반화된 성능을 보여주었다.    
- 이와 같은 이유는 14개의 channel에 대한 EEG 이미지를 동시에 사용하므로 일반적인 2D CNN보다 parallel 2D CNN에서는 14배 더 적은 data가 training에 사용되기 때문일 수 있으며, channel별 이미지 간에 유의미한 차이가 존재하지 않아, 오히려 성능을 떨어트리는 요소로 작용했다고 생각할 수 있다. 보다 신뢰성을 높이며, accuracy 성능을 향상시키기 위해 아래와 같은 작업을 수행할 예정이다.
- Future works
    - Systematic evalution by varying wavelet functions
    - HAHV/HALV/LAHV/LALV 4-class classification
    - InceptionV3가 아닌 다른 CNN backbone model 테스트
    - Supervised iNat2021 pre-trained model 테스트
    - Self-supervised ImageNet pre-trained model 테스트




