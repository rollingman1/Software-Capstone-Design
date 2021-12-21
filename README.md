# 2021-2 Software-Capstone-Design
# 뇌파(EEG)를 이용한 딥러닝 기반 감정 분류 시스템 SW 개발

# 1. 개요
## 1.1. 과제 선정 배경
- (EEG) 뇌파 는 뇌의 전기적인 활동을 머리 표면에 부착한 전극에 의해 측정한 전기신호로써 측정, 대상자의 심신 상태에 따라 다르게 나타나며 뇌의 활동상황을 측정하는 중요한 지표이다 이러한, 뇌파를 활용한 다양한 연구가 진행되고 있다. 루게릭병이나 파킨슨병과 같은 뇌 질환에 관련된 연구, 신체를 물리적으로 움직이지 않고 동작을 단지 상상하는 정신적인 작업인 Motor Imagery 연구, 집중력 측정이나 감정인식과 같은 대상의 상태분석 연구 등이 있다.
- 감정은 논리적인 의사결정, 인지, 지능과 관련돼 있어 일상생활과 건강에 밀접한 관련이 있다. EEG신호를 통해 정확한 감정분석이 가능해지면 불안, 분노와 같이 인간의 면역체계를 방해하고 질병의 위험을 증가시키는 감정들을 탐지해내고 개선해냄으로써 삶의 전반적인 질을 향상시키는데 기여할 수 있다.
- 표정, 음성, 동작 등을 통한 감정 분석과 달리 뇌파는 본인의 의지로 억누르거나 숨길 수 없기 때문에 뇌파를 이용한 감정 분석은 보다 객관적인 분석 결과를 얻을 수 있다.

## 1.2. 과제 주요내용
- EEG 1D signal data를 Continuous wavelet transform(CWT)를 통해 2D scalogram으로 변환 후, 이미지화하여 Time과 Frequency 모두에 걸쳐 표현된 data 획득
- Convolutional Neural Network(CNN)을 통해 감정 분류에 적절한 feature 자동 추출
* Parallel 2D CNN을 통한 feature extraction (all-channel)
* 일반적인 2D CNN을 통한 feature extraction (single-channel)
