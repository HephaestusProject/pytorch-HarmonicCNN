# template

[![Code Coverage](https://codecov.io/gh/HephaestusProject/template/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/template)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

Audio Representation은 Music Tagging, Keyword Spotting, Sound Event Tagging 등 다양한 테스크에서 중요합니다. 고차원의 Deep Embedding Space에 음악이나, Audio Signal의 도메인 요소를 적용시키는 방법론을 제안합니다. Inherent Harmonic Structure은 인간의 인지과정의 핵심적인 요소입니다. 본 논문에서는 이러한 인지모델링을 Harmonic filter를 통해 구현합니다. Harmonic filter는 spectro-temporal locality를 유지하면서 Harmonic Relationship을 유지합니다.

## Table

* 구현하는 paper에서 제시하는 benchmark dataset을 활용하여 구현하여, 논문에서 제시한 성능과 비교합니다.
  + benchmark dataset은 하나만 골라주세요.
    1. 논문에서 제시한 hyper-parameter와 architecture로 재현을 합니다.
    2. 만약 재현이 안된다면, 본인이 변경한 사항을 서술해주세요.

## Training history

* tensorboard 또는 weights & biases를 이용, 학습의 로그의 스크린샷을 올려주세요.

## OpenAPI로 Inference 하는 방법

* curl ~~~

## Usage

### Environment

* install from source code
* dockerfile 이용

### Training & Evaluate

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.
    - single-gpu, multi-gpu

### Inference

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.

### Project structure

* 터미널에서 tree커맨드 찍어서 붙이세요.

### License

* Licensed under an MIT license.
