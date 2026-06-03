# Mini GPT 발표 Q&A 백업

이 문서는 4분 발표 본문에서 뺀 질문 대응용 자료다.

## Q1. validation loss가 accuracy인가?

아니다. 사전학습 validation loss는 validation 문장에서 정답 다음 token에 얼마나 높은 확률을 줬는지 보는 cross entropy다.

accuracy는 정답을 1등으로 맞혔는지 보는 값이다. GPT 사전학습에서는 vocab 후보가 많기 때문에 accuracy보다 loss를 중심으로 보았다.

## Q2. 1392 epoch면 과적합 아닌가?

train loss가 validation loss보다 낮으므로 과적합 신호는 있다. 하지만 구간 평균 기준 validation loss가 6.1680에서 4.5587까지 낮아졌기 때문에, validation 성능이 나빠지는 전형적인 과적합으로 보지는 않았다.

발표에서는 “과적합이 전혀 없다”가 아니라 “train/validation gap은 있지만 장기 validation 추세는 개선됐다”로 말하는 것이 안전하다.

## Q3. SiLU가 GELU보다 낮은데 왜 GELU인가?

50 epoch activation ablation에서 SiLU와 GELU의 best val_loss 차이는 0.0076이었다. 단일 seed의 작은 차이만으로 SiLU가 항상 낫다고 결론내리기 어렵다.

과제의 GPT FFN 구조는 GELU를 기준으로 했고, 추가 분석에서 GELU는 sigmoid의 작은 gradient 문제와 ReLU의 0으로 끊기는 경로 문제를 피하는 근거를 보였다.

## Q4. ReLU를 피하는 근거는 무엇인가?

ReLU는 `z <= 0`에서 output과 derivative가 정확히 0이다. 실험에서도 epoch 30 기준 ReLU는 activation output `== 0` 비율과 derivative `== 0` 비율이 모두 0.450이었다.

반면 GELU는 음수 입력도 정확히 0으로 자르지 않기 때문에, 그 구간의 gradient 경로가 완전히 끊기지 않는다.

| 지표 | GELU | ReLU | 해석 |
| --- | ---: | ---: | --- |
| `z <= 0` | 0.444 | 0.450 | pre-activation 음수 비율은 비슷함 |
| activation output `== 0` | 0.000 | 0.450 | ReLU는 음수 구간의 출력을 끊음 |
| activation derivative `== 0` | 0.000 | 0.450 | ReLU는 음수 구간의 gradient 경로도 끊음 |

## Q5. vocab_size 1000 loss가 낮은데 왜 2000을 썼나?

vocab_size가 바뀌면 tokenization 단위와 예측 class 수가 같이 바뀐다. 따라서 cross entropy loss 숫자만으로 vocab_size를 직접 비교하면 안 된다.

vocab_size 비교는 loss만이 아니라 bits-per-byte, 생성 품질, token 수를 함께 봐야 한다.

## Q6. 5 epoch씩 나눠 학습하는 것과 500 epoch 한 번에 학습하는 것이 같은가?

총 학습량은 비슷하지만 optimizer state를 이어받지 않으면 달라질 수 있다.

AdamW는 `exp_avg`, `exp_avg_sq`, `step` 같은 내부 상태를 사용한다. 따라서 checkpoint에는 model state뿐 아니라 optimizer state도 같이 저장해야 한다.

## Q7. fine-tuning 성능이 사전학습 덕분이라고 말할 수 있나?

이번 결과만으로는 사전학습 효과의 크기를 분리해 말할 수 없다. random initialization backbone이나 frozen backbone baseline이 없기 때문이다.

그래서 발표에서는 fine-tuning pipeline이 정상 동작했다는 범위로만 주장한다.

## Q8. dropout 0.0이 왜 위험한가?

full data 500 epoch 실험에서 dropout 0.0은 epoch 20에서 best val_loss 5.0969를 찍은 뒤, epoch 500에서 val_loss 6.7176까지 상승했다.

반면 dropout 0.1과 0.3은 epoch 500까지 validation loss가 계속 낮아졌다.

| dropout | best val_loss | best epoch | final val_loss | final train-val gap |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 5.0969 | 20 | 6.7176 | 3.2413 |
| 0.1 | 4.5912 | 500 | 4.5912 | 0.4723 |
| 0.3 | 4.5582 | 500 | 4.5582 | 0.0472 |

이는 train loss는 계속 낮아지지만 validation loss가 올라가는 전형적인 과적합 패턴이다. 이 결과 때문에 장기 사전학습에서는 dropout 같은 regularization이 실제로 필요하다고 해석했다.
