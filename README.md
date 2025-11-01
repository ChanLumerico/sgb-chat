# 🤖 SGB-Chat

대한민국 군 병영시설 중 하나인 *사이버지식정보방(이하 사지방)* 에서 직접 개발한 NumPy/MLX 기반 딥러닝 프레임워크인 [`Lucid`](https://github.com/ChanLumerico/lucid)의 실질적인 성능 검증을 위해 수행한 간단한 Transformer 모델 학습

---

## ✨ Transformer

Tranformer는 시퀀스 입력을 처리하기 위해 **순환 구조(RNN)** 없이 전적으로 **어텐션(Attention)** 메커니즘과 **병렬 연산** 에 의존하는 인코더-디코더(Encoder-Decoder) 아키텍처다.

원 논문은 *"Attention Is All You Need" (Vaswani et al., 2017)* 이며, 핵심 철학은 다음과 같다:

1. 입력 토큰(token)을 고정 차원의 벡터 공간로 **임베딩(embedding)** 한다.
2. 시퀀스 내에서 각 위치가 다른 위치를 *'얼마나 볼지(attend)'* 를 학습적으로 결정한다. (Self-Attention)
3. 이 과정을 **여러 헤드(head)로 병렬화** 하여 다양한 표현 하위공간을 본다 (Multi-Head Attention).
4. **인코더** 는 입력 문장 표현을 만드는 역할, **디코더** 는 그 표현을 조건으로 출력 문장을 생성하는 역할을 한다.
5. 모든 것은 *Residual Connection + LayerNorm* 으로 안정화된다.

### 1️⃣ Embedding

#### 🔹 What for?

자연어의 단어(또는 sub-word 토큰)는 **정수 ID** 로 표현된다. 예를 들어

$$
\mathbf{x}=\left[\text{'나는'},~\text{'밥을'},~\text{'먹는다'}\right]\quad\rightarrow\quad\left[1032, ~481,~77\right]
$$

처럼 vocabularay index가 된다.

하지만 신경망은 정수 index 자체에는 직접적으로 의미를 부여할 수 없다. 그래서 각 토큰 ID를 *학습 가능한(learnable) 고차원 벡터* 로 매핑한다. 이 매핑을 **임베딩(embedding)** 이라 한다.

#### 🔸 Definition

어휘 집합(단어 집합)의 크기를 $V$, 임베딩 차원을 $d_{model}$이라고 하자.

- 임베딩 행렬(Embedding Matrix):

  $$
  \mathbf{E}\in\mathbb{R}^{V\times d_{model}}
  $$

- 입력 시퀀스 길이를 $T$라 하면, one-hot 벡터 $\mathbf{o}_t\in\mathbb{R}^V$ (단어 인덱스에만 $1$, 나머지 $0$)를 곱해 임베딩을 얻는다:

  $$
  \mathbf{e}_t=\mathbf{o}_t^\top\mathbf{E}\in\mathbb{R}^{d_{model}}
  $$

  실제로 구현은 one-hot을 만들지 않고 그냥 `E[token_id]`를 가져오는 gather 연산을 취한다.

전체 시퀀스를 모으면:

$$
\mathbf{X}_{emb}=\left[\mathbf{e}_1~\mathbf{e}_2~\cdots~\mathbf{e}_T\right]^\top\in\mathbb{R}^{T\times d_{model}}
$$

#### 🔹 Scaling

원 논문에서는 임베딩에 $\sqrt{d_{model}}$를 곱해준다:

$$
\mathbf{X}_{emb,~scaled}=\sqrt{d_{model}}~\cdot~\mathbf{X}_{emb}
$$

그 이유는 어텐션에 들어가기 전, *위치 정보(Positional Encoding)* 를 더하는데 두 값의 scale를 비슷하게 맞춰 **학습을 안정화시키기 위함** 이다.

### 2️⃣ Positional Encoding

#### 🔹 What for?

Transformer는 RNN처럼 순차적으로 처리하지 않는다. 즉, 토큰 순서를 구조적으로 ***모른다***.

따라서 모델이 예를 들어 *"3번째 단어는 2번째 단어 다음에 온다"* 같은 순서 정보를 알 수 있게, 각 위치 $t$에 대해 **위치 인코딩 벡터** $\text{PE}(t)$를 만들어 임베딩에 더해준다.

$$
\mathbf{Z}_t=\mathbf{X}_{emb,~t}+\text{PE}(t)\in\mathbb{R}^{d_{model}}
$$

#### 🔸 Sine/Cosine-Based Absolute Positional Encoding

$$
\text{PE}(t)=
\begin{cases}
\sin{\frac{t}{10000^{i/d_{model}}}}\quad\text{if}\;i\mod 2=0 \\
\cos{\frac{t}{10000^{i/d_{model}}}}\quad\text{if}\;i\mod 2=1 \\
\end{cases}
$$

- $t$: 시퀀스 내 위치 $(0, 1, 2, \ldots)$
- $i$: 채널 인덱스 $(0, 1, 2, \ldots)$
- 짝수 채널에는 $\sin$, 홀수 채널에는 $\cos$

이를 직관적으로 해석하자면:

- 분모 $10000^{2i/d_{model}}$는 채널마다 다른 파장(frequency)을 준다.
- 즉, 어떤 차원은 *"느리게 변하는 위치 정보"*, 어떤 차원은 *"빠르게 변하는 미세 위치 정보"* 를 담는다.
- 이 조합으로 모델은 상대적 거리 정보까지 유추할 수 있다.

  즉, $\text{PE}(t+k)-\text{PE}(t)$가 일정한 구조를 갖는다 $\rightarrow$ *"$k$만큼 떨어져 있음"* 을 계산 가능하다.

#### 🔹 Implementation Perspective

최종 입력은

$$
\mathbf{Z}=\mathbf{X}_{emb,~scaled}+\text{PE}\in\mathbb{R}^{T\times d_{model}}
$$

이 $\mathbf{Z}$가 Encoder/Decoder 블록으로 들어가는 최소 시퀀스 표현이 된다.

### 3️⃣ Multi-Head Self Attention

Self-Attention은 시퀀스 안의 각 위치가, 같은 시퀀스의 다른 위치들을 **바라보고(attend) 가중합(weighted-sum)** 을 만드는 과정이다.

#### 🔹 Core Idea

토큰 $t$의 표현이, 전체 토큰들과의 관계를 반영하도록 만든다. 예를 들어, 대명사 'it'이라는 단어가 문장 내에서 가리키는 대상(antecedent)을 찾는 데 유리하다.

#### 🔸 Query/Key/Value (Q, K, V)

각 입력 벡터 $\mathbf{z}_t\in\mathbb{R}^{d_{model}}$에 대해, 세 개의 서로 다른 **선형 변환(linear transformation)** 을 적용한다.

$$
\mathbf{q}_t=\mathbf{z}_t\mathbf{W}^Q,\quad\mathbf{k}_t=\mathbf{z}_t\mathbf{W}^K,\quad\mathbf{v}_t=\mathbf{z}_t\mathbf{W}^V
$$

여기서

$$
\mathbf{W}^Q\in\mathbb{R}^{d_{model}\times d_k},\quad\mathbf{W}^K\in\mathbb{R}^{d_{model}\times d_k},\quad\mathbf{W}^V\in\mathbb{R}^{d_{model}\times d_v}
$$

보통 $h$가 헤드 수일 때, $d_k=d_v=d_{model}/h$이다.

행렬 형태로 쓰명, 입력 전체를 모아

$$
\mathbf{Z}\in\mathbb{R}^{T\times d_{model}}
$$

이면 다음과 같다:

$$
\mathbf{Q}=\mathbf{Z}\mathbf{W}^Q\in\mathbb{R}^{T\times d_k},\quad
\mathbf{K}=\mathbf{Z}\mathbf{W}^K\in\mathbb{R}^{T\times d_k},\quad
\mathbf{V}=\mathbf{Z}\mathbf{W}^V\in\mathbb{R}^{T\times d_v}
$$

#### 🔹 Attention Score

토큰 $t$가 토큰 $s$를 *"얼마나 볼지(attend)"* 는 두 벡터의 **유사도(similarity)** 로 결정한다.

유사도는 $\mathbf{q}_t$와 $\mathbf{k}_s$의 내적(dot-product)으로 계산한다:

$$
a_{t,s}=\mathbf{q}_t\cdot\mathbf{k}_s^\top,\quad \mathbf{A}=\mathbf{Q}\mathbf{K}^\top\in\mathbb{R}^{T\times T}
$$

하지만 차원이 커질수록 내적 값이 커져서 softmax가 **saturate** 되기 쉬우므로 $\sqrt{d_k}$로 나눠준다.

$$
\mathbf{A}_{scaled}=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}
$$

#### 🔸 Masking

*To be continued ...*
