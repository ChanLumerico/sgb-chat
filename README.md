# 🤖 SGB-Chat

대한민국 군 병영시설 중 하나인 *사이버지식정보방(이하 사지방)* 에서 군 생활동안 직접 개발한 NumPy/MLX 기반 딥러닝 프레임워크인 [`💎Lucid`](https://github.com/ChanLumerico/lucid)의 실질적인 성능 검증을 위해 수행한 간단한 채팅용 Transformer 모델 학습

추가적으로 상대적으로 성능이 열악한, 외장 GPU 조차 있지 않은 사지방 PC의 CPU로 Transformer 모델을 학습시켜보고 싶은 약간의 도전정신(?) 또한 이 프로젝트를 진행하는 동기가 됨.

---

## ✨ Transformer

Tranformer는 시퀀스 입력을 처리하기 위해 **순환 구조(RNN)** 없이 전적으로 **어텐션(Attention)** 메커니즘과 **병렬 연산** 에 의존하는 인코더-디코더(Encoder-Decoder) 아키텍처이다.

원 논문은 [*"Attention Is All You Need" (Vaswani et al., 2017)*](https://arxiv.org/pdf/1706.03762) 이며, 핵심 철학은 다음과 같다:

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
\sin\left({\frac{t}{10000^{i/d_{model}}}}\right)\quad &\text{if}~i\mod 2=0 \\
\cos\left({\frac{t}{10000^{i/d_{model}}}}\right)\quad &\text{if}~i\mod 2=1 \\
\end{cases}
$$

- $t$: 시퀀스 내 위치 $(0, 1, 2, \ldots)$
- $i$: 채널 인덱스 $(0, 1, 2, \ldots)$
- 짝수 채널에는 $\sin$, 홀수 채널에는 $\cos$

이를 직관적으로 해석하자면:

- 분모 $10000^{2i/d_{model}}$는 채널마다 다른 주파수(frequency)을 준다.
- 즉, 어떤 차원은 *"느리게 변하는 위치 정보"*, 어떤 차원은 *"빠르게 변하는 미세 위치 정보"* 를 담는다.
- 이 조합으로 모델은 상대적 거리 정보까지 유추할 수 있다.

  즉, $\text{PE}(t+k)-\text{PE}(t)$가 일정한 구조를 갖는다 $\rightarrow$ $k$ 만큼 떨어져 있음 을 계산 가능하다.

#### 🔹 Implementation Perspective

최종 입력은

$$
\mathbf{Z}=\mathbf{X}_{emb,~scaled}+\text{PE}\in\mathbb{R}^{T\times d_{model}}
$$

이 $\mathbf{Z}$가 Encoder/Decoder 블록으로 들어가는 최소 시퀀스 표현이 된다.

### 3️⃣ Multi-Head Self-Attention

Self-Attention은 시퀀스 안의 각 위치가, 같은 시퀀스의 다른 위치들을 **바라보고(attend) 가중합(weighted-sum)** 을 만드는 과정이다.

#### 🔹 Core Idea

토큰 $t$의 표현이, 전체 토큰들과의 관계를 반영하도록 만든다. 예를 들어, 대명사 'it'이라는 단어가 문장 내에서 가리키는 대상(antecedent)을 찾는 데 유리하다.

#### 🔸 Query/Key/Value (Q, K, V)

각 입력 벡터 $\mathbf{z}_t$ 에 대해, 세 개의 서로 다른 **선형 변환(linear transformation)** 을 적용한다.

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

토큰 $t$가 토큰 $u$를 *"얼마나 볼지(attend)"* 는 두 벡터의 **유사도(similarity)** 로 결정한다.

유사도는 $\mathbf{q}_t$와 $\mathbf{k}_u$의 내적(dot-product)으로 계산한다:

$$
S_{t,u}=\mathbf{q}_t\cdot\mathbf{k}_u^\top,\quad \mathbf{S}=\mathbf{Q}\mathbf{K}^\top\in\mathbb{R}^{T\times T}
$$

하지만 차원이 커질수록 내적 값이 커져서 softmax가 **saturate** 되기 쉬우므로 $\sqrt{d_k}$로 나눠준다.

$$
\mathbf{S}_{scaled}=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}
$$

#### 🔸 Masking

- **인코더 Self-Attention**: 일반적으로 마스크 없이 모든 위치가 서로 볼 수 있다.
- **디코더 Self-Attention**: 미래 시점의 단어를 참조하면 안 되므로 *causal mask* 를 사용한다.

Causal Mask $\mathbf{M}$은

$$
M_{t,s}=
\begin{cases}
0\quad &s\le t \\
-\infty\quad &s> t
\end{cases}
$$

이것을 $\mathbf{S}_{scaled}$에 더해서 미래 위치에 대한 softmax 확률이 $0$이 되게 만든다.

$$
\tilde{\mathbf{S}}=\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}+\mathbf{M}
$$

#### 🔹 Attention Softmax

각 query 위치 $t$별로 softmax를 취해 확률 분포를 얻는다.

$$
\alpha_{t,s}=\frac{\exp(\tilde{S}_{t,s})}{\sum_{u=1}^T\exp(\tilde{S}_{t,u})}
\quad\Rightarrow\quad
\mathbf{A}=\text{softmax}(\tilde{\mathbf{S}})\in\mathbb{R}^{T\times T}
$$

이 $\mathbf{A}$는 $t$ 번째 토큰이 $s$ 번째 토큰에 얼마나 집중(attend)하는가 를 나타내는 **attention matrix** 이다.

#### 🔸 Output Vector via Weighted-Sum

이게 각 위치 $t$의 최종 표현은 value들의 가중합:

$$
\mathbf{o}_t=\sum_{s=1}^T\alpha_{t,s}\mathbf{v}_s,\quad \mathbf{O}=\mathbf{A}\mathbf{V}\in\mathbb{R}^{T\times d_v}
$$

이것이 한 개의 어텐션 헤드(head) 결과다.

#### 🔹 Multi-Head Attention

단일 헤드만 사용하면 모델은 한 종류의 관계 패턴만 볼 수 있다.

여러 헤드를 두고 서로 다른 $\mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V$를 학습시키면 문법적 관계, 의미적 관계, 위치적 관계 등 서로 다른 시각을 병렬로 볼 수 있다.

각 헤드 $i\in\{1,\ldots,h\}$의 output을 계산하고

$$
\text{head}_i=\text{Attention}\left(\mathbf{Z}\mathbf{W}^Q_i,\mathbf{Z}\mathbf{W}^K_i,\mathbf{Z}\mathbf{W}^V_i\right)\in\mathbb{R}^{T\times d_v}
$$

그 다음 모든 헤드를 concatenate 한다.

$$
\text{Concat}\left(\text{head}_1,\ldots,\text{head}_h\right)\in\mathbb{R}^{T\times(h\cdot d_v)}
$$

마지막으로 이를 다시 원래 차원 $d_{model}$로 투사(projection) 시킨다.

$$
\text{MHA}(\mathbf{Z})=\text{Concat}\left(\text{head}_1,\ldots,\text{head}_h\right)\mathbf{W}^O\in\mathbb{R}^{T\times d_{model}}
$$

여기서 $\mathbf{W}^O\in\mathbb{R}^{(h\cdot d_v)\times d_{model}}$이다.

요약하자면, Multi-Head Self-Attention은

$$
\text{MHA}(\mathbf{Z})=\text{Concat}\left(\text{head}_1,\ldots,\text{head}_h\right)\mathbf{W}^O
$$

$$
\text{where}\quad\text{head}_i=\text{softmax}\left(\frac{\mathbf{Z}\mathbf{W}^Q_i(\mathbf{Z}\mathbf{W}^K_i)^\top}{\sqrt{d_k}}+\mathbf{M}\right)\left(\mathbf{Z}\mathbf{W}^V_i\right)
$$

### 4️⃣ Encoder

Transformer Encoder는 동일한 블록을 $N$번 반복한 stack이다. 각 블록은 **두 개의 핵심 sub-layer** 로 구성된다:

1. Multi-Head Self-Attention
2. Position-wise Feed-Forward-Network(FFN)

각 sub-layer에는 **Residual Connection + LayerNorm** 이 있다.

텍스트로 표현하자면 인코더 레이어 하나는:

1. 입력 $\mathbf{Z}^{(l)}$ $\rightarrow$ MHA $\rightarrow$ Dropout $\rightarrow$ Residual Add $\rightarrow$ LayerNorm $\rightarrow$ $\mathbf{H}^{(l)}$

2. $\mathbf{H}^{(l)}$ $\rightarrow$ FFN $\rightarrow$ Dropout $\rightarrow$ Residual Add $\rightarrow$ LayerNorm $\rightarrow$ $\mathbf{Z}^{(l+1)}$

여기서 $l$은 레이어 인덱스이다.

#### 🔹 Residual + LayerNorm

한 sub-layer를 함수 $\text{sublayer}(\cdot)$라고 할 때,

$$
\text{LayerOutput} = \text{LayerNorm}\left(\mathbf{X}+\text{Dropout}\left(\text{sublayer}\left(\mathbf{X}\right)\right)\right)
$$

Residual은 gradient 흐름을 **안정화** 시켜 deep-stack 학습을 돕는다.

LayerNorm은 **채널 차원 기준 정규화(regularization)** 으로, 각 토큰 벡터를 평균 $0$, 분산 $1$ 근처로 맞춰 **분포 폭주(explosion)** 를 막는다.

LayerNorm은 입력 $\mathbf{u}\in\mathbb{R}^{d_{model}}$에 대해:

$$
\text{LayerNorm}(\mathbf{u})=\gamma\odot\frac{\mathbf{u}-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

여기서

$$
\mu=\frac{1}{d_{model}}\sum_{j=1}^{d_{model}}u_j,\quad\sigma^2=\frac{1}{d_{model}}\sum_{j=1}^{d_{model}}(u_j-\mu)^2
$$

이고, $\gamma,\beta\in\mathbb{R}^{d_{model}}$은 학습 가능한(learnable) 파라미터이다.

#### 🔸 Position-wise Feed-Forward-Network(FFN)

각 위치(토큰)별로 독립적으로 적용되는 2층 MLP:

$$
\text{FFN}(\mathbf{h})=\max(0,\mathbf{h}\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2
$$

보통 차원 확장 후 축소하는 과정을 거친다.

$$
\mathbf{W}_1\in\mathbb{R}^{d_{model}\times d_{ff}},\quad\mathbf{W}_2\in\mathbb{R}^{d_{ff}\times d_{model}}
$$

여기서 $d_{ff}$ (예: 2048)은 $d_{model}$ (예: 512)보다 훨씬 크게 잡는다. 이는 토큰별 **비선형 변환(non-linear transformation)** 능력을 강화시킨다.

행렬 형태로, 시퀀스 전체 $\mathbf{H}\in\mathbb{R}^{T\times d_{model}}$에 대해:

$$
\text{FFN}(\mathbf{H})=\text{ReLU}\left(\mathbf{H}\mathbf{W}_1+\mathbf{b}_1\right)\mathbf{W}_2+\mathbf{B}_2\in\mathbb{R}^{T\times d_{model}}
$$

#### 🔹 Encoder Layer Summary

한 인코더 레이어 $l$는 다음 과정으로 진행된다:

1. **Self-Attention Sub-Layer**

$$
\mathbf{U}^{(l)}=\text{LayerNorm}\left(\mathbf{Z}^{(l)}+\text{Dropout}\left(\text{MHA}\left(\mathbf{Z}^{(l)}\right)\right)\right)
$$

2. **FFN Sub-Layer**

$$
\mathbf{Z}^{(l+1)}=\text{LayerNorm}\left(\mathbf{U}^{(l)}+\text{Dropout}\left(\text{FFN}\left(\mathbf{U}^{(l)}\right)\right)\right)
$$

여기서 $\mathbf{Z}^{(0)}=\mathbf{Z}=\mathbf{Z}_{emb,~scaled}+\text{PE}$이다.

최종 인코더 출력:

$$
\text{EncOut}=\mathbf{Z}^{(N)}\in\mathbb{R}^{T\times d_{model}}
$$

은 디코더로 넘어가서 *"인풋 문장의 의미 요약"* 으로 쓰인다.

### 5️⃣ Decoder

디코더는 언어 생성 및 번역 등에 사용된다. 인코더와 작동 방식은 비슷하지만, 한 layer마다 sub-layer가 **3개씩** 존재한다.

1. Masked Multi-Head Self-Attention *(미래 토큰 차단)*
2. Cross-Attention *(인코더 출력과 attending)*
3. Position-wise FFN

인코더와 마찬가지로 모든 sub-layer는 **Residual Connection + LayerNorm** 의 형태이다.

#### 🔹 Decoder Input

타겟 시퀀스의 토큰을 임베딩하고 positional encoding을 더한다. 이걸 $\mathbf{Y}^{(0)}\in\mathbb{R}^{T_{tgt}\times d_{model}}$이라 하자.

#### 🔸 Masked Self-Attention

인코더의 self-attention과 동일하지만, 미래 단어를 보지 못하게 causal mask $\mathbf{M}_{causal}$를 적용한다. 즉, 디코더가 시점 $t$ 단어를 예측할 때, $t+1,t+2,\ldots$를 **참조할 수 없게** 된다.

수식은 인코더 self-attention과 동일하되 $\mathbf{M}_{causal}$ 마스크를 상용한다.

이 결과에 **Residual + LayerNorm** 를 적용한다.

$$
\mathbf{U}^{(l)}_1=\text{LayerNorm}\left(\mathbf{Y}^{(l)}+\text{Dropout}\left(\text{MaskedMHA}\left(\mathbf{Y}^{(l)}\right)\right)\right)
$$

#### 🔹 Cross-Attention

이 단계가 디코더가 인코더를 attend하는 부분이다.

여기서 Query는 디코더의 중간 표현(intermediate representation)에서 오고, Key/Value는 인코더 출력에서 온다.

- 디코더 중간 표현: $\mathbf{U}_1^{(l)}$
- 인코더 출력: $\text{EncOut}$

$$
\begin{aligned}
\mathbf{Q}=\mathbf{U}_1^{(l)}\mathbf{W}_{dec}^Q\quad&\in\quad\mathbb{R}^{T_{tgt}\times d_k} \\
\mathbf{K}=\text{EncOut}\mathbf{W}_{enc}^K\quad&\in\quad\mathbb{R}^{T_{src}\times d_k} \\
\mathbf{V}=\text{EncOut}\mathbf{W}_{enc}^V\quad&\in\quad\mathbb{R}^{T_{src}\times d_v} \\
\end{aligned}
$$

Self-Attention과 동일한 방식으로 softmax를 거친 뒤 가중합을 구한다.

Multi-Head인 경우 마찬가지로 head별로 위의 연산을 하고 **Concatenation + Linear Transformation** 을 거친다.

그 다음엔 인코더 출력과 cross attention을 진행한다.

$$
\mathbf{U}^{(l)}_2=\text{LayerNorm}\left(\mathbf{U}^{(l)}_1+\text{Dropout}\left(\text{CrossMHA}\left(\mathbf{U}^{(l)}_1,\text{EncOut}\right)\right)\right)
$$

직관적으로 이해해보자면:

- 디코더는 지금까지 생성한 **단어 맥락** ($\mathbf{U}_1^{(l)}$)을 가지고
- 인코더가 이해한 **소스 문장** ($\text{EncOut}$) 중 어디를 참고할지 동적으로 *"포인팅"* 한다.

#### 🔸Feed-Forward Network(FFN)

인코더와 동일한 position-wise FFN을 거쳐 디코더 레이어 $l$의 출력을 만든다.

$$
\mathbf{Y}^{(l+1)}=\text{LayerNorm}\left(\mathbf{U}^{(l)}_2+\text{Dropout}\left(\text{FFN}\left(\mathbf{U}^{(l)}_2\right)\right)\right)
$$

#### 🔹 Decoder Stack and Final Logit

이 디코더 레이러를 $N$번 반복 후 최종 출력 $\mathbf{Y}^{(N)}\in\mathbb{R}^{T_{tgt}\times d_{model}}$를 얻는다.

이것을 vocab 크기 $V$로 projection 해서 각 시점의 단어 분포(logits)를 만든다.

$$
\text{Logits}=\mathbf{Y}^{(N)}\mathbf{W}^{\text{vocab}}+\mathbf{b}^{\text{vocab}}\in\mathbb{R}^{T_{tgt}\times V}
$$

이후, softmax를 이용해 다음 단어의 확률을 구한다.

$$
P(\text{token}_t=v~|~\text{context})=\frac{\exp\left(\text{Logits}_{t,v}\right)}{\sum_{u=1}^V\exp\left(\text{Logits}_{t,u}\right)}
$$

학습 시에는 **teacher forcing** 으로 정답 시퀀스를 한 시점씩 shift하여 다음 토큰을 예측하게 하고, Cross-Entorypy Loss를 사용한다.

$$
\mathcal{L}
= -\sum_{t=1}^{T_{\text{tgt}}}
\log P\left(
    y_{t}^{\star} \mid y_{<t}^{\star},\, \text{EncOut}
\right)
$$

여기서 $y_t^*$는 정답 토큰이다.

### ✅ Transformer Summary

1. **Embedding**

   단어 ID $\rightarrow$ 연속 벡터 표현, $\mathbb{R}^{V\times d_{model}}$ 임베딩 행렬로 lookup 후 스케일 $\sqrt{d_{model}}$ 곱함.

2. **Positional Encoding**

   순서를 모르는 Self-Attention에게 위치 정보를 주기 위해 $\sin,\cos$ 주파수 기반 벡터 $\text{PE}(t)$를 더함.

3. **Multi-Head Self-Attention**

   각 위치가 전체 시퀀스의 다른 위치를 attend하고 가중합을 만들도록 서로 다른 시각(head)을 병렬로 학습.

$$
\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

4. **Encoder**

   (Self-Attention $\rightarrow$ FFN) + Residual + LayerNorm을 $N$층 쌓아 입력 문장을 풍부한 표현 공간(representation space)로 인코딩.

5. **Decoder**

   (Masked Self-Attention $\rightarrow$ Cross-Attention with Encoder $\rightarrow$ FFN) + Residual + LayerNorm을 $N$층 쌓아 한 토큰씩 생성, 마지막은 vocab projection으로 확률 분포 계산.

---

## 💻 Code Implementation

*To be continued ...*
