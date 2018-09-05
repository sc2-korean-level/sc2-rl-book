# Advantage Actor Critic Review

Advantage Actor Critic\(A2C\)는 강화학습에서 가장 기본인 Policy Gradient Theorem에서 발전된 알고리즘입니다. 먼저 A2C에 대해 자세히 알아보기 이전에 Policy Gradient Theorem 알고리즘을 간단히 알아보겠습니다. 

Policy Gradient Theorem는 어떠한 모델의 Gradient를 구하고 그에 맞게 모델의 파라미터를 갱신하여 높은 Reward를 받게 하는 모델로 만듭니다. 간단하게 목표함수\(Object Function\)는 다음과 같이 표현할 수 있습니다.

$$
J(\theta) = E_{\pi_\theta}[r] = \Sigma_{s\in S} d(s) \Sigma_{a \in A} \pi_\theta(s,a)R^s_a
$$

위의 식에서 Object Function의 뜻은 한 에피소드에 대한 상태에서 얻을 수 있는 Reward의 기대값이며 오른쪽의 식으로 표현될 수 있습니다. $$d(s)$$ 는 상태의 분포, $$\pi_\theta(s,a)$$ 는 파라미터\( $$\theta$$ \)에 기반하여 상태\(s\)에서 각 행동을 선택할 확률을 뜻하며 $$R^s_a$$ 는 상태\(s\)에서 행동\(a\)를 선택했을 때 얻는 Reward를 뜻합니다. 전체 식이 의미하는 것은 \(상태의 확률\)x\(행동의 확률\)x\(Reward\)이기에 결국 전체 에피소드에 대한 Reward의 기대값으로 귀결될 수 있습니다.

이를 최대화 하기 위해 각 파라미터에 대해 Object Function을 미분하며 양의 방향으로 업데이트합니다. 이를 해석적으로 표현하면 다음과 같습니다.

$$
\bigtriangledown_\theta J(\theta) = \Sigma_{s\in S} d(s) \Sigma_{a \in A} \bigtriangledown_\theta \pi_\theta(s,a)R^s_a
$$

위의 로그 미분법을 이용하여 계산하기 쉽게 표현하면 다음과 같습니다.

$$
\Sigma_{s\in S} d(s) \Sigma_{a \in A} \bigtriangledown_\theta \pi_\theta(s,a)R^s_a \\
=  \Sigma_{s\in S} d(s) \Sigma_{a \in A} \dfrac{\pi_\theta(s,a)}{\pi_\theta(s,a)}\bigtriangledown_\theta \pi_\theta(s,a)R^s_a \\
= \Sigma_{s\in S} d(s) \Sigma_{a \in A} \dfrac{\bigtriangledown_\theta\pi_\theta(s,a)}{\pi_\theta(s,a)} \pi_\theta(s,a)R^s_a \\ 
=\Sigma_{s\in S} d(s) \Sigma_{a \in A} \pi_\theta(s,a) \bigtriangledown_\theta log \pi_\theta(s,a) R^s_a\\
=E[\bigtriangledown_\theta log \pi_\theta(s,a) R^s_a]
$$

결국 Object Function의 Gradient는 $$\bigtriangledown_\theta log \pi_\theta(s,a) R^s_a$$ 의 기대값이며 계산이 많이 간단하게 변하였습니다. 위의 Gradient를 이용하여 파라미터\( $$\theta$$ \)를 업데이트하면 Object Function을 최대화하는 모델을 얻을 수 있습니다.

하지만 Policy Gradient Theorem는 문제가 있습니다. Reward\( $$R^s_a$$ \)를 그대로 사용하기 때문에 baised된 값으로 업데이트하기 때문에 high variance의 문제가 있습니다.

이러한 문제점을 해결하기 위해 Advantage Actor Critic이라는 알고리즘을 고안했습니다. 간단히 말하자면 Policy Gradient Theorem에서의 문제점은 모두 Reward에서 나오기 때문에 이를 추정하는 하나의 모델을 더 만들어 업데이트하는 것입니다. 그리고 Advantage라는 개념을 이용하여 high variance 문제를 해결합니다. Advantage 다음 상태와 현재 상태의 '차이'로 정의를 하기 때문에 baised되지 않은 값을 가집니다. 이를 통해 unbiased 문제를 해결하였습니다. Advantage는 아래와 같은 형태를 가집니다.

$$
A(s,a) = R^s_a + \gamma V_v(s') - V_v(s)
$$

위의 식에서 $$V_v(s)$$ 는 상태\(s\)에서의 가치를 뜻하며 $$\theta$$ 와 다른 파라미터\($$v$$\)를 이용하여 새로 추정합니다. Bellman 방정식을 이용하여 다음 상태\(s'\)와 현재상태\(s\) 사이의 차이\(Advantage\)를 구합니다. 만약 Advantage가 양수로 나온다면 현재 상태보다 다음 상태가 더 좋다는 뜻이며 음수일 경우 다음 상태보다 현재 상태가 더 좋다는 뜻입니다.

그리고 파라미터\($$v$$\)는 아래의 식을 최소화 하는 방향으로 업데이트합니다. 일반적으로 Deep Q Network에서 Bellman 방정식을 통해서 업데이트하는 방식과 같습니다.

$$
\delta = (R^s_a + \gamma V_v(s') - V_v(s))^2
$$

파라미터\( $$v$$ \)를 이용해서 구한 Advantage를 이용하여 Policy Gradient Theorem에서 사용한 정책 업데이트 방식과 같이 파라미터\( $$\theta$$ \)를 업데이트합니다.

$$
\bigtriangledown_\theta J(\theta) = E[\bigtriangledown_\theta log \pi_\theta(s,a) A(s,a)]
$$

이를 [코드](https://github.com/sc2-korean-level/MoveToBeacon/blob/master/4wayBeacon_a2c/a2c.py)로 구현하면 다음과 같습니다.

