# Advantage Actor Critic Review

Advantage Actor Critic\(A2C\)는 강화학습에서 가장 기본인 Reinforce에서 발전된 알고리즘입니다. 먼저 A2C에 대해 자세히 알아보기 이전에 Reinforce 알고리즘을 간단히 알아보겠습니다. 

Reinforce는 Monte-Carlo Policy Gradient라고도 불리며 해석적으로 어떠한 모델의 Gradient를 구하고 그에 맞게 모델의 파라미터를 갱신하 높은 Reward를 받게 하는 모델로 만듭니다. 간단하게 목표함수\(Object Function\)는 다음과 같이 표현할 수 있습니다.

$$
J(\theta) = E_{\pi_\theta}[r] = \Sigma_{s\in S} d(s) \Sigma_{a \in A} \pi_\theta(s,a)R^s_a
$$

위의 식에서 Object Function의 뜻은 한 에피소드에 대한 상태에서 얻을 수 있는 Reward의 기대값이며 오른쪽의 식으로 표현될 수 있습니다. $$d(s)$$ 는 상태의 분포, $$\pi_\theta(s,a)$$ 는 파라미터\( $$\theta$$ \)에 기반하여 상태\(s\)에서 각 행동을 선택할 확률을 뜻하며 $$R^s_a$$ 는 상태\(s\)에서 행동\(a\)를 선택했을 때 얻는 Reward를 뜻합니다. 전체 식이 의미하는 것은 \(상태의 확률\)x\(행동의 확률\)x\(Reward\)이기에 결국 전체 에피소드에 대한 Reward의 기대값으로 귀결될 수 있습니다.

이를 최대화 하기 위해 각 파라미터에 대해 Object Function을 미분하며 양의 방향으로 업데이트합니다. 이를 해석적으로 표현하면 다음과 같습니다.

$$
\bigtriangledown_\theta J(\theta) = \Sigma_{s\in S} d(s) \Sigma_{a \in A} \bigtriangledown_\theta \pi_\theta(s,a)R^s_a
$$

위의 로그 미분법을 이용하여 계산하기 쉽게 표현하면 다음과 같습니다.



