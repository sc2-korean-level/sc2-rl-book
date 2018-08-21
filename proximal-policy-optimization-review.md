# Proximal Policy Optimization Review

기본적으로 Reinforcement Learning의 Proximal Policy Optimization\(이하 PPO\)은 Trust Region Policy Optimization\(이하 TRPO\)을 실용적으로, 컴퓨터 친화적으로\(연산에 용이하도록\) 알고리즘을 수정한 것입니다. 또한 Actor와 Critic이라는 두 가지 네트워크를 이용하여 수행하는 알고리즘이며 이를 ㅇ최적으로 업데이트 하는 방법을 제안합니다.

일반적인 Reinforcement Learning의 목표는 대부분 목표함수인 Expected Reward를 최대화 할 수 있는 정책을 구성하는 파라미터\( $$\theta$$ \)를 찾는 것 입니다.

Reinforcement Learning에서의 목표함수 Expected Reward는 다음과 같이 표현됩니다.

$$
\eta(\pi) = \hat{E}_t[\Sigma log \pi_\theta(a_t|s_t)\hat{A}_t]
$$

위의 식에서 $$\pi_\theta$$ 는 stochastic policy이며 $$\hat{A}_t$$는 가치를 평가하는 네트워크에 의해 t의 시점에서 추정되는 Advantage입니다. 이 목표함수를 최대화 하기 위해 Reinforcement Learning에서는 파라미터\( $$\theta$$ \)에 대해서 Gradient를 사용하여 최대화하는 방향으로 업데이트합니다.

PPO는 TRPO에서 파생되어 나오는 알고리즘이기에 TRPO에 대해 먼저 간단히 알아보겠습니다. 자세히 알고 싶다면 [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) 을 직접 읽어보시는 것을 권합니다.

