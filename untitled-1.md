# 4WayBeacon PPO Code Review

## Policy\_net.py

전체 코드는 [다음](https://github.com/sc2-korean-level/MoveToBeacon/blob/master/4wayBeacon_ppo/policy_net.py)을 참조하세요.

### Class Structure

이 파일은 PPO의 Policy\_net이라는 클래스를 가지고 있으며 이 클래스는 다음과 같은 구조를 가지고 있습니다. 클래스 내부의 함수를 하나씩 설명하겠습니다.

```text
└── Policy_net(Class)
    ├── __init__(Function)
    ├── act(Function)
    ├── get_action_prob(Function)
    ├── get_variables(Function)
    └── get_trainable_variables(Function)
```

### \_\_init\_\_\(self, name: str, temp=0.1\)

이 함수는 처음 클래스를 정의할 때 호출이 되는 함수이며 주로 클래스 내부에서 필요한 함수나 변수들을 정의하는 곳입니다.

```python
def __init__(self, name: str, temp=0.1):
    with tf.variable_scope(name):
        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='obs')
        with tf.variable_scope('policy_net'):
            layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
            layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
            layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.tanh)
            layer_4 = tf.layers.dense(inputs=layer_3, units=4, activation=tf.tanh)
            self.act_probs = tf.layers.dense(inputs=tf.divide(layer_4, temp), units=4, activation=tf.nn.softmax)
        with tf.variable_scope('value_net'):
            layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
            layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
            layer_3 = tf.layers.dense(inputs=layer_2, units=30, activation=tf.tanh)
            self.v_preds = tf.layers.dense(inputs=layer_3, units=1, activation=None)
        self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
        self.act_deterministic = tf.argmax(self.act_probs, axis=1)
        self.scope = tf.get_variable_scope().name
```

먼저 클래스를 정의하면서 name, temp의 변수를 받습니다. 이 변수들 중 쓰이는 변수는 name입니다. tf.variable\_scope\(name\)를 통해 네트워크의 이름을 지정합니다. 4WayBeacon의 경우 \(x, y\)의 입력을 받아 4가지의 action 중 하나를 선택합니다. 4WayBeacon의 경우 마린과 Beacon 사이의 상대거리를 입력 \(x, y\)로 정의합니다. 또한 4가지의 action은 각각 상하좌우로의 일정거리만큼 이동을 뜻합니다.

self.obs는 입력의 형상을 지정합니다. 입력 \(x, y\)에 맞추어 self.obs를 정의합니다.

with tf.variable\_scope\('policy\_net'\) 이하에서 Actor 네트워크의 형상을 지정합니다. 4WayBeacon에서는 입력이 상대거리이며 상대거리는 음수부터 양수까지 존재할 수 있습니다. 그렇기 때문에 모든 활성함수는 음수와 양수 모두를 반영할 수 있는 tanh 함수를 사용하였습니다. 총 3개의 은닉층을 가지고 있으며 각 64개의 노드를 가지고 있으며 마지막으로 self.act\_probs에서 4개\(action의 개수\)를 출력 개수로 지정합니다. PPO에서 볼 수 있듯이 $$\pi_\theta(s)$$ 는 상태에 대한 각 action의 선택 확률을 뜻하기 때문에 마지막 활성화 함수는 softmax로 하여 각 action에 대한 확률값을 출력하도록 정의하였습니다.

with tf.variable\_scope\('value\_net'\) 이하에서 Critic 네트워크의 형상을 지정합니다. Actor 네트워크와 같은 이유로 활성화 함수를 tanh 함수를 사용하였습니다. 3개의 은닉층을 가지며 각각 64, 64, 30개의 노드를 가지고 있으며 Critic 네트워크는 하나의 value를 출력해야 하기에 self.v\_preds는 1개의 마지막 출력 노드를 가지고 있습니다.

self.act\_stochastic은 변수 그대로 stochatic하게 action을 선택하기 위해 정의되었습니다. tf.multinomial 함수를 통해 확률 행렬에서 num\_samples의 개수만큼 샘플링을 합니다.

self.act\_determinisitic은 stochastic처럼 확률적으로 action을 선택하는 것이 아닌 가장 확률이 높은 것을 tf.argmax를 통해 선택합니다.

self.scope는 이 클래스가 어떤 이름으로 네트워크의 이름을 지정했는지 확인하기 위해 정의됩니다.

### act\(self, obs, stochastic=True\)

이 함수는 클래스 내의 Actor 네트워크에 출력에 의해 정의되는 $$\pi_\theta(s)$$ 에 대한 값을 얻기 위한 함수입니다.

```python
def act(self, obs, stochastic=True):
    if stochastic:
        return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
    else:
        return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})
```

이 함수는 obs와 stochatic를 입력으로 받습니다. obs는 $$\pi_\theta(s)$$ 에서 $$s$$ 를 뜻합니다. stochastic은 True나 False 중 하나로 정의되면 True일 경우 \_\_init\_\_함수 내 self.act\_stochastic에 의해 확률적으로 action을 선택합니다. False일 경우 self.act\_deterministic에 의해 가장 높은 확률을 가지는 action을 선택합니다. 또한 이 함수는 action만 정의하는 것이 아니라 obs에 대한 $$V(s)$$ 도 출력합니다.

### get\_action\_probs\(self, obs\)

이 함수는 obs에 대한 모든 action에 대한 확률값을 얻기 위한 함수입니다.

```python
def get_action_prob(self, obs):
    return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})
```

get\_action\_prob 함수는 모든 액션에 대한 확률을 출력한다는 점에서 act함수와 차이를 가집니다. 예를들어 상태\(obs\) \[1, 2\]에 대해서 get\_action\_prob는 \[0.3, 0.7\]을 출력합니다. act 함수는 stochastic이 True이면 0.3와 0.7의 확률 값에 의해 0 또는 1을 출력하며 False이면 가장 높은 확률 값을 가지는 action인 1을 출력합니다.

### get\_variables\(self\)

이 함수는 클래스 내에 존재하는 모든 파라미터들을 출력합니다.

```python
def get_variables(self):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
```

### get\_trainable\_variables\(self\)

이 함수는 클래스 내에 존재하는 학습가능한 모든 파라미터들을 출력합니다.

```python
def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
```

## ppo.py

전체 코드는 [다음](https://github.com/sc2-korean-level/MoveToBeacon/blob/master/4wayBeacon_ppo/ppo.py)을 참조하세요.

### Class Structure

이 파일은 Policy\_net.py 파일에서 정의한 네트워크를 학습하기 위한 클래스가 있으며 다음과 같은 구조를 가지고 있습니다.

```text
└── PPOTrain(Class)
    ├── __init__(Function)
    ├── train(Function)
    ├── get_summary(Function)
    ├── assign_policy_parameters(Function)
    └── get_gaes(Function)
```

### \_\_init\_\_\(self, Policy, Old\_Policy, gamma, clip\_value, c\_1, c\_2\)

이 함수는 처음 클래스를 정의할 때 호출이 되는 함수이며 주로 클래스 내부에서 필요한 함수나 변수들을 정의하는 곳입니다. Policy와 Old\_Policy는 클래스 Policy\_net의 객체로 각각 현재 네트워크 \($$\pi_{\theta}, V_\theta$$\) 와 이전 네트워크\( $$$\pi_{\theta old}, V_{\theta old}$$ \)를 뜻합니다. gamma는 일반적으로 강화학습에서 쓰이는 Bellman Equation에서의 감가율\( $$\gamma$$ \)와 [General Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)에서의 감가율\( $$\gamma$$ \) 두 가지 모두에 사용됩니다. clip\_value는 PPO 논문에서 나오는 $$\hat{E}_t[min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$ 에서의  $$\epsilon$$ 을 뜻합니다. c\_1과 c\_2는 각각 $$maximize\; L^{CLIP+VF+S}=\hat{E}_t[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta(s_t)]]$$ 에서의 $$c_1$$ 과 $$c_2$$ 를 뜻합니다.

PPOTrain클래스 내부의 \_\_init\_\_함수는 매우 길기에 조금씩 분할해서 설명을 하겠습니다. 

```python
def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):

    self.Policy = Policy
    self.Old_Policy = Old_Policy
    self.gamma = gamma

    pi_trainable = self.Policy.get_trainable_variables()
    old_pi_trainable = self.Old_Policy.get_trainable_variables()

    with tf.variable_scope('assign_op'):
        self.assign_ops = []
        for v_old, v in zip(old_pi_trainable, pi_trainable):
            self.assign_ops.append(tf.assign(v_old, v))
```

self.Policy는 입력받은 현재 네트워크 Policy를 해당 클래스의 내부 인스턴스로 재정의합니다. 또한 self.Old\_Policy, self.gamma도 마찬가지로 내부 인스턴스로 재정의합니다.  
내부 인스턴스로 재정의한 self.Policy와 self.Old\_policy 네트워크의 파라미터들을 pi\_trainable과 old\_pi\_trainable로 정의합니다.  
with tf.variable\_scope\('assign\_op'\)는 target 네트워크의 파라미터들\(old\_pi\_trainable,  $$\theta_{old}$$ \)을 main 네트워크의 파라미터\(pi\_trainable,  $$\theta$$ \)로 덮어쓰는 것입니다.

```python
with tf.variable_scope('train_inp'):
    self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
    self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
    self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
    self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
```

학습에 사용할 실제 선택한 action, rewards\( $$r^s_a$$ \), v\_preds\_next\( $$V(s_{t+1})$$ \), gaes를 각각 tf.placeholder로 받아오는 변수들입니다.

```python
act_probs = self.Policy.act_probs
    act_probs_old = self.Old_Policy.act_probs

    # probabilities of actions which agent took with policy
    act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
    act_probs = tf.reduce_sum(act_probs, axis=1)

    # probabilities of actions which agent took with old policy
    act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
    act_probs_old = tf.reduce_sum(act_probs_old, axis=1)
```

Policy\_net 클래스로부터 정해진 self.Policy와 self.Old\_Policy로 부터 $$\pi_\theta(s_t)$$ , $$a = b$$ 를 act\_probs와 act\_probs\_old로 정의합니다. 첫번째로 정해진 act\_probs와 act\_probs\_old는 각 액션에 대한 모든 확률값을 출력으로 가집니다. 하지만 실제 파라미터를 업데이트할 때에는 실제 선택한 action에 대한 확률값을 이용하여 계산하므로 위에서 정의한 self.actions와 확률값을 가지는 act\_probs를 곱연산하여 실제로 선택한 action의 확률값만을 가지도록 act\_probs를 재정의합니다.아래의 act\_probs\_old도 같은 과정을 가집니다.

```python
with tf.variable_scope('loss/clip'):
    ratios = tf.exp(tf.log(act_probs) - tf.log(act_probs_old))
    clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
    loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
    loss_clip = tf.reduce_mean(loss_clip)
    tf.summary.scalar('loss_clip', loss_clip)
```

ratios는 PPO논문에서 볼 수 있는 $$r_t(\theta)=(\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta old}(a_t|s_t)})$$ 를 뜻합니다. $$clip(r_t(\theta), 1-\epsilon, 1+\epsilon)$$ 를 구하여 clipped\_ratios를 정의합니다. 이 중 ratios와 clipped\_ratios를 각각 $$\hat{A}_t$$ 와 곱연산을 한 후 작은 값을 선택합니다. 그리고 그 둘중 작은 값을 loss\_clip으로 정의합니다. 이는 결국 $$min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)$$ 을 뜻합니다. 그 후 tf.reduce\_mean을 통해 마지막으로 loss\_clip을 $$\hat{E}_t[min(r_t(\theta)\hat{A}_t, clip(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$ 로 재정의 합니다. 결국 loss\_clip은 $$L^{CLIP}$$ 을 뜻합니다.

```python
with tf.variable_scope('loss/vf'):
    v_preds = self.Policy.v_preds
        loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
        loss_vf = tf.reduce_mean(loss_vf)
    tf.summary.scalar('loss_vf', loss_vf)
```

v\_preds는 $$V(s)$$ 를 뜻합니다. loss\_vf는 $$(r^s_a + \gamma V_\theta(s_{t+1})-V_\theta(s_t))^2$$ 을 뜻하며 tf.reduce\_mean을 통해 $$\hat{E}$$ 를 씌워 loss\_vf를 결국 $$L^{VF}$$ 를 뜻합니다.

```python
with tf.variable_scope('loss/entropy'):
    entropy = -tf.reduce_sum(self.Policy.act_probs * tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
    entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
    tf.summary.scalar('entropy', entropy)
```

현재 Policy Gradient 기법들에 자주 사용되는 Exploration 방법입니다. 네트워크의 출력으로 나오는 확률 값들의 cross-entropy를 최대화하는 방향으로 학습합니다. 이것은 더 좋은 방법이 있을 수 있는 경로를 탐색하기 위해 필요한 부분입니다. 마지막으로 entropy는 논문에서 $$S[\pi_\theta(s_t)]$$ 를 뜻합니다.

```python
with tf.variable_scope('loss'):
    loss = loss_clip - c_1 * loss_vf + c_2 * entropy
    loss = -loss  # minimize -loss == maximize loss
    tf.summary.scalar('loss', loss)
```

이는 위에서 구한 $$L^{CLIP}$$, $$L^{VF}$$, $$S[\pi_\theta(s_t)]$$ 를 하나로 합쳐 $$L^{CLIP+VF+S}=\hat{E}_t[L_t^{CLIP}(\theta)-c_1L_t^{VF}(\theta)+c_2S[\pi_\theta(s_t)]]$$ 를 뜻합니다. $$loss = -loss$$ 를 넣은 이유는 $$tensorflow$$ 에서는 Gradent-Ascent는 지원하지 않고 Gradient-Descent만을 지원하기에 부호를 바꾸어 최소화 하는 방법으로 실질적으로는 Gradient-Ascent를 구현합니다.

```python
self.merged = tf.summary.merge_all()
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
self.train_op = optimizer.minimize(loss, var_list=pi_trainable)
```

이는 모든 변수들을 $$tensorboard$$ 에서 보기 위해 tf.summary.merge\_all\(\)을 통해 하나로 보여주기 위함이며 optimizer를 통해 위에서 구한 loss를 최소화하는 방향으로 학습합니다.

### train\(self, obs, actions, rewards, v\_preds\_next, gaes\)

이 함수는 학습을 진행하기 위해 사용됩니다. 입력으로 학습을 진행할때 필요한 obs\( $$s_t$$ \), actions\( $$a_t$$ \), rewards\( $$r_s^a$$ \), v\_preds\_next\( $$V(s_{t+1})$$ \), gaes\( $$\hat{A}_t$$ \)를 받습니다.

```python
def get_summary(self, obs, actions, rewards, v_preds_next, gaes):
    return tf.get_default_session().run([self.merged], feed_dict={self.Policy.obs: obs,
                                                                      self.Old_Policy.obs: obs,
                                                                      self.actions: actions,
                                                                      self.rewards: rewards,
                                                                      self.v_preds_next: v_preds_next,
                                                                      self.gaes: gaes})
```

### assign\_policy\_parameters\(self\)

이 함수는 \_\_init\_\_에서 정의한 target 네트워크에 main 네트워크의 파라미터를 덮어쓰는 변수를 실행하기 위한 함수입니다.

```python
def assign_policy_parameters(self):
    # assign policy parameter values to old policy parameters
    return tf.get_default_session().run(self.assign_ops)
```

### get\_gaes\(self, rewards, v\_preds, v\_preds\_next\)

이 함수는 Bellman Equation에서 얻은 가치의 값을 이용해 General Advantage Estimation을 수행하는 함수합니다. 구체적으로 General Advantage Estimation에 대해 설명하지는 않지만 본 코드에서는 $$TD(\lambda)$$ 를 1로 하며 $$\gamma$$ 도 0.99로 설정합니다.

```text
def get_gaes(self, rewards, v_preds, v_preds_next):
    deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
    return gaes
```

먼저 Bellman Equation의 값들을 delta\( $$\delta_t$$ \)에 정의합니다. 그 후 $$\hat{A}_t = \delta_t+\gamma\delta_{t+t}+\dots+\gamma^{T-t+1}\delta_{T-1}$$ 을 수행하여 gaes\( $$\hat{A}_t$$ \)를 정의합니다.

