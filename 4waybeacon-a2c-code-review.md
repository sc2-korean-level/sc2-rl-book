# 4WayBeacon A2C Code Review

## a2c.py

전체 코드는 [다음](https://github.com/sc2-korean-level/MoveToBeacon/blob/master/4wayBeacon_a2c/a2c.py)을 참조하세요.

### Class Structure

이 파일은 a2c라는 클래스를 가지고 있으며 이 클래스는 다음과 같은 구조를 가지고 있습니다.

```text
└── Policy_net(Class)
    ├── __init__(Function)
    ├── learn(Function)
    ├── _build_net(Function)
    └── choose_action(Function)
```

### \_\_init\_\_\(self, sess, exp\_rate\)

```python
def __init__(self, sess, exp_rate):
        self.sess = sess
        self.state_size = 2
        self.action_size = 4
        self.exp_rate = exp_rate

        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.a = tf.placeholder(tf.float32, [None, self.action_size])
        self.r = tf.placeholder(tf.float32, [None, 1])
        self.v_ = tf.placeholder(tf.float32, [None, 1])
        self.actor, self.critic = self._bulid_net()

        self.td_error = self.r + 0.99 * self.v_ - self.critic
        self.closs = tf.square(self.td_error)
        self.train_cop = tf.train.AdamOptimizer(0.0001).minimize(self.closs)

        self.log_lik = self.a * tf.log(self.actor)
        self.log_lik_adv = self.log_lik * self.td_error
        self.exp_v = tf.reduce_mean(tf.reduce_sum(self.log_lik_adv, axis=1))
        self.entropy = -tf.reduce_sum(self.actor * tf.log(self.actor))
        self.obj_func = self.exp_v + self.exp_rate * self.entropy
        self.loss = -self.obj_func
        self.train_aop = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
```

sess는 $$tensorflow$$ 에서 사용하는 tf.Session\(\)이며 exp\_rate는 Policy Gradient는 엔트로피를 이용해서 탐험을 하는데 그 정도를 정의하는 변수입니다. self.X는 상태를 받는 변수, self.a는 행동을 받는 변수, self.r은 Reward를 받는 변수, self.v\_는 다음 상태의 가치\( $$V_v(s')$$ \)를 받는 변수이며 self.actor와 self.critic은 각각 정책 네트워크와 가치 네트워크를 뜻합니다.

self.td\_error는 $$\delta = r+\gamma V_v(s') - V_v(s)$$ 를 뜻합니다. self.closs는 $$(r + \gamma V_v(s') - V_v(s))^2$$ 인 가치 네트워크의 최소화할 항을 뜻합니다.

self.log\_lik는 $$\pi_\theta(s,a)$$ , self.log\_lik\_adv는 $$log\pi_\theta(s,a)A(s,a)$$ , self.exp\_v는 $$E[log\pi_\theta(s,a)A(s,a)]$$ 를 뜻합니다. self.entropy는 정책 네트워크의 엔트로피이며 이를 최대화 시켜 행동이 하나에 집중되는 것을 막아 탐험을 수행하도록합니다. self.entropy와 self.exp\_v를 하나로 합쳐 self.obj\_func으로 지정한 후 이를 최대화 하는 방향으로 파라미터를 업데이트 합니다. 하지만 $$tensorflow$$ 에서는 최소화 하는 방향으로만 학습을 진행할 수 있습니다. 그렇기 때문에 반대로 최소화 하도록 하여 실질적으로는 최대화하도록 학습합니다.

### learn\(self, state, next\_state, reward, action\)

```python
def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _, _ = self.sess.run([self.train_cop, self.train_aop],
                          feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action})

```

학습에 필요한 모든 변수들을 입력으로 받습니다. v\_는 다음 상태\(s'\)를 입력으로 하여 가치 네트워크를 통해 다음 상태의 가치\( $$V_v(s')$$ \)를 뜻하게 됩니다. 다음 상태의 가치를 구한 후 해당하는 변수의 입력으로 하여 정책 네트워크와 가치 네트워크를 학습합니다.

### \_build\_net\(self\)

```python
def _bulid_net(self):
        layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.tanh)
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
        layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.tanh)
        layer_4 = tf.layers.dense(inputs=layer_3, units=4, activation=tf.tanh)
        actor = tf.layers.dense(inputs=layer_4, units=self.action_size, activation=tf.nn.softmax)

        layer_1 = tf.layers.dense(inputs=self.X, units=64, activation=tf.tanh)
        layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)
        layer_3 = tf.layers.dense(inputs=layer_2, units=30, activation=tf.tanh)
        critic = tf.layers.dense(inputs=layer_3, units=1, activation=None)

        return actor, critic
```

정책 네트워크와 가치 네트워크의 형태를 정의하는 함수입니다. 정책 네트워크는 5개의 은닉층으로 구성되어 있으며 각각 64, 64, 64, 4, 2의 노드를 가지고 있는 동시에 활성화 함수는 각각 tanh, tanh, tanh, tanh, softmax로 이루어져 있습니다. 가치 네트워크는 4개의 은닉층을 가지며 64, 64, 30, 1의 노드를 가지고 있습니다. 각 활성화 함수는 tanh, tanh, tanh, None으로 구성되어 있습니다.

### choose\_action\(self, s\)

```python
def choose_action(self, s):
        act_prob = self.sess.run(self.actor, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action
```

상태 값을 입력으로 받아 act\_prob로 저장합니다. 이는 $$\pi_\theta(s,a)$$ 를 뜻하며 각 행동에 대한 확률을 가지고 있습니다. 이 확률에 기반하여 하나의 행동을 선택하고 action에 저장합니다.

