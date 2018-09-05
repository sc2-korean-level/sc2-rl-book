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

### learn\(self, state, next\_state, reward, action\)

```python
def learn(self, state, next_state, reward, action):
        v_ = self.sess.run(self.critic, feed_dict={self.X: next_state})
        _, _ = self.sess.run([self.train_cop, self.train_aop],
                          feed_dict={self.X: state, self.v_: v_, self.r: reward, self.a: action})

```

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

### choose\_action\(self, s\)

```python
def choose_action(self, s):
        act_prob = self.sess.run(self.actor, feed_dict={self.X: [s]})
        action = np.random.choice(self.action_size, p=act_prob[0])
        return action
```

상태 값을 입력으로 받아 act\_prob로 저장합니다. 이는 $$\pi_\theta(s,a)$$ 를 뜻하며 각 행동에 대한 확률을 가지고 있습니다. 이 확률에 기반하여 하나의 행동을 선택하고 action에 저장합니다.

