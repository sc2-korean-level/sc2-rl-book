# 4WayBeacon PPO Code Review

## Policy\_net.py

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

