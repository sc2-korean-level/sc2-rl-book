# 4WayBeacon A2C Script Review

전체 코드는 [다음](https://github.com/sc2-korean-level/MoveToBeacon/blob/master/4wayBeacon_a2c/ex.py)을 참고하세요

이번 코드는 대부분 4WayBeacon PPO Script Review와 대부분 같은 코드를 사용하고 있으며 몇가지 다른 점에 대해서만 설명을 하겠습니다.

```python
states = np.empty(shape=[0, 2])
actions_list = np.empty(shape=[0, 4])
next_states = np.empty(shape=[0, 2])
rewards = np.empty(shape=[0, 1])
```

학습에 필요한 정보들을 저장한 빈 array를 만듭니다.

```python
marine_y, marine_x = (obs[0].observation.feature_screen.base[5] == 1).nonzero()
beacon_y, beacon_x = (obs[0].observation.feature_screen.base[5] == 3).nonzero()
marine_x, marine_y, beacon_x, beacon_y = np.mean(marine_x), np.mean(marine_y), np.mean(beacon_x), np.mean(beacon_y)
state = [marine_x*10/63 - beacon_x*10/63,  marine_y*10/63 - beacon_y*10/63]
```

마린의 좌표, 비콘의 좌표를 구하여 marine\_x, marine\_y, beacon\_x, beacon\_y의 변수에 각각 저장합니다. 그리고 상대좌표를 구한 후 의미있는 값을 만든 후\(좌표 X 10 / 63\) state에 상태값을 저장합니다.

```python
if global_step == 200 or distance < 0.3: done = True
if global_step == 200: reward = -1
if distance < 0.3: reward = 0            
states = np.vstack([states, state])
next_states = np.vstack([next_states, next_state])
rewards = np.vstack([rewards, reward])
action = np.zeros(4)
action[act] = 1
actions_list = np.vstack([actions_list, action])
```

한 에피소드가 끝나는 조건입니다. 특정 시간이 지나거나 비콘에 다다랐을 경우 에피소드를 종료합니다. 에피소드가 특정 시간이 지나서 종료된 것\(global\_step == 200\)과  비콘에 다다랐을 경우\(distance &lt; 0.3\)을 정의하고 그에 맞는 Reward를 배정합니다. 스텝이 진행되면서 나온 현재 상태, 다음 상태, Reward, 행동 값을 저장합니다. choose\_action에 의해 나온 행동 값은 0~3을 가지지만 학습에 필요한 형태인 one-hot vector형태로 바꾸어 action\_list에 쌓습니다.

```python
if done:
    A2C.learn(states, next_states, rewards, actions_list)
    saver.save(sess, "4wayBeacon_a2c/tmp/model.ckpt")
    print(sum(rewards), episodes)
    open_file_and_save('4wayBeacon_a2c/reward.csv', [sum(rewards)])                
```

에피소드가 종료되을 경우 정책 네트워크와 가치 네트워크를 학습하고 그 모델

