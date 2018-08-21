# Testing

$$\mathcal{L}_{\text{nstep-Q}} = \sum_\text{envs} \sum_{j=1}^n \left ( \sum_{k=1}^j [\gamma^{j-l}r_{t+n-k}] + \gamma^j \max_{a'} Q(s_{t+n}, a' , \theta^{-}) - Q (s_{t+n-j}, a_{t+n-j},\theta)\right)^2.$$hello

