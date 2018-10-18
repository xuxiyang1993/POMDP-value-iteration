# POMDP value iteration implementation

import matplotlib.pyplot as plt
import numpy as np
import time

# record running time
start_time = time.time()

# observation probability: O(o|s') = ob[o][s']
ob = ((0.9, 0.2),
      (0.1, 0.8))
# transition probability: T(s'|s, a) = t[s'][s][a]
t = (((0.9, 1), (0, 1)),
     ((0.1, 0), (1, 0)))
# reward: R(s, a) = r[s][a]
r = ((0, -5),
     (-10, -15))

# P(o|s, a) = p[o][s][a]
# P(o|s, a) = sum_s' (O(o|s')*P(s'|s,a))
#p = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
#for o in range(2):
 #   for s in range(2):
  #      for a in range(2):
   #         for ss in range(2):
    #            p[o][s][a] += t[ss][s][a] * ob[o][ss]
     #       p[o][s][a] = round(p[o][s][a], 2)


# we use b(s_1) to denote belief state, since b(s_0) + b(s_1) = 1
def belief_update(b0, a, o):
    """
    perform belief update, follows eqn(6.11) on page 137, Decision Making Under Uncertainty.
    :param b0: the belief before update
    :param a: the action we took
    :param o: the observance we observed
    :return: the belief after update
    """
    b10 = ob[o][0] * (t[0][0][a]*(1-b0) + t[0][1][a]*b0)
    b11 = ob[o][1] * (t[1][0][a]*(1-b0) + t[1][1][a]*b0)
    b1 = b11/(b10 + b11)    # normalization
    return b1


def q(b, a, n):
    """
    q value for belief b, action a
    this algorithm is based on eqn(7.7) and eqn(7.8) on Page 196, Markov Decision Processes in Artificial Intelligence
    :param b: belief
    :param a: action
    :param n: time horizon
    :return:
    """
    if n > 1:
        u = r[0][a] * (1-b) + r[1][a] * b
        for s in range(2):  # sum over current state
            for ss in range(2):  # sum over next state (ss denotes next state)
                for o in range(2):  # sum over observation
                    m = max(q(belief_update(b, a, o), 0, n - 1), q(belief_update(b, a, o), 1, n - 1))
                    if s == 1:
                        u += gamma * b * t[ss][s][a] * ob[o][ss] * m
                    else:
                        u += gamma * (1-b) * t[ss][s][a] * ob[o][ss] * m
        return u
    else:
        return r[0][a] * (1-b) + r[1][a] * b


gamma = 0.9  # discount factor

# sample distribution
b = np.linspace(0, 1, 100)
utility0 = []   # utility of action 0
utility1 = []   # utility of action 1

n = 3   # number of horizon
for i in b:
    utility0.append(q(i, 0, n))
for i in b:
    utility1.append(q(i, 1, n))

plt.plot(b, utility0, label='not feed')
plt.plot(b, utility1, label='feed')
plt.legend(loc='upper right')
plt.title('utility with horizon %s (belief)' % n)
plt.xlabel('probability of hungry')
plt.ylabel('utility')
plt.show()

print('--- %s seconds ---' % (time.time() - start_time))
