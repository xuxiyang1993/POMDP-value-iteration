import matplotlib.pyplot as plt
import numpy as np


# observation probability: O(o|s') = ob[o][s']
ob = ((0.9, 0.2), (0.1, 0.8))
# transition probability: T(s'|s, a) = t[s'][s][a]
t = (((0.9, 1), (0, 1)),
     ((0.1, 0), (1, 0)))
# reward: R(s, a) = r[s][a]
r = ((0, -5), (-10, -15))

# P(o|s, a) = p[o][s][a]
p = [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
for o in range(2):
    for s in range(2):
        for a in range(2):
            for ss in range(2):
                p[o][s][a] += t[ss][s][a] * ob[o][ss]
            p[o][s][a] = round(p[o][s][a], 2)


def beliefupdate(b0, a, o):
    b10 = ob[o][0]*( t[0][0][a]*(1-b0) + t[0][1][a]*b0 )
    b11 = ob[o][1]*( t[1][0][a]*(1-b0) + t[1][1][a]*b0 )
    b1 = b11/(b10 + b11)    # normalization
    return b1

def U(a, b, n):
    if n>1:
        u = r[0][a] * (1-b) + r[1][a] * b
        for s in range(2):
            for o in range(2):
                if s == 1:
                    if o == 0:
                        u += 0.9 * b * p[o][s][a] * (0.18*b+0.02)/(-0.63*b+0.83)*(-10)
                    elif o ==1:
                        u += 0.9 * b * p[o][s][a] * (0.72*b+0.08)/(0.63*b+0.17)*(-10)
                elif s==0:
                    if o == 0:
                        u += 0.9 * (1-b) * p[o][s][a] * (0.18*b+0.02)/(-0.63*b+0.83)*(-10)
                    elif o==1:
                        u += 0.9 * (1-b) * p[o][s][a] * (0.72*b+0.08)/(0.63*b+0.17)*(-10)
        return u
    else:
        return r[0][a] * (1-b) + r[1][a] * b


b = np.arange(0, 1, 0.05)
y = []
utility = []
for i in b:
    total = 0
    for s in range(1):
        for o in range(2):
            bb = beliefupdate(i, 0, o)
            v1 = bb*r[1][0] + (1-bb)*r[0][0]
            td = p[o][s][0]*v1
            total += td
    y.append(0)
    utility.append(U(0, i, 2))

inst = (1-b)*0 + b*(-10)
frac_a = (0.18*b+0.02)/(-0.63*b+0.83)
frac_b = (0.72*b+0.08)/(0.63*b+0.17)
t = (-8.3*(0.18*b+0.02)/(-0.63*b+0.83) - 1.7*(0.72*b+0.08)/(0.63*b+0.17))*(1-b) + (- 2*(0.18*b+0.02)/(-0.63*b+0.83) - 8*(0.72*b+0.08)/(0.63*b+0.17))*b
z = (0.9 * b * 0.2 * (0.18*b+0.02)/(-0.63*b+0.83)*(-10)+
     0.9 * b * 0.8 * (0.72 * b + 0.08) / (0.63 * b + 0.17) * (-10)+
     0.9 * (1-b) * 0.83 * (0.18*b+0.02)/(-0.63*b+0.83)*(-10)+
     0.9 * (1-b) * 0.17 * (0.72*b+0.08)/(0.63*b+0.17)*(-10)
     )

plt.plot(b, t, lw=2)
plt.xlabel('belief state')
plt.ylabel('update')
plt.show()

