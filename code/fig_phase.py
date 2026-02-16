from lotka import lotka_simulation
import matplotlib.pyplot as plt
import sys

args = sys.argv[1:]
if args:
    [filename] = args
else:
    filename = None

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)
t, X = lotka_simulation()
ax.plot(X[:, 0], X[:, 1], color='red')
ax.set_xlabel('Predator')
ax.set_ylabel('Prey')
plt.tight_layout()
if filename is None:
    plt.show()
else:
    fig.savefig(filename)

