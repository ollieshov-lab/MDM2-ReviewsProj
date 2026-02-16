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
ax.plot(t, X[:, 0], 'b', label='Prey (x)')
ax.plot(t, X[:, 1], 'r', label='Predator (y)')
ax.set_xlabel('Time')
ax.set_ylabel('Population')
ax.legend()
plt.tight_layout()
if filename is None:
    plt.show()
else:
    fig.savefig(filename)

