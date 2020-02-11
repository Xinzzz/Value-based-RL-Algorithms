import matplotlib.pyplot as plt

fig = plt.figure()

fig.patch.set_facecolor('cyan')
fig.patch.set_alpha(0.7)

ax = fig.add_subplot(111)

ax.plot(range(10))

ax.patch.set_facecolor('red')
ax.patch.set_alpha(0.5)

# If we don't specify the edgecolor and facecolor for the figure when
# saving with savefig, it will override the value we set earlier!
fig.savefig('temp.png', facecolor='none', edgecolor='none')

plt.show()