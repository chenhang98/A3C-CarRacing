from matplotlib import pyplot as plt
import csv
import numpy as np 

def moving_average(arr, alpha = 0.05):
	newarr = []
	temp = 0

	for v in arr:
		temp = (1 - alpha) * temp + alpha * v
		newarr.append(temp)

	return newarr


rs, ls = [], []
cr = csv.reader(open('bak/log.csv', encoding='utf-8'))

# jump title
next(cr)
# get rewards and epsiode length
for row in cr:
    rs.append(float(row[-2]))
    # ls.append(float(row[-1]))

# plot
# plt.subplot(211)
plt.plot(rs, label = "raw reward", color = 'g')
plt.plot(moving_average(rs, 0.05), label = "moving average", color = 'r')

plt.plot(750 * np.ones(len(rs) + 200), label = "human level estimate", color = "c", linestyle='dashed')
plt.plot(950 * np.ones(len(rs) + 200), label = "max score estimate", color = "m", linestyle='dashed')

plt.xlabel("training time / 100s")
plt.ylabel("episode reward")
plt.legend()

# plt.subplot(212)
# plt.plot(ls, label = "epsiode length")
# plt.legend()

plt.show()