#!/usr/bin/env python3

import matplotlib.pyplot as plt

out = open("./temp/sample", "w+")

dat = []
lines = open("./data/SPXL_1min_sample.txt", "r").readlines()
for line in lines:
    out.write(line.split(",")[4])
    dat.append(float(line.split(",")[4]))
    if line != lines[-1]:
        out.write("\n")
out.close()

plt.plot(dat)
plt.show()
