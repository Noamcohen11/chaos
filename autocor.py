from statsmodels.graphics.tsaplots import plot_acf
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# Initialize empty lists to store data
import csv


def main():
    time = []
    theta1 = []
    speed1 = []

    with open(
        "/Users/noamcohen/Google Drive/My Drive/כאוס/test/4b.csv",
        "r",
    ) as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i < 3:
                i += 1
                continue
            time.append(float(row[0]))
            theta1.append(float(row[1]))
            speed1.append(float(row[2]))
    # Plot the autocorrelation function
    print(180 * max(theta1) / math.pi)
    plt.plot(time, theta1)
    plt.show()
    x = pd.plotting.autocorrelation_plot(speed1)

    # fig, ax1 = plt.subplots()
    x_vals = x.get_xticks()
    x.set_xticklabels([int(x * (time[1] - time[0])) for x in x_vals])

    x.plot()
    plt.xlabel(r"$\tau$ [secs]")

    x.yaxis.label.set_fontsize(18)
    x.xaxis.label.set_fontsize(18)
    plt.show()


if __name__ == "__main__":
    main()
