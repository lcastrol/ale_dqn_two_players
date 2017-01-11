"""
Thanks Josh Hemann for the example
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def loadGameScore(data_file):
    # Open data file
    iFile = open(data_file, 'rb')
    reader = csv.reader(filter(lambda row: row[0]!='#', iFile))
    #reader = csv.reader(iFile)

    game_score_list = []

    score_column = 5

    # Read values from csv file
    for row in reader:
        game_score_list.append(int(row[score_column]))

    iFile.close()

    return np.array(game_score_list)

set1 = loadGameScore("exp_0_1/boxing_1484055874.csv")
set2 = loadGameScore("exp_0_2/boxing_1484055874.csv")
set3 = loadGameScore("exp_0_3/boxing_1484055874.csv")

data = [set1, set2, set3]

# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax1 = fig.add_subplot(111)

# Create the boxplot
bp = ax1.boxplot(data, notch=0, sym='+', vert=1, whis=1.5, showmeans=True, meanline=True)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')

# Add a horizontal grid to the plot, but make it very light in color
# so we can use it for reading data values but not be distracting
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)

# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Game score distributions')
ax1.set_xlabel('Sample set')
ax1.set_ylabel('Differential score')

numBoxes = 3
medians = list(range(numBoxes))
for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    # Now draw the median lines back over what we just filled in
    med = bp['medians'][i]
    medianX = []
    medianY = []
    for j in range(2):
        medianX.append(med.get_xdata()[j])
        medianY.append(med.get_ydata()[j])
        plt.plot(medianX, medianY, 'k')
        medians[i] = medianY[0]
    # Finally, overplot the sample averages, with horizontal alignment
    # in the center of each box
    plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
             color='yellow', marker='*', markeredgecolor='k')

# Set the axes ranges (and axes labels)
ax1.set_xlim(0.5, numBoxes + 0.5)
top = 0
bottom = -30
ax1.set_ylim(bottom, top)

# Due to the Y-axis scale being different across samples, it can be
# hard to compare differences in medians across the samples. Add upper
# X-axis tick labels with the sample medians to aid in comparison
# (just use two decimal places of precision)
pos = np.arange(numBoxes) + 1
upperLabels = [str(np.round(s, 2)) for s in medians]
for tick, label in zip(range(numBoxes), ax1.get_xticklabels()):
    ax1.text(pos[tick], top - 1.5, upperLabels[tick],
             horizontalalignment='center', size='x-small', weight='bold',
             color='black')

# Save the figure
#plt.show()
plt.savefig("box_plot.png")
plt.close()
