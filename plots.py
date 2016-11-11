#!/usr/bin/python2

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

import argparse

def plot():

    args = parse_arguments()
    destiny = args.destiny

    # Create folder if it doesn't exits
    if not os.path.isdir(destiny):
        os.makedirs(destiny)

    # Open data file
    iFile = open(args.data_file, 'rb')
    #reader = csv.reader(filter(lambda row: row[0]!='#', iFile))
    reader = csv.reader(iFile)

    game_score_list = []
    episode_number_list = []

    score_column = 5
    episode_column = 0

    # Read values from csv file
    for row in reader:
        if row[1] == '1786': #TODO this is need of time, missing KO case
            game_score_list.append(int(row[score_column]))
            episode_number_list.append(int(row[episode_column]))

    iFile.close()

    episode_number = np.array(episode_number_list)
    game_score = np.array(game_score_list)

    if args.plot_histogram == True:
        plot_histogram(game_score)

    plot_frame_skip(episode_number, game_score, destiny)


def plot_histogram(data):

    fig, ax = plt.subplots()

    # histogram our data with numpy
    n, bins = np.histogram(data, 50)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n


    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(
        barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.show()

# plot average difference / frame skip opponent graphs with fixed frame skip of frame_skip
def plot_frame_skip(x_list, y_list, destiny):

    plt.plot(x_list, y_list, 'ro', label='Game Score')

    legend = plt.legend(loc='lower left', shadow=True)
    plt.xlabel('frame skip opponent (frames)')
    plt.ylabel('average result difference (points)')
    plt.title('Evolution of the game score')
    plt.grid(True)

    plt.savefig("%s/game_score_evolution.png" % destiny)
    plt.close()

def parse_arguments():

    parse = argparse.ArgumentParser()

    parse.add_argument("data_file", type=str, help="data file in csv format")
    parse.add_argument("--destiny", type=str, default="./plot", help="real path to the destiny folder to save the plots, if it doesn't exists it will be created")

    parse.add_argument("--histogram", dest="plot_histogram", action='store_true', default=False, help="Generate a histogram")

    #Parse the arguments
    args = parse.parse_args()
    return args

# Execute the program
if __name__ == "__main__":
    plot()
