#!/usr/bin/python2

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
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
    
    plot_frame_skip(episode_number, game_score, destiny)

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

    #Parse the arguments
    args = parse.parse_args()
    return args

# Execute the program
if __name__ == "__main__":
    plot()
