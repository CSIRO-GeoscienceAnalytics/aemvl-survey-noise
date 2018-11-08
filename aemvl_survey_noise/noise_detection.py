import time
from statistics import stdev
import numpy as np


# Calculate the smoothness (No comparison between channels but works ok)
# Perhaps the ideal method given a single channel of data
# Prone to picking up small wave patterns
def detect_noise_single(axes, data, threshold=0.5, windsize=10, step=2):
    fid = data[:, 0]

    columns = data.shape[0]
    rows = data.shape[1]

    print("Columns = " + str(columns))
    print("Rows = " + str(rows))

    for i in range(1, rows):
        print("I = " + str(i))
        cnl = data[:, i]
        for j in range(0, columns, step):
            if (j + windsize) < columns:
                wind = cnl[j : j + windsize]
                deltas = getDeltasFromList(wind)
                sd = stdev(deltas)
                # If Noisy
                if sd > threshold:
                    axes.plot(fid[j : j + windsize], wind, "-")
            else:
                break


# Creates a matrix of delta values
# Uses a window on the x and y axis
# For a given window compare the delta values on the y axis and get a value for standard deviation
# Get a mean value of the standard deviations accross the x axis
# Compare the mean value to a threshold to dectect if the window is "Noisy"
# There is also a "low threshold" which requires at least on of the channel lines to cross
def detect_noise_multi(
    axes,
    data,
    low_threshold=0.003,
    high_threshold=0.01,
    windsize=20,
    step=10,
    channel_step=2,
    channel_group_size=4,
):
    fid = data[:, 0]
    sig = data[:, 1:]

    delta_matrix = getDeltasFromMatrix(sig)

    columns = delta_matrix.shape[0]
    rows = delta_matrix.shape[1]

    for j in range(0, columns, step):
        if (j + windsize) > columns:
            break

        cgs = channel_group_size

        for i in range(0, rows, channel_step):

            if (i + channel_group_size) > rows:
                cgs = rows - i
                # ignore the orphaned channels on the end
                break

            delta_window = delta_matrix[j : j + windsize, i : i + cgs]
            avgsd = np.mean(delta_window.std(axis=1))

            # Definate noise
            if avgsd > high_threshold:
                print("High Noise Detected")
                for l in range(i, i + cgs):
                    axes.plot(fid[j : j + windsize], sig[j : j + windsize, l], "-")
                    # Might be noise
            elif avgsd > low_threshold and hasChannelCrossOver(
                sig[j : j + windsize, i : i + cgs]
            ):
                print("Low Noise Detected with crossover")
                for l in range(i, i + cgs):
                    axes.plot(fid[j : j + windsize], sig[j : j + windsize, l], "-")


def detect_noise_sections(
    data,
    low_threshold=0.003,
    high_threshold=0.01,
    windsize=20,
    step=10,
    channel_step=2,
    channel_group_size=4,
):
    sig = data[:, 1:]

    delta_matrix = getDeltasFromMatrix(sig)
    columns = delta_matrix.shape[0]
    rows = delta_matrix.shape[1]

    # Array of noise index ranges
    noise_array = []

    noise_start = None
    noise_end = None

    for j in range(0, columns, step):
        if (j + windsize) > columns:
            break

        cgs = channel_group_size
        noise = False
        for i in range(0, rows, channel_step):

            if (i + channel_group_size) > rows:
                cgs = rows - i
                # ignore the orphaned channels on the end
                break

            delta_window = delta_matrix[j : j + windsize, i : i + cgs]
            avgsd = np.mean(delta_window.std(axis=1))

            # If window is noisey
            if avgsd > high_threshold or (
                avgsd > low_threshold
                and hasChannelCrossOver(sig[j : j + windsize, i : i + cgs])
            ):

                noise = True
                if noise_start is None:
                    noise_start = j

                noise_end = j + windsize
                break

        # Record noise from previous window(s) if this window is not noisy or it is the last window
        if noise_start is not None and (
            noise is False or (j + windsize + step > columns)
        ):
            noise_array.append([noise_start, noise_end])
            noise_start = None
            noise_end = None

    return noise_array


def getDeltasFromList(data_list):
    if data_list.size < 2:
        raise ValueError

    deltaList = []

    for i in range(data_list.size - 1):
        deltaList.append(data_list[i + 1] - data_list[i])

    return deltaList


def getDeltasFromMatrix(matrix):
    if matrix.shape[0] < 2:
        raise ValueError

    delta_matrix = np.zeros((matrix.shape[0] - 1, matrix.shape[1]))

    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0] - 1):
            delta_matrix[i][j] = matrix[i][j] - matrix[i + 1][j]

    return delta_matrix


def hasChannelCrossOver(matrix):
    if matrix.shape[0] < 2:
        raise ValueError

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] - 1, 1, -1):
            if matrix[i][j] > matrix[i][j - 1]:
                return True

    return False
