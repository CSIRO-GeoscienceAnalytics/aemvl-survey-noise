import numpy as np
from statistics import stdev
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import signal, optimize, stats
from collections import deque
from itertools import islice
import plotly		

def sliding_window(iterable, size=2, step=1, fillvalue=None):
    if size < 0 or step < 1:
        raise ValueError
    it = iter(iterable)
    q = deque(islice(it, size), maxlen=size)
    if not q:
        return  # empty iterable or size == 0
    q.extend(fillvalue for _ in range(size - len(q)))  # pad to size
    while True:
        yield iter(q)  # iter() to avoid accidental outside modifications
        try:
            q.append(next(it))
        except StopIteration: # Python 3.5 pep 479 support
            return
        q.extend(next(it, fillvalue) for _ in range(step - 1))

def getDeltasFromList(list):
	if list.size < 2:
		raise ValueError
	
	deltaList = []

	for i in range(list.size-1):
		deltaList.append(list[i+1] - list[i])

	return deltaList

def getDeltasFromMatrix(matrix):
	if matrix.shape[0] < 2:
		raise ValueError
	
	deltaMatrix = np.zeros((matrix.shape[0]-1, matrix.shape[1]))

	for j in range(matrix.shape[1]):
		for i in range(matrix.shape[0]-1):
			deltaMatrix[i][j] = matrix[i][j] - matrix[i+1][j]

	return deltaMatrix

def hasChannelCrossOver(matrix):
	if matrix.shape[0] < 2:
		raise ValueError

	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]-1, 1, -1):
			if(matrix[i][j] > matrix[i][j-1]):
				return True
	
	return False

def read_line(f):
	df = pd.read_csv(f, sep=",")
	return df.as_matrix()


def plot_all(m):
	fid = m[:, 0]
	for i in range(1, m.shape[1]):
		plt.plot(fid, m[:, i], "-", color=".9")


def detect_method1(m, draw_lines=True, draw_fit_lines=True):

	FILTER_WINDOW = 151
	DEV_THRESHOLD = .4

	fid = m[:, 0]
	sig = m[:, 1]

	if draw_lines:
		for i in range(1, m.shape[1]):
			plt.plot(fid, m[:, i], "-")

	# Apply a Savitzky-Golay filter.
	savgol = signal.savgol_filter(sig, window_length=FILTER_WINDOW, polyorder=3, deriv=0)
	if draw_fit_lines:
		plt.plot(fid, savgol, "-.")

	# Compute gradient and its slope.
	grad = np.gradient(savgol)
	dy = np.roll(grad, -1) - grad
	dx = np.roll(fid, -1) - fid
	slope = np.sign(dy / dx)

	# plt.plot(fid, np.fabs(grad) * 1e2, "g-")

	ix_threshold = np.nonzero(np.fabs(sig - savgol) > DEV_THRESHOLD)
	ix_mask = np.zeros(fid.size)

	for ix in ix_threshold[0]:

		# Skip if already masked out.
		if ix_mask[ix]:
			continue

		# Search for slope inversion points.
		l = r = None
		for i in range(ix, 1, -1):
			if slope[i - 1] != slope[i] or ix_mask[ix]:
				l = i
				break
		for i in range(ix, fid.size):
			if slope[i + 1] != slope[i] or ix_mask[ix]:
				r = i
				break
		if l is None:
			l = 0
		if r is None:
			r = fid.size - 1

		if draw_fit_lines:
			plt.plot(fid[l], savgol[l], ">k", fid[r], savgol[r], "<k")
		ix_mask[l:(r + 1)] = 1

	# Plot mask.
	if np.count_nonzero(ix_mask):
		ix_mask[np.nonzero(ix_mask == 0)] = None
		plt.plot(fid, sig * ix_mask, "-m", linewidth=3.)


def detect_method2(m):

	fid = m[:, 0]
	sig = m[:, 1:]

	TIMES = [8.35e-05, 8.95e-05, 9.75e-05, 0.0001075, 0.0001205, 0.0001365, 0.0001565, 0.0001815, 0.0002125, 0.0002525,
		0.0003035, 0.0003665, 0.0004475, 0.0005485, 0.0006755, 0.0008365, 0.0010385, 0.0012935, 0.0016155, 0.0020215,
		0.0025325, 0.0031775, 0.0039905, 0.0050155, 0.0063075, 0.0079355, 0.0099895]

	window = sig[900:1000, :]
	# window = sig[:100, :]

	for i in range(0, window.shape[0]):
		plt.plot(TIMES, window[i, :], "-", color=".9")
	plt.xlim(0, .01)

	mmean = np.mean(window, axis=0)
	mstd = np.std(window, axis=0)

	SIGMA = 2
	plt.plot(TIMES, mmean, "-", color=".7")
	plt.plot(TIMES, mmean - SIGMA * mstd, "-r", linewidth=.7)
	# plt.plot(TIMES, mmean + SIGMA * mstd, "-r", linewidth=.7)

	mm = np.tile((mmean - SIGMA * mstd), (window.shape[0], 1))
	noisy = np.count_nonzero(np.where(window < mm)) != 0
	print(noisy)


def detect_method2_full(m, draw_lines=True, draw_fit_lines=True):

	fid = m[:, 0]
	sig = m[:, 1:]

	WINDOW_SIZE = 40
	SIGMA = 2

	TIMES = [8.35e-05, 8.95e-05, 9.75e-05, 0.0001075, 0.0001205, 0.0001365, 0.0001565, 0.0001815, 0.0002125, 0.0002525,
		0.0003035, 0.0003665, 0.0004475, 0.0005485, 0.0006755, 0.0008365, 0.0010385, 0.0012935, 0.0016155, 0.0020215,
		0.0025325, 0.0031775, 0.0039905, 0.0050155, 0.0063075, 0.0079355, 0.0099895]

	for it in sliding_window(range(m.shape[0]), size=WINDOW_SIZE, step=WINDOW_SIZE, fillvalue=-1):
		ix = list(it)

		# Take the last N elements for the last window.
		if np.count_nonzero(np.asarray(ix) < 0):
			ix = list(range(m.shape[0])[-WINDOW_SIZE:])

		print(ix)

		window = sig[ix, :]
		mmean = np.mean(window, axis=0)
		mstd = np.std(window, axis=0)

		mm = np.tile(mmean - SIGMA * mstd, (window.shape[0], 1))
		noisy = np.count_nonzero(np.where(window < mm)) != 0
		print(noisy)

		if noisy:
		 	for i in range(window.shape[1]):
		 		plt.plot(fid[ix], window[:, i], "-")

#Compares the standard deviation of each channel in the segment
#Helps eliminate the detection of the smooth waves but not great at detecting smaller noise
def detect_method3(m, threshold=2, windsize=10, step=2):
	fid = m[:, 0]
	
	columns = m.shape[0]
	rows = m.shape[1]

	print("Columns = " + str(columns))
	print("Rows = " + str(rows))

	for j in range(0,columns,step):
		if((j + windsize) < columns):
			#A segment with all channels
			segment = m[j:j + windsize,:]
			sdArray = []
			for i in range(1,rows):							
				sdArray.append(stdev(segment[:,i]))
			
			#If Noisy
			if(stdev(sdArray) > threshold):
				print("Noise Detected")		 			
				for i in range(1,rows):
					plt.plot(fid[j:j + windsize], segment[:,i], "-")
		else:
			break

# Calculate the smoothness (No comparison between channels but works well)
# Perhaps the ideal method given a single channel of data
# Prone to picking up small wave patterns
def detect_method4(m, threshold=0.5, windsize=10, step=2):
	fid = m[:, 0]
	
	columns = m.shape[0]
	rows = m.shape[1]

	print("Columns = " + str(columns))
	print("Rows = " + str(rows))

	for i in range(1,rows):
		print("I = " + str(i))
		cnl = m[:,i]
		for j in range(0,columns,step):
			if((j + windsize) < columns):
				wind = cnl[j:j + windsize]
				deltas = getDeltasFromList(wind)
				sd = stdev(deltas)
				#If Noisy
				if(sd > threshold):
					print("Noise Detected")		 			
					plt.plot(fid[j:j + windsize], wind, "-")
			else:
				break

# Creates a matrix of delta values
# Uses a window on the x and y axis
# For a given window compare the delta values on the y axis and get a value for standard deviation  
# Get a mean value of the standard deviations accross the x axis 
# Compare the mean value to a threshold to dectect if the window is "Noisy"
# There is also a "low threshold" which requires at least on of the channel lines to cross 
def detect_method4_full(m, lowThreshold=0.5, highThreshold=.1, windsize=20, step=10, channelStep=2, channelGroupSize=4):
	fid = m[:, 0]
	sig = m[:, 1:]

	deltaMatrix = getDeltasFromMatrix(sig)
	columns = deltaMatrix.shape[0]
	rows = deltaMatrix.shape[1]


	for j in range(0,columns,step):
		if((j + windsize) > columns):
			break
		
		cgs = channelGroupSize

		for i in range(0,rows,channelStep):

			if((i + channelGroupSize) > rows):
				cgs = rows - i
				#ignore the orphaned channels on the end
				break

			deltaArray = deltaMatrix[j:j+windsize,i:i+cgs]
			sdArray = []
			for k in range(0,windsize):
				sdArray.append(stdev(deltaArray[k,:].tolist()))

			avgsd = np.mean(sdArray)		

			#Definate noise
			if(avgsd > highThreshold):
				print("High Noise Detected")
				for l in range(i,i+cgs):		 			
					plt.plot(fid[j:j + windsize], sig[j:j + windsize,l], "-")
			#Might be noise
			elif(avgsd > lowThreshold and hasChannelCrossOver(sig[j:j + windsize,i:i+cgs])):
				print("Low Noise Detected with crossover")
				for l in range(i,i+cgs):		 			
					plt.plot(fid[j:j + windsize], sig[j:j + windsize,l], "-")
				




def run_by_channel(m, fn):
	for i in range(1, m.shape[1]):
		fn(m[:, np.r_[0, i]])


def main():
	FILENAME = "data/200301_hmx.csv"
	m = read_line(FILENAME)

	# Transform.
	m[:, 1:] = np.arcsinh(m[:, 1:])

	# Setup figure.
	fig = plt.figure(figsize=(18, 6), tight_layout=False)
	plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.05)
	plt.title(FILENAME)

	plot_all(m)

	# Get channels of interest.
	# detect_method1(m[:, np.r_[0, 12]])
	# detect_method2(m)

	# Detect all.
	# run_by_channel(m, lambda x: detect_method1(x, draw_lines=False, draw_fit_lines=False))
	# detect_method2_full(m)
	# detect_method3(m,.1, 20, 5)
	# detect_method4(m,.012, 20, 10)
	
	#Calibrated for non _hmx
	detect_method4_full(m,.003, .01, 20, 10, 2, 4)

  # Calibrated for _hmx using a high threshold works to detect the obvious noise
	#detect_method4_full(m,.03, .03, 20, 10, 2, 4)

	# Convert to Plotly.
	plotly.offline.plot_mpl(mpl_fig=fig, strip_style=False, show_link=False, auto_open=True)


if __name__ == "__main__":
	main()
