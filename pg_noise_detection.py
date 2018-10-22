# Pavel's Noise Detection code

import numpy as np
from collections import deque
from itertools import islice
import matplotlib.pyplot as plt
import scipy
from scipy import signal, optimize, stats

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
	

def run_by_channel(data, fn):
	for i in range(1, data.shape[1]):
		fn(data[:, np.r_[0, i]])