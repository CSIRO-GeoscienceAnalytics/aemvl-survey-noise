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

	for i in range(1,rows):
		print("I = " + str(i))
		cnl = data[:,i]
		for j in range(0,columns,step):
			if((j + windsize) < columns):
				wind = cnl[j:j + windsize]
				deltas = getDeltasFromList(wind)
				sd = stdev(deltas)
				#If Noisy
				if(sd > threshold):
					print("Noise Detected")		 			
					axes.plot(fid[j:j + windsize], wind, "-")
			else:
				break

# Creates a matrix of delta values
# Uses a window on the x and y axis
# For a given window compare the delta values on the y axis and get a value for standard deviation  
# Get a mean value of the standard deviations accross the x axis 
# Compare the mean value to a threshold to dectect if the window is "Noisy"
# There is also a "low threshold" which requires at least on of the channel lines to cross 
def detect_noise_multi(axes, data, lowThreshold=0.5, highThreshold=.1, windsize=20, step=10, channelStep=2, channelGroupSize=4):
	fid = data[:, 0]
	sig = data[:, 1:]

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
					axes.plot(fid[j:j + windsize], sig[j:j + windsize,l], "-")
			#Might be noise
			elif(avgsd > lowThreshold and hasChannelCrossOver(sig[j:j + windsize,i:i+cgs])):
				print("Low Noise Detected with crossover")
				for l in range(i,i+cgs):		 			
					axes.plot(fid[j:j + windsize], sig[j:j + windsize,l], "-")

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