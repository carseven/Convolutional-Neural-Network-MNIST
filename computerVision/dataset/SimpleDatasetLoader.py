import numpy as np
import cv2
import os


class SimpleDatasetLoader(object):
	"""docstring for SimpleDatasetLoader"""
	def __init__(self, preprocessors=None):
		self.preporcessors = preporcessors

		if self.preporcessors is None:
			self.preporcessors = []

	def load(self, imagePath, verbose=1):
		data = []
		labels = []

		for (i, imagePath) in enumerate(imagePath):
			image = cv2.imread(imagePath)
			label = imagePath.split(os.path.sep)[-2]

			if self.preprocessors is not None:
				for p in self.preprocessors:
					image = p.preprocess(image)	

			data.append(image)
			labels.append(label)

			if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
				print("[INFO] processed {}/{}".format(i + 1, len(imagePath)))

		return (np.array(data), np.array(labels))

		
