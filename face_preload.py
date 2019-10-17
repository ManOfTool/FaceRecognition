import face_recognition
import json
import numpy as np
import os

def loadFaces(face_src, k_encodings=[], k_names=[]):
	'''
	Load face from given directory as known data, or add new face to known data
	:param face_src: Directory contains known face images, or new known face with parent path
	:param k_encodings: A list contains face encodings
	:param k_names: A list contains face names
	:return: Returns a list of face encodings and a list of names
	'''

	# Get files in given directory
	face_src = os.path.abspath(face_src)
	face_list = []
	if os.path.isdir(face_src):
		# Load from directory
		face_list = [os.path.join(face_src, fp) for fp in os.listdir(face_src)]
	else:
		# Load single file
		face_list.append(face_src)

	for face_image in face_list:
		img = face_recognition.load_image_file(face_image)
		k_encodings.append(face_recognition.face_encodings(img, num_jitters=10)[0])
		k_names.append(os.path.splitext(os.path.split(face_image)[-1])[0])

	return k_encodings, k_names

def saveData(k_encodings, k_names, f_target):
	'''
	Given lists of known face encodings and known face names, save to file as json array
	:param k_encodings: A list contains face encodings
	:param k_names: A list contains face names
	:param f_target: saved file name
	'''

	# Convert numpy.ndarrays to python list
	for i in range(len(k_encodings)):
		k_encodings[i] = k_encodings[i].tolist()

	# Zip known faces and known encodings
	c = list(zip(k_names, k_encodings))

	# Save to file as json array
	with open(f_target, 'w') as f:
		f.write(json.dumps(c))

def loadPreload(f_target):
	"""
	Load known face encodings and name from given file
	:param f_target: The file contains json array of known face names and known face encodings
	:return: return known face encodings and known face names
	"""

	# Read json array from file
	with open(f_target, 'r') as f:
		m = f.read()

	# Deserialize string (json array) to python object (list)
	m = json.loads(m)

	# Seperate encodings and names
	k_encodings = []
	k_names = []
	for i in m:
		k_encodings.append(np.asarray(i[1])) # Convert python list to numpy.ndarray
		k_names.append(i[0])

	return k_encodings, k_names

if __name__ == '__main__':

	k_encodings, k_names = loadFaces('./known')
	print(k_encodings[0])
	print(k_names[0])

	saveData(k_encodings, k_names, 'preload_data_2.json')

	k_encodings, k_names = loadPreload('preload_data_2.json')
	print(k_encodings[0])
	print(k_names[0])