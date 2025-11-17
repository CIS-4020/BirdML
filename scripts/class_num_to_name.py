import os
# used to convert a class number from our training model to the actual name of the bird
def convertClassNumToClassName(classNum):

	PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	classes_path = os.path.join(PROJECT_ROOT, "processed_data", "classes.txt")

	with open(classes_path) as classFile:

		for line in classFile:

			splitLine = line.split(" ")

			_classNum = splitLine[0]

			if int(_classNum) == int(classNum):
				return " ".join(splitLine[1:]).strip()

	return "Could not determine class name."