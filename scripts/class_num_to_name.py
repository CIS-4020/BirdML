# used to convert a class number from our training model to the actual name of the bird
def convertClassNumToClassName(classNum):

    with open("./processed_data/classes.txt", "r") as classFile:

        for line in classFile:

            splitLine = line.split(" ")

            _classNum = splitLine[0]

            if int(_classNum) == int(classNum):
                return " ".join(splitLine[1:]).strip()
            
    return "Could not determine class name."