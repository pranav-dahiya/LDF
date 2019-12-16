import os
import glob
import random

files = []
for i in range(1, 11):
    files.extend(glob.glob("lingspam/part"+str(i)+"/*.txt"))

random.shuffle(files)

split_index = int(0.9*len(files))

for index, file in enumerate(files):
    temp = file.split("/")
    new_file = temp[0] + "/lingspam_full/"
    if index < split_index:
        new_file += "train/"
    else:
        new_file += "test/"
    new_file += temp[-1]
    command = "cp " + file + " " + new_file
    os.system(command)
