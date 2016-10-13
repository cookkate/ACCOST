import numpy as np

def readfile(filename, eliminator, dictionary, bin_size):
    # find the max length of chromosome
    max_length = 0
    for key in dictionary:
        max_length = max(max_length, int(dictionary[key]) )

    # construct numpy array with max length + 1
    # row 1 - count
    # row 2 - sum
    # row 3 - sum of square
    # each column stores the data of corresponding distance
    # last column stores data of 'no distance'
    array = np.zeros((3, max_length + 1))

    # loop through the array and update corresponding data
    input_file = open(filename, 'r')
    for lines in input_file:
        lines = lines[:-1]
        data = lines.split(eliminator)
        if data[0] != data[2]:
            index = max_length
        else:
            index = (int(data[3]) - int(data[1])) / bin_size
        array[0][index] += 1
        array[1][index] += int(data[4])
        array[2][index] += int(data[4]) ** 2

    # calculate the mean and variance
    # row 1 - mean
    # row 2 - variance
    result = np.zeros((2, max_length + 1))
    for x in range(0, max_length + 1):
        mean = array[1][x] / array[0][x]
        result[0][x] = mean
        result[1][x] = (array[2][x] / array[0][x]) - mean ** 2
    return result

# construct a dictionary storing the length of each chromosome
# may instead loop through and find the max
chr_size_file = open('chr_sizes.tab', 'r')
chr_size_file.readline()
dictionary = dict()
for line in chr_size_file:
    line = line[:-1]
    pair = line.split('\t')
    dictionary[pair[0]] = pair[1]

resultArray = readfile('TROPHOZOITES-XL', '\t', dictionary, 10000)
print str(resultArray)


