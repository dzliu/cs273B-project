import csv

f = open("../../data/DEP/train_input_data_dep.txt", "r")
o = open("../../data/DEP/train_input_data_dep.csv", "w")
wr = csv.writer(o, quoting=csv.QUOTE_NONE)

for line in f:
    l = line.replace("\n", "")
    l = line.replace(" ", "")
    l = list(l)[1:-1]
    # print len(l)
    # l = map(int, l)
    wr.writerow(l)

