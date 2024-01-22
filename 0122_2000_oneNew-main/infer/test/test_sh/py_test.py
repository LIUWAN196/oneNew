import numpy as np
import sys

param_1 = sys.argv[1]
param_2 = sys.argv[2]
param_3 = sys.argv[3]

st = int(param_1)
ed = int(param_2)
test_mode = param_3

if test_mode == "branch_1":
    print("this is python, the barnch is branch_1 to generate input data")
    print("the param_1 is {}, param_2 is {}".format(param_1, param_2))

    # gen the input data
    in_bin = np.array(range(st, ed))
    input_data_s8 = in_bin.astype(np.int8)
    # dump input data
    input_data_s8.tofile("input_0.bin")

elif test_mode == "branch_2":
    print("this is python, the barnch is branch_2 to compare the correct of C")
    print("the param_1 is {}, param_2 is {}".format(param_1, param_2))

    input_from_bin = np.fromfile("input_0.bin", dtype=np.int8)
    sum_bin = np.sum(input_from_bin)
    print("the sum is {}".format(sum_bin))

else:
    print("please input the branch_1 or branch_2")
    sys.exit(0)


