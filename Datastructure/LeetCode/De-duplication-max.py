import sys
import numpy as np


def fuc():
    i_input = sys.stdin.readline().strip('\n')
    i_input = i_input.strip('[').strip(']')
    i_list = list(map(int, i_input.split(',')))

    i_array = np.array(i_list)
    max_num = i_array.max()
    max_pos = np.argwhere(i_array == i_array.max()).tolist()

    if len(max_pos) == 1:
        sub_max = sub_array_max(i_array)
        # convert 'an_integer' to a string
        digit_string = str(sub_max)
        # convert each character of 'digit_string' to an integer
        digit_map = map(int, digit_string)
        # convert 'digit_map' to a list
        digit_list = list(digit_map)
        print(digit_list)
    else:
        list_max = []
        for i in max_pos:
            sub_array = (np.delete(i_array, max_pos)).tolist()
            sub_max = sub_array_max(sub_array)
            list_max.append(sub_max)
        list_max.sort()
        # convert 'an_integer' to a string
        digit_string = str(list_max[-1])
        # convert each character of 'digit_string' to an integer
        digit_map = map(int, digit_string)
        # convert 'digit_map' to a list
        digit_list = list(digit_map)
        print(digit_list)


def sub_array_max(sub_array):
    max_list = []

    while len(sub_array) > 1:
        r_index = 0
        max_num = sub_array.max()
        while sub_array[r_index] != max_num:
            max_pos = sub_array.argmax()
            index_pos = (np.argwhere(sub_array == sub_array[r_index])).tolist()
            if len(index_pos) > 1:
                list_e = []
                for e in index_pos:
                    if e < max_pos:
                        list_e.append(e)
                sub_array = np.delete(sub_array, list_e)
            else:
                r_index += 1

        max_pos = sub_array.argmax()
        right_array = sub_array[0: max_pos+1].tolist()
        left_array = sub_array[max_pos+1:]
        max_list.append(right_array)
        sub_array = left_array

    max_list.append(sub_array.tolist())

    # convert list of list to a list
    max = []
    for i in max_list:
        for j in i:
            max.append(j)
    # convert a list of int to a int number
    strings = []
    for integer in max:
        strings.append(str(integer))
    a_string = "".join(strings)
    an_integer = int(a_string)
    return an_integer


if __name__ == "__main__":
    fuc()