import sys

class sort_string():
    def __init__(self, i_string):
        self.digit_val = []
        self.digit_pos = []

        self.alpha_val = []
        self.alpha_pos = []

        self.i_string = list(i_string)
        self.o_string = None

    def split_string(self):
        i_string_pos = 0

        for i in self.i_string:
            if i.isdigit():
                self.digit_val.append(i)
                self.digit_pos.append(i_string_pos)

            if i.isalpha():
                self.alpha_val.append(i)
                self.alpha_pos.append(i_string_pos)

            i_string_pos += 1

    def sort(self):
        self.digit_val.sort()
        self.alpha_val.sort()

    def merge_string(self):
        for i in range(len(self.digit_pos)):
            self.i_string[self.digit_pos[i]] = self.digit_val[i]

        for i in range(len(self.alpha_pos)):
            self.i_string[self.alpha_pos[i]] = self.alpha_val[i]

        self.o_string = "".join(self.i_string)


if __name__ == "__main__":
    i_string = sys.stdin.readline().strip('\n')
    sort_i_string = sort_string(i_string)
    sort_i_string.split_string()
    sort_i_string.sort()
    sort_i_string.merge_string()
    print(sort_i_string.o_string)