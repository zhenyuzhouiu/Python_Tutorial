import sys
import numpy as np

class Solution():
    def __init__(self, k, n):
        self.k = k
        self.n = n

    def Keyboard(self):
        dp = np.zeros([self.n+1, 27])
        for j in range(27):
            dp[0][j] = 1
        for i in range(1, self.n+1):
            for j in range(1, 27):
                for x in range(self.k+1):
                    if (i-x) >= 0:
                        dp[i][j] += dp[i-x][j-1]*self.combination(i,x)

        return int(dp[n][26])

    def combination(self, i, x):
        k = 1
        ans = 1
        while x >= k:
            ans = ((i-k+1)*ans)/k
            k += 1
        return ans


    # def combination(self, i, x):
    #     return self.factorial(i)/(self.factorial(i-x)*self.factorial(x))
    #
    # def factorial(self, y):
    #     if y == 0 or y == 1:
    #         return 1
    #     else:
    #         return y * self.factorial(y-1)


if __name__ == "__main__":
    # k = int(input("Please input each key can pressed k times value: "))
    # n = int(input("Please input pressed times n: "))
    print("Please give value to k and n!")
    input = sys.stdin.readlines()

    k = int(input[0].strip('\n'))
    n = int(input[1].strip('\n'))
    solution = Solution(k, n)
    num = solution.Keyboard()
    print("There is {a} different output".format(a = num))