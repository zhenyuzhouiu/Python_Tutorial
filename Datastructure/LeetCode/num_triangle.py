import sys

n = int(sys.stdin.readline().strip('\n'))
a = [[0 for i in range(n)] for j in range(n)]
Maxsum = [[0 for i in range(n)] for j in range(n)]
for i in range(n):
    a_n = list(map(int, sys.stdin.readline().strip('\n').split()))
    for j in range(len(a_n)):
        a[i][j] = a_n[j]
Maxsum[n-1] = a[n-1]

for i in range(n-2, -1, -1):
    for j in range(i+1):
        if Maxsum[i+1][j] > Maxsum[i+1][j+1]:
            Maxsum[i][j] = Maxsum[i+1][j] + a[i][j]
        else:
            Maxsum[i][j] = Maxsum[i+1][j+1] + a[i][j]
print(Maxsum)
print(Maxsum[0][0])
