import numpy as np
#index | weight     |   value
#------|------------|----------
# 0    |    15      |   2
# 1    |    13      |   5
# 2    |    13      |   9
# 3    |    12      |   5
# 4    |    12      |   9
# 5    |    11      |   7
#(weight, value)
items = [(15, 2), (13, 5), (13, 9), (12, 5), (12, 9), (11, 7)]
#Weight capacity
W = 60

def knapSack(W, wt, val, n):
    #Create table
    C = np.sum(val)
    x = np.zeros(shape=(n+1, C+1))
    x[:, 1:] = np.inf
    s = np.zeros(shape=(n, C+1))
    # Build table K[][] in bottom up manner
    for j in range(1, n+1):
        for k in range(val[j-1], C+1):
            if x[j-1, k-val[j-1]] + wt[j-1] <= min(W, x[j, k]):
                x[j, k] = x[j-1, k-val[j-1]] + wt[j-1]
                s[j-1, k] = 1
    x[n, x[n, :] == np.inf] = -np.inf
    k = np.argmax(x[n, :])
    items = []
    for j in reversed(range(n)):
        if s[j, k] == 1:
            items.append(j)
            k = k - val[j]
    return np.sum(np.array(val)[items]), items
wt, val = zip(*items)
n = len(items)
sack_val, sack = knapSack(W, wt, val, n)
print(sack_val, sack)