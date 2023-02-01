import numpy as np
from secret_sharing import generateShares, reconstructSecret

n,t = 2,2

# Mersenne Prime 4th 5th 7th
l_Min = 2**7 - 1
l_Max = 2**13 - 1
PRIME = 2**19 - 1

class RandomnessGenerator:

    def __init__(self):

        self.a = np.random.randint(10)
        self.b = np.random.randint(10)
        self.ab = self.a * self.b
        
        self.shareA = generateShares(n, t, self.a)
        self.shareB = generateShares(n, t, self.b)
        self.shareAB = generateShares(n, t, self.ab)
        
    def for_P1(self):       # share value in [3, a1], [3, b1], [3, ab1]
        res = [ self.shareA[0][1], self.shareB[0][1], self.shareAB[0][1] ]
        return res

    def for_P2(self):       # share value in [5, a2], [5, b2], [5, ab2]
        res = [ self.shareA[1][1] , self.shareB[1][1] , self.shareAB[1][1] ]
        return res
    
def multiply(x , y):

    RG = RandomnessGenerator()
    r = RG.for_P1()
    _r = RG.for_P2()

    # point_X1 = 3 , point_X2 = 5
    point_X1 = x[0][0]
    point_X2 = x[1][0]

    # [x-a]
    x1 = x[0][1] - r[0]
    x2 = x[1][1] - _r[0]
    
    # [y-b]
    y1 = y[0][1] - r[1]
    y2 = y[1][1] - _r[1]
    
    x_a = []
    y_b = []

    x_a.append([point_X1, x1])
    x_a.append([point_X2, x2])
    y_b.append([point_X1, y1])
    y_b.append([point_X2, y2])

    e = reconstructSecret(x_a)
    p = reconstructSecret(y_b)
    
    xy1 = r[2] + e*r[1] + p*r[0] + e*p
    xy2 = _r[2] + e*_r[1] + p*_r[0] + e*p

    xy = []
    xy.append([point_X1, xy1])
    xy.append([point_X2, xy2])
    
    ans = reconstructSecret(xy)

    return ans
    
if __name__=='__main__':

    print("\n====\n")

    n,t = 2,2
    
    secret1 = 4
    secret2 = 6
    print('Secret 1:' , secret1)
    print('Secret 2:' , secret2)

    ss1 = generateShares(n, t, secret1)
    ss2 = generateShares(n, t, secret2)

    SSM = multiply(ss1, ss2)
    print('SSM:' , SSM)

    print()

    # Test with comparison
    correct = True
    for i in range(100000):
        record = []
        
        ss1 = generateShares(n, t, secret1)
        ss2 = generateShares(n, t, secret2)
        
        x1 = ss1[0][1]
        y1 = ss2[0][1]
        
        x2 = ss1[1][1]
        y2 = ss2[1][1]
        
        l = np.random.randint(l_Min , l_Max)
        l_share = generateShares(n, t, l)
        l1 = l_share[0][1]
        l2 = l_share[1][1]

        record.append(['l:', l])
        
        z1 = x1 - y1 + l1
        z2 = x2 - y2 + l2

        r = np.random.randint(1 , 50)
        _r = np.random.randint(1 , 50)
        
        record.append(['r:', r])
        record.append(['_r:', _r])

        r_share = generateShares(n , t , r)
        _r_share = generateShares(n , t , _r)

        # point_X1 = 3 , point_X2 = 5
        point_X1 = r_share[0][0]
        point_X2 = r_share[1][0]
        
        # r1 = r_share[0][1]
        # r2 = r_share[1][1]
        
        _r1 = _r_share[0][1]
        _r2 = _r_share[1][1]

        z = []
        z.append([point_X1, z1])
        z.append([point_X2, z2])

        record.append(['z:', reconstructSecret(z)])

        zr = multiply(z , r_share)

        lr = multiply(l_share , r_share)
        
        s1 = zr + _r1
        s2 = zr + _r2
        
        h1 = lr + _r1
        h2 = lr + _r2

        record.append(['zr:', zr])
        record.append(['lr:', lr])
        
        S = []
        H = []

        S.append([point_X1, s1])
        S.append([point_X2, s2])
        H.append([point_X1, h1])
        H.append([point_X2, h2])

        s = reconstructSecret(S)
        h = reconstructSecret(H)
    
        if(s > h):
            correct = False
            print("Error: s={} h={}" .format(s,h))
            print("record: ", record)

    if(correct):
        print("Comparison all correct！")
    else:
        print("Some Comparison wrong！")

    print("\n====\n")
