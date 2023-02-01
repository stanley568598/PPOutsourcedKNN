# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:57:26 2021

@author: 123
"""

import numpy as np
from secret_sharing import generateShares,reconstructSecret

t , n = 2 , 2

class RandomnessGenerator:
    
    def __init__(self):
        self.a = np.random.randint(10)
        self.b = np.random.randint(10)
        self.ab = self.a * self.b
        
        self.shareA = generateShares( t , n , self.a )
        self.shareB = generateShares( t , n , self.b )
        self.shareAB = generateShares( t , n , self.ab )
        
    def forA(self):

        res = [ self.shareA[0][1] , self.shareB[0][1] , self.shareAB[0][1] ]
        return res

    def forB(self):
        res = [ self.shareA[1][1] , self.shareB[1][1] , self.shareAB[1][1] ]
        return res
    
def multiply( a , b ):
    
    rg = RandomnessGenerator()
    r = rg.forA()
    _r = rg.forB()

    x1 = a[0][1] - r[0]
    x2 = a[1][1] - _r[0]
    
    y1 = b[0][1] - r[1]
    y2 = b[1][1] - _r[1]
    
    e = reconstructSecret( [ [ 3 , x1 ] , [ 5 , x2 ] ] )
    p = reconstructSecret( [ [ 3 , y1 ] , [ 5 , y2 ] ] )
    
    ab1 = r[2] + e * r[1] + p * r[0] + e * p
    ab2 = _r[2] + e * _r[1] + p * _r[0] + e * p
    
    ans = reconstructSecret( [ [ 3 , ab1 ] , [ 5 , ab2 ] ] )
    return ans
    
if __name__ == '__main__' :

    t , n = 2 , 2
    
    secret1 = 4.1
    secret2 = 6.1
    
    for i in range(10000) :                         # 做 10000 次 偵錯，輸出 1 為 使用 multiple 的 compare 錯誤。 
        
        ss1 = generateShares( t , n , secret1 )
        ss2 = generateShares( t , n , secret2 )
        # print( 'secret : ' , ss1 , ss2 )

        x1 = ss1[0][1]
        y1 = ss2[0][1]
        
        x2 = ss1[1][1]
        y2 = ss2[1][1]

        # print( 'shares : ' , x1 , x2 , y1 , y2 )

        l1 = np.random.randint( max( x1 , y1 ) , max( x1 , y1 ) + 20 )
        l2 = np.random.randint( max( x2 , y2 ) , max( x2 , y2 ) + 20 )
        # print( 'l : ' , l1 , l2 )
        
        z1 = x1 - y1 + l1
        z2 = x2 - y2 + l2
        #  print( 'z : ' , z1 , z2 )
        
        r = np.random.randint( 1 , 50 )
        _r = np.random.randint( 1 , 50 )
        # print( 'r : ' , r , _r )
        
        r_share = generateShares( t , n , r )
        _r_share = generateShares( t , n , _r )
        # print( 'r_share : ' , r_share , _r_share )

        r1 = r_share[0][1]
        r2 = r_share[1][1]
        zr = multiply( [ [ 3 , z1 ] , [ 5 , z2 ] ] , [ [ 3 , r1 ] , [ 5 , r2 ] ] )
        # print( 'zr : ' , zr )

        _r1 = _r_share[0][1]
        _r2 = _r_share[1][1]
        lr = multiply( [ [ 3 , l1 ] , [ 5 , l2 ] ] , [ [ 3 , r1 ] , [ 5 , r2 ] ] )
        # print( 'lr : ' , lr )

        s1 = zr + _r1
        s2 = zr + _r2
        # print( 's : ' , s1 , s2 )

        h1 = lr + _r1
        h2 = lr + _r2
        # print( 'h : ' , h1 , h2 )

        s = reconstructSecret( [ [ 3 , s1 ] , [ 5 , s2 ] ] )
        h = reconstructSecret( [ [ 3 , h1 ] , [ 5 , h2 ] ] )
        
        if( s > h ) :
            print(1)
        