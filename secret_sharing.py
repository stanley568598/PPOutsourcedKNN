import random 
import numpy as np
from math import ceil 
from decimal import *

global field_size
field_size = 100

def computeDelta( s1 , s2 ):

    y1 = ( s1[0][1] - s2[0][1] )
    y2 = ( s1[1][1] - s2[1][1] )
    x = s1[0][0] - s2[1][0]

    r = ( y1 - y2 ) / x

    delta = y1 - s1[0][0] * r

    if( delta < 0 ):

        y1 = ( s2[0][1] - s1[0][1] )
        y2 = ( s2[1][1] - s1[1][1] )
    
        x = s1[0][0] - s2[1][0]
        
        r = ( y1 - y2 ) / x

        delta = y1 - s1[0][0] * r  

    return(delta)


def reconstructSecret( shares ): 
      
    # Combines shares using Lagranges interpolation 拉格朗日插值.  
    # Shares is an array of shares being combined.

    sums , prod_arr = 0 , [] 
      
    for j in range( len(shares) ) :

        xj , yj = shares[j][0] , shares[j][1]

        prod = Decimal(1) 
          
        for i in range( len(shares) ): 
            xi = shares[i][0] 
            if i != j : 
                prod *= Decimal( Decimal(xi) / ( xi - xj ) )  

        prod *= Decimal(yj) 
        sums += Decimal(prod) 

    return round( Decimal(sums) , 1 )     
    # return int( round( Decimal(sums) , 0 ) ) 
   
def polynom( x , coeff ): 
      
    # Evaluates a polynomial in x with coeff being the coefficient list 
    # 計算 x 中的多項式，其中 coeff 是係數列表

    return sum( [ x ** ( len(coeff) - i - 1 ) * coeff[i] for i in range( len(coeff) ) ] )
   
def coeff( t , secret ): 
    
    # Randomly generate a coefficient array for a polynomial with degree t-1 whose constant = secret
    # 隨機生成一個多項式的係數數組，次數為 t-1，其常數 = secret

    coeff = [ random.randrange( 0 , field_size ) for _ in range( t - 1 ) ]

    coeff.append( secret ) 

    return coeff 

def genX( n ) :
    x = []
    for i in range(n):
        x.append( 3 + i * 2 )
    return x

def generateShares( t , n , secret ) : 
    
    x = genX(n)

    # Split secret using SSS into n shares with threshold t 
    cfs = coeff( t , secret ) 
    shares = [] 
    # print( cfs )

    for i in range( 0 , n ) : 
        # r = random.randrange( 1 , field_size ) 
        shares.append( [ x[i] , round( polynom ( x[i] , cfs ) , 1 ) ] ) 

    return shares 
  

# Driver code  
if __name__ == '__main__': 
      
    # (t,n) sharing scheme 
    a = 0
    b = 0
    t , n = 2 , 2
    secret = 55
    print( 'Original Secret:' , secret ) 
   
    # Phase I : Generation of shares 
    shares = generateShares( t , n , secret ) 
    a += shares[0][1]
    b += shares[1][1]
    print( 'Shares:' , *shares )

    secret = 105
    print()
    print( 'Original Secret:' , secret ) 
    shares = generateShares( t , n , secret ) 
    print( 'Shares:' , *shares )

    a += shares[0][1]
    b += shares[1][1]

    # Phase II : Secret Reconstruction 
    # Picking t shares randomly for reconstruction 
    pool = random.sample( shares , t ) 
    print( '\nCombining shares: ' , *pool ) 
    print( "Reconstructed secret:" , reconstructSecret([ [ 3 , a ] , [ 5 , b ] ]) ) 

