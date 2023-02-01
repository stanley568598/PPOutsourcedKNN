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

    # x = [ 3 , 5 , 7 , ... ]
    return x

def generateShares( t , n , secret ) : 

    x = genX(n)

    # Split secret using SSS into n shares with threshold t 
    cfs = coeff( t , secret ) 
    # print( cfs )

    shares = [] 
    for i in range( 0 , n ) : 
        # r = random.randrange( 1 , field_size ) 
        shares.append( [ x[i] , round( polynom ( x[i] , cfs ) , 1 ) ] ) 

    return shares 
  

# Driver code  
if __name__ == '__main__': 
      
    # (t,n) sharing scheme 
    t , n = 2 , 2

    # Phase I : Generation of shares 
    secret = 55
    print( 'Original Secret:' , secret ) 
    shares_1 = generateShares( t , n , secret ) 
    print( 'Shares:' , *shares_1 )
    print( "Reconstructed secret:" , reconstructSecret( random.sample( shares_1 , t ) ) ) 
    # Phase II : Secret Reconstruction，Picking t shares randomly for reconstruction。

    secret = 105
    print()
    print( 'Original Secret:' , secret ) 
    shares_2 = generateShares( t , n , secret ) 
    print( 'Shares:' , *shares_2 )
    print( "Reconstructed secret:" , reconstructSecret( random.sample( shares_2 , t ) ) ) 

    shares = shares_1 + shares_2
    print()
    print( 'Shares:' , *shares )

    # Phase III : Combining shares with 2,2
    a_1 = shares_1[0][0]    
    b_1 = shares_2[1][0]   

    a_2 = 0
    a_2 += shares_1[0][1]
    a_2 += shares_2[0][1]

    b_2 = 0
    b_2 += shares_1[1][1]
    b_2 += shares_2[1][1]

    combine = [ [ a_1 , a_2 ] , [ b_1 , b_2 ] ]

    print()
    print( 'Combining shares:' , *combine )
    print( "Reconstructed secret:" , reconstructSecret( combine ) ) 

    # print( "Reconstructed secret:" , reconstructSecret([ [ 3 , a ] , [ 5 , b ] ]) ) 

    print()
