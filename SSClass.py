# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:03:08 2021

@author: 123
"""
import random 
from decimal import Decimal

global field_size 
field_size = 100

def genX(n):
    x =[]
    for i in range(n):
        x.append( 3 + i * 2 )
    return x

class secretSharing:

    def __init__( self , secret , t , n ):
        self.secret = secret
        self.t = t
        self.n = n
        self.coeff = self.coeff()
        self.shares = self.generateShares()
        
    def coeff(self):
        coeff = [ random.randrange( 0 , field_size ) for _ in range( self.t - 1 )]
        coeff.append( self.secret ) 
        return coeff 
    
    def generateShares(self): 
        x = genX( self.n )
        # Split secret using SSS into n shares with threshold t 
        cfs = self.coeff
        # print(cfs)

        shares = [] 
        for i in range( 0 , self.n ): 
            # r = random.randrange( 1 , field_size ) 
            shares.append( [ x[i] , self.polynom( x[i] , cfs ) ] ) 
        return shares 
    
    def polynom( self , x , coeff ): 
        # Evaluates a polynomial in x with coeff being the coefficient list 
        return sum( [ x ** ( len(coeff) - i - 1 ) * coeff[i] for i in range( len(coeff) ) ] )
    
    def reconstructSecret(self): 
        # Combines shares using Lagranges interpolation.  
        # Shares is an array of shares being combined 
        sums , prod_arr = 0 , [] 
      
        for j in range( len(self.shares) ):

            xj , yj = self.shares[j][0] ,self.shares[j][1] 
            prod = Decimal(1) 
          
            for i in range( len(self.shares) ): 
                xi = self.shares[i][0] 
                if i != j : 
                    prod *= Decimal( Decimal(xi) / ( xi - xj ) )          
            prod *= Decimal(yj) 
            sums += Decimal(prod) 

        return round( Decimal(sums) , 1 )     

if __name__=='__main__':
    secret = 12.5
    ss = secretSharing( secret , 2 , 2 )
    print(ss.shares)
    