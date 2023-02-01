import numpy as np


class Node:
    
    def __init__( self , value , index = None ) :
        self.value = value
        self.index = index
        self.left = None
        self.right = None

    def set_left( self , left_node ) :
        self.left = left_node
    
    def set_right( self , right_node ) :
        self.right = right_node

        
class BinaryTree:
    
    def __init__( self , root ):
        self.root = root

    def traversal( self , node ):         # 遍歷

        if( node != None ):
            
            self.traversal( node.left )

            self.traversal( node.right )


    def findNearestGroup( self , value ):

        node = self.root

        while( node.index == None ):

            if( value > node.value ):
                if( node.right != None ):
                    node = node.right
                else:
                    node = node.left

            elif( value == node.value ):
                ld = abs( value - node.left.value )
                rd = abs( value - node.right.value )
                if( ld < rd ):
                    node = node.left
                else:
                    node = node.right

            else:
                if( node.left != None ):
                    node = node.left
                else:
                    node = node.right
                    
        return node.index

