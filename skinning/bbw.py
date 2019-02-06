from numpy import *
import scipy.sparse.linalg

def cvxopt_sparse( scipy_sparse_matrix ):
    '''
    Converts a scipy.sparse matrix to a cvxopt sparse matrix.
    '''
    import cvxopt
    P = scipy.sparse.coo_matrix( scipy_sparse_matrix )
    M = cvxopt.spmatrix( P.data, asarray( P.row, dtype = int ), asarray( P.col, dtype = int ), P.shape )
    return M

def linear_blend_skin_2D( vertices, weights, transforms ):
    '''
    Given an array of 2D `vertices`,
    an array of #vertices-by-#transforms `weights`,
    and an array of 3x3 matrices `transforms`,
    returns the linear blend skinning deformation of each vertex.
    
    NOTE: The input 2D vertices are not homogeneous.
          They are just [ [ x0, y0 ], [ x1, y1 ], ... ].
          You must convert them to homogeneous coordinates to see the effects of translation.
          You must return non-homogeneous 2D coordinates.
    '''
    
    ## Add your code here.
    
    return array( vertices )

def laplacian_and_mass_matrices( faces, vertices ):
    '''
    Given `vertices`, an array of N 2D points pi = [xi, yi] (equivalently, an N-by-2 array),
    and `faces`, an array of F triplets of integer indices into vertices,
    where the triplet faces[f][0], faces[f][1], faces[f][2]
    are the indices of the three vertices that make up triangle f,
    return two N-by-N sparse matrices: Laplacian, Mass.
    '''
    
    ### 1 Create an N-by-N matrix A, where N is the number of vertices, that is initially zero.
    ### 2 Iterate over all edges (i,j), setting a 1 at the corresponding (i,j) and (j,i) location in A.
    ### 3 Create an N-by-N diagonal Mass matrix, where the i-th diagonal is the sum of the i-th row of A.
    ### 4 The Laplacian matrix is inverse( Mass )*(Mass - A). In other words,
    ###   it is (Mass-A) followed by dividing each row by its diagonal element.
    
    ### Add your code here.
    
    N = len(vertices)
    return scipy.sparse.eye(N), scipy.sparse.eye(N)

def bbw( faces, vertices, handles, laplacian_mode, solver_mode ):
    '''
    Given `faces` and `vertices` as would be passed to laplacian_and_mass_matrices(),
    an array of H integer indices into `vertices` representing the handle vertices,
    a string `laplacian_mode` which will be one of "graph" or "cotangent",
    and a string `solver_mode` which will be one of "bounded" or "unbounded",
    return an #vertices-by-#handles weight matrix W, where W[i,j] is the influence weight
    of the j-th handle on the i-th vertex.
    Each row of W must sum to 1.
    If mode == "bounded", apply inequality constraints so that the weights are
    all between 0 and 1.
    '''
    
    ### 1 Create the laplacian L and mass M matrices.
    ### 2 The bilaplacian B is L.T @ M @ L.
    ### 3 Create the constraint matrix. There will be an equality constraint for
    ###   every handle vertex, and 2 inequality constraints for the remaining vertices.
    ### 4 Solve once for each handle, setting each handles constraint value to 1 in turn.
    ### 5 Normalize each vertex's weights so that they sum to 1.
    
    ### Add your code here.
    N = len( vertices )
    H = len( handles )
    weights = (1./H)*ones( N, H )
    return weights
