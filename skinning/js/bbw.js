"use strict";

/*
When debugging:

numeric.prettyPrint( x );

// To print only non-zeros of a dense matrix B:
for( var row = 0; row < B.length; ++row ) for( var col = 0; col < B[0].length; ++col ) if( Math.abs( B[row][col] ) > 1e-5 ) console.log( "( " + row + ", " + col + " ): " + B[row][col] );
// This can be written succinctly as:
console.log( numeric.transpose( numeric.ccsGather( numeric.ccsSparse( B ) ) ) );
*/

// Patch numeric to add a sparse matrix transpose routine.
numeric.ccsTranspose = function( A )
{
    var rows_cols_vals = numeric.ccsGather( A );
    // Swap the rows and cols.
    return numeric.ccsScatter( [ rows_cols_vals[1], rows_cols_vals[0], rows_cols_vals[2] ] );
}
// Patch numeric to add a sparse matrix diagonal routine.
numeric.ccsDiag = function( diag )
{
    var ij = [];
    for( var i = 0; i < diag.length; ++i ) ij.push( i );
    return numeric.ccsScatter( [ ij, ij, diag ] );
}
// Patch numeric to add a function to extract the diagonal of a CCS sparse matrix.
numeric.ccsGetDiag = function( A )
{
    var N = A[0].length-1;
    
    var diag = Array(N);
    for( var col = 0; col < N; ++col ) diag[col] = A[2][ A[0][col] ];
    return diag;
}
// Patch numeric to add a sparse matrix identity.
numeric.ccsIdentity = function( N )
{
    return numeric.ccsDiag( numeric.rep( [N], 1. ) );
}
// Patch numeric to add a sparse matrix column sum.
// NOTE: There is no sparse matrix row sum, because it would have to first compute the transpose each time.
numeric.ccsSumColumn = function( A, col )
{
    var sum = 0.;
    for( var k = A[0][col]; k < A[0][col+1]; ++k ) sum += A[2][k];
    return sum;
}
// Patch numeric to add a sparse matrix column max.
// NOTE: There is no sparse matrix row max, because it would have to first compute the transpose each time.
numeric.ccsMaxColumn = function( A, col )
{
    var max = A[2][ A[0][col] ];
    for( var k = A[0][col]+1; k < A[0][col+1]; ++k ) max = Math.max( A[2][k], max );
    return max;
}
// Patch numeric to get the maximum value in each column across all columns.
numeric.ccsMaxColumns = function( A, col )
{
    var num_columns = numeric.ccsDim(A)[1];
    var maxs = Array(num_columns);
    for( var col = 0; col < num_columns; ++col ) maxs[col] = numeric.ccsMaxColumn( A, col );
    return maxs;
}
// Patch numeric to add a function to compute the product of a CCS matrix and a dense 1D array as a column vector.
numeric.ccsDotVector = function( A, vec )
{
    // return numeric.transpose( numeric.ccsFull( numeric.ccsDot( A, numeric.ccsTranspose( numeric.ccsSparse( [vec] ) ) ) ) )[0];
    // return numeric.dot( numeric.ccsFull( A ), vec );
    return numeric.transpose( numeric.ccsFull( numeric.ccsDot( A, numeric.ccsColumnVector( vec ) ) ) )[0];
}
// Patch numeric to add a function to create a CCS column matrix from a 1D dense array.
numeric.ccsColumnVector = function( vec )
{
    var result = [ [ 0, vec.length ], Array(vec.length), numeric.clone(vec) ];
    for( var i = 0; i < vec.length; ++i ) result[1][i] = i;
    return result;
}
// Patch numeric to add a function to return a copy of a CCS matrix whose elements are
// the element-wise maximum of the matrix's value and a scalar.
numeric.ccsmax = function( scalar, A )
{
    var result = [ numeric.clone(A[0]), numeric.clone(A[1]), numeric.clone(A[2]) ];
    result[2] = numeric.max( scalar, result[2] );
    return result;
}

// Returns argmin_x 1/2 x^T Q x - x^T h such that x >= 0.
function Brand_Chen_qp( Q, h )
{
    /*
    r = max(diag(Q),max(-Q,[],2));
    s = ones(n,1);
    
    Qp = max(Q,0) + diag(r);
    
    Qm = max(-Q,0) + diag(r);
    hp = max(h,0) + s;
    
    hm = max(-h,0) + s;
    x = x0;
    
    for it = 1:max_iter
        x_prev = x
        x = x.*(hp+Qm*x)./(hm+Qp*x);
        if max(abs(x-x_prev)) <=tol
            return;
        end
    end
    */
    
    var n = h.length;
    var r = numeric.max( numeric.ccsGetDiag( Q ), numeric.ccsMaxColumns( numeric.ccsmul( -1., Q ) ) );
    var s = numeric.rep( [n], 1. );
    
    var Qp = numeric.ccsadd( numeric.ccsmax( 0., Q ), numeric.ccsDiag(r) );
    var Qm = numeric.ccsadd( numeric.ccsmax( 0., numeric.ccsmul( -1., Q ) ), numeric.ccsDiag(r) );
    var hp = numeric.add( numeric.max( h, 0. ), s );
    var hm = numeric.add( numeric.max( numeric.mul( -1., h ), 0. ), s );
    
    var x = numeric.rep( [n], .5 );
    var x_prev;
    
    var tol = 1e-7;
    var MAX_ITER = 1000;
    for( var it = 0; it < MAX_ITER; ++it )
    {
        x_prev = x;
        x = numeric.mul(
            x,
            numeric.div(
                numeric.add( hp, numeric.ccsDotVector( Qm, x ) ),
                numeric.add( hm, numeric.ccsDotVector( Qp, x ) )
                )
            );
        if( Math.max.apply( Math, numeric.abs( numeric.sub( x, x_prev ) ) ) <= tol ) {
            return { 'message': 'Converged after ' + it + ' iterations.', 'solution': x };
        }
    }
    
    console.error( "Maximum iterations reached." );
    return { 'message': 'Maximum iterations reached (' + MAX_ITER + ').', 'solution': x };
}

// From: http://stackoverflow.com/questions/15313418/javascript-assert
function assert(condition, message) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}

/*
Convert an array of arrays into a row-major flat buffer.
*/
function flatten_Float32Array( array ) {
    const nrows = array.length;
    const ncols = array[0].length;
    
    var result = new Float32Array( nrows*ncols );
    for( let row = 0; row < nrows; ++row ) {
        for( let col = 0; col < ncols; ++col ) {
            result[row*ncols+col] = array[row][col];
        }
    }
    return result;
}
function flatten_Int32Array( array ) {
    const nrows = array.length;
    const ncols = array[0].length;
    
    var result = new Int32Array( nrows*ncols );
    for( let row = 0; row < nrows; ++row ) {
        for( let col = 0; col < ncols; ++col ) {
            result[row*ncols+col] = array[row][col];
        }
    }
    return result;
}
/*
Expand a row-major flat buffer into an array of arrays with n rows.
*/
function inflate_Float32Array( data, nrows ) {
    let array = new Float32Array( data );
    
    if( array.length % nrows !== 0 ) console.error( "inflate_Float32Array() called but dimensions are impossible." );
    
    var ncols = array.length / nrows;
    var result = Array(nrows);
    for( let row = 0; row < nrows; ++row ) {
        result[row] = Array(ncols);
        for( let col = 0; col < ncols; ++col ) {
            result[row][col] = array[row*ncols+col]
        }
    }
    
    return result;
}

/*
Given a ccsSparse matrix Dmat and vector dvec for an expression in terms of unknowns x
    .5 x^T Dmat x - dvec^T x
and given two arrays, 'indices' and 'values', containing indices into x and corresponding
constrained values,
returns a modified [ Dmat, dvec ] such that the value constraints are incorporated
and the matrix stays symmetric.
*/
function update_quadratic_and_linear_with_equality_constraints( Dmat, dvec, indices, values )
{
    // Update the right-hand-side elements wherever a fixed degree-of-freedom
    // is involved in a row (the columns of constrained degrees-of-freedom).
    // Do this via:
    //     dvec_out = dvec - R*Dmat
    // where R is a row matrix containing the constrained values.
    // var R = numeric.ccsScatter( [ numeric.rep( [ indices.length ], 0. ), indices, values ] );
    // var dvec_out = numeric.sub( dvec, numeric.ccsFull( numeric.ccsDot( R, Dmat ) ) );
    var R = numeric.rep( [ dvec.length ], 0. );
    for( var i = 0; i < indices.length; ++i ) R[ indices[i] ] = values[i];
    var dvec_out = numeric.sub( dvec, numeric.ccsDotVector( numeric.ccsTranspose( Dmat ), R ) );
    
    // Set corresponding entries of the right-hand-side vector to the constraint value.
    for( var i = 0; i < indices.length; ++i ) dvec_out[ indices[i] ] = values[i];
    
    // Finally, zero the constrained rows and columns, and set the diagonal to 1.
    
    // Zero the constrained rows and columns.
    var D = numeric.rep( [ dvec.length ], 1. );
    for( var i = 0; i < indices.length; ++i ) D[ indices[i] ] = 0.;
    D = numeric.ccsDiag( D );
    var Dmat_out = numeric.ccsDot( D, numeric.ccsDot( Dmat, D ) );
    
    // Set the diagonal to 1.
    var D = numeric.rep( [ dvec.length ], 0. );
    for( var i = 0; i < indices.length; ++i ) D[ indices[i] ] = 1.;
    D = numeric.ccsDiag( D );
    Dmat_out = numeric.ccsadd( Dmat_out, D );
    
    return [ Dmat_out, dvec_out ];
}       
/*
Given `vertices`, an array of N 2D points pi = [xi, yi] (equivalently, an N-by-2 array),
and `faces`, an array of F triplets of integer indices into vertices,
where the triplet faces[f][0], faces[f][1], faces[f][2]
are the indices of the three vertices that make up triangle f,
return two N-by-N sparse matrices, [ Laplacian, Mass ].
*/
function laplacian_and_mass_matrices( faces, vertices )
{
    /// 1 Create an N-by-N matrix A, where N is the number of vertices, that is initially zero.
    /// 2 Iterate over all edges (i,j), setting a 1 at the corresponding (i,j) and (j,i) location in A.
    /// 3 Create an N-by-N diagonal Mass matrix, where the i-th diagonal is the sum of the i-th row of A.
    /// 4 The Laplacian matrix is inverse( Mass )*(Mass - A). In other words,
    ///   it is (Mass-A) followed by dividing each row by its diagonal element.
    
    /// Add your code here.
    var l,m,n,i=0,j=0,x=vertices.length;
    var P=[],Q=[],R=[],S=[];
	do{
		P[i]=[];
        Q[i]=[];
        R[i]=[];
        j=0;
		do{
			P[i][j]=0;
            Q[i][j]=0;
            R[i][j]=0;
            j++;
        }while(j<x);
        i++;
	}while(i<x);
    for(i=0;i<faces.length;i++){
        l=faces[i][0];
		m=faces[i][1];
		n=faces[i][2];	
		P[l][m]=1;
        P[l][n]=1;
		P[m][l]=1;
		P[m][n]=1;
		P[n][l]=1;
		P[n][m]=1;
    }
    for(i=0;i<x;i++){
        var temp=0;
        for(j=0;j<x;j++){
            temp=numeric.add(temp,P[i][j]);
            S[i]=temp;
        }
    }
    Q=numeric.diag(S);
    R=numeric.ccsFull(numeric.ccsDot(numeric.ccsSparse(numeric.inv(Q)),numeric.ccsSparse(numeric.sub(Q,P))));
    return [R,Q];
}
/*
Given `faces` and `vertices` as would be passed to laplacian_and_mass_matrices(),
an array of H integer indices into `vertices` representing the handle vertices,
a string `laplacian_mode` which will be one of "graph" or "cotangent",
and a string `solver_mode` which will be one of "bounded" or "unbounded",
return an #vertices-by-#handles weight matrix W, where W[i][j] is the influence weight
of the j-th handle on the i-th vertex.
Each row of W must sum to 1.
If mode === "bounded", apply inequality constraints so that the weights are
all between 0 and 1.
*/
async function bbw( faces, vertices, handles, laplacian_mode, solver_mode )
{
    /// Comment out this code.
    /*socket.send( "bbw " + faces.length + " " + vertices.length + " " + laplacian_mode + " " + solver_mode );
    socket.send( flatten_Int32Array( faces ) );
    socket.send( flatten_Float32Array( vertices ) );
    socket.send( new Int32Array( handles ) );
    
    var result = await socket.receive();
    result = inflate_Float32Array( result, vertices.length );
    return result;*/
    
    
    /// 1 Create the laplacian L and mass M matrices.
    /// 2 The bilaplacian B is L.T * M * L.
    /// 3 Create the constraint matrix. There will be an equality constraint for
    ///   every handle vertex, and 2 inequality constraints for the remaining vertices.
    /// 4 Solve once for each handle, setting each handles constraint value to 1 in turn.
    /// 5 Normalize each vertex's weights so that they sum to 1.
    
    /// Add your code here.
    var L=laplacian_and_mass_matrices(faces,vertices);    
    var U=L[0],V=L[1],f=[],vh=[],nh=[],a=[],b=[],c=[];
	var p_in=0,age_in=0,i,j,index=0;
    var fl=true,cons=[[]];;
    var Z=numeric.ccsFull(numeric.ccsDot(numeric.ccsDot(numeric.ccsSparse(numeric.transpose(U)),numeric.ccsSparse(V)),numeric.ccsSparse(U))); 
    var x=vertices.length;
    var y=handles.length;   
    i=0;
    do{
        f[i]=[];
        j=0;
        do{        
            f[i][j]=0;
            if(vertices[handles[i]]==vertices[j])
                f[i][j]=1;
            j++;
        }while(j<x);
        i++;
    }while(i<y);

    i=0;
    do{
        fl=true;
        j=0;
        do{
            if(vertices[i]==vertices[handles[j]])
                fl=false;
            j++;
        }while(j<y);
        if(fl==true){
            nh[index]=i;
            index++;
        }
        i++;
    }while(i<x);

    i=0;
	do{
		a[i]=[];
		if((i%2)!=0){
            j=0;
			do{
				a[i][j]=0;
				a[i][p_in]=-1;
                j++;
			}while(j<x);
		}
		else{
            j=0;
			do{
				a[i][j] = 0;
				if(vertices[nh[i]]==vertices[j]){
					a[i][j]=1;
					p_in=j;
				}
                j++;	
			}while(j<x);
		}
    i++;	
	}while(i<2*nh.length);

    i=0;
	do{
		if(i%2==0)
			b[i] = 0;
		else
			b[i]= -1;
        i++;		
	}while(i<2*nh.length);
	var cons=f;
	var g=f.length+a.length;
    i=f.length;
	do{
		cons[i]=a[age_in];
		age_in++;
        i++;
	}while(i<g);
    var cons_trans=numeric.transpose(cons);
    i=0;
	do{
		c[i] = 0;
        i++;
    }while(i<y);
    var h=c;
    var h_in=0;
	var hs=c.length+b.length; 
    i=h.length;
	do{
		h[i]=b[h_in];
		h_in++;
        i++;
	}while(i<hs);
    i=0;
	do{
		vh[i]=[];
        j=0;
		do{
			vh[i][j]=0;
            j++;
		}while(j<x);
        i++;
	}while(i<y);
    var d=numeric.rep([x],0);
    if(solver_mode=="bounded"){
        i=0;
		do{
			h[i] = 1;
			var res=numeric.solveQP(Z,d,cons_trans,h,y);
			var s=res.solution;
			vh[i]=s;
			h[i]=0;
            i++;
		}while(i<y);
	}else{
        i=0;
		do{
			c[i]=1;
			var a_trans=numeric.transpose(f);
			var res=numeric.solveQP(Z,d,a_trans,c,y);
			var s=res.solution;
            c[i]=0;
			vh[i]=s;
            i++;
		}while(i<y);	
	}
    var E=numeric.transpose(vh);
    i=0;
    do{
		var sum=0;
        j=0;
		do{
			sum=numeric.add(sum,E[i][j]);
            j++;
		}while(j<y);
		E[i]=numeric.div(E[i],sum);
        i++;
	}while(i<E.length);
    return E;
}

/*
Given an array of 2D `vertices`,
an array of #vertices-by-#transforms `weights`,
and an array of 3x3 matrices `transforms`,
returns the linear blend skinning deformation of each vertex.

NOTE: The input 2D vertices are not homogeneous.
      They are just [ [ x0, y0 ], [ x1, y1 ], ... ].
      You must convert them to homogeneous coordinates to see the effects of translation.
      You must return non-homogeneous 2D coordinates.
*/
async function linear_blend_skin_2D( vertices, weights, transforms )
{
    /// Comment out this code:
    //socket.send( "linear_blend_skin_2D " + vertices.length );
    //socket.send( flatten_Float32Array( vertices ) );
    //socket.send( flatten_Float32Array( weights ) );
    //socket.send( JSON.stringify( transforms ) );
    //var result = await socket.receive()
    //result = inflate_Float32Array( result, vertices.length );
    //return result;   
    /// Add your code here.
    var result=[],p=vertices.length,q=transforms.length,X,i=0,j=0;
    do{
        result[i]=[];
        j=0;
        do{
            result[i][j]=0;
            j++;
        }while(j<2);
        i++;
    }while(i<p)  
    i=0;
    do{
        var X=[[0,0,0],[0,0,0],[0,0,0]];
        j=0;
        do{
            X=numeric.add(X,numeric.mul(weights[i][j],transforms[j]));
            j++;
        }while(j<q);
        result[i]=numeric.dot(X,[vertices[i][0],vertices[i][1],1.0]);
        i++;
    }while(i<p);  
    return result;
}
