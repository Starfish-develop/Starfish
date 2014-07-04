/* cov.h
 * Ian Czekala 2014
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <cholmod.h>

#define PI 3.14159265358979323846
#define c_kms 2.99792458E5
#define TRUE 1
#define FALSE 0

double k_3_2 (double r, double a, double l, double r0)
{
    //Hanning-tapered covariance kernel
    double taper = (0.5 + 0.5 * cos(PI * r/r0));
    return taper * a*a * (1 + sqrt(3) * r/l) * exp(-sqrt(3) * r/l);
}

double k_region (double x0, double x1, double a, double mu, double sigma)
{
    //Hanning-tapered covariance kernel

    //printf("x0 %.4f x1 %.4f\n" , x0, x1);
    //Hanning taper is defined using radial velocity distance from mu
    double rx0 = c_kms / mu * fabs(x0 - mu);
    double rx1 = c_kms / mu * fabs(x1 - mu);
    double r_tap = rx0 > rx1 ? rx0 : rx1; //choose the larger distance
    //double r_mu = sqrt((x0 - mu)*(x0 - mu) + (x1 - mu)*(x1 - mu));
    //printf("rx0 %.4f rx1 %.4f r_tap %.4f\n", rx0, rx1, r_tap);
    double r0 = 4.0 * sigma; //where the kernel goes to 0

    //You shouldn't be sending any values of r that are greater to this function, 
    //because otherwise you would be initializing a sparse matrix with a 0, adding in 
    //an unneccessary element.
    assert(r_tap <= r0);
    double taper = (0.5 + 0.5 * cos(PI * r_tap/r0));

    return  taper * a*a /(2. * PI * sigma * sigma) * exp(-0.5 *
        (c_kms * c_kms) / (mu * mu) * ((x0 - mu)*(x0 - mu) + 
            (x1 - mu)*(x1 - mu))/(sigma * sigma)); 
}


// function to initialize an array of wl
void linspace (double *wl, int N, double start, double end)
{       
    double j; //double index
    double Ndist = (double) N;
    double increment = (end - start)/(Ndist -1.);
    
    //initialize 
    int i;
    for (i = 0; i < N; i++) 
    {
        j = (double) i;
        wl[i] = start + j * increment;
    }
}

double get_min_sep (double *wl, int N)
{
    double min_sep, sep=0;
    int i;
    min_sep = (c_kms / wl[1]) * (wl[1] - wl[0]);
    for (i = 0; i < (N - 1); i++)
    {
        sep = (c_kms / wl[i+1]) * (wl[i+1] - wl[i]);
        if (sep < min_sep)
        {
            min_sep = sep;
        }
    }
    return min_sep;
}

// given an array of sigma values, square them and then stuff them into the
// diagonal of a sparse matrix
cholmod_sparse *create_sigma(double *sigma, int N, cholmod_common *c)
{
    cholmod_triplet *T = cholmod_allocate_triplet(N, N, N, 1, CHOLMOD_REAL, c);

    if (T == NULL || T->stype == 0)		    /* T must be symmetric */
    {
      cholmod_free_triplet (&T, c) ;
      cholmod_finish (c) ;
      return (0) ;
    }
    
    int * Ti = T->i;
    int * Tj = T->j;
    double * Tx = T->x;
    int k=0;
    int i=0;

    //k and i could serve the same purpose here as a (row, column) index
    //and a nnz index, but that would be 
    //confusing with later code

    for (i = 0; i < N; i++)
    {
        Ti[k] = i;
        Tj[k] = i;
        Tx[k] = sigma[i]*sigma[i];
        k++;
    }

    T->nnz = k;
    
    //The conversion will transpose the entries and add to the upper half.
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, k, c);
    cholmod_free_triplet(&T, c);
    return A;
}


// create and return a sparse matrix using a wavelength array and parameters
// for a covariance kernel.
cholmod_sparse *create_sparse(double *wl, int N, double min_sep, double a, double l, cholmod_common *c)
{
    double r0 = 6.0 * l;
    //calculate how big the matrix will be
    int i=0, j=0;
    int M=N; //How many non-zeros this matrix will have. We initialize
    //to the number of elements already on the diagonal
    //how many matrix indicies away will we need to go to get to r0?
    int max_offset = (int) ceil(r0/min_sep);
    //count all of the elements on the off-diagonals until we fill the matix
    for (i=1; (i<=max_offset) && (M < N*N); i++)
    {
        M += 2*(N - i);
    }

    //printf("N^2=%d \n", N*N);
    //printf("M  =%d \n", M);

    /* Initialize a cholmod_triplet matrix, which we will subsequently fill with
     * values. This matrix is NxN sparse with M total non-zero elements. 1 means we
     * want a square and symmetric matrix. */
    cholmod_triplet *T = cholmod_allocate_triplet(N, N, M, 1, CHOLMOD_REAL, c);

    if (T == NULL || T->stype == 0)		    /* T must be symmetric */
    {
      cholmod_free_triplet (&T, c) ;
      cholmod_finish (c) ;
      return (0) ;
    }
    
    int * Ti = T->i;
    int * Tj = T->j;
    double * Tx = T->x;
    int k = 0;
    double r;

    //Loop for the first block
    for (i = 0; (i < max_offset) && (i < N); i++)
    {
        for (j = 0; j <= i; j++)
        {
            //printf("(%d,%d)\n", i, j);
            r = fabs(wl[i] - wl[j]) * c_kms /wl[i]; //Now in velocity

                if (r < r0) //If the distance is below our cutoff, initialize
                {
                    Ti[k] = i;
                    Tj[k] = j;
                    Tx[k] = k_3_2(r, a, l, r0);
                    k++;
                }
        }
    }
    

    //Loop for the second block
    for (i = max_offset; i < N; i++)
    {
        for (j = (i - max_offset); j <= i; j++)
        {
            //printf("(%d,%d)\n", i, j);
            r = fabs(wl[i] - wl[j]) * c_kms/wl[i];

                if (r < r0) //If the distance is below our cutoff, initialize
                {
                    Ti[k] = i;
                    Tj[k] = j;
                    Tx[k] = k_3_2(r, a, l, r0);
                    k++;
                }

        }
    }

    T->nnz = k;
    
    //The conversion will transpose the entries and add to the upper half.
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, k, c);
    cholmod_free_triplet(&T, c);
    return A;
	
}


cholmod_sparse *create_sparse_region(double *wl, int N, double a, 
        double mu, double sigma, cholmod_common *c)
{
    //Create a sparse region in the covariance matrix, to be added/updated

    //determine the wl indices that are within 4 sigma of mu
    double r0 = 4.0 * sigma;
    int i = 0, j;
    int first_ind, last_ind;
    double r;

    //this loop should exit with first_ind being equal to the first row,col occurence of the region
    do {
        r = c_kms/mu * (mu - wl[i]); //how far away are we from mu?
        first_ind = i;
        i++;
    } while (r > r0);

    if (r <= 0.)
    {
        //if r had to go negative to escape the previous loop, then we know 
        //the sigma is too small for any region to exist on the current 
        //wavelength spacing, return null
        return NULL;
    }

    //this loop should exit with last_ind being equal to the last row,col occurence (inclusive)
    do {
        last_ind = i;
        i++;
        r = c_kms/mu * fabs(wl[i] - mu); //how far away are we from mu?
    } while (r < r0);
    

    int M = (last_ind - first_ind + 1) * (last_ind - first_ind + 1); //how many non-zero elements

    // we want a square and symmetric matrix
    cholmod_triplet *T = cholmod_allocate_triplet(N, N, M, 1, CHOLMOD_REAL, c);

    if (T == NULL || T->stype == 0)		    /* T must be symmetric */
    {
      cholmod_free_triplet (&T, c) ;
      cholmod_finish (c) ;
      return (0) ;
    }
    
    int * Ti = T->i;
    int * Tj = T->j;
    double * Tx = T->x;
    int k = 0;

    //Only fill in the lower entries (and diagonal). 
    for (i = first_ind; i <= last_ind; i++)
    {
        for (j = first_ind; j <= i; j++)
        {
            Ti[k] = i;
            Tj[k] = j;
            Tx[k] = k_region(wl[i], wl[j], a, mu, sigma);
            k++;
        }
    }
    T->nnz = k;

    if (k == 0)
    {
        //the parameters for the region were too small, and no region was
        //initialized, return null
        cholmod_free_triplet(&T, c);
        return NULL;
    }
    
    //The conversion will transpose the entries and add to the upper half.
    cholmod_sparse *A = cholmod_triplet_to_sparse(T, k, c);
    cholmod_free_triplet(&T, c);
    return A;
}


/* Given a cholesky_factor, calculate the log determinant. Can accept supernodal or  simplicial factorizations and LDLt or LLt factorizations. Output is checked against  Julia's SuiteSparse wrapping. Inspiration for this code drawn from
 * https://github.com/JuliaLang/julia/blob/master/base/linalg/cholmod.jl
 * https://github.com/dfm/george/blob/93042f84ceb86afe43454cb66092cde58fe58112/include/sparse.h
 *
 * The supernodal factorization stores the sparse elements as dense, column-major, rectangular blocks. For these blocks, nrows > ncols (at least in my experience). Extracting the diagonals of these blocks gives you the diagonal for the full Cholesky factor.
 *
 * For example, supernode 2 may contain colums 89 through 97, but rows 89 through 97, and *also* rows 112 through 132. Therefore, it always makes sense to extract the diagonals and stop once you are out of columns in the supernode. Columns 112 through 132 will appear in a later supernode, where you can capture their diagonals. For more information, run the function explore_supernodal() */
double get_logdet(cholmod_factor *L)
{
    int is_ll = L->is_ll; //is it Cholesky LLt or LDLt?
    int n = L->n; //L is n-by-n
    double *x = (double *) L->x; //array of numerical values. Has size L->xsize
    double v, logdet = 0;

    //Determine if L is simplicial or supernodal
    if (L->is_super)
    {
        int *super = L->super; //array listing the first column in each supernode
        int *pi = L->pi; //array of the pointers to integer patterns
        int *px = L->px; //array of pointers to real parts
        int *s = L->s; //integer part of supernodes

        int k = 0; //supernodal index
        int ncol = 0; //number of columns in each supernode
        int nrow = 0; //number of rows in each supernode
        int base; //The pointer to the base of the supernode in memory
        int pos = 0; //the position tracker in memory
        int ii,jj=0; //row and column indices for the supernode

        for (k = 0; k < L->nsuper; k++) //iterate through each supernode
        {
            ncol = super[k + 1] - super[k];
            nrow = pi[k + 1] - pi[k];
            base = px[k]; //pointer to (0,0) of supernode
            
            //If it is an LL factorization, you must square the diagonal, otherwise, take it as is
            if (!is_ll)//According to pg 76 of the user manual, this should never occur
            {                
                return 0;
                /* //if it ever is required, here's the code
                for (pos = base; pos < base + (nrow * ncol); pos += (nrow + 1))
                {
                    //calculate indices for the supernode
                    ii = (pos - base) % nrow;
                    jj = (pos - base)/nrow;
                    //calculate indices for the Cholesky factor
                    //row is s[pi[k] + row_index]
                    //column is super[k] + column_index
                    assert(s[pi[k] + ii] == super[k] + jj); //make sure we are reading the diagonal
                    v = x[pos];
                    logdet += log(fabs(v));
                }
                */
            }
            
            //start at the base of the supernode and increment along the diagonal
            //from (0,0) to (1,1), to (2,2), etc...
            for (pos = base; pos < base + (nrow * ncol); pos += (nrow + 1))
            {
                //calculate indices for the supernode
                ii = (pos - base) % nrow;
                jj = (pos - base)/nrow;
                //calculate indices for the Cholesky factor
                //row is s[pi[k] + row_index]
                //column is super[k] + column_index
                assert(s[pi[k] + ii] == super[k] + jj); //make sure we are reading the diagonal
                v = x[pos];
                logdet += log(fabs(v * v));
            }
        }
    }
    //simplicial factorization
    else
    {
	//column pointers
	int *c0 = L->p;
	//row indices
	int *r0 = L->i;

	int j = 0; //column iterator
	int jj = 0;//column index

	//we want to extract values from (0,0), (1,1), etc..
        if (is_ll)
        {
            for (j = 0; j<n; j++)
            {
                    jj = c0[j]; //column for j
                    assert(r0[jj] == j); //row and column indices should be equal
                    v = x[jj]; //value for j,j
                    logdet += log(fabs(v * v));
            }
        }
        else
        {
            for (j = 0; j<n; j++)
            {
                    jj = c0[j]; //column for j
                    assert(r0[jj] == j); //row and column indices should be equal
                    v = x[jj]; //value for j,j
                    logdet += log(fabs(v));
            }

        }
    }
    return logdet;
}

//Evaluate r^T C^(-1) r^T for the log probability function
double chi2 (cholmod_dense *r, cholmod_factor *L, cholmod_common *c)
{
    cholmod_dense *x; 
    x = cholmod_solve (CHOLMOD_A, L, r, c) ;  // solve Cx=r 

    int i;
    double ans = 0;

    //x is N-by-1
    //r is N-by-1
    double *xx = x->x;
    double *rx = r->x;
    //go through x and r to calculate the dot product
    for (i = 0; i < L->n; i++)
    {
        ans += xx[i] * rx[i];
    }
    cholmod_free_dense(&x, c);
    return ans;
}


void explore_supernodal(cholmod_factor *L)
{
    printf("L->is_ll = %d\n", L->is_ll);
    /* Julia implementation in base/linalg/cholmod.jl
    res = zeros(Tv,L.c.n)
    xv  = L.x
    if bool(L.c.is_super)
        nec = decrement!(diff(L.super))  # number of excess columns per supernode
        #nec = ncol - 1
        dstride = increment!(diff(L.pi)) # stride of diagonal elements (nrow + 1)
        px = L.px
        pos = 1
        for i in 1:length(nec) #do once for each supernode
            base = px[i] + 1 #get the base pointer for the supernode
            res[pos] = xv[base] #read off the (0, 0) value for the supernode
            pos += 1
            for j in 1:nec[i] #for each remaining column (1 to ncol)
                res[pos] = xv[base + j*dstride[i]] #read off (1,1)
                pos += 1
            end
        end
    else
    */

    /* Python version from scikits-sparse
    d = np.empty(self._factor.n, dtype=dtype) #create a result vector
    filled = 0 #how many numbers have you read so far
    for k in xrange(self._factor.nsuper): #once for each supernode
        ncols = super_[k + 1] - super_[k] #number of columns
        nrows = pi[k + 1] - pi[k] #number of rows

        #Read off all of the diagonal elements in the supernode, striding by nrows + 1
        d[filled:filled + ncols] = x[px[k]
                                     :px[k] + nrows * ncols
                                     :nrows + 1]
        filled += ncols
    */

    /*
    C version from 
    https://github.com/dfm/george/blob/93042f84ceb86afe43454cb66092cde58fe58112/include/sparse.h

    double extract_logdet (cholmod_factor *L)
    {
            int is_ll = L_->is_ll
            int n = L_->n
            int i;
            int k;
            double v, logdet = 0.0;
            double *x = (double*)(L_->x);

    if (L_->is_super) {
        int nsuper = L_->nsuper, *super = (int*)(L_->super),
            *pi = (int*)(L_->pi), *px = (int*)(L_->px), ncols, nrows, inc;
        for (i = 0; i < nsuper; i++){ //once for each supernode
            ncols = super[i+1] - super[i];
            nrows = pi[i+1] - pi[i];
            inc = nrows + 1; //stride by the number of rows + 1, to capture just diagonals
            if (is_ll)
                for (k = 0; k < ncols * nrows; k += inc) {
                    v = x[px[i]+k];
                    logdet += log(v * v);
                }
            else
                for (k = 0; k < ncols * nrows; k += inc)
                    logdet += log(x[px[i]+k]);
        }
    } else {
        int* p = (int*)(L_->p);
        if (is_ll)
            for (i = 0; i < n; ++i) {
                v = x[p[i]];
                logdet += log(v * v);
            }
        else
            for (i = 0; i < n; ++i)
                logdet += log(x[p[i]]);
    }
    return logdet;
    };
    */

    int *pi = L->pi; //pointers to integer patterns
    int *s = L->s; //integer part of supernodes
    double *x = (double *) L->x; //numerical values. Has size L->xsize
    int *super = L->super; //first column in each supernode
    int *px = L->px; //pointers to real parts
    
    int k = 0; //supernodal index
    int ncol = 0; //number of columns in each supernode
    int nrow = 0; //number of rows in each supernode
    int ipointer = 0;
    int base; //The pointer to the base of the supernode
    int pos = 0; //the position tracker
    double logdet = 0;
    int ii,jj = 0;
    for (k = 0; k < L->nsuper; k++) //step through each supernode
    {
        ncol = super[k + 1] - super[k];
        nrow = pi[k + 1] - pi[k];

        printf("Supernode %d contains columns %d through %d for a total of %d columns.\n", k, super[k], super[k+1] -1, ncol);
        printf("Supernode %d has %d rows. These are rows\n", k, nrow);
        for (ipointer = pi[k]; ipointer < pi[k+1]; ipointer++)
        {
                printf("%d ", s[ipointer]);
        }
        printf("\n");

        printf("Numerical pointers range from %d to %d\n", px[k], px[k + 1] -1);
        printf("for a total of %d values.\n", px[k + 1] - px[k]);
        printf("ncol * nrow = %d\n", ncol * nrow);
        
        base = px[k]; //pointer to (0,0) of supernode
        printf("Supernode %d has base pointer %d. Value %.2f\n", k, base, x[base]);

        for (pos = base; pos < base + (nrow * ncol); pos += (nrow + 1))
        {
                //column is super[k] + column_index
                //row is s[pi[k] + row_index]
                //where column_index and row_index are both the values for the supernode
                jj = (pos - base)/nrow;
                ii = (pos - base) % nrow;
                printf("Reading off %d, %d.\n", s[pi[k] + ii], super[k] + jj);
                logdet += log(fabs(x[pos]*x[pos]));
        }

    }
    printf("logdet = %.2f", logdet);
}
