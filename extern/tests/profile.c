#include <stdio.h>
#include "../cov.h"

/* compile with
gcc profile.c -pg -o profile -lm -lcholmod -lamd -lcolamd -lblas -llapack -lsuitesparseconfig
 * then run 
 * ./profile
 * then do
 * gprof profile gmon.out -p
 * and
 * gprof profile gmon.out -q
*/

int main(void)
{

    int N=3000;
    double wl[N];
    linspace(wl, N, 5100., 5200.);
    double min_sep = get_min_sep(wl, N);

    cholmod_common c;
    cholmod_start(&c);
    c.print = 5;

    cholmod_sparse *A = create_sparse(wl, N, min_sep, 1.0, 2.0, &c); //often slow
    cholmod_factor *L = cholmod_analyze (A, &c) ;		    
    cholmod_factorize (A, L, &c) ;  //tends to be slow    
    //also it seems like the permutation is also slow

    //get_logdet(L); //very fast

    //cholmod_dense *r = cholmod_ones (A->nrow, 1, A->xtype, &c) ;   
    //chi2(r, L, &c); //very fast


    //cholmod_sparse *C = create_sparse_region(wl, N, 3, 1.0, 5150., 1., &c);
    //instead of doing a cholesky update, try adding and refactoring?
    //double alpha [2] = {1,0}, beta [2] = {1,0} ;	    // basic scalars 
    //cholmod_sparse *F = cholmod_add(A, C, alpha, beta, TRUE, TRUE, &c);
    //cholmod_factor *L = cholmod_analyze (F, &c) ;		    
    //cholmod_factorize (F, L, &c) ;  //tends to be slow    

    
    /*
    cholmod_sparse *F = cholmod_allocate_sparse(A->nrow, A->ncol, A->nzmax, A->sorted, A->packed, A->stype, A->xtype, &c);
    cholmod_transpose_sym(C, 1, L->Perm, F, &c);
    cholmod_sort(F, &c);
    cholmod_updown(TRUE, F, L, &c);
*/

    cholmod_free_sparse(&A, &c); 
    //cholmod_free_sparse(&C, &c); 
    //cholmod_free_sparse(&F, &c); 
    cholmod_free_factor(&L, &c);
    //cholmod_free_dense(&r, &c);
    cholmod_finish(&c);


    return 0;
}

/* 
 * It looks like we are going to need faster methods for creating the sparse matrices, since doing two loops through a 3000x3000 matrix (and computing r_mu
 * that many times) will be slow. 
 *
 * It also looks like cholmod_updown might be pretty slow (0.14 s).  cholmod_transpose_sym took 0.05 s, create_sparse_region took 0.11, and the other shit only took about half of this time.
 *
 * When you do adding, create_sparse_region is still 0.09, now cholmod_transpose_sym has gone up to 0.09 s, and cholmod_add has an extra 0.01 s. So yes, we will have to be factoring the whole matrix again, but it seems like the small cholmod_update stuff is not fast enough.
 */

/* After rewriting create_sparse, it only takes 0.01 s. 
 * After rewriting create_sparse_region, it takes < 0.01s.
 */

// cholmod_transpose_sym is called from inside create_sparse, probably when it converts from triplet to sparse.
//
//
//How long does the factorization take?
//Creation of the sparse matrix and factorizing it takes 0.1 s
//The python wrapper of this function takes 0.138 s, not bad.
