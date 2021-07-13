#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include <mpi.h> 
#include <omp.h> 

#define MAXVARS (250)   /* max # of variables	     */
#define RHO_BEGIN (0.5) /* stepsize geometric shrink */
#define EPSMIN (1E-6)   /* ending value of stepsize  */
#define IMAX (5000)     /* max # of iterations	     */

/* global variables */
unsigned long funevals = 0;

/* Rosenbrocks classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    for (i = 0; i < n - 1; i++) /* rosenbrock */
        fv = fv + 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);

    return fv;
}

/* given a point, look for a better one nearby, one coord at a time */
double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars)
{
    double z[MAXVARS];
    double minf, ftmp;
    int i;
    minf = prevbest;

    for (i = 0; i < nvars; i++)
    {
        z[i] = point[i];
    }

    for (i = 0; i < nvars; i++)
    {
        z[i] = point[i] + delta[i];
        ftmp = f(z, nvars);
        if (ftmp < minf)
            minf = ftmp;
        else
        {
            delta[i] = 0.0 - delta[i];
            z[i] = point[i] + delta[i];
            ftmp = f(z, nvars);
            if (ftmp < minf)
                minf = ftmp;
            else
                z[i] = point[i];
        }
    }
    for (i = 0; i < nvars; i++)
        point[i] = z[i];

    return (minf);
}

int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax)
{
    double delta[MAXVARS];
    double newf, fbefore, steplength, tmp;
    double xbefore[MAXVARS], newx[MAXVARS];
    int i, j, keep;
    int iters, iadj;

    for (i = 0; i < nvars; i++)
    {
        newx[i] = xbefore[i] = startpt[i];
        delta[i] = fabs(startpt[i] * rho);
        if (delta[i] == 0.0)
            delta[i] = rho;
    }
    iadj = 0;
    steplength = rho;
    iters = 0;
    fbefore = f(newx, nvars);
    newf = fbefore;
    while ((iters < itermax) && (steplength > epsilon))
    {
        iters++;
        iadj++;
#if DEBUG
        printf("\nAfter %5d funevals, f(x) =  %.4le at\n", funevals, fbefore);
        for (j = 0; j < nvars; j++)
            printf("   x[%2d] = %.4le\n", j, xbefore[j]);
#endif

        /* find best new point, one coord at a time */
        for (i = 0; i < nvars; i++)
        {
            newx[i] = xbefore[i];
        }
        newf = best_nearby(delta, newx, fbefore, nvars);
        /* if we made some improvements, pursue that direction */
        keep = 1;
        while ((newf < fbefore) && (keep == 1))
        {
            iadj = 0;
            for (i = 0; i < nvars; i++)
            {
                /* firstly, arrange the sign of delta[] */
                if (newx[i] <= xbefore[i])
                    delta[i] = 0.0 - fabs(delta[i]);
                else
                    delta[i] = fabs(delta[i]);
                /* now, move further in this direction */
                tmp = xbefore[i];
                xbefore[i] = newx[i];
                newx[i] = newx[i] + newx[i] - tmp;
            }
            fbefore = newf;
            newf = best_nearby(delta, newx, fbefore, nvars);
            /* if the further (optimistic) move was bad.... */
            if (newf >= fbefore)
                break;

            /* make sure that the differences between the new */
            /* and the old points are due to actual */
            /* displacements; beware of roundoff errors that */
            /* might cause newf < fbefore */
            keep = 0;
            for (i = 0; i < nvars; i++)
            {
                keep = 1;
                if (fabs(newx[i] - xbefore[i]) > (0.5 * fabs(delta[i])))
                    break;
                else
                    keep = 0;
            }
        }
        if ((steplength >= epsilon) && (newf >= fbefore))
        {
            steplength = steplength * rho;
            for (i = 0; i < nvars; i++)
            {
                delta[i] *= rho;
            }
        }
    }
    for (i = 0; i < nvars; i++)
        endpt[i] = xbefore[i];

    return (iters);
}

double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

int main(int argc, char *argv[])
{

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int rank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //  Get total number of ranks to split the work between them.
    int _size;
    MPI_Comm_size(MPI_COMM_WORLD, &_size);

    omp_set_num_threads(4);

    
    double t0, t1; //  Used by main rank for time calculation.

    int itermax = IMAX;
    double rho = RHO_BEGIN;
    double epsilon = EPSMIN;
    int nvars = 16; //  Number of variables (problem dimension).
    int trial;
    int ntrials = 128 * 1024;                     // Number of trials.
    int mpi_ntrials = ntrials / (_size - 1); // Distribute for-loop evenly between secondary ranks (rank 0 is main rank).

    double best_fx = 1e10;
    double best_pt[MAXVARS];
    int best_trial = -1;
    int best_jj = -1;

    //  Initialize best_pt
    for (int i = 0; i < MAXVARS; i++)
    {
        best_pt[i] = 0.0;
    }

    srand48(time(0));

    t0 = get_wtime();

    double fx;
    int jj;
    double startpt[MAXVARS], endpt[MAXVARS];
    short seed = (short)get_wtime(); //seed for erand()
    unsigned short randBuffer[3];
    randBuffer[0] = 0;
    randBuffer[1] = 0;
    randBuffer[2] = seed + rank;

    if (rank != 0) // only secondary ranks
    {

#pragma omp parallel //  Split each secondary rank to threads
        {

            double fx;
            int jj;
            double startpt[MAXVARS], endpt[MAXVARS];

            
            short seed = (short)get_wtime(); //  Seed for erand().
            unsigned short randBuffer[3];
            randBuffer[0] = 0;
            randBuffer[1] = 0;
            randBuffer[2] = seed + omp_get_thread_num();

#pragma omp for
            for (trial = 0; trial < mpi_ntrials; trial++)
            {
                /* starting guess for rosenbrock test function, search space in [-4, 4) */
                for (int i = 0; i < nvars; i++)
                {
                    startpt[i] = 4.0 * erand48(randBuffer) - 4.0;
                }

                jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);

#if DEBUG
                printf("\n\n\nHOOKE %d USED %d ITERATIONS, AND RETURNED\n", trial, jj);
                for (int i = 0; i < nvars; i++)
                {
                    printf("x[%3d] = %15.7le \n", i, endpt[i]);
                }
#endif

                fx = f(endpt, nvars);

                //  Send the results to main rank(all threads can send data to the main rank), one at a time(send fx and block )
#pragma omp critical
                {
                    MPI_Send(&fx, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
                    MPI_Send(&trial, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                    MPI_Send(&jj, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
                    MPI_Send(&endpt, 250, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
                }

#if DEBUG
                printf("f(x) = %15.7le\n", fx);
#endif
            }
        }
    }
    else if (rank == 0)  // only main rank enters
    {

        t0 = get_wtime();

        for (int i = 0; i < ntrials; i++)
        {
            //  Receive the values from any process
            MPI_Recv(&fx, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&trial, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&jj, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&endpt, 250, MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

            //  Compares and determines the best point found.
            if (fx < best_fx)
            {
                best_trial = trial;
                best_jj = jj;
                best_fx = fx;
                for (int i = 0; i < nvars; i++)
                {
                    best_pt[i] = endpt[i];
                }
            }
        }

        t1 = get_wtime();

        printf("\n\nFINAL RESULTS:\n");
        printf("Elapsed time = %.3lf s\n", t1 - t0);
        printf("Total number of trials = %d\n", ntrials);
        printf("Total number of function evaluations = %ld\n", funevals);
        printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
        for (int i = 0; i < nvars; i++)
        {
            printf("x[%3d] = %15.7le \n", i, best_pt[i]);
        }
        printf("f(x) = %15.7le\n", best_fx);
    }

    
    MPI_Finalize();    // clean up and terminate the environment


    return 0;
}
