/////////////////////////////////////////////////////////////
//                    외판원 문제를 위한                   //
//       GPU기반 병렬 유전자 알고리즘의 설계 및 구현       //
//                     (GPU 알고리즘)                      //
//                                                         //
//                     20140167 김현성                     //
/////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* CUDA libraries */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Helper functions and utilities to work with CUDA
#include "curand.h"
#include "curand_kernel.h"

/* When experimenting with other samples, change the value of MAX_CITIES, GOAL, and fname. */
#define MAX_CITIES 130					// The number of cities in TSP
#define MAX_POPULATION 20				// The number of population
#define MAX_GENERATION 1000				// The maximum number of generation
#define GOAL 6110						// The optimal solution of TSP
#define RATE 1.005						// Acceptance error rate
#define fname "./TSP Samples/ch130.tsp" // TSP Sample

int population;							// The number of population
int num_cities;							// The number of cities in TSP
double cities[MAX_CITIES * 2];			// The list of cities in TSP
int pop[MAX_POPULATION * MAX_CITIES];	// Current population
double score[MAX_POPULATION];			// Length of the travel path of individuals in current population
double best_length;						// Best length in current population
int best_index;							// The index of the best individual in current population

void tsp_genetic_2opt();
__global__ void kernel_step1(int *p_pop, double *p_score, double *p_cities, int *p_num_cities, curandState *state); // Crossover kernel
__global__ void kernel_step2(double *p_score_cm, double *p_cities, int *p_num_cities);								// 2-OPT local search kernel
__global__ void kernel_step3(int *p_pop, double *p_score, double *p_cities, int *p_num_cities);						// Selection kernel

/* Calculates length of the travel path */
double get_length(int i) {
	double length = 0;
	i = i*num_cities;
	double d[MAX_CITIES][2];

	for (int j = 0; j < num_cities; j++) {
		d[j][0] = cities[2 * pop[i + j]];
		d[j][1] = cities[2 * pop[i + j] + 1];
	}

	for (int j = 0; j < num_cities; j++) {
		length += sqrt(pow(d[j][0] - d[(j + 1) % num_cities][0], 2) + pow(d[j][1] - d[(j + 1) % num_cities][1], 2));
	}

	return length;
}

/* Set up curand for random number generation */
__global__ void setup_kernel(curandState *state, unsigned long seed) {
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

/* Get input data from the file */
void input()
{
	FILE *in = fopen(fname, "r");

	int x;
	double a, b;

	fscanf(in, "%d", &num_cities);

	for (int i = 0; i < num_cities; i++) {
		fscanf(in, "%d %lf %lf", &x, &a, &b);
		cities[2 * i] = a;
		cities[2 * i + 1] = b;
	}
	fclose(in);

	population = MAX_POPULATION;
}

/* Generates random permutation */
void myShuffle(int S[])
{
	int A[MAX_CITIES], i, j;

	for (i = 0; i < num_cities; i++) A[i] = i;

	for (i = num_cities - 1; i > 0; i--) {
		j = rand() % (i + 1);
		S[i] = A[j];
		A[j] = A[i];
	}
	S[0] = A[0];
}

/* Generates initial population */
void initial_pops()
{
	for (int i = 0; i < population; i++) {
		myShuffle(pop + (i*num_cities));
	}
}

int main()
{
	srand(time(NULL));
	input();
	initial_pops();

	/* Calculate length of the travel paths in initial population */
	for (int i = 0; i < population; i++) {
		score[i] = get_length(i);
	}

	/* Calculate the initial best individual */
	best_length = score[0];
	best_index = 0;
	for (int i = 1; i < population; i++) {
		if (score[i] < best_length) {
			best_length = score[i];
			best_index = i;
		}
	}

	printf("\ngen 0: best length = %.2f\n", best_length);

	// If the initial best individual is acceptable, print the length and exit program
	if (best_length > GOAL*RATE) {
		tsp_genetic_2opt();
	}
	else {
		printf("\ngen 0: best length = %.2f\n", best_length);
		printf("\n");
	}
	system("pause");
	return 0;
}

/* The function operating genetic algorithm with GPU */
void tsp_genetic_2opt()
{
	clock_t begin, end;

	/* Device memories used in the kernel */
	int *d_num_cities;	// The number of cities
	double *d_cities;	// The list of cities
	int *d_pop;			// Current population
	double *d_score;	// Length of travel path of individuals in current population
	double *d_score_cm;	// Length of travel path of individuals after crossover and 2-OPT local search

	/* Set up curand for random number generation */
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));
	setup_kernel << <population, num_cities >> > (d_state, unsigned(time(NULL)));

	double score_cm[MAX_POPULATION]; // Length of travel path of individuals after crossover and 2-OPT local search

	/* Memory allocation of device memories */
	cudaMalloc((void**)&d_num_cities, sizeof(int));
	cudaMalloc((void**)&d_cities, sizeof(double)*num_cities * 2);
	cudaMalloc((void**)&d_pop, sizeof(int)*population*num_cities);
	cudaMalloc((void**)&d_score, sizeof(double)*population);
	cudaMalloc((void**)&d_score_cm, sizeof(double)*population);

	/* Copying the memory from host to device */
	cudaMemcpy(d_num_cities, &num_cities, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cities, &cities, sizeof(double)*num_cities * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pop, pop, sizeof(int)*population*num_cities, cudaMemcpyHostToDevice);
	cudaMemcpy(d_score, score, sizeof(double)*population, cudaMemcpyHostToDevice);

	/* Grid and block dimension in kernel */
	dim3 block1(num_cities);
	dim3 block2(num_cities, num_cities);
	dim3 grid1(population);
	dim3 grid2(population, num_cities);

	/* Clock set */
	begin = clock();

	int generation = 1;

	/* Generation loop */
	while (generation <= MAX_GENERATION) {

		/* Crossover */
		kernel_step1 << < grid1, block1 >> > (d_pop, d_score, d_cities, d_num_cities, d_state);
		cudaDeviceSynchronize();

		/* 2-OPT Local Search */
		double min_length = 1.0E10; // length difference of reversed path and original path
		int cnt;
		for (cnt = 0; cnt < 2 * num_cities; cnt++) {
			kernel_step2 << < grid2, block1 >> > (d_score_cm, d_cities, d_num_cities);
			cudaDeviceSynchronize();
			cudaMemcpy(score_cm, d_score_cm, sizeof(double)*population, cudaMemcpyDeviceToHost);

			min_length = score_cm[0];
			for (int i = 1; i < population; i++) {
				if (score_cm[i] < min_length) {
					min_length = score_cm[i];
				}
			}

			if (min_length >= -0.001) break;
		}

		/* Selection */
		kernel_step3 << < grid1, block1 >> > (d_pop, d_score, d_cities, d_num_cities);
		cudaDeviceSynchronize();

		cudaMemcpy(pop, d_pop, sizeof(int)*population*num_cities, cudaMemcpyDeviceToHost);
		cudaMemcpy(score, d_score, sizeof(double)*population, cudaMemcpyDeviceToHost);

		/* Update best length */
		best_length = score[0];
		best_index = 0;
		for (int i = 1; i < population; i++) {
			if (score[i] < best_length) {
				best_length = score[i];
				best_index = i;
			}
		}

		/* If the best individual is acceptable, exit the function */
		if (best_length <= GOAL*RATE) break;

		/* Print current best individual and elasped time */
		if (generation % 100 == 0) { 
			end = clock();
			printf("gen %d: best length = %.1f, time = %.3f\n", generation, best_length, (double)(end - begin) / CLOCKS_PER_SEC);
		}

		generation++;
	}

	/* If the best individual is unacceptable, print 'No Solution' */
	if (best_length > GOAL*RATE) printf("No Solution\n\n");
	else {

		/* Clock stop */
		end = clock();
		printf("\ngen %d: best length = %.1f, time = %.3f\n", generation, best_length, (double)(end - begin) / CLOCKS_PER_SEC);

		printf("path : ");
		for (int j = 0; j < num_cities; j++) printf("%d ", pop[best_index*num_cities + j]);
		printf("\n");
		int Z[500] = { 0 };
		int jj;
		for (jj = 0; jj < num_cities; jj++) {
			if (Z[pop[best_index*num_cities + jj]] > 0) break;
			Z[pop[best_index*num_cities + jj]] = 1;
		}
		if (jj < num_cities) puts("Incorrect result!!");
		else puts("Correct result!!");
	}

	cudaFree(d_num_cities);
	cudaFree(d_cities);
	cudaFree(d_pop);
	cudaFree(d_score);
	cudaFree(d_score_cm);

}

/*************************************************************************/

__device__ double score_ox[MAX_POPULATION];

/* Calculates length of the travel path of an individual */
__device__ void d_get_length(int path[MAX_CITIES], double *score_ox, double *p_cities, int N) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;

	__shared__ double length_array[MAX_CITIES];

	double x1, y1, x2, y2;
	x1 = p_cities[2 * path[tx]];
	y1 = p_cities[2 * path[tx] + 1];
	x2 = p_cities[2 * path[(tx + 1) % N]];
	y2 = p_cities[2 * path[(tx + 1) % N] + 1];
	__syncthreads();

	length_array[tx] = sqrtf(powf(x2 - x1, 2) + powf(y2 - y1, 2));
	__syncthreads();

	/* Reduction algorithm of array sum */
	for (int j = 1; j < N; j <<= 1) {
		if (tx % (2 * j) == 0 && tx + j < N) {
			length_array[tx] += length_array[tx + j];
		}
		__syncthreads();
	}

	if (tx == 0) {
		score_ox[bx] = length_array[0];
	}
	__threadfence();
}

// Crossover function
// Do order crossover on path1, path2. The result is new_path
__device__ void d_crossover(int *new_path, int *path1, int *path2, int N, curandState *state) {
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int idx = tx + bx*blockDim.x;

	__shared__ int cut1, cut2;
	__shared__ bool flag[MAX_CITIES];
	__shared__ int d[MAX_CITIES];

	/* Set random cut1, cut2 s.t. 0 <= cut1 < cut2 < num_city */
	if (tx == 0) {
		do {
			cut1 = curand_uniform(state + idx) * gridDim.x;
			cut2 = curand_uniform(state + idx) * gridDim.x;
		} while (cut1 >= cut2);
	}

	flag[tx] = true;
	d[tx] = 1;
	__syncthreads();

	/* Fill [cut1, cut2] of new_path with path2 */
	if (cut1 <= tx && tx <= cut2) {
		flag[path2[tx]] = false;
		new_path[tx] = path2[tx];
	}
	__syncthreads();

	if (flag[path1[tx]]) d[tx] = 1;
	else d[tx] = 0;
	__syncthreads();

	// Reduction algorithm of prefix sum
	for (int j = 1; j < N; j <<= 1) {
		if (tx + j < N) d[tx + j] += d[tx];
		__syncthreads();
	}

	/* Fill the rest part of new_path in same order with path1 */
	if (flag[path1[tx]]) {
		if (d[tx] <= cut1)
			new_path[d[tx] - 1] = path1[tx];
		else
			new_path[cut2 - cut1 + d[tx]] = path1[tx];
	}
	__syncthreads();
}

__device__ int p_path[MAX_POPULATION][MAX_CITIES];	// Global memory used to save paths after crossover

/* Crossover kernel */
__global__ void kernel_step1(int *p_pop, double *p_score, double *p_cities, int *p_num_cities, curandState *state)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int N = *p_num_cities;

	int idx = bx*blockDim.x + tx;
	int j;
	__shared__ int bx2;
	
	/* Set bx2 randomly s.t. bx2 != bx */
	if (tx == 0) {
		do {
			j = curand_uniform(state + idx) * gridDim.x;
		} while (j == bx);
		bx2 = j;
	}
	__syncthreads();

	p_path[bx][tx] = p_pop[bx*N + tx];
	__threadfence();

	__shared__ int path1[MAX_CITIES];
	__shared__ int path2[MAX_CITIES];
	__shared__ int new_path[MAX_CITIES];
	__syncthreads();

	path1[tx] = p_path[bx][tx];
	path2[tx] = p_path[bx2][tx];
	__syncthreads();

	d_crossover(new_path, path1, path2, N, state);
	__syncthreads();

	/* Move values to global memory */
	if (tx < N) {
		p_path[bx][tx] = new_path[tx];
	}
	__threadfence();

}

/* Calculates the distance between two points */
__device__ double distance(double a[], double b[]) {
	return sqrtf(powf(a[0] - b[0], 2) + powf(a[1] - b[1], 2));
}

__device__ double d_length[MAX_POPULATION][MAX_CITIES];
__device__ int d_a[MAX_POPULATION][MAX_CITIES], d_b[MAX_POPULATION][MAX_CITIES];

/* 2-OPT local search kernel */
__global__ void kernel_step2(double *p_length_cm, double *p_cities, int *p_num_cities)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int N = *p_num_cities;

	__shared__ int path[MAX_CITIES];
	__shared__ double city[MAX_CITIES][2];
	__shared__ int cut1, cut2;
	__shared__ double min;
	__shared__ double length[MAX_CITIES];
	__shared__ int a[MAX_CITIES], b[MAX_CITIES];
	__syncthreads();

	path[tx] = p_path[bx][tx];
	__syncthreads();


	city[tx][0] = p_cities[2 * tx];
	city[tx][1] = p_cities[2 * tx + 1];
	__syncthreads();

	double dist1, dist2;

	length[tx] = 0.0;
	a[tx] = b[tx] = by;
	__syncthreads();

	if (tx == 0) {
		min = 0.0;
		cut1 = cut2 = 0;
		d_length[bx][by] = 0.0;
	}
	__threadfence();

	// For every thread tx in each block (bx, by) 
	// calculates the length change when reversed the path [j, k]
	dist1 = dist2 = 0.0;
	if (by <= N - 3 && tx >= by + 2 && tx < N) {
		dist1 = distance(city[path[by]], city[path[tx]]) +
			distance(city[path[by + 1]], city[path[(tx + 1) % N]]);
		dist2 = distance(city[path[by]], city[path[by + 1]]) +
			distance(city[path[tx]], city[path[(tx + 1) % N]]);
	}
	__syncthreads();

	if (length[tx] > (dist1 - dist2)) {
		length[tx] = dist1 - dist2;
		a[tx] = by;
		b[tx] = tx;
	}
	__syncthreads();

	/* Reduction algorithm of calculating minimum in an array */
	for (int j = 1; j < N; j <<= 1) {
		if (tx % (2 * j) == 0 && tx + j < N) {
			if (length[tx] > length[tx + j]) {
				length[tx] = length[tx + j];
				a[tx] = a[tx + j];
				b[tx] = b[tx + j];
			}
		}
		__syncthreads();
	}
	__syncthreads();

	// Each block (bx, by) calculates the most decreased length change
	// Save the minimum value to d_length[bx][by]
	if (tx == 0) {
		d_length[bx][by] = length[0];
		d_a[bx][by] = a[0];
		d_b[bx][by] = b[0];
	}
	__threadfence();

	// Each block (bx, 0) calculates the minimum value among d_length[bx][0:N]
	// and saves it to d_length[bx][0]
	if (by == 0) {
		length[tx] = d_length[bx][tx];
		a[tx] = d_a[bx][tx];
		b[tx] = d_b[bx][tx];
	}
	__syncthreads();

	/* Reduction algorithm of calculating minimum in an array */
	for (int j = 1; j < N; j <<= 1) {
		if (tx % (2 * j) == 0 && tx + j < N) {
			if (length[tx] > length[tx + j]) {
				length[tx] = length[tx + j];
				a[tx] = a[tx + j];
				b[tx] = b[tx + j];
			}
		}
		__syncthreads();
	}

	if (tx == 0 && by == 0) {
		min = length[0];
		cut1 = a[0];
		cut2 = b[0];
	}
	__syncthreads();

	/* If the found minimum length change is under 0, update the path */
	if (min < -0.001) {
		if (by == 0) {
			int mid = (cut1 + cut2) / 2;
			if (tx > cut1 && tx <= mid) {
				int tmp = path[tx];
				path[tx] = path[cut2 - tx + cut1 + 1];
				path[cut2 - tx + cut1 + 1] = tmp;
			}
		}
	}
	__syncthreads();

	if (by == 0) {
		p_path[bx][tx] = path[tx];
	}
	__threadfence();

	if (by == 0) {
		p_length_cm[bx] = min;
	}
	__threadfence();

	path[tx] = p_path[bx][tx];
	__syncthreads();
	__threadfence();
}

/* Selection kernel */
__global__ void kernel_step3(int *p_pop, double *p_score, double *p_cities, int *p_num_cities)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	int N = *p_num_cities;
	int M = gridDim.x;
	__shared__ int path[MAX_CITIES];

	path[tx] = p_path[bx][tx];
	__syncthreads();

	d_get_length(path, score_ox, p_cities, N);
	__syncthreads();

	if (tx == 0 && bx == 0) {
		for (int i = 0; i < M; i++) {
			if (p_score[i] > score_ox[i]) {
				p_score[i] = score_ox[i];
				for (int j = 0; j < N; j++)
					p_pop[i*N + j] = p_path[i][j];
			}
		}
	}
	__threadfence();
}

