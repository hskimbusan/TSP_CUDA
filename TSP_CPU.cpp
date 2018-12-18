/////////////////////////////////////////////////////////////
//                    외판원 문제를 위한                   //
//       GPU기반 병렬 유전자 알고리즘의 설계 및 구현       //
//                     (CPU 알고리즘)                      //
//                                                         //
//                     20140167 김현성                     //
/////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#pragma warning(disable:4996)

#define MAX_CITIES 500				// The maximum number of cities in TSP
#define MAX_POP 60					// The number of population
#define ACCEPTABLE 6110				// The optimal solution of TSP
#define RATE 1.005					// Acceptance error rate

#define random(m) (rand() % (m))

/* Structure of travel path and its length of an individual */
typedef struct {
	int route[MAX_CITIES];
	double cost;
} POP;

/* Structure of 2d Coordinate of cities */
typedef struct {
	double x, y;
} Point;

POP P[MAX_POP];						// Current population 
Point city[MAX_CITIES];				// The list of cities in TSP
									   
int num_city;						// The number of cities in TSP
int pop;							// The number of population

int genetic();						// The function operating genetic algorithm
void input();						// Get input data from the file
double eval(int S[]);				// Calculates length of the travel path of an individual
void OX(int A[], int B[], int D[]);	// Crossover function
void TwoOpt1(int D[]);				// 2-OPT local search function
int initialize_pop();				// Generates initial population
double distant(Point, Point);		// Calculates the distance between two points

clock_t start, stop;

int main()
{
	srand(time(NULL));
	pop = MAX_POP;

	input();

	/* Generate the initial population and check if there is accepted individual in the population */
	int i = initialize_pop();

	/* Clock set */
	start = clock();

	/* If no individual is accepted as a solution, proceed the genetic algorithm */
	if (i < 0) i = genetic();

	// If no individual is accepted as a solution, print "No solution"
	// Else, print the travel path and length of the solution
	if (i < 0) printf("No solution\n");
	else {
		for (int j = 0; j < num_city; j++) {
			printf("%d ", P[i].route[j]);
		}
		printf("\n경로 길이 = %.1f\n", P[i].cost);
	}

	/* Clock stop */
	stop = clock();

	/* Print the elasped time */
	printf("소요 시간 = %.3f\n", (double)(stop - start) / CLOCKS_PER_SEC);

	system("pause");
}

// The function operating genetic algorithm
// Returns the index of accepted individual in the population
// If no individual is accepted, returns -1
int genetic()
{
	POP Q;						// population of individuals after crossover and 2-OPT local search
	int final = 1000;			// The number of maximum generation
	double best = P[0].cost;	// Current best length

	/* Generation loop */
	for (int t = 1; t <= final; t++) {

		/* Process for all individuals in the population */
		for (int i = 0; i < pop; i++) {
		
			/* Set j randomly s.t. j != i */
			int j;
			do {
				j = random(pop);
			} while (j == i);

			/* Crossover */
			OX(P[i].route, P[j].route, Q.route);

			/* 2-OPT local search */
			TwoOpt1(Q.route);

			/* Calcuate the length of Q */
			Q.cost = eval(Q.route);

			/* Select the better individual */
			if (P[i].cost > Q.cost) {
				for (int k = 0; k < num_city; k++) P[i].route[k] = Q.route[k];
				P[i].cost = Q.cost;

				/* If the individual is acceptable, return the index and exit the function */
				if (P[i].cost < ACCEPTABLE*RATE) {
					printf("gen %d: best = %.1f\n", t, P[i].cost);
					return i;
				}
			}

			/* Update the best length */
			if (best > P[i].cost) {
				best = P[i].cost;
			}
		}

		// Print the current best length and elapsed time
		if (t < 5 || t % 100 == 0) {
			stop = clock();
			printf("gen %d: best = %.1f (%.0fs)\n", t, best, (double)(stop - start) / CLOCKS_PER_SEC);
		}
	}

	/* If no individual is accepted, return -1 */
	return -1;
}

/* Generates random permutation */
void myShuffle(int S[])
{
	int A[MAX_CITIES], i, j;

	for (i = 0; i < num_city; i++) A[i] = i;

	for (i = num_city - 1; i > 0; i--) {
		j = random(i + 1);
		S[i] = A[j];
		A[j] = A[i];
	}
	S[0] = A[0];
}

// Generates initial population
// Returns the index of accepted individual in the population
// If no individual is accepted, returns -1
int initialize_pop()
{
	int i, j = -1;
	double best = 1.0E64;

	for (i = 0; i < pop; i++) {
		myShuffle(P[i].route);

		/* Calculate and update the initial best length */
		P[i].cost = eval(P[i].route);
		if (P[i].cost < best) {
			best = P[i].cost;
		}

		/* If the individual is acceptable, return the index and exit the function */
		if (P[i].cost < ACCEPTABLE*RATE) {
			j = i;
			break;
		}
	}

	printf("Initial best length = %.2f\n", best);
	return j;
}

// Crossover function
// Do order crossover on A[], B[]. The result is D[]
void OX(int A[], int B[], int D[])
{
	int cut1, cut2;
	int flag[MAX_CITIES] = { 0, };

	/* Set random cut1, cut2 s.t. 0 <= cut1 < cut2 < num_city */
	do {
		cut1 = random(num_city);
		cut2 = random(num_city);
	} while (cut1 >= cut2);

	/* Fill [cut1, cut2] of D with B */
	for (int j = cut1; j <= cut2; j++) {
		flag[B[j]] = 1;
		D[j] = B[j];
	}

	/* Fill the rest part of D in same order with A */
	int i = 0;
	for (int j = 0; j < cut1; j++) {
		while (flag[A[i]]) i++;
		D[j] = A[i++];
	}

	for (int j = cut2 + 1; j < num_city; j++) {
		while (flag[A[i]]) i++;
		D[j] = A[i++];
	}

}

/* 2-OPT local search function */
void TwoOpt1(int D[]) {
	double best_delta, dist1, dist2;
	int a, b;

	while (true) {
		best_delta = 0.0;	// length difference of reversed path and original path

		// For every combination of i and j s.t. 0 <= i < j < num_city and j > i+1,
		// calculate the length difference when reversed the path [i+1, j]
		// Save the best i and j, to a and b
		for (int i = 0; i < num_city - 2; i++) {
			for (int j = i + 2; j < num_city; j++) {

				dist1 = distant(city[D[i]], city[D[j]]) +
					distant(city[D[i + 1]], city[D[(j + 1) % num_city]]);
				dist2 = distant(city[D[i]], city[D[i + 1]]) +
					distant(city[D[j]], city[D[(j + 1) % num_city]]);

				if (best_delta >(dist1 - dist2)) {
					best_delta = dist1 - dist2;
					a = i;
					b = j;
				}
			}
		}

		// If the better solution exists, reverse the path [a+1, b]
		// Else, exit the function
		if (best_delta < -0.001) {
			for (a++; a < b; a++, b--) {
				int tmp = D[a];
				D[a] = D[b];
				D[b] = tmp;
			}
		}
		else break;
	}

}

/* Calculates length of the travel path of an individual */
double eval(int S[])
{
	int i;
	double cost = 0.0;

	for (i = 1; i < num_city; i++) {
		cost += distant(city[S[i]], city[S[i - 1]]);
	}
	cost += distant(city[S[0]], city[S[num_city - 1]]);

	return cost;
}

/* Calculates the distance between two points */
double distant(Point A, Point B)
{
	double dx = (double)(B.x - A.x);
	double dy = (double)(B.y - A.y);
	return sqrt(dx*dx + dy*dy);
}

/* Get input data from the file */
void input()
{
	int i, t;
	char name[31] = "./TSP Samples/ch130.tsp";	// TSP Sample
	FILE *fp;

	fp = fopen(name, "r");
	fscanf(fp, "%d", &num_city);
	for (i = 0; i < num_city; i++) {
		fscanf(fp, "%d %lf %lf", &t, &(city[i].x), &(city[i].y));
	}
}