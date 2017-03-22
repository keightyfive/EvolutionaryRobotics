#include "population.h"
#include "genotype.h"
#include "random.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

// part of the population that is cloned from one generation to the next
static const double ELITE_PART = 0.1;
static const double KEEP_PART=0.5;

struct _Population_ {
  Genotype *genotypes;  // genotypes
  int size;             // population size
};

Population population_create(int pop_size, int gen_size) {
  Population p = malloc(sizeof(struct _Population_));
  p->size = pop_size;
  genotype_set_size(gen_size);
  p->genotypes = (Genotype*)malloc(p->size * sizeof(Genotype));
  int i;
  for (i = 0; i < p->size; i++)
    p->genotypes[i] = genotype_create();
    
  return p;
}


void population_destroy(Population p) {
  int i;
  for (i = 0; i < p->size; i++)
    genotype_destroy(p->genotypes[i]);

  free(p->genotypes);
}

// rank selection: the population need to be sorted by fitness
Genotype population_select_parent(Population p) {
  while (1) {
    int index = random_get_integer(p->size);
    if (index <= random_get_integer(p->size))
      return p->genotypes[index];
  }
}

Genotype tournament(Population p){
  int size=2;
  int indices[size];
  int i;
  for (i=0;i<size;i++){
    indices[i]=random_get_integer(p->size);
  }
  int max_i=0;
  double max_fit=-999;
  for (i=0;i<size;i++){
    double fit=genotype_get_fitness(p->genotypes[indices[i]]);
    if (fit>max_fit){
      max_fit=fit;
      max_i=i;
    }
  }
  return p->genotypes[indices[max_i]];
  
}

// comparison function for qsort()
static int compare_genotype(const void *a, const void *b) {
  return genotype_get_fitness(*((Genotype*)a)) > genotype_get_fitness(*((Genotype*)b)) ? -1 : +1;
}

void population_reproduce(Population p) {

  // quick sort for rank selection
  qsort(p->genotypes, p->size, sizeof(Genotype), compare_genotype);
  
  
  int i;
  for (i = 0; i < p->size; i++) {
    printf("fitness %d: %lf\n",i, genotype_get_fitness(p->genotypes[i]));
  }
  
  // create new generation
  Genotype *next_generation = malloc(p->size * sizeof(Genotype));
  for (i = 0; i < p->size; i++) {
    Genotype child;
    
    if (i < ELITE_PART * p->size) {
      // cloned elite
      //printf("** select fitness %d: %lf\n",i, genotype_get_fitness(p->genotypes[i]));
      child = genotype_clone(p->genotypes[i]);
    }
    else if (i< KEEP_PART * p->size) {
      Genotype selec=tournament(p);
      child=genotype_clone(selec);
      printf("*** select fitness %lf \n",genotype_get_fitness(selec));
      genotype_mutate(child);
    }else{
      // sexual reproduction
      // or asexual if both parents are the same individual
      Genotype mom = tournament(p);
      Genotype dad = tournament(p);
      
      printf("*** mate fitness %lf, %lf \n", genotype_get_fitness(mom),genotype_get_fitness(dad));
      child = genotype_crossover2(mom, dad); 
      genotype_mutate(child);
    }

    genotype_set_fitness(child, 0.0);
    next_generation[i] = child;
  }

  // destroy old generation
  for (i = 0; i < p->size; i++)
    genotype_destroy(p->genotypes[i]);

  free(p->genotypes);

  // switch generation pointers
  p->genotypes = next_generation;
  
  //for (i = 0; i < p->size; i++) {
    //printf("fitness %d: %lf\n",i, genotype_get_fitness(p->genotypes[i]));
  //}
  
}

Genotype population_get_fittest(Population p) {
  Genotype fittest = p->genotypes[0];
  int i;
  for (i = 1; i < p->size; i++) {
    Genotype candidate = p->genotypes[i];
    if (genotype_get_fitness(candidate) > genotype_get_fitness(fittest))
      fittest = candidate;
  }
  
  return fittest;
}

Genotype population_get_genotype(Population p, int index) {
  return p->genotypes[index];
}

double population_compute_average_fitness(Population p) {
  double sum_fitness = 0.0;
  int i;
  for (i = 0; i < p->size; i++)
    sum_fitness += genotype_get_fitness(p->genotypes[i]);
    
  return sum_fitness / p->size;
}
