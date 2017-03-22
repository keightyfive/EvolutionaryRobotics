//   Description:   Supervisor code for genetic algorithm

#include "genotype.h"
#include "population.h"
#include "rnn.h"

#include <webots/supervisor.h>
#include <webots/robot.h>
#include <webots/emitter.h>
#include <webots/display.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include <stdlib.h>

static const int INPUT_NUM=8;
static const int OUTPUT_NUM=2;

static const int HIDDEN_LAYER_NUM=1;
static const int HIDDEN_LAYER_NEURON_NUM=5;

static const int POPULATION_SIZE = 50;
static const int NUM_GENERATIONS = 120;
static const char *FILE_NAME = "fittest.txt";

// must match the values in the advanced_genetic_algorithm.c code
static const int NUM_SENSORS = 8;
static const int NUM_WHEELS  = 2;

int GENOTYPE_SIZE=-1;
// index access
enum { X, Y, Z };

static int time_step;
static WbDeviceTag emitter;   // to send genes to robot
static WbDeviceTag display;   // to display the fitness evolution
static int display_width, display_height;

// the GA population
static Population population;

// for reading or setting the robot's position and orientation
WbFieldRef robot_translation;
WbFieldRef robot_rotation;
static double robot_trans0[3];  // a translation needs 3 doubles
static double robot_rot0[4];    // a rotation needs 4 doubles
  
// start with a demo until the user presses the 'O' key
// (change this if you want)
static bool demo = true;

//current id
int robot_node_id;
bool enable_log=false;
void log_(char* str){
  if (!enable_log) return;
    FILE *outfile = fopen("log.txt", "a");
  if (outfile) {
      fprintf(outfile, "%s\n",str);
      fclose(outfile);
      //printf("log: %s\n", str);
    }
    else
      printf("unable to write %s\n", FILE_NAME);
}
void draw_scaled_line(int generation, double y1, double y2) {
  const double XSCALE = (double)display_width / NUM_GENERATIONS;
  const double YSCALE = 10.0;
  wb_display_draw_line(display, (generation - 0.5) * XSCALE, display_height - y1 * YSCALE,
    (generation + 0.5) * XSCALE, display_height - y2 * YSCALE);
}

// plot best and average fitness
void plot_fitness(int generation, double best_fitness, double average_fitness) {
  static double prev_best_fitness = 0.0;
  static double prev_average_fitness = 0.0;
  if (generation > 0) {  
    wb_display_set_color(display, 0xff0000); // red
    draw_scaled_line(generation, prev_best_fitness, best_fitness);

    wb_display_set_color(display, 0x00ff00); // green
    draw_scaled_line(generation, prev_average_fitness, average_fitness);
  }

  prev_best_fitness = best_fitness;
  prev_average_fitness = average_fitness;
}
 
// run the robot simulation for the specified number of seconds
double distanceP2P(double* p1, double* p2){
  double dx=p1[0]-p2[0];
  double dy=p1[1]-p2[1];
  return sqrt(dx*dx+dy*dy);
}

void run_seconds(double seconds, Genotype genotype) {

  log_("run sec\n");
  //award points for exploration
  int award_point_num=10;
  double award_distance=0.05;
  double award_points[award_point_num][2];
  double award_value[award_point_num];
  int award_point_i=0;
  int z;
  for (z=0;z<award_point_num;z++){
    award_points[z][0]=-0.5+1.0/award_point_num*z;
    award_points[z][1]=0;
    award_value[z]=1;
  }
  
  
  
  double award=0;
  
  
  WbNodeRef robot = wb_supervisor_node_get_from_def("ROBOT");
  double robot_trans[3];
  int punish_step=0;
  
  double robot_p_old[2];
  robot_p_old[0]=0;
  robot_p_old[1]=0;
  double punish=0.996;
  
  int i, n = 1000.0 * seconds / time_step;
  for (i = 0; i < n; i++) {
  
    
    WbFieldRef current_translation = wb_supervisor_node_get_field(robot, "translation");
    memcpy(robot_trans, wb_supervisor_field_get_sf_vec3f(current_translation), sizeof(robot_trans));
    //check punishment for going dangerous
    if (robot_trans[2]<-0.05 || robot_trans[2]>0.05){
      punish_step++;
    }
    //check if already fall off
    if (robot_trans[1]<0.4){
      //punish_step+=n-i;
      break;
    }
    //check if reward
    double robot_p[2];
    robot_p[0]=robot_trans[0];
    robot_p[1]=robot_trans[2];
    
    double movement=distanceP2P(robot_p, robot_p_old);
    if (movement<0.00001){
      punish*=0.996;
    }
    robot_p_old[0]=robot_p[0];
    robot_p_old[1]=robot_p[1];
    
    int k;
    //check if arraive to a new award point
    for (k=0;k<award_point_num;k++){
      double dist=distanceP2P(award_points[k], robot_p);
      if (dist<award_distance && !(k==award_point_i)){
        award_point_i=k;
        award+=award_value[award_point_i];
        award_value[award_point_i]-=0.1;
      }
    }
    
    
    if (demo && wb_robot_keyboard_get_key() == 'O') {
      demo = false;
      return; // interrupt demo and start GA optimization
    }

    wb_robot_step(time_step);
  }
  //final award
  WbFieldRef current_translation = wb_supervisor_node_get_field(robot, "translation");
  memcpy(robot_trans, wb_supervisor_field_get_sf_vec3f(current_translation), sizeof(robot_trans));
  double robot_p[2];
  robot_p[0]=robot_trans[0];
  robot_p[1]=robot_trans[2];
  double dist=distanceP2P(award_points[award_point_i], robot_p);
  award+=(1-dist);
  //take out init reward
  award-=0.5;
  award=award*punish;
  double fitness=award*((double)(n-punish_step))/(double)n;
  
  genotype_set_fitness(genotype, fitness);
}

void sleep(int sec){
  
  int i, n = 1000.0 * sec / time_step;
  for (i = 0; i < n; i++) {
  
    wb_robot_step(time_step);
  
  }
}

void reset_robot(){

  
  log_("reset_robot\n");
  //wb_supervisor_field_set_sf_rotation(robot_rotation, robot_rot0);
  //wb_supervisor_field_set_sf_vec3f(robot_translation, robot_trans0);
  int i=0;
  wb_emitter_send(emitter, &i, sizeof(int));
  
  sleep(2);
  
  WbNodeRef robot = wb_supervisor_node_get_from_id(robot_node_id);
  
  log_("reset_robot REMOVE\n");
  wb_supervisor_node_remove (robot);
  
  
  
  log_("reset_robot A\n");
  WbNodeRef root_node = wb_supervisor_node_get_root();
  WbFieldRef root_children_field = wb_supervisor_node_get_field(root_node, "children");
  
  log_("reset_robot before import\n");
  wb_supervisor_field_import_mf_node(root_children_field,4,"E-puck.wbo");

      
  log_("reset_robot import ded\n");
  robot = wb_supervisor_node_get_from_def("ROBOT");
  robot_node_id=wb_supervisor_node_get_id(robot);
  
  
  sleep(1);
  
  log_("reset_robot done\n");
}

// evaluate one genotype at a time
void evaluate_genotype(Genotype genotype, int generation_num) {
  
  log_("evaluate_genotype \n");
  // send genotype to robot for evaluation
  //int k;
  //for (k=0;k<GENOTYPE_SIZE;k++){
    //printf("sending %lf,",genotype_get_genes(genotype)[k]);
        
  //}
  
 // printf("\n");
  reset_robot();
  wb_emitter_send(emitter, genotype_get_genes(genotype), GENOTYPE_SIZE * sizeof(double));
  
  //printf("sent\n");
  // reset robot and load position
  // evaluation genotype during one minute
  double t=(double)(generation_num+1)*10;
  if (t>60) t=60;
  if (generation_num==-1){
    t=300;
  }
  run_seconds((double)t, genotype);
  
  // measure fitness
  double fitness = genotype_get_fitness(genotype);

  printf("fitness: %g\n", fitness);
  
  log_("evaluate_genotype done\n");
}

void run_optimization() {

  log_("run_optimization\n");
  wb_robot_keyboard_disable();

  printf("---\n");
  printf("starting GA optimization ...\n");
  printf("population size is %d, genome size is %d\n", POPULATION_SIZE, GENOTYPE_SIZE);

  int i, j;
  for  (i = 0; i < NUM_GENERATIONS; i++) {    
    for (j = 0; j < POPULATION_SIZE; j++) {
      printf("generation: %d, genotype: %d\n", i, j);

      // evaluate genotype
      Genotype genotype = population_get_genotype(population, j);
      
      evaluate_genotype(genotype, i);
    }
  
    double best_fitness = genotype_get_fitness(population_get_fittest(population));
    double average_fitness = population_compute_average_fitness(population);
    
    // display results
    plot_fitness(i, best_fitness, average_fitness);
    printf("best fitness: %g\n", best_fitness);
    printf("average fitness: %g\n", average_fitness);
    Genotype fittest = population_get_fittest(population);
    FILE *outfile = fopen(FILE_NAME, "w");
    if (outfile) {
      genotype_fwrite(fittest, outfile);
      fclose(outfile);
      printf("wrote best genotype into %s\n", FILE_NAME);
    }
    else
      printf("unable to write %s\n", FILE_NAME);
    // reproduce (but not after the last generation)
    if (i < NUM_GENERATIONS - 1)
      population_reproduce(population);
  }
  
  printf("GA optimization terminated.\n");

  // save fittest individual
  
  
  population_destroy(population);
}
  
// show demo of the fittest individual
void run_demo() {
  wb_robot_keyboard_enable(time_step);
  
  printf("---\n");
  printf("running demo of best individual ...\n");
  printf("select the 3D window and push the 'O' key\n");
  printf("to start genetic algorithm optimization\n");

  FILE *infile = fopen(FILE_NAME, "r");
  if (! infile) {
    printf("unable to read %s\n", FILE_NAME);
    return;
  }
  
  Genotype genotype = genotype_create();
  genotype_fread(genotype, infile);
  fclose(infile);
  
  while (demo)
    evaluate_genotype(genotype,-1);
}

int main(int argc, const char *argv[]) {
  
  // initialize Webots
  srand((unsigned)time(NULL));
  wb_robot_init();
  // get simulation step in milliseconds
  time_step = wb_robot_get_basic_time_step();

  // the emitter to send genotype to robot
  emitter = wb_robot_get_device("emitter");
  
  // to display the fitness evolution
  display = wb_robot_get_device("display");
  display_width = wb_display_get_width(display);
  display_height = wb_display_get_height(display);
  wb_display_draw_text(display, "fitness", 2, 2);

  //create a dummy network to measure encode size
  
  NeuralNet dummy = new_net(INPUT_NUM,OUTPUT_NUM,HIDDEN_LAYER_NUM,HIDDEN_LAYER_NEURON_NUM);
  GENOTYPE_SIZE=get_encode_length(dummy);
  printf("genosize= %d\n",GENOTYPE_SIZE);
  // initial population
  population = population_create(POPULATION_SIZE, GENOTYPE_SIZE);
  
  // find robot node and store initial position and orientation
  WbNodeRef robot = wb_supervisor_node_get_from_def("ROBOT");
  robot_node_id=wb_supervisor_node_get_id(robot);
  
  robot_translation = wb_supervisor_node_get_field(robot, "translation");
  robot_rotation = wb_supervisor_node_get_field(robot, "rotation");
  memcpy(robot_trans0, wb_supervisor_field_get_sf_vec3f(robot_translation), sizeof(robot_trans0));
  memcpy(robot_rot0, wb_supervisor_field_get_sf_rotation(robot_rotation), sizeof(robot_rot0));

  
  if (demo){
    run_demo();
    
  }else{
  // run GA optimization
  run_optimization();
  }
  // cleanup Webots
  wb_robot_cleanup();
  return 0;  // ignored
}
