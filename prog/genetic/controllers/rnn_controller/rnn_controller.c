// Description:   Robot execution code for genetic algorithm


#include <webots/robot.h>
#include <webots/differential_wheels.h>
#include <webots/receiver.h>
#include <webots/distance_sensor.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "../advanced_genetic_algorithm_supervisor/rnn.h"

#define NUM_SENSORS 8
#define NUM_WHEELS 2
#define INPUT_NUM 8
#define OUTPUT_NUM 2
#define HIDDEN_LAYER_NUM 1
#define HIDDEN_LAYER_NEURON_NUM 5

//The neural network of current run
NeuralNet network;
int GENOTYPE_SIZE=-1;


WbDeviceTag sensors[NUM_SENSORS];  // proximity sensors
WbDeviceTag receiver;              // for receiving genes from Supervisor
int receive_num=0;

int is_exit=0;


// check if a new set of genes was sent by the Supervisor
// in this case start using these new genes immediately
void check_for_new_genes() {
  if (wb_receiver_get_queue_length(receiver) > 0) {
    
    if (wb_receiver_get_data_size(receiver)==sizeof(int)){
       int* i;
       memcpy(i, wb_receiver_get_data(receiver), 1 * sizeof(int));
       is_exit=1;
       return;
    }
    
    wb_differential_wheels_set_speed(0,0);
    printf("receiving new genotype!\n");
    // check that the number of genes received match what is expected
    assert(wb_receiver_get_data_size(receiver) == GENOTYPE_SIZE * sizeof(double));
    
    double geno[GENOTYPE_SIZE];
    // copy new genes directly in the sensor/actuator matrix
    // we don't use any specific mapping nor left/right symmetry
    // it's the GA's responsability to find a functional mapping
    memcpy(geno, wb_receiver_get_data(receiver), GENOTYPE_SIZE * sizeof(double));
    
    destroy_net(network);
    
    network=decode_net(geno, INPUT_NUM,OUTPUT_NUM,HIDDEN_LAYER_NUM,HIDDEN_LAYER_NEURON_NUM);
    receive_num++;
    // prepare for receiving next genes packet
    wb_receiver_next_packet(receiver);
    
  }
}

static double clip_value(double value, double min_max) {
  if (value > min_max)
    return min_max;
  else if (value < -min_max)
    return -min_max;

  return value;
}

void sense_compute_and_actuate() {
  // read sensor values
  if (receive_num==0) return;
  double sensor_values[NUM_SENSORS];
  int i;
  for (i = 0; i < NUM_SENSORS; i++){
    sensor_values[i] = wb_distance_sensor_get_value(sensors[i])/(double)512;
    //printf("s%d=%lf,",i,sensor_values[i]);
  }

  // compute actuation using Braitenberg's algorithm:
  // The speed of each wheel is computed by summing the value
  // of each sensor multiplied by the corresponding weight of the matrix.
  // By chance, in this case, this works without any scaling of the sensor values nor of the
  // wheels speed but this type of scaling may be necessary with a different problem
  double* wheel_speed;
  wheel_speed = evaluate_net(sensor_values, network);
  //printf("- %lf,%lf\n",wheel_speed[0],wheel_speed[1]);
  // clip to e-puck max speed values to avoid warning
  wheel_speed[0] = clip_value(wheel_speed[0]*500, 1000.0);
  wheel_speed[1] = clip_value(wheel_speed[1]*500, 1000.0);

  //printf("= %lf,%lf\n",wheel_speed[0],wheel_speed[1]);
  // actuate e-puck wheels
  wb_differential_wheels_set_speed(wheel_speed[0], wheel_speed[1]);
  
  //wb_differential_wheels_set_speed(0,0);
}

int main(int argc, const char *argv[]) {
  
  //measure the network encoding length
  
  NeuralNet dummy = new_net(INPUT_NUM,OUTPUT_NUM,HIDDEN_LAYER_NUM,HIDDEN_LAYER_NEURON_NUM);
  GENOTYPE_SIZE=get_encode_length(dummy);
  network=dummy;
  wb_robot_init();  // initialize Webots

  // find simulation step in milliseconds (WorldInfo.basicTimeStep)
  int time_step = wb_robot_get_basic_time_step();
    
  // find and enable proximity sensors
  char name[32];
  int i;
  for (i = 0; i < NUM_SENSORS; i++) {
    sprintf(name, "ps%d", i);
    //printf("[%s, %d]\n",name, i);
    sensors[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(sensors[i], time_step);
  }
    
  // find and enable receiver
  receiver = wb_robot_get_device("receiver");
  wb_receiver_enable(receiver, time_step);
  printf("run_controller starts!\n");
  // run until simulation is restarted
  while (wb_robot_step(time_step) != -1) {
    check_for_new_genes();
    if (is_exit==1) break;
    sense_compute_and_actuate();
    
  }
  
  wb_robot_cleanup();  // cleanup Webots
  return 0;            // ignored
}
