#ifndef RNN_H
#define RNN_H

#include <stdio.h>

typedef struct _neuron_ *Neuron;

typedef struct _neuron_layer_ *NeuronLayer;

typedef struct _neural_net_ *NeuralNet;

Neuron new_neuron(int finput, int binput);

void destroy_neuron(Neuron n);

NeuronLayer new_layer(int neuron_num, int finput, int binput);

void destroy_layer(NeuronLayer nl);

NeuralNet new_net(int input_num, int output_num, int hidden_layer_num, int hidden_layer_neuron_num);

void destroy_net(NeuralNet nn);

double* evaluate_net(double* inputs, NeuralNet nn);

int get_encode_length(NeuralNet nn);

double* encode_net(NeuralNet nn);

NeuralNet decode_net(double* d_array, int input_num, int output_num, int hidden_layer_num, int hidden_layer_neuron_num);

#endif