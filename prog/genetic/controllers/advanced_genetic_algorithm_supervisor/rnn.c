#include "rnn.h"
#include "random.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>

static const bool isBackwardRecurrent=false;
static const bool isSelfRecurrent=true;
double fast_sigmoid(double x){
    return x/(1+abs(x));
}

struct _neuron_ {
    int forward_InputNum;
    int backward_InputNum;
    double cur_val;
    double new_val;
    double self_weight;
    double* fVec;
    double* bVec;
};

struct _neuron_layer_ {
    int neuron_num;
    Neuron* neurons;
};

struct _neural_net_ {
    int layer_num;
    int input_num;
    int output_num;
    int hidden_layer_num;
    int hidden_layer_neuron_num;
    NeuronLayer* layers;
};

double rand_weight(){
	return ((double)rand()/(double)RAND_MAX)-0.5;
}

Neuron new_neuron(int finput, int binput){
    Neuron n=malloc(sizeof(struct _neuron_));
    n->forward_InputNum=finput;
    n->backward_InputNum=binput;
    n->cur_val=0;
    n->new_val=0;
    //weight for self connection
    
    n->self_weight=rand_weight();
    n->fVec=(double*)malloc(finput*sizeof(double));
    n->bVec=(double*)malloc(binput*sizeof(double));
    
    int i;
    for (i=0;i<finput;i++){
      double random=rand_weight();
      n->fVec[i]=random;
    }
    for (i=0;i<binput;i++){
      double random=rand_weight();
      n->bVec[i]=random;
    }
    
    return n;
}

double eval_neuron(Neuron n, double* finputs, double* binputs){
    int i;
    double sum=0;
    //printf("fin_n=%d\n",n->forward_InputNum);
    //printf("(neuron val=");
    for (i=0;i<n->forward_InputNum;i++){
        double v=n->fVec[i]*finputs[i];
        sum+=v;
       // printf("%lf * %lf +",n->fVec[i],finputs[i]);
    }
    
    for (i=0;i<n->backward_InputNum;i++){
        sum+=n->bVec[i]*binputs[i];
    }
    if (isSelfRecurrent){
        sum+=n->cur_val*n->self_weight;
    }
    //activation function
    sum=fast_sigmoid(sum);
    //printf("= %lf)",sum);
    n->new_val=sum;
    return sum;
}

void destroy_neuron(Neuron n){
    free(n->fVec);
    free(n->bVec);
    free(n);
}

NeuronLayer new_layer(int neuron_num, int finput, int binput){
    NeuronLayer nl=malloc(sizeof(struct _neuron_layer_));
    nl->neuron_num=neuron_num;
    nl->neurons=(Neuron*)malloc(sizeof(Neuron)*neuron_num);
    int i;
    for (i=0;i<neuron_num;i++){
        nl->neurons[i]=new_neuron(finput,binput);
    }
    return nl;
}

void destroy_layer(NeuronLayer nl){
    int i;
    for (i=0;i<nl->neuron_num;i++){
        destroy_neuron(nl->neurons[i]);
    }
    free(nl);
}

NeuralNet new_net(int input_num, int output_num, int hidden_layer_num, int hidden_layer_neuron_num){
    NeuralNet nn=malloc(sizeof(struct _neural_net_));
    nn->input_num=input_num;
    nn->output_num=output_num;
    nn->hidden_layer_num=hidden_layer_num;
    nn->hidden_layer_neuron_num=hidden_layer_neuron_num;
    //input+hidden+output
    nn->layer_num=2+hidden_layer_num;
    nn->layers=(NeuronLayer*)malloc(nn->layer_num*sizeof(NeuronLayer));
    //prepare layer neuron number
    int layer_neuron_num[nn->layer_num];
    layer_neuron_num[0]=input_num;
    layer_neuron_num[nn->layer_num-1]=output_num;
    int i;
    for (i=1;i<nn->layer_num-1;i++){
      layer_neuron_num[i]=hidden_layer_neuron_num;
    }
    
    //create input layer
    
    
    int binput_firstlayer;
    if (isBackwardRecurrent){
        binput_firstlayer=layer_neuron_num[1];
    }else{
        binput_firstlayer=0;
    }
    nn->layers[0]=new_layer(input_num, 0, binput_firstlayer);
    //create output layer
    nn->layers[nn->layer_num-1]=new_layer(output_num, layer_neuron_num[nn->layer_num-2], 0);
    
    //create hiddenlayers
    for (i=1;i< nn->layer_num-1;i++){
        int finput=nn->layers[i-1]->neuron_num;
        int binput;
        if (i< nn->layer_num-2) {
            binput=hidden_layer_neuron_num;
        }else{
            binput=output_num;
        }
        if (!isBackwardRecurrent){
            binput=0;
        }
        
        nn->layers[i]=new_layer(hidden_layer_neuron_num, finput, binput);
    }
    return nn;
}

void destroy_net(NeuralNet nn){
    int i;
    for (i=0;i< nn->layer_num;i++){
        destroy_layer(nn->layers[i]);
    }
    free(nn);
}

double* evaluate_net(double* inputs, NeuralNet nn){
    //layer index
    int li;
    //neuron index at each layer
    int ni;
    double* result=(double*)malloc(sizeof(double)*nn->output_num);
    //calculate value of each neuron for current time step from last step
    
    //printf("eval net.\n");
    for (li=0; li<nn->layer_num; li++){
        if (li==0){
            //set input 
            
            //printf("ln=%d, inputs=[",nn->layer_num);
            for(ni=0;ni<nn->layers[li]->neuron_num;ni++){
                
               // printf("%lf,",inputs[ni]);
                nn->layers[li]->neurons[ni]->cur_val=inputs[ni];
                nn->layers[li]->neurons[ni]->new_val=inputs[ni];
            }
            //printf("]\n");
        }else{
            double* finputs;
            double* binputs;
            //forward input number, which is the neuron number of previous layer
            int fi_num=nn->layers[li-1]->neuron_num;
            finputs=(double*)malloc(sizeof(double)*fi_num);
            int i;
            for (i=0;i<fi_num;i++){
              finputs[i]=nn->layers[li-1]->neurons[i]->cur_val;
            }
            
            //backward input num, which is the neuron number of next layer
            int bi_num;
            if (li<nn->layer_num-1){
              bi_num=nn->layers[li+1]->neuron_num;
              binputs=(double*)malloc(sizeof(double)*bi_num);
            }else{
              bi_num=0;
            } 
            for (i=0;i<bi_num;i++){
              binputs[i]=nn->layers[li+1]->neurons[i]->cur_val;
            }
           // printf("fi=%d,bi=%d\n",fi_num,bi_num);
            //printf("layer %d, val=[",li);
            for(ni=0;ni<nn->layers[li]->neuron_num;ni++){
                double val=eval_neuron(nn->layers[li]->neurons[ni],finputs,binputs);
                
             //   printf("%lf\n,",val);
                
                if (li==nn->layer_num-1){
                    //if already the last layer, take the value for output 
                    result[ni]=val;
                }
            }
            //printf("]\n");
        }
    }
    //update neuron value with the new value
    for (li=0;li<nn->layer_num;li++){
        for (ni=0;ni<nn->layers[li]->neuron_num;ni++){
            Neuron n=nn->layers[li]->neurons[ni];
            n->cur_val=n->new_val;
        }
    }
    return result;
}

int get_encode_length(NeuralNet nn){
    int li;
    int ni;
    int encode_length=0;
    //compute length
    for (li=0;li<nn->layer_num;li++){
        NeuronLayer layer=nn->layers[li];
        for (ni=0;ni<layer->neuron_num;ni++){
            Neuron n=layer->neurons[ni];
            encode_length+=n->backward_InputNum+n->forward_InputNum+1;
        }
    }
    return encode_length;
}

double* encode_net(NeuralNet nn){
    int li;
    int ni;
    int encode_length=get_encode_length(nn);
    //compute length
    /*
    for (li=0;li<nn->layer_num;li++){
        NeuronLayer layer=nn->layers[li];
        for (ni=0;ni<layer->neuron_num;ni++){
            Neuron n=layer->neurons[ni];
            encode_length+=n->backward_InputNum+n->forward_InputNum+1;
        }
    }
    */
    //
    
    double* result=(double*)malloc(sizeof(double)*encode_length);
    int encode_i=0;
    for (li=0;li<nn->layer_num;li++){
        NeuronLayer layer=nn->layers[li];
        for (ni=0;ni<layer->neuron_num;ni++){
            Neuron n=layer->neurons[ni];
            //
            int i;
            for (i=0;i<n->forward_InputNum;i++){
                result[encode_i]=n->fVec[i];
                encode_i++;
            }
            for (i=0;i<n->backward_InputNum;i++){
                result[encode_i]=n->bVec[i];
                encode_i++;
            }
            result[encode_i]=n->self_weight;
			encode_i++;
        }
    }
    return result;
}

NeuralNet decode_net(double* d_array, int input_num, int output_num, int hidden_layer_num, int hidden_layer_neuron_num){
	NeuralNet nn=new_net(input_num, output_num, hidden_layer_num, hidden_layer_neuron_num);
    int li;
    int ni;
	int encode_i=0;
	for (li=0;li<nn->layer_num;li++){
        NeuronLayer layer=nn->layers[li];
        for (ni=0;ni<layer->neuron_num;ni++){
            Neuron n=layer->neurons[ni];
            //
            int i;
            for (i=0;i<n->forward_InputNum;i++){
                n->fVec[i]=d_array[encode_i];
                encode_i++;
            }
            for (i=0;i<n->backward_InputNum;i++){
                n->bVec[i]=d_array[encode_i];
                encode_i++;
            }
            n->self_weight=d_array[encode_i];
			encode_i++;
        }
    }
	return nn;
}

