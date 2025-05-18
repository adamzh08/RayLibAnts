#ifndef NEURAL_H
#define NEURAL_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Layer structure defining a single neural network layer
struct Layer
{
  int amount_of_neurons;             // Number of neurons in this layer
  float (*activation)(float);        // Activation function for this layer
};

// Network structure containing the full neural network architecture
struct Network
{
  int amount_of_layers;             // Total number of layers
  struct Layer *layers;             // Array of layer definitions
  float ***weights;                 // 3D array of weights [layer][output_neuron][input_neuron]
};

// Load network weights from a binary file
int loadWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "rb");
  if (!file)
    return 0;

  // Load weights for each layer
  for (int layer = 0; layer < net->amount_of_layers - 1; layer++)
  {
    for (int output_neuron = 0; output_neuron < net->layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      fread(net->weights[layer][output_neuron], sizeof(float), 
            net->layers[layer].amount_of_neurons + 1, file);
    }
  }
  fclose(file);
  return 1;
}

// Save network weights to a binary file
int saveWeights(struct Network *net, const char *filename)
{
  FILE *file = fopen(filename, "wb");
  if (!file)
    return 0;

  // Save weights for each layer
  for (int layer = 0; layer < net->amount_of_layers - 1; layer++)
  {
    for (int output_neuron = 0; output_neuron < net->layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      fwrite(net->weights[layer][output_neuron], sizeof(float),
             net->layers[layer].amount_of_neurons + 1, file);
    }
  }
  fclose(file);
  return 1;
}

// Initialize weights using Xavier/Glorot initialization
void randomizeWeights(struct Network *net)
{
  for (int layer = 0; layer < net->amount_of_layers - 1; layer++)
  {
    // Calculate Xavier/Glorot scaling factor
    float scale = sqrtf(2.0f / (net->layers[layer].amount_of_neurons + 
                               net->layers[layer + 1].amount_of_neurons));

    for (int output_neuron = 0; output_neuron < net->layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      for (int input_neuron = 0; input_neuron < net->layers[layer].amount_of_neurons + 1; input_neuron++)
      {
        // Generate random weight between -1 and 1, then scale
        float random_weight = ((float)rand() / RAND_MAX * 2.0f - 1.0f);
        net->weights[layer][output_neuron][input_neuron] = random_weight * scale;
      }
    }
  }
}

// Apply random mutations to network weights
void mutateWeights(struct Network *net)
{
  for (int layer = 0; layer < net->amount_of_layers - 1; layer++)
  {
    for (int output_neuron = 0; output_neuron < net->layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      for (int input_neuron = 0; input_neuron < net->layers[layer].amount_of_neurons + 1; input_neuron++)
      {
        // 30% chance to mutate each weight
        if ((float)rand() / RAND_MAX < 0.3)
        {
          // Add random value between -0.5 and 0.5
          net->weights[layer][output_neuron][input_neuron] += ((float)rand() / RAND_MAX) - 0.5;
        }
      }
    }
  }
}

// Initialize a new neural network with given architecture
struct Network initNetwork(struct Layer layers[], int amount_of_layers)
{
  struct Network net;
  net.layers = layers;
  net.amount_of_layers = amount_of_layers;

  // Allocate memory for network weights
  net.weights = (float ***)malloc((amount_of_layers - 1) * sizeof(float **));

  for (int layer = 0; layer < amount_of_layers - 1; layer++)
  {
    net.weights[layer] = (float **)malloc((layers[layer + 1].amount_of_neurons) * sizeof(float *));

    for (int output_neuron = 0; output_neuron < layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      // Add 1 for bias weight
      net.weights[layer][output_neuron] = (float *)malloc((layers[layer].amount_of_neurons + 1) * sizeof(float));
    }
  }

  return net;
}

// Hyperbolic tangent activation function
static inline float tanh_func(float x)
{
  return tanhf(x);
}

// Forward propagation through the network
static inline float *feed_forward(struct Network net, float input[])
{
  // Initialize first layer with input values
  float *current_layer = (float *)malloc(net.layers[0].amount_of_neurons * sizeof(float));
  memcpy(current_layer, input, net.layers[0].amount_of_neurons * sizeof(float));

  float *next_layer = NULL;

  // Process each layer
  for (int layer = 0; layer < net.amount_of_layers - 1; layer++)
  {
    next_layer = (float *)calloc(net.layers[layer + 1].amount_of_neurons, sizeof(float));

    // Calculate each neuron's output
    for (int output_neuron = 0; output_neuron < net.layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      // Sum weighted inputs
      for (int input_neuron = 0; input_neuron < net.layers[layer].amount_of_neurons; input_neuron++)
      {
        next_layer[output_neuron] += current_layer[input_neuron] * 
                                    net.weights[layer][output_neuron][input_neuron];
      }

      // Add bias and apply activation function
      next_layer[output_neuron] = net.layers[layer + 1].activation(
          next_layer[output_neuron] + 
          net.weights[layer][output_neuron][net.layers[layer].amount_of_neurons]
      );
    }

    free(current_layer);
    current_layer = next_layer;
    next_layer = NULL;
  }

  return current_layer;
}

// Free all allocated memory for the network
void freeNetwork(struct Network *net)
{
  for (int layer = 0; layer < net->amount_of_layers - 1; layer++)
  {
    for (int output_neuron = 0; output_neuron < net->layers[layer + 1].amount_of_neurons; output_neuron++)
    {
      free(net->weights[layer][output_neuron]);
    }
    free(net->weights[layer]);
  }
  free(net->weights);
}

#endif