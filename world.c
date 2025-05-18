#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Window dimensions
#define SCREEN_WIDTH 1200
#define SCREEN_HEIGHT 800

// Neural network and simulation parameters
#define AMOUNT_OF_RAYS 20    // Number of sensor rays for each ant
#define RAYS_RADIUS 100      // How far each ray can detect
#define AMOUNT_OF_ANTS 3000  // Population size
#define MAX_SPEED 1          // Maximum movement speed of ants

#include "neural.h"
#include "world.h"

int main(void)
{
  // Initialize random seed
  srand(time(NULL));
  
  // Initialize lookup tables for ray calculations
  init_trigonometry_tables();
  
  // Create window and set up simulation environment
  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Ant Colony Neural Evolution");

  // Load and prepare image resources
  Image background = LoadImage("Labyrint.png");  // Maze/environment image
  Image ant = LoadImage("ant.png");             // Ant sprite

  // Convert images to textures for rendering
  Texture2D background_texture = LoadTextureFromImage(background);
  UnloadImage(background);  // Free original image after texture creation

  // Initialize background for collision detection
  init_bg(background_texture);

  Texture2D ant_texture = LoadTextureFromImage(ant);
  UnloadImage(ant);

  // Define neural network architecture:
  // Input layer: AMOUNT_OF_RAYS neurons (sensor inputs)
  // Hidden layers: 3 layers of 8 neurons each using tanh activation
  // Output layer: 2 neurons (movement directions) using tanh activation
  struct Layer network_architecture[] = {
      {AMOUNT_OF_RAYS, NULL},      // Input layer
      {8, tanh_func},             // Hidden layer 1
      {8, tanh_func},             // Hidden layer 2
      {8, tanh_func},             // Hidden layer 3
      {2, tanh_func}              // Output layer (movement control)
  };

  int num_layers = sizeof(network_architecture) / sizeof(network_architecture[0]);

  // Initialize ant population
  struct Ant ants[AMOUNT_OF_ANTS];
  for (int ant_index = 0; ant_index < AMOUNT_OF_ANTS; ant_index++)
  {
    // Create neural network for each ant
    ants[ant_index].network = initNetwork(network_architecture, num_layers);
    
    // Place ant at center of screen
    ants[ant_index].x = (SCREEN_WIDTH / 2);
    ants[ant_index].y = (SCREEN_HEIGHT / 2);

    // Either load existing weights or create new ones
    if (loadWeights(&ants[ant_index].network, "adam.bin") == 0)
    {
      // If no weights file exists, initialize random weights
      randomizeWeights(&ants[ant_index].network);
      saveWeights(&ants[ant_index].network, "adam.bin");
    }
    else
    {
      // If weights exist, apply mutations for evolution
      mutateWeights(&ants[ant_index].network);
    }
  }

  SetTargetFPS(1200);  // Set high FPS for faster simulation

  // Main simulation loop
  while (!WindowShouldClose())
  {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    // Draw maze/environment
    DrawTexture(background_texture,
                SCREEN_WIDTH / 2 - background_texture.width / 2,
                SCREEN_HEIGHT / 2 - background_texture.height / 2,
                WHITE);

    DrawFPS(SCREEN_WIDTH - 100, SCREEN_HEIGHT - 50);

    // Update and render each ant
    for (int ant_index = 0; ant_index < AMOUNT_OF_ANTS; ant_index++)
    {
      // Get sensor readings from ray-casting
      float *sensor_inputs = get_rays(ant_index, ants[ant_index].x, ants[ant_index].y);

      // Process inputs through neural network
      float *movement_output = feed_forward(ants[ant_index].network, sensor_inputs);

      // Move ant based on neural network output
      cartesian_move(&ants[ant_index], movement_output);

      // Render ant sprite
      DrawTexture(ant_texture, 
                  ants[ant_index].x - ant_texture.width / 2, 
                  ants[ant_index].y - ant_texture.height / 2, 
                  WHITE);
      
      free(movement_output);
    }

    EndDrawing();
  }

  // Cleanup resources
  cleanup_world();
  UnloadTexture(background_texture);
  UnloadTexture(ant_texture);
  CloseWindow();

  // Free neural networks
  for (int ant_index = 0; ant_index < AMOUNT_OF_ANTS; ant_index++)
  {
    freeNetwork(&ants[ant_index].network);
  }

  return 0;
}