#ifndef WORLD_H
#define WORLD_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

// Ant structure containing its neural network and position
struct Ant {
    struct Network network;  // Neural network controlling ant behavior
    float x;                // Current X position
    float y;                // Current Y position
};

// Optimization constants for performance
static float HALF_MAX_SPEED = MAX_SPEED / 2.0;                    // Pre-calculated half of max speed
static float RAY_ANGLE_INCREMENT = PI * 2.0 / AMOUNT_OF_RAYS;     // Angle between each sensor ray

// Pre-calculated trigonometric values for ray calculations
static float ray_cos_table[AMOUNT_OF_RAYS];  // Cosine values for each ray angle
static float ray_sin_table[AMOUNT_OF_RAYS];  // Sine values for each ray angle

// Cached background image for collision detection
static Image cached_background;

// Initialize background image for collision detection
void init_bg(Texture2D background)
{
    cached_background = LoadImageFromTexture(background);
}

// Pre-calculate trigonometric values for ray calculations
void init_trigonometry_tables()
{
    for (int ray_index = 0; ray_index < AMOUNT_OF_RAYS; ray_index++)
    {
        ray_cos_table[ray_index] = cos(ray_index * RAY_ANGLE_INCREMENT);
        ray_sin_table[ray_index] = sin(ray_index * RAY_ANGLE_INCREMENT);
    }
}

// Array to store ray sensor readings for each ant
static float ray_sensors[AMOUNT_OF_ANTS][AMOUNT_OF_RAYS];

// Calculate sensor readings for an ant's position
float *get_rays(int ant_index, int pos_x, int pos_y)
{
    float *ray_distances = ray_sensors[ant_index];

    for (int ray_index = 0; ray_index < AMOUNT_OF_RAYS; ray_index++)
    {
        float ray_dir_x = ray_cos_table[ray_index];
        float ray_dir_y = ray_sin_table[ray_index];
        int ray_length;

        // Cast ray until obstacle is hit or max distance reached
        for (ray_length = 0; ray_length < RAYS_RADIUS; ray_length++)
        {
            float ray_x = ray_dir_x * ray_length;
            float ray_y = ray_dir_y * ray_length;

            if (pos_x + ray_x > 0 && pos_x + ray_x < SCREEN_WIDTH && 
                pos_y + ray_y > 0 && pos_y + ray_y < SCREEN_HEIGHT)
            {
                // Check if ray hit obstacle (non-white pixel)
                if (!ColorIsEqual((Color)GetImageColor(cached_background, pos_x + ray_x, pos_y + ray_y), WHITE))
                {
                    break;
                }
            }
        }
        // Normalize ray distance to [0,1] range
        ray_distances[ray_index] = (float)(RAYS_RADIUS - ray_length) / (float)RAYS_RADIUS;
    }

    return ray_distances;
}

// Cleanup resources
void cleanup_world()
{
    UnloadImage(cached_background);
}

// Move ant based on neural network output, with collision detection
void cartesian_move(struct Ant *ant, float movement[])
{
    // Calculate new position
    float new_x = ant->x + movement[0] * MAX_SPEED;
    float new_y = ant->y + movement[1] * MAX_SPEED;

    // Clamp position to screen boundaries
    new_x = fmaxf(0, fminf(new_x, SCREEN_WIDTH));
    new_y = fmaxf(0, fminf(new_y, SCREEN_HEIGHT));

    // Only move if new position is not colliding with obstacle
    if (ColorIsEqual((Color)GetImageColor(cached_background, (int)new_x, (int)new_y), WHITE))
    {
        ant->x = new_x;
        ant->y = new_y;
    }
}

#endif