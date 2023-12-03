// #ifndef FRAMEH
// #define FRAMEH
// #include "vec3.h"
// #include "ray.h"

// class Frame {
//     __host__ __device__ Frame() {}
//     __host__ __device__ Frame(vec3* p, int w, int h) {
//         pixels = p;
//         width = w;
//         height = h;
//     }

//     __device__ bool outOfBounds(int i, int j) {
//         return (i >= width) || (j >= height)
//     }

//     vec3* pixels;
//     int width;
//     int height;
// }

// #endif