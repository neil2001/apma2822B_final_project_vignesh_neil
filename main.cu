#include <iostream>
#include <time.h>
#include "vec3.h"

__global__ void render(vec3 *frame, int x_max, int y_max) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= x_max) || (j >= y_max)) {
        return;
    }

    int pixel_index = j * x_max + i;
    frame[pixel_index] = vec3(float(i) / x_max, float(j) / y_max, 0.1);
}

int main() {
    int nx = 1200;
    int ny = 600;

    int tx = 8;
    int ty = 8;

    int num_pixels = nx * ny;
    size_t frame_size = num_pixels * sizeof(vec3);

    // allocating image frame
    vec3 *frame;
    cudaMallocManaged((void **) &frame, frame_size);

    // dim3 nthreads(256, 1, 1);
    // dim3 nblocks( (N+nthreads.x-1)/nthreads.x, 1, 1);
    dim3 nthreads(tx, ty);
    dim3 nblocks(nx/tx + 1, ny/ty + 1);

    render<<<nblocks, nthreads>>>(frame, nx, ny);
    cudaDeviceSynchronize();

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*frame[pixel_index].r());
            int ig = int(255.99*frame[pixel_index].g());
            int ib = int(255.99*frame[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(frame);
}