#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"
#include "camera.h"
#include "triangle.h"
#include "stlobject.h"
#include "stlparser.h"

__device__ vec3 color(const ray& r, StlObject obj) {
    ray_hit rec;

    if (obj.hit(r, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }

    vec3 normalized = unit_vector(r.direction());
    float t = 0.5f*(normalized.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *frame, int x_max, int y_max, Camera camera, StlObject obj) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= x_max) || (j >= y_max)) {
        // printf("out of bounds\n");
        return;
    }

    int pixel_index = j * x_max + i;    
    float u = float(i) / float(x_max);
    float v = float(j) / float(y_max);

    ray toTrace = camera.make_ray(u, v);
    // printf("%g, %g, %g\n", toTrace.direction().x(), toTrace.direction().y(), toTrace.direction().z());
    
    frame[pixel_index] = color(toTrace, obj);
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

    Camera camera(
        vec3(0.0, 0.0, 0.0), 
        vec3(-2.0, -1.0, -1.0), 
        vec3(4.0, 0.0, 0.0), 
        vec3(0.0, 2.0, 0.0));

    // Camera *camera_d;
    // cudaMalloc ( (void**) &camera_d, sizeof(Camera));
    // cudaMemcpy (camera_d, &camera_h, sizeof(Camera), cudaMemcpyHostToDevice); 

    // Triangle tetrahedron[4] = {
    //     Triangle(vec3(0, -0.5, -1), vec3(0.5, 0.5, -1), vec3(-0.5, 0.5, -1), vec3(0, 0, 1)),
    //     Triangle(vec3(0, -0.5, -1), vec3(0.5, 0.5, -1), vec3(0, 0, -0.5), vec3(0, 0, 1)),
    //     Triangle(vec3(0, 0, -0.5), vec3(0.5, 0.5, -1), vec3(-0.5, 0.5, -1), vec3(0, 0, 1)),
    //     Triangle(vec3(0, -0.5, -1), vec3(0, 0, -1), vec3(-0.5, 0.5, -1), vec3(0, 0, 1))
    // };

    // Triangle *tetra_d;
    // cudaMalloc ( (void**) &tetra_d, sizeof(Triangle)*4);
    // cudaMemcpy (tetra_d, tetrahedron, sizeof(Triangle)*4, cudaMemcpyHostToDevice);  
    
    // StlObject tetraObj(tetra_d, 4);

    

    render<<<nblocks, nthreads>>>(frame, nx, ny, camera, tetraObj);
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