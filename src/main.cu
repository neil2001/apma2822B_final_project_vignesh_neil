#include <iostream>
#include <time.h>
#include <vector>
#include <sys/time.h>

#include "tracing/vec3.h"
#include "tracing/ray.h"
#include "tracing/camera.h"
#include "tracing/triangle.h"
#include "tracing/stlobject.h"
#include "tracing/stlparser.h"
#include "acceleration/kdtree.h"

#define NUM_REFLECTIONS 10
#define WARP_SIZE 32
#define N_THREAD 32



#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
/**
 * TODO:
 * - maybe do triangles as hostmalloc
 * - use streams and 1024 threads per block (switch to row-wise)
*/

__device__ vec3 color(const ray& r, StlObject obj) {

    // NORMAL SHADING 
    /* 
    ray_hit rec;

    if (obj.hit(r, rec)) {
        return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    }
    vec3 normalized = unit_vector(r.direction());
    float t = 0.5f*(normalized.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0); 
    */

    // LAMBERTIAN
    vec3 kd(1.0, 1.0, 0.1);
    ray_hit rec;
    if (obj.hit(r, rec)) {
        vec3 rayDir = r.direction() - 2 * rec.normal * dot(r.direction(), rec.normal);
        rayDir.make_unit_vector();
        // printf("rayDir: %g, %g, %g \n", rayDir.x(), rayDir.y(), rayDir.z());
        float dotProd = dot(rec.normal, rayDir);
        // printf("dotProd:%g \n", dotProd);
        return kd * dotProd;
    }

    vec3 normalized = unit_vector(r.direction());
    float t = 0.5f*(normalized.x() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0); 

    // REFLECTIONS
    /*
    ray curr = r;
    float f_att = 1.0f;

    vec3 rayDir;
    for (int i=0; i<NUM_REFLECTIONS; i++) {
        ray_hit rec;
        if (obj.hit(curr, rec)) {
            f_att *= 0.5f;
            rayDir = curr.direction() - 2 * rec.normal * dot(curr.direction(), rec.normal);
            curr = ray(rec.p, rayDir);
        } else {
            vec3 normalized = unit_vector(curr.direction());
            float t = 0.5f*(normalized.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0); 
            return f_att * c;
        }
    }

    return vec3(0,0,0);
    */
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
    vec3 colorResult = color(toTrace, obj);
    frame[pixel_index] = colorResult;
}

int main() {
    int n_cols = 1200;
    int n_rows = 2400;

    int tx = 8;
    int ty = 8;

    int num_pixels = n_cols * n_rows;
    size_t frame_size = num_pixels * sizeof(vec3);

    // allocating image frame
    vec3 *frame;
    cudaMallocManaged((void **) &frame, frame_size);

    // dim3 nthreads(256, 1, 1);
    // dim3 nblocks( (N+nthreads.x-1)/nthreads.x, 1, 1);
    dim3 nthreads(tx, ty);
    dim3 nblocks(n_cols/tx + 1, n_rows/ty + 1);

    // Pikachu
    Camera camera(
        vec3(0.0, -64.0, 32.0), 
        vec3(-16.0, 0.0, -16.0), 
        vec3(48.0, 0.0, 0.0), 
        vec3(0.0, 0.0, 96.0));

    // Mandalorian
    // Camera camera(
    //     vec3(0.0, -16.0, 16.0), 
    //     vec3(-16.0, 0.0, -16.0), 
    //     vec3(24.0, 0.0, 0.0), 
    //     vec3(0.0, 0.0, 48.0));

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

    struct timeval startTime;
    struct timeval endTime;

    gettimeofday(&startTime, nullptr);
    std::vector<Triangle> triangles = StlParser::parseFile("examples/pikachu.stl");
    // std::vector<Triangle> triangles = StlParser::parseFile("examples/F-15.stl");
    // std::vector<Triangle> triangles = StlParser::parseFile("examples/pikachu.stl");
    gettimeofday(&endTime, nullptr);

    int millis = (endTime.tv_sec - startTime.tv_sec) * 1000 + (endTime.tv_usec - startTime.tv_usec) / 1000;

    std::cerr << "Parsing time: " << millis << "ms" << std::endl;

    size_t triangle_count = triangles.size();
    std::cerr << "Triangle count: " << triangle_count << std::endl;

    Triangle *object_h; //= triangles.data();
    // Triangle *object_d;
    
    std::cerr << "mallocing triangles: " << std::endl;

    checkCudaErrors(cudaMallocManaged ( (void**) &object_h, sizeof(Triangle)*triangle_count));
    // checkCudaErrors(cudaMemcpy (object_d, object_h, sizeof(Triangle)*triangle_count, cudaMemcpyHostToDevice));  // TODO: Maybe use cuda host malloc? share the memory?
    std::memcpy(object_h, triangles.data(), sizeof(Triangle)*triangle_count);

    std::cerr << "making object: " << std::endl;

    // TODO: think about what this looks like
    StlObject object(object_h, triangle_count);

    // copy over GPU Tree
    // copy over GPU TreeNodes
    // set pointers and fields

    TreeNodeGPU *treeNodesGPU_h;
    int node_count = object.treeGPU->node_count;
    checkCudaErrors(cudaMallocManaged ( (void**) &treeNodesGPU_h, sizeof(TreeNodeGPU) * node_count));
    // checkCudaErrors(cudaMemcpy (treeNodesGPU_d, object.treeGPU->nodes, sizeof(TreeNodeGPU)*node_count, cudaMemcpyHostToDevice));  // TODO: Maybe use cuda host malloc? share the memory?
    std::memcpy(treeNodesGPU_h, object.treeGPU->nodes, sizeof(TreeNodeGPU)*node_count);

    KdTreeGPU treeGPU_h(object_h, triangle_count, treeNodesGPU_h, node_count);

    KdTreeGPU *treeGPU_u;
    checkCudaErrors(cudaMallocManaged ( (void**) &treeGPU_u, sizeof(KdTreeGPU)));
    // checkCudaErrors(cudaMemcpy (treeGPU_d, &treeGPU_h, sizeof(KdTreeGPU), cudaMemcpyHostToDevice));
    std::memcpy(treeGPU_u,&treeGPU_h, sizeof(KdTreeGPU));

    // treeGPU_d->nodes = treeNodesGPU_d;
    // treeGPU_d->allTriangles = object_d;

    object.treeGPU = treeGPU_u;
    // object.triangles = object_d;

    // MAKING CAMERA
    // vec3 bboxMin = object.tree->root->box.min;
    // vec3 bboxMax = object.tree->root->box.max;
    // std::cerr << "bboxMin:" << bboxMin << std::endl;
    // std::cerr << "bboxMax:" << bboxMax << std::endl;
    // vec3 centroid = (bboxMin + bboxMax) / 2.0f;
    // std::cerr << "centroid:" << centroid << std::endl;

    // vec3 bmoPos(300, -300, 200);
    // Camera camera(bmoPos, centroid, 300, 150);

    std::cerr << "starting render" << std::endl;
    gettimeofday(&startTime, nullptr); 
    render<<<nblocks, nthreads>>>(frame, n_cols, n_rows, camera, object);
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        // Handle kernel launch error
        std::cerr << "Kernel launch error: " << cudaGetErrorString(kernelError) << std::endl;
    }
    cudaDeviceSynchronize();
    fflush(stdout);
    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Synchronize error: " << cudaGetErrorString(kernelError) << std::endl;
    }
    gettimeofday(&endTime, nullptr);

    millis = (endTime.tv_sec - startTime.tv_sec) * 1000 + (endTime.tv_usec - startTime.tv_usec) / 1000;

    std::cerr << "Rendering time: " << millis << "ms" << std::endl;

    gettimeofday(&startTime, nullptr); 
    std::cout << "P3\n" << n_cols << " " << n_rows << "\n255\n";
    for (int j = n_rows-1; j >= 0; j--) {
        for (int i = 0; i < n_cols; i++) {
            size_t pixel_index = j*n_cols + i;
            int ir = int(255.99*frame[pixel_index].r());
            int ig = int(255.99*frame[pixel_index].g());
            int ib = int(255.99*frame[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    gettimeofday(&endTime, nullptr);

    millis = (endTime.tv_sec - startTime.tv_sec) * 1000 + (endTime.tv_usec - startTime.tv_usec) / 1000;

    std::cerr << "File Output Time: " << millis << "ms" << std::endl;

    cudaFree(frame);
    cudaFree(treeNodesGPU_h);
    cudaFree(treeGPU_u);
    // cudaFree(object_d);
}