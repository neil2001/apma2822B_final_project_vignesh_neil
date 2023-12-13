#include <iostream>
#include <time.h>
#include <vector>
#include <sys/time.h>
#include <omp.h>

#include "tracing/vec3.h"
#include "tracing/ray.h"
#include "tracing/camera.h"
#include "tracing/triangle.h"
#include "tracing/stlobject.h"
#include "tracing/stlparser.h"
// #include "tqdm/tqdm.h"

#define NUM_REFLECTIONS 10
#define WARP_SIZE 32
#define N_THREAD 32

/**
 * TODO:
 * - maybe do triangles as hostmalloc
 * - use streams and 1024 threads per block (switch to row-wise)
*/

vec3 color(const ray& r, StlObject obj) {

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
    if (obj.hitTreeGPU(r, rec)) {
        // std::cout << "ray origin:" << r.A << ", ray dir:" << r.B << endl;
        vec3 rayDir = unit_vector(r.direction() - 2 * rec.normal * dot(r.direction(), rec.normal));
        return kd * dot(rec.normal, rayDir);
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

void render(vec3 *frame, int n_cols, int n_rows, Camera camera, StlObject obj) {

    // for (int j : tqdm::range(n_rows)) {
    #pragma omp parallel for
    for (int j = 0; j < n_rows; j++) {
        for (int i=0; i<n_cols; i++) {
            float v = float(j) / float(n_rows);
            int pixel_index = j * n_cols + i;    
            float u = float(i) / float(n_cols);
            ray toTrace = camera.make_ray(u, v);
            frame[pixel_index] = color(toTrace, obj);
        } 
    }

    // printf("%g, %g, %g\n", toTrace.direction().x(), toTrace.direction().y(), toTrace.direction().z());    
}

int main() {
    int n_rows = 1200;
    int n_cols = 600;

    // int num_pixels = n_cols * n_rows;

    // allocating image frame
    vec3 frame[n_cols * n_rows];
 
    // Pikachu
    // Camera camera(
    //     vec3(0.0, -64.0, 32.0), 
    //     vec3(-16.0, 0.0, -16.0), 
    //     vec3(48.0, 0.0, 0.0), 
    //     vec3(0.0, 0.0, 96.0));

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
    std::vector<Triangle> triangles = StlParser::parseFile("examples/neuron_ball.stl");
    // std::vector<Triangle> triangles = StlParser::parseFile("examples/low_drogon.stl");
    gettimeofday(&endTime, nullptr);

    int millis = (endTime.tv_sec - startTime.tv_sec) * 1000 + (endTime.tv_usec - startTime.tv_usec) / 1000;

    std::cerr << "Parsing time: " << millis << "ms" << std::endl;

    size_t triangle_count = triangles.size();
    std::cerr << "Triangle count: " << triangle_count << std::endl;

    // Triangle *object_h = triangles.data();
    // Triangle *object_d;
    
    // cudaMalloc ( (void**) &object_d, sizeof(Triangle)*triangle_count);
    // cudaMemcpy (object_d, object_h, sizeof(Triangle)*triangle_count, cudaMemcpyHostToDevice);  // TODO: Maybe use cuda host malloc? share the memory?

    StlObject object(triangles.data(), triangle_count);

    vec3 bboxMin = object.tree->root->box.min;
    vec3 bboxMax = object.tree->root->box.max;
    std::cerr << "bboxMin:" << bboxMin << std::endl;
    std::cerr << "bboxMax:" << bboxMax << std::endl;
    vec3 centroid = (bboxMin + bboxMax) / 2.0f;
    std::cerr << "centroid:" << centroid << std::endl;

    // vec3 dragCamPos(30, -30, 20);
    // vec3 mandoPos(-40, -40, 20);
    vec3 bmoPos(300, -300, 200);
    // vec3 neuronPos();

    // Camera camera(dragCamPos, centroid, 20, 40);
    // Camera camera(mandoPos, centroid, 40, 20);
    Camera camera(bmoPos, centroid, 300, 150);

    gettimeofday(&startTime, nullptr);
    render(frame, n_cols, n_rows, camera, object);
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

    std::cerr << "File writing time: " << millis << "ms" << std::endl;


    return 0;
}