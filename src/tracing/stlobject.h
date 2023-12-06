#ifndef STLOBJECTH
#define STLOBJECTH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "triangle.h"

class StlObject {

public:
    __host__ __device__ StlObject() {}
    __host__ __device__ StlObject(Triangle *ts, int n) {
        triangles = ts;
        count = n;
    }

    __device__ bool hit(const ray& r, ray_hit& finalHitRec) {
        ray_hit rec;
        bool hasHit = false;
        float t_max = INFINITY;
        for (int i=0; i < count; i++) {
            if (triangles[i].hit(r, t_max, rec)) {
                hasHit = true;
                t_max = rec.t;
                finalHitRec = rec;
            }
        }

        return hasHit;
    }
    __device__ bool hitTree(const ray& r, ray_hit& finalHitRec) {
        return root.traverse(r, finalHitRec);
    }
    
    Triangle *triangles;
    int count;
    TreeNode *root;
};

#endif