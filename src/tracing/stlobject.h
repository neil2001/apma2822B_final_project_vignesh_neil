#ifndef STLOBJECTH
#define STLOBJECTH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "triangle.h"
#include "../acceleration/kdtree.h"

class StlObject {

public:
    StlObject() {}
    StlObject(Triangle *ts, int n) {
        triangles = ts;
        count = n;
        // std::cout << "making kdtree" << std::endl;
        tree = new KdTree();
        tree->init(ts, n);
    }

    bool hit(const ray& r, ray_hit& finalHitRec) {
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

    bool hitTree(const ray& r, ray_hit& finalHitRec) {
        return tree->hit(r, finalHitRec);
    }
    
    Triangle *triangles;
    int count;
    KdTree *tree;
};

#endif