#ifndef TRIANGLEH
#define TRIANGLEH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "vec3.h"
#include "ray.h"

#define EPSILON 0.0001

class Triangle {

public:
    __host__ __device__ Triangle() {}
    __host__ __device__ Triangle(const vec3& v1, const vec3& v2, const vec3& v3, const vec3& normal) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        
        n = normal;
        n = cross(v[1]-v[0], v[2]-v[0]); //normal; 
        area = 0.5f * n.length();
        n.make_unit_vector();
    }

    __host__ __device__ Triangle(const vec3& v1, const vec3& v2, const vec3& v3) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        
        n = cross(v[1]-v[0], v[2]-v[0]); //normal; 
        area = 0.5f * n.length();
        n.make_unit_vector();
    }

    __host__ __device__ inline bool hit(const ray& r, float t_max, ray_hit& hitRec) {
        // return true;
        float nDotDir = dot(r.direction(), n);
        if (fabs(nDotDir) < EPSILON) {
            return false; // ray parallel to triangle
        }

        float d = -dot(n, v[0]);
        float t = -(dot(n, r.origin()) + d) / nDotDir;
        if (t < 0) {
            return false;
        }

        if (t > t_max) {
            return false;
        }

        vec3 p = r.origin() + t * r.direction();
        vec3 c;

        for (int i=0; i < 3; i++) {
            vec3 edge = v[(i+1) % 3] - v[i];
            vec3 vp = p - v[i];
            c = cross(edge, vp);
            if (dot(n, c) < 0) {
                return false;
            }
        }

        // if (t < t_max) {
            hitRec.t = t - 0.01f;
            hitRec.p = r.point_at_parameter(hitRec.t);
            hitRec.normal = n;
            return true;
        // }

        // return false;
    }

    vec3 n;
    vec3 v[3];
    float area;
}; 

inline std::ostream& operator<<(std::ostream &os, const Triangle &t) {
    os << t.v[0] << ", " << t.v[1] << ", " << t.v[2];
    return os;
}

#endif