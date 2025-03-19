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

    __host__ __device__ bool hit(const ray& r, float t_max, ray_hit& hitRec); 

    vec3 n;
    vec3 v[3];
    float area;
    vec3 vn[3];

}; 

inline std::ostream& operator<<(std::ostream &os, const Triangle &t) {
    os << t.v[0] << ", " << t.v[1] << ", " << t.v[2];
    return os;
}

__host__ __device__ inline bool Triangle::hit(const ray& r, float t_max, ray_hit& hitRec) {
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

    // for (int i=0; i < 3; i++) {
    //     vec3 edge = v[(i+1) % 3] - v[i];
    //     vec3 vp = p - v[i];
    //     c = cross(edge, vp);
    //     if (dot(n, c) < 0) {
    //         return false;
    //     }
    // }

    float u, w;

    vec3 edge = v[1] - v[0];
    vec3 vp = p - v[0];
    c = cross(edge, vp);
    if (dot(n, c) < 0) {
        return false;
    }

    edge = v[2] - v[1];
    vp = p - v[1];
    c = cross(edge, vp);
    if ((u = dot(n, c)) < 0) {
        return false;
    }

    edge = v[0] - v[2];
    vp = p - v[2];
    c = cross(edge, vp);
    if ((w = dot(n, c)) < 0) {
        return false;
    }

    u /= (area*2);
    w /= (area*2);

    hitRec.t = t - 0.01f;
    hitRec.p = r.point_at_parameter(hitRec.t);
    // hitRec.normal = n; // normal
    // hitRec.normal = vn[0] + vn[1] + vn[2] / 3.f; // mean
    // hitRec.normal = (1-u-w) * vn[0] + u * vn[1] + w * vn[2]; // X barycentric coordinates
    // hitRec.normal = (1-u-w) * vn[0] + w * vn[1] + u * vn[2]; // X barycentric coordinates
    // hitRec.normal = w * vn[0] + (1-u-w) * vn[1] + u * vn[2]; // X barycentric coordinates
    // hitRec.normal = u * vn[0] + (1-u-w) * vn[1] + w * vn[2]; // X barycentric coordinates
    hitRec.normal = u * vn[0] + w * vn[1] + (1-u-w) * vn[2]; // barycentric coordinates
    // hitRec.normal = w * vn[0] + u * vn[1] + (1-u-w) * vn[2]; // X barycentric coordinates

    return true;
}



#endif