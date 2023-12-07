#ifndef CAMERAH
#define CAMERAH
#include "vec3.h"
#include "ray.h"

class Camera {

public:
    Camera() {}
    Camera(const vec3& pos, const vec3& ll, const vec3& h, const vec3& v) {
        position = pos;
        lowerLeft = ll;
        horizontal = h;
        vertical = v;
    }
    
    inline ray make_ray(float u, float v) { 
        return ray(position, (lowerLeft + u*horizontal + v*vertical) - position); 
    }
    
    vec3 lowerLeft;
    vec3 horizontal;
    vec3 vertical;
    vec3 position;
};

#endif