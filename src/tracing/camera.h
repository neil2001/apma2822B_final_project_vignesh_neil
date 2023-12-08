#ifndef CAMERAH
#define CAMERAH
#include "vec3.h"
#include "ray.h"

class Camera {

public:
    __host__ Camera() {}
    __host__ Camera(const vec3& pos, const vec3& ll, const vec3& h, const vec3& v) {
        position = pos;
        lowerLeft = ll;
        horizontal = h;
        vertical = v;
    }
    __host__ Camera(const vec3& pos, const vec3& target, float view_height, float view_width) {
        vec3 look = unit_vector(target - pos);
        // float d = -dot(look, target);

        // alternate computation of up vec
        // vec3 up = vec3(0, 1, 0);
        // up = unit_vector(cross(look, cross(look, right)));

        vec3 up = vec3(0, 0, 1);
        vec3 right = unit_vector(cross(look, up));
        //TODO: don't really know if this points right or left technically
        up = unit_vector(cross(right, look));

        position = pos;
        lowerLeft = target - up*(view_height / 2.f) - right*(view_width/2.f);
        horizontal = right*view_width;
        vertical = up*view_height;

        std::cerr << "look:" << look << std::endl;
        std::cerr << "pos:" << pos << std::endl;
        std::cerr << "lowerLeft:" << lowerLeft << std::endl;
        std::cerr << "horizontal:" << horizontal << std::endl;
        std::cerr << "up:" << vertical << std::endl;
    }

    
    __host__ __device__ inline ray make_ray(float u, float v) { 
        return ray(position, unit_vector((lowerLeft + u*horizontal + v*vertical) - position)); 
    }
    

    vec3 lowerLeft;
    vec3 horizontal;
    vec3 vertical;
    vec3 position;
};

#endif