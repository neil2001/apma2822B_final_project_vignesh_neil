#ifndef RAYH
#define RAYH
#include "vec3.h"

struct ray_hit {
    float t;
    vec3 p;
    vec3 normal;
};

class ray
{
    public:
        ray() {}
        ray(const vec3& a, const vec3& b) { A = a; B = unit_vector(b); }
        vec3 origin() const       { return A; }
        vec3 direction() const    { return B; }
        vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif