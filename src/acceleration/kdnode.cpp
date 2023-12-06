#include "tracing/ray.h"
#include "tracing/vec3.h"
#include "acceleration/kdtree.h"

__device__ bool hit(const ray& r) {

    // r.dir is unit direction vector of ray
    // float dirfrac_x = 1.0f / r.direction()[0];
    // float dirfrac_y = 1.0f / r.direction()[1];
    // float dirfrac_z = 1.0f / r.direction()[2];
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (this.min[0] - r.origin()[0]) / r.direction()[0];
    float t2 = (this.max[0] - r.origin()[0]) / r.direction()[0];
    float t3 = (this.min[1] - r.origin()[1]) / r.direction()[1];
    float t4 = (this.max[1] - r.origin()[1]) / r.direction()[1];
    float t5 = (this.min[2] - r.origin()[2]) / r.direction()[2];
    float t6 = (this.max[2] - r.origin()[2]) / r.direction()[2];

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        t = tmax;
        return false;
    }

    t = tmin;
    return true;
}