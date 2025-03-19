#ifndef KDTREEGPU
#define KDTREEGPU

#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"
#include "util.h"

class TreeNodeGPU {

public:
    __host__ __device__ TreeNodeGPU() {}
    __host__ __device__ TreeNodeGPU(bool leafBool, int trisArraySize, int* trisArray, 
                bbox newBox, int curIdx, int leftIdx, int rightIdx) {
        isLeaf = leafBool;
        box = newBox;
        numTris = trisArraySize;
        for (int i = 0; i < numTris; i++) {
            t_idxs[i] = trisArray[i];
        }
        idx = curIdx;
        leftNodeIdx = leftIdx;
        rightNodeIdx = rightIdx;
    }
    
    __host__ __device__ bool hit(const ray& r);

    bool isLeaf;
    bbox box;
    int numTris;
    int idx;
    int leftNodeIdx;
    int rightNodeIdx;
    int t_idxs[LEAF_SIZE];
    
};

class KdTreeGPU {

public: 
    __host__ __device__ KdTreeGPU() {}
    __host__ __device__ KdTreeGPU(Triangle *ts, int nts, TreeNodeGPU *ns, int nns) {
        tri_count = nts;
        node_count = nns;

        allTriangles = ts;
        nodes = ns;
    }

    __device__ bool hit(const ray& r, ray_hit& finalHitRec);

    int tri_count;
    int node_count; 
    Triangle *allTriangles;
    TreeNodeGPU *nodes;
};

__device__ bool KdTreeGPU::hit(const ray& r, ray_hit& finalHitRec) {
    // thrust::device_vector<int> toVisit;
    // std::vector<int> toVisit;
    // toVisit.push_back(0);

    int toVisit[BUF_SIZE];

    int visitIdx = 0;
    int pushIdx = 1;
    // std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;
    ray_hit rec;

    while (visitIdx < pushIdx) {
        TreeNodeGPU *curr = &(this->nodes[toVisit[visitIdx]]);
        visitIdx++;
        visitIdx %= BUF_SIZE;
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            Triangle t;
            for (int i = 0; i < curr->numTris; i++) {
                t = this->allTriangles[curr->t_idxs[i]];
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    // finalHitRec = rec;
                    finalHitRec.t = t_max;
                    finalHitRec.p = rec.p;
                    finalHitRec.normal = rec.normal;
                }
                // std::cerr << "hit " << hitCount << " triangle(s) in leaf node, level:" << curr->level << std::endl;
            }

            continue;
        }

        bool hitLeft = this->nodes[curr->leftNodeIdx].hit(r);
        if (hitLeft) {
            toVisit[pushIdx] = curr->leftNodeIdx;
            pushIdx++;
            pushIdx %= BUF_SIZE;
            if (pushIdx == visitIdx) {
                // TODO: Remove this
                printf("BUFFER too small \n");
            }
        }

        bool hitRight = this->nodes[curr->rightNodeIdx].hit(r);

        if (hitRight) {
            toVisit[pushIdx] = curr->rightNodeIdx;
            pushIdx++;
            pushIdx %= BUF_SIZE;
            if (pushIdx == visitIdx) {
                // TODO: Remove this
                printf("BUFFER too small \n");
            }
        }
        
    }

    return has_hit;
}

__host__ __device__ bool TreeNodeGPU::hit(const ray& r) {

    // r.dir is unit direction vector of ray
    // float dirfrac_x = 1.0f / r.direction()[0];
    // float dirfrac_y = 1.0f / r.direction()[1];
    // float dirfrac_z = 1.0f / r.direction()[2];
    // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
    // r.org is origin of ray
    float t1 = (this->box.min[0] - r.origin()[0]) / r.direction()[0];
    float t2 = (this->box.max[0] - r.origin()[0]) / r.direction()[0];
    float t3 = (this->box.min[1] - r.origin()[1]) / r.direction()[1];
    float t4 = (this->box.max[1] - r.origin()[1]) / r.direction()[1];
    float t5 = (this->box.min[2] - r.origin()[2]) / r.direction()[2];
    float t6 = (this->box.max[2] - r.origin()[2]) / r.direction()[2];

    float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
    float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

    // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
    if (tmax < 0)
    {
        // t = tmax;
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        // t = tmax;
        return false;
    }

    // t = tmin;
    return true;
}



#endif