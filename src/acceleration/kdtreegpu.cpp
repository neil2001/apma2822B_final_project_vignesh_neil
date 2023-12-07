#include "kdtree.h"
// #include <thrust/device_vector.h>

bool KdTreeGPU::hit(const ray& r, ray_hit& finalHitRec) {
    // thrust::host_vector<int> toVisit;
    std::vector<int> toVisit;
    toVisit.push_back(0);
    int visitIdx = 0;
    // std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;
    ray_hit rec;

    while (visitIdx < int(toVisit.size())) {
        TreeNodeGPU *curr = &(this->nodes[toVisit[visitIdx]]);
        visitIdx++;
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            Triangle t;
            for (int ti : curr->t_idxs) {
                t = this->allTriangles[ti];
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    finalHitRec = rec;
                }
                // std::cerr << "hit " << hitCount << " triangle(s) in leaf node, level:" << curr->level << std::endl;
            }

            continue;
        }

        bool hitLeft = this->nodes[curr->leftNodeIdx].hit(r);
        if (hitLeft) {
            // std::cerr << "hit left tree bounding box, level:" << curr->left->level << std::endl;
            toVisit.push_back(curr->leftNodeIdx);
        }

        bool hitRight = this->nodes[curr->rightNodeIdx].hit(r);

        if (hitRight) {
            // std::cerr << "hit right tree bounding box, level:" << curr->right->level << std::endl;
            toVisit.push_back(curr->rightNodeIdx);
        }

        if (!hitLeft && !hitRight) {
            // std::cerr << "how did we hit neither box, but we hit the box above" << std::endl;
        }
    }

    return has_hit;
}