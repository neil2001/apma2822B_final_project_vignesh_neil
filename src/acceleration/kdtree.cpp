#include "kdtree.h"

#define LEAF_SIZE 10

TreeNode* KdTree::init(StlObject obj){
    // median of first dimension, entire list
    // two leaves, 
    std::vector<Triangle> ts()
    return initHelper()

}

TreeNode* KdTree::initHelper(std::vector<Triangle> ts, Axis a, int l) {
    if (ts.size() < LEAF_SIZE) {
        TreeNode leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, NULL);
    }
    float s;
    s = quickSelectHelper(ts, a);
    // split on dim
    // create subvectors
    // store and pass in subvectrs to function, call on children
    // create treenode 
    // return treenode

}



float KdTree::bound(Triangle *t) {
    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = 0;
    float max_y = 0;
    float max_z = 0;

    max_x = max(max_x, t->v1[0]);
    max_x = max(max_x, t->v2[0]);
    max_x = max(max_x, t->v3[0]);

    max_y = max(max_y, t->v1[1]);
    max_y = max(max_y, t->v2[1]);
    max_y = max(max_y, t->v3[1]);

    max_z = max(max_z, t->v1[2]);
    max_z = max(max_z, t->v2[2]);
    max_z = max(max_z, t->v3[2]);

    min_x = min(min_x, t->v1[0]);
    min_x = min(min_x, t->v2[0]);
    min_x = min(min_x, t->v3[0]);

    min_y = min(min_y, t->v1[1]);
    min_y = min(min_y, t->v2[1]);
    min_y = min(min_y, t->v3[1]);

    min_z = min(min_z, t->v1[2]);
    min_z = min(min_z, t->v2[2]);
    min_z = min(min_z, t->v3[2]);

    vec3 maxVec(max_x, max_y, max_z);
    vec3 minVec(min_x, min_y, min_z);

    float xLen = fabs(max_x - min_x);
    float yLen = fabs(max_y - min_y);
    float zLen = fabs(max_z - min_z);

    float surfaceArea = 2*xLen*yLen + 2*xLen*zLen + 2*yLen*zLen;
    vec3 dims(xLen, yLen, zLen);
    
    return bbox{maxVec, minVec, surfaceArea, dims};
}

float KdTree::boundFromList(std::vector<Triangle> *items) {
    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = 0;
    float max_y = 0;
    float max_z = 0;

    int count = items->size();

    Triangle t;

    for (int i=0; i < count; i++) {
        t = (*items)[i];

        max_x = max(max_x, t.v1[0]);
        max_x = max(max_x, t.v2[0]);
        max_x = max(max_x, t.v3[0]);

        max_y = max(max_y, t.v1[1]);
        max_y = max(max_y, t.v2[1]);
        max_y = max(max_y, t.v3[1]);

        max_z = max(max_z, t.v1[2]);
        max_z = max(max_z, t.v2[2]);
        max_z = max(max_z, t.v3[2]);

        min_x = min(min_x, t.v1[0]);
        min_x = min(min_x, t.v2[0]);
        min_x = min(min_x, t.v3[0]);

        min_y = min(min_y, t.v1[1]);
        min_y = min(min_y, t.v2[1]);
        min_y = min(min_y, t.v3[1]);

        min_z = min(min_z, t.v1[2]);
        min_z = min(min_z, t.v2[2]);
        min_z = min(min_z, t.v3[2]);
    }

    vec3 maxVec(max_x, max_y, max_z);
    vec3 minVec(min_x, min_y, min_z);

    float xLen = fabs(max_x - min_x);
    float yLen = fabs(max_y - min_y);
    float zLen = fabs(max_z - min_z);

    float surfaceArea = 2*xLen*yLen + 2*xLen*zLen + 2*yLen*zLen;
    vec3 dims(xLen, yLen, zLen);
    
    return bbox{maxVec, minVec, surfaceArea, dims};
}
