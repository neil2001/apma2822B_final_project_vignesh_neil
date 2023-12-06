#include "kdtree.h"

#define LEAF_SIZE 10

TreeNode* KdTree::init(StlObject obj){
    // median of first dimension, entire list
    // two leaves, 

    std::vector<Triangle> ts(obj.Triangles, obj.count*sizeof(Triangle));
    return initHelper(ts, 0, 0);

}

TreeNode* KdTree::initHelper(std::vector<Triangle> ts, Axis a, int l) {
    bbox* newBbox = boundFromList(ts);
    if (ts.size() < LEAF_SIZE) {
        TreeNode leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, 
                                    NULL, newBbox);
        return &leaf;
    }
    float s;
    s = quickSelectHelper(ts, a);
    std::vector<Triangle> leftVector;
    std::vector<Triangle> rightVector;
    for (Triangle t : ts){
        if (t.v[a] <= s) {
            leftVector.push_back(t);
        } else {
            rightVector.push_back(t);
        }
    }

    TreeNode node = new TreeNode(l, a, s, false, std::vector<Triangle>(),
                                NULL, NULL, newBbox);
    a++;
    a%=3;
    l++;
    TreeNode *leftLeaf = initHelper(leftVector, a, l);
    TreeNode *rightLeaf = initHelper(rightVector, a, l);
    node.left = leftLeaf;
    node.right = rightLeaf;

    return &node;
}

TreeNode* KdTree::traverse(ray r, TreeNode *curNode, ray_hit rec) {
    // check if ray hits bounding box of curNode
    // check if ray hits bounding box of left or right child
    // traverse again with either the left or right child

    if (curNode.isLeaf) {

    }
}

bbox* KdTree::boundFromList(std::vector<Triangle> *items) {
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
    
    return &bbox{maxVec, minVec, surfaceArea, dims};
}
