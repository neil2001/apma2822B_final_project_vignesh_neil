#ifndef KDTREEH
#define KDTREEH
#include <vector>
#include <random>
#include <algorithm>
#include <deque>

#include "../tracing/vec3.h"
#include "../tracing/triangle.h"
#include "../tracing/ray.h"
#include "util.h"
#include "kdtreegpu.h"



using namespace std;

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator

class TreeNode {

public:
    TreeNode() {}
    TreeNode(int l, Axis a, float s, bool leaf, 
               std::vector<int> ts, TreeNode *ltree, 
               TreeNode *rtree, bbox newBox, int nid) {
        level = l;
        axis = a;
        split = s;
        isLeaf = leaf;
        tri_idxs = ts;
        left = ltree;
        right = rtree;
        box = newBox;
        id = nid;
    }

    bool hit(const ray& r);

    bool isLeaf;
    float split;
    Axis axis;
    int level;
    int id;

    std::vector<int> tri_idxs;

    TreeNode *left;
    TreeNode *right;
    bbox box;
};

class KdTree {

public: 
    __host__ KdTree() {}
    __host__ void init(Triangle *triangles, int n);
    __host__ bool hit(const ray& r, ray_hit& finalHitRec);
    __host__ void printTree();

    int numNodes;
    TreeNode *root;
    // TreeNodeGPU *nodeArray;
    std::vector<TreeNodeGPU> nodeArray;
    Triangle *allTriangles;

    // TreeNodeGPU *rootGPU;

private:
    // const int MAXDEPTH = 10;
    // const int MINOBJS = 2;

    __host__ TreeNode* initHelper(std::vector<int> ts, Axis a, int l, int prevId);
    __host__ void renumber();
    __host__ void createNodeArray();
    __host__ bbox boundFromList(std::vector<int> *items);
    __host__ float quickSelectHelper(std::vector<float> &data, int k);
    __host__ float quickSelect(std::vector<int> ts, Axis a);
    __host__ void printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft);
    __host__ void printGPUTreeHelper(const std::string& prefix, const TreeNodeGPU* node, bool isLeft);
};

__host__ void KdTree::init(Triangle *triangles, int n) {
    // median of first dimension, entire list
    // two leaves, 
    this->numNodes = 0;

    this->allTriangles = triangles;

    std::vector<int> ts(n);
    std::iota(ts.begin(), ts.end(), 0);

    // std::vector<Triangle> ts(triangles, triangles + n);
    std::cerr << "starting tree init" << std::endl;
    this->root = initHelper(ts, X, 0, 1);
    // std::cerr << "finished tree init" << std::endl;
    this->renumber();
    // std::cerr << "renumbered tree" << std::endl;
    this->createNodeArray();
    this->printTree();
    return;
}

__host__ TreeNode* KdTree::initHelper(std::vector<int> ts, Axis a, int l, int nextId) {
    // if (l == 0) {
    //     std::cout << "num triangles: " << ts.size() << std::endl;
    // }
    this->numNodes++;

    bbox newBbox = boundFromList(&ts);

    // if (l == 0) {
    //     std::cerr << "bounding box on init min: " << newBbox.min << std::endl;
    //     std::cerr << "bounding box on init max: " << newBbox.max << std::endl;
    // }

    if (ts.size() <= LEAF_SIZE) {
        TreeNode* leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, 
                                    NULL, newBbox, nextId);
        return leaf;
    }
    float s;
    s = quickSelect(ts, a);
    std::vector<int> leftVector;
    std::vector<int> rightVector;
    Triangle t;
    for (int ti : ts){
        t = this->allTriangles[ti];
        float center = (t.v[0][a] + t.v[1][a] + t.v[2][a]) / 3.0;
        if (center <= s) {
            leftVector.push_back(ti);
        } else {
            rightVector.push_back(ti);
        }
    }

    TreeNode* node = new TreeNode(l, a, s, false, std::vector<int>(),
                                NULL, NULL, newBbox, nextId);
                                
    
    int axisNumRep = static_cast<int>(a);
    axisNumRep++;
    axisNumRep%=3;
    a = static_cast<Axis>(axisNumRep);
    l++;
    TreeNode *leftLeaf = initHelper(leftVector, a, l, nextId*2);
    TreeNode *rightLeaf = initHelper(rightVector, a, l, nextId*2 + 1);
    node->left = leftLeaf;
    node->right = rightLeaf;

    return node;
}

__host__ void KdTree::renumber() {
    std::deque<TreeNode*> toVisit = {this->root};
    int counter = 0;
    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        toVisit.pop_front();

        curr->id = counter;
        counter++;

        if (curr->isLeaf) {
            continue;
        }

        if (curr->left != NULL) {
            toVisit.push_back(curr->left);
        }

        if (curr->right != NULL) {
            toVisit.push_back(curr->right);
        }
    }
}

__host__ void KdTree::createNodeArray() {
    // this->nodeArray = new TreeNodeGPU[this->numNodes];
    std::deque<TreeNode*> toVisit = {this->root};
    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        toVisit.pop_front();

        if (curr->isLeaf) {
            // std::cerr << "creating leaf node for GPU" << std::endl;
            int t_count = curr->tri_idxs.size();
            this->nodeArray.push_back(TreeNodeGPU(true, t_count, curr->tri_idxs.data(),
                                                curr->box, curr->id, -1, -1));
            continue;
        }

        int leftNum = -1;
        int rightNum = -1;

        if (curr->left != NULL) {
            // newNode->leftNodeIdx = curr->left->id; 
            leftNum = curr->left->id;
            toVisit.push_back(curr->left);
        }

        if (curr->right != NULL) {
            // newNode->rightNodeIdx = curr->right->id; 
            rightNum = curr->right->id;
            toVisit.push_back(curr->right);
        }

        // this->nodeArray[curr->id] = *newNode;
        // this->nodeArray.push_back(*newNode);
        this->nodeArray.push_back(TreeNodeGPU(false, 0, NULL, curr->box, curr->id, leftNum, rightNum));
    }

    std::cerr << "creating tree size: " << this->nodeArray.size() << std::endl;

    // this->nodeArray = allNodes;
}

__host__ bool KdTree::hit(const ray& r, ray_hit& finalHitRec) {
    // check if ray hits bounding box of curNode
    // check if ray hits bounding box of left or right child
    // traverse again with either the left or right child
    /*
    bounding box max: 18.7272 -1.29747 1.01005e-08
    ounding box min: 16.1273 -1.26498 1.01005e-08
    */

    std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;
    ray_hit rec;

    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();

        // if (curr->level == 0) {
        //     std::cerr << "bounding box min: " << curr->box.min << std::endl;
        //     std::cerr << "bounding box max: " << curr->box.max << std::endl;
        // }


        toVisit.pop_front();
        if (!curr->hit(r) && curr->level > 0) {
            // std::cerr << "how did we get here, level:" << curr->level << std::endl;
        }
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            Triangle t;
            for (int ti : curr->tri_idxs) {
                t = this->allTriangles[ti];
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    finalHitRec = rec;
                }
            }

            continue;
        }

        bool hitLeft = curr->left->hit(r);
        if (hitLeft) {
            toVisit.push_back(curr->left);
        }

        bool hitRight = curr->right->hit(r);

        if (hitRight) {
            toVisit.push_back(curr->right);
        }

        if (!hitLeft && !hitRight && curr->level > 0) {
        }
    }

    return has_hit;
}

__host__ bbox KdTree::boundFromList(std::vector<int> *items) {
    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    float max_z = -INFINITY;

    int count = items->size();

    Triangle t;

    for (int i=0; i < count; i++) {
        t = this->allTriangles[(*items)[i]];
        for (int j = 0; j < 3; j++) {
            max_x = max(max_x, t.v[j][0]);
            max_y = max(max_y, t.v[j][1]);
            max_z = max(max_z, t.v[j][2]);

            min_x = min(min_x, t.v[j][0]);
            min_y = min(min_y, t.v[j][1]);
            min_z = min(min_z, t.v[j][2]);
        }
    }

    vec3 maxVec(max_x, max_y, max_z);
    vec3 minVec(min_x, min_y, min_z);
    return bbox{minVec, maxVec};
}

__host__ void KdTree::printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft)
{
    if( node != nullptr )
    {
        std::cerr << prefix;

        std::cerr << (isLeft ? "├──" : "└──" );

        // print the value of the node
        // std::cerr << node->id << std::endl;
        if (node->isLeaf) {
            std::cerr << "hit leaf node with " << node->tri_idxs.size() << " leaves" << std::endl;
        } else {
            std::cerr << node->id << std::endl;
        }
        // std::cerr << node->level << std::endl;

        // enter the next tree level - left and right branch
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->left, true);
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}

__host__ void KdTree::printGPUTreeHelper(const std::string& prefix, const TreeNodeGPU* node, bool isLeft)
{   
    std::cerr << prefix;

    std::cerr << (isLeft ? "├──" : "└──" );

    // print the value of the node
    if (node->isLeaf) {
        std::cerr << "hit leaf node";
        std::cerr << node->numTris << std::endl;
    } else {
        std::cerr << node->idx << std::endl;
    }
    // std::cerr << node->level << std::endl;

    // enter the next tree level - left and right branch
    if (node->leftNodeIdx != -1) {
        printGPUTreeHelper( prefix + (isLeft ? "│   " : "    "), &(this->nodeArray[node->leftNodeIdx]), true);
    }

    if (node->rightNodeIdx != -1) {
        printGPUTreeHelper( prefix + (isLeft ? "│   " : "    "), &(this->nodeArray[node->rightNodeIdx]), false);
    }
}

__host__ void KdTree::printTree()
{
    // this->printTreeHelper("", this->root, false);
    std::cerr << "num nodes: " << this->numNodes << std::endl;
    // std::cerr << "node array size: " << this->nodeArray.size() << std::endl;

    // this->printGPUTreeHelper("", &(this->nodeArray[0]), false);
}

// From stack overflow: https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
__host__ bool TreeNode::hit(const ray& r) {
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
        return false;
    }

    // if tmin > tmax, ray doesn't intersect AABB
    if (tmin > tmax)
    {
        return false;
    }

    return true;
}

__host__ float KdTree::quickSelect(std::vector<int> ts, Axis a) {
    std::vector<float> data;
    int axis = static_cast<int>(a);
    int count = ts.size();

    Triangle t;
    for (int i=0; i<count; i++) {
        t = this->allTriangles[ts[i]];
        data.push_back(t.v[0][axis]);
        data.push_back(t.v[1][axis]);
        data.push_back(t.v[2][axis]);
    }
    std::sort(data.begin(), data.end());
    int dSize = data.size();
    if (dSize % 2 == 0) {
        return (data[dSize / 2 - 1] + data[dSize / 2]) / 2.0;
    } else {
        return data[dSize / 2];
    }
}

#endif

