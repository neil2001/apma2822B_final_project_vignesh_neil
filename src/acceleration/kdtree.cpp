#include <vector>
#include <deque>

#include "kdtree.h"

#define LEAF_SIZE 4

using namespace std;

void KdTree::init(Triangle *triangles, int n) {
    // median of first dimension, entire list
    // two leaves, 

    std::vector<Triangle> ts(triangles, triangles + n);
    std::cerr << "starting tree init" << std::endl;
    this->root = initHelper(ts, X, 0);
    std::cerr << "finished tree init" << std::endl;
    this->printTree();
    return;
}

TreeNode* KdTree::initHelper(std::vector<Triangle> ts, Axis a, int l) {
    bbox newBbox = boundFromList(&ts);
    if (ts.size() <= LEAF_SIZE) {
        TreeNode* leaf = new TreeNode(l, a, INFINITY, true, ts, NULL, 
                                    NULL, newBbox);
        return leaf;
    }
    float s;
    s = quickSelect(ts, a);
    std::vector<Triangle> leftVector;
    std::vector<Triangle> rightVector;
    for (Triangle t : ts){

        // TODO: use the center of the triangle instead
        float center = (t.v[0][a] + t.v[1][a] + t.v[2][a]) / 3.0;
        if (center <= s) {
            leftVector.push_back(t);
        } else {
            rightVector.push_back(t);
        }
        // bool inLeft = false;
        // bool inRight = false;

        // if (t.v[0][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (t.v[1][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (t.v[2][a] <= s) {
        //     inLeft = true;
        // } else {
        //     inRight = true;
        // }

        // if (inLeft) {
        //     leftVector.push_back(t);
        // }
        // if (inRight) {
        //     rightVector.push_back(t);
        // }
    }

    TreeNode* node = new TreeNode(l, a, s, false, std::vector<Triangle>(),
                                NULL, NULL, newBbox);
    
    // std::cerr << "s:" << s << std::endl;
    // std::cerr << "initial list size:" << ts.size() << ", left size:" << leftVector.size() << ", right size:" << rightVector.size() << std::endl;
    int axisNumRep = static_cast<int>(a);
    axisNumRep++;
    axisNumRep%=3;
    a = static_cast<Axis>(axisNumRep);
    l++;
    TreeNode *leftLeaf = initHelper(leftVector, a, l);
    TreeNode *rightLeaf = initHelper(rightVector, a, l);
    node->left = leftLeaf;
    node->right = rightLeaf;

    return node;
}

bool KdTree::hit(const ray& r, ray_hit finalHitRec) {
    // check if ray hits bounding box of curNode
    // check if ray hits bounding box of left or right child
    // traverse again with either the left or right child

    std::deque<TreeNode*> toVisit = {this->root};

    bool has_hit = false;
    float t_max = INFINITY;

    ray_hit rec;

    while (!toVisit.empty()) {
        TreeNode *curr = toVisit.front();
        toVisit.pop_front();
        if (!curr->hit(r) && curr->level > 0) {
            std::cerr << "how did we get here, level:" << curr->level << std::endl;
        }
        if (curr->isLeaf) {
            // LEAF NODE
            int hitCount = 0;
            for (Triangle t : curr->triangles) {
                if (t.hit(r, t_max, rec)) {
                    hitCount++;
                    has_hit = true;
                    t_max = rec.t;
                    finalHitRec = rec;
                }
                std::cerr << "hit " << hitCount << " triangle(s) in leaf node, level:" << curr->level << std::endl;
            }

            continue;
        }

        bool hitLeft = curr->left->hit(r);
        if (hitLeft) {
            std::cerr << "hit left tree bounding box, level:" << curr->left->level << std::endl;
            toVisit.push_back(curr->left);
        }

        bool hitRight = curr->right->hit(r);

        if (hitRight) {
            std::cerr << "hit right tree bounding box, level:" << curr->right->level << std::endl;
            toVisit.push_back(curr->right);
        }

        if (!hitLeft && !hitRight && curr->level > 0) {
            std::cerr << "how did we hit neither box, but we hit the box above" << std::endl;
        }
    }

    return has_hit;
}

bbox KdTree::boundFromList(std::vector<Triangle> *items) {
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

        max_x = max(max_x, t.v[0][0]);
        max_x = max(max_x, t.v[1][0]);
        max_x = max(max_x, t.v[2][0]);

        max_y = max(max_y, t.v[0][1]);
        max_y = max(max_y, t.v[1][1]);
        max_y = max(max_y, t.v[2][1]);

        max_z = max(max_z, t.v[0][2]);
        max_z = max(max_z, t.v[1][2]);
        max_z = max(max_z, t.v[2][2]);

        min_x = min(min_x, t.v[0][0]);
        min_x = min(min_x, t.v[1][0]);
        min_x = min(min_x, t.v[2][0]);

        min_y = min(min_y, t.v[0][1]);
        min_y = min(min_y, t.v[1][1]);
        min_y = min(min_y, t.v[2][1]);

        min_z = min(min_z, t.v[0][2]);
        min_z = min(min_z, t.v[1][2]);
        min_z = min(min_z, t.v[2][2]);
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

void KdTree::printTreeHelper(const std::string& prefix, const TreeNode* node, bool isLeft)
{
    if( node != nullptr )
    {
        std::cerr << prefix;

        std::cerr << (isLeft ? "├──" : "└──" );

        // print the value of the node
        std::cerr << node->level << std::endl;

        // enter the next tree level - left and right branch
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->left, true);
        printTreeHelper( prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}

void KdTree::printTree()
{
    this->printTreeHelper("", this->root, false);    
}