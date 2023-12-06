#include <vector>
#include "acceleration/kdtree.h"
#include "tracing/vec3.h"
#include <cmath>

std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator

__host__ __device__ float KdTree::quickSelectHelper(std::vector<float> &data, int k) {
    if (data.size() == 1) {
        return data[0];
    }
    // choosing a random pivot
    std::uniform_int_distribution<> dist (0, data.size()-1);
    int idx = dist(randomPicker);
    float pivot = data[idx];

    std::vector<float> less;
    std::vector<float> equal;
    std::vector<float> greater;
    // partitioning the list
    for (float val : data) {
        if (val < pivot) {
            less.push_back(val);
        } else if (val > pivot) {
            greater.push_back(val);
        } else {
            equal.push_back(val);
        }
    }
    // recursive calls to quickselect
    if (k <= less.size()) {
        return quickSelectHelper(less, k);
    } else if (k <= less.size() + equal.size()) {
        return pivot;
    } else {
        return quickSelectHelper(greater, k - less.size() - equal.size());
    }
}

__host__ __device__ float KdTree::quickSelect(std::vector<Triangle> ts, Axis a) {
    std::vector<float> data;
    int axis = static_cast<int>(a);
    int count = ts.size();

    for (int i=0; i<count; i++) {
        data.push_back(ts[i].v[0][axis]);
        data.push_back(ts[i].v[1][axis]);
        data.push_back(ts[i].v[2][axis]);
    }

    return quickSelectHelper(data, count/2);
}


