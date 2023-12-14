#ifndef STLPARSERH
#define STLPARSERH

#include <cmath>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "vec3.h"
#include "triangle.h"
#include "stlobject.h"

#define HEADER_LENGTH 80

using namespace std;

struct Facet
{
    float normal[3];
    float v1[3];
    float v2[3];
    float v3[3];
};

// template <typename T>
class StlParser{

public:
    StlParser() {}
    static std::vector<Triangle> parseFile(const char* filename);

private: 
    static std::ifstream open_binary_stl(const char* filename) {
        return std::ifstream{filename, std::ifstream::in | std::ifstream::binary};
    }

    template <typename T>
    static void read_binary_value(std::ifstream& in, T* dst) {
        in.read(as_char_ptr(dst), sizeof(T));
    }

    template <typename T>
    static char* as_char_ptr(T* pointer) {
        return reinterpret_cast<char*>(pointer);
    }

    template <typename T>
    static void read_binary_array(std::ifstream& in, T* dst, size_t array_length) {
        size_t n_bytes = array_length * sizeof(T);
        in.read(as_char_ptr(dst), n_bytes);
    }
};

std::vector<Triangle> StlParser::parseFile(const char* filename) {
    std::ifstream in = open_binary_stl(filename);

    char header[HEADER_LENGTH] = "";
    read_binary_array<char>(in, header, 80);

    unsigned int triangle_count;
    read_binary_value<unsigned int>(in, &triangle_count);
    std::cerr << triangle_count << std::endl;

    std::vector<Triangle> triangles;
    triangles.reserve(triangle_count);

    char bc[2];

    float min_x = INFINITY;
    float min_y = INFINITY;
    float min_z = INFINITY;
    
    float max_x = -INFINITY;
    float max_y = -INFINITY;
    float max_z = -INFINITY;

    for (int i=0; i<int(triangle_count); i++) {
        Facet f{};

        read_binary_array<float>(in, f.normal, 3);
        read_binary_array<float>(in, f.v1, 3);
        read_binary_array<float>(in, f.v2, 3);
        read_binary_array<float>(in, f.v3, 3);
        
        max_x = max(max_x, f.v1[0]);
        max_x = max(max_x, f.v2[0]);
        max_x = max(max_x, f.v3[0]);

        max_y = max(max_y, f.v1[1]);
        max_y = max(max_y, f.v2[1]);
        max_y = max(max_y, f.v3[1]);

        max_z = max(max_z, f.v1[2]);
        max_z = max(max_z, f.v2[2]);
        max_z = max(max_z, f.v3[2]);

        min_x = min(min_x, f.v1[0]);
        min_x = min(min_x, f.v2[0]);
        min_x = min(min_x, f.v3[0]);

        min_y = min(min_y, f.v1[1]);
        min_y = min(min_y, f.v2[1]);
        min_y = min(min_y, f.v3[1]);

        min_z = min(min_z, f.v1[2]);
        min_z = min(min_z, f.v2[2]);
        min_z = min(min_z, f.v3[2]);

        Triangle newTriangle(vec3(f.v1), vec3(f.v2), vec3(f.v3), vec3(f.normal));
        
        triangles.emplace_back(
            Triangle(vec3(f.v1), vec3(f.v2), vec3(f.v3), vec3(f.normal))
        );

        read_binary_array<char>(in, bc, 2);

        // std::cerr << newTriangle << std::endl;
    }
    std::cerr << vec3(max_x, max_y, max_z) << std::endl;
    std::cerr << vec3(min_x, min_y, min_z) << std::endl;

    min_x *= 1.5;
    min_y *= 1.5;
    min_z *= 1.5;
    max_x *= 1.5;
    max_y *= 1.5;
    max_z *= 1.5;

    triangles.push_back(Triangle(vec3(min_x, min_y, min_z), vec3(max_x, min_y, min_z), vec3(max_x, max_y, min_z)));
    triangles.push_back(Triangle(vec3(min_x, min_y, min_z), vec3(max_x, max_y, min_z), vec3(min_x, max_y, min_z)));

    return triangles;
}

#endif