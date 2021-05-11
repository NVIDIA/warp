#include "core.h"

#include "stdlib.h"
#include "string.h"

void init()
{
}

void shutdown()
{
}

void* alloc_host(size_t s)
{
    return malloc(s);
}

void free_host(void* ptr)
{
    free(ptr);
}

void memcpy_h2h(void* dest, void* src, size_t n)
{
    memcpy(dest, src, n);
}

void memset_host(void* dest, int value, size_t n)
{
    memset(dest, value, n);
}


// impl. files
#include "bvh.cpp"
#include "mesh.cpp"
//#include "spline.inl"
//#include "volume.inl"

