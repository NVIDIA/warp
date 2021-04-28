#pragma once


struct BVHPackedNodeHalf
{
	float x;
	float y;
	float z;
	unsigned int i : 31;
	unsigned int b : 1;
};

struct BVH
{
    BVHPackedNodeHalf node_lowers;
    BVHPackedNodeHalf node_uppers;

    int num_nodes;
    int* root;
};