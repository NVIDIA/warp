// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/sm_builder.h"

namespace cuBQL {
  namespace radixBuilder_impl {
    using gpuBuilder_impl::atomic_grow;
    using gpuBuilder_impl::_ALLOC;
    using gpuBuilder_impl::_FREE;
    
    template<typename T, int D> struct Quantizer;

    template<int D> struct numMortonBits;
#if 0
    template<> struct numMortonBits<2> { enum { value = 31 }; };
    template<> struct numMortonBits<3> { enum { value = 21 }; };
    template<> struct numMortonBits<4> { enum { value = 15 }; };
#else
    // force 32-bit morton codes ... only for testing/evaluation
    template<> struct numMortonBits<2> { enum { value = 15 }; };
    template<> struct numMortonBits<3> { enum { value = 10 }; };
    template<> struct numMortonBits<4> { enum { value =  7 }; };
#endif
    
    template<int D>
    struct Quantizer<float,D> {
      using vec_t = cuBQL::vec_t<float,D>;
      using box_t = cuBQL::box_t<float,D>;
      
      inline __device__ void init(cuBQL::box_t<float,D> centBounds)
      {
        quantizeBias
          = centBounds.lower;
        quantizeScale
          = vec_t(1u<<numMortonBits<D>::value)
          * rcp(max(vec_t(reduce_max(centBounds.size())),vec_t(1e-20f)));
      }
        
      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        using vec_ui = cuBQL::vec_t<uint32_t,D>;

        vec_ui cell = vec_ui((P-quantizeBias)*quantizeScale);
        cell = min(cell,vec_ui(uint32_t(((1u<<numMortonBits<D>::value)-1))));
        return cell;
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
        quantization operation that does
        `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
        centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t quantizeBias;
      vec_t quantizeScale;
    };

    template<int D>
    struct Quantizer<double,D> {
      using vec_t = cuBQL::vec_t<double,D>;
      using box_t = cuBQL::box_t<double,D>;
      
      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        using vec_ui = cuBQL::vec_t<uint32_t,D>;

        vec_ui cell = vec_ui((P-quantizeBias)*quantizeScale);
        cell = min(cell,vec_ui(uint32_t(((1u<<numMortonBits<D>::value)-1))));
        return cell;
      }
        
      inline __device__ void init(cuBQL::box_t<double,D> centBounds)
      {
        quantizeBias
          = centBounds.lower;
        quantizeScale
          = vec_t(1u<<numMortonBits<D>::value)
          * rcp(max(vec_t(reduce_max(centBounds.size())),vec_t(1e-20f)));
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
        quantization operation that does
        `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
        centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t quantizeBias;
      vec_t quantizeScale;
    };

    template<int D>
    struct Quantizer<int,D> {
      using vec_t = cuBQL::vec_t<int,D>;
      using box_t = cuBQL::box_t<int,D>;
      
      inline __device__ void init(cuBQL::box_t<int,D> centBounds)
      {
        quantizeBias = centBounds.lower;
        int maxValue = reduce_max(centBounds.size());
        shlBits = __clz(maxValue);
      }

      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        cuBQL::vec_t<uint32_t,D> cell = cuBQL::vec_t<uint32_t,D>(P-quantizeBias);
        // move all relevant bits to top
        cell = cell << shlBits;
        return cell >> (32-numMortonBits<D>::value);
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
        quantization operation that does
        `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
        centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t quantizeBias;
      int   shlBits;
    };
    
    template<int D>
    struct Quantizer<int64_t,D> {
      using vec_t = cuBQL::vec_t<int64_t,D>;
      using box_t = cuBQL::box_t<int64_t,D>;
      
      inline __device__ void init(cuBQL::box_t<int64_t,D> centBounds)
      {
        quantizeBias = centBounds.lower;
        uint64_t maxValue = reduce_max(centBounds.size());
        shlBits = __clzll(maxValue);
      }

      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        cuBQL::vec_t<uint64_t,D> cell = cuBQL::vec_t<uint64_t,D>(P-quantizeBias);
        // move all relevant bits to top
        cell = cell << shlBits;
        return cuBQL::vec_t<uint32_t,D>(cell >> (64-numMortonBits<D>::value));
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
        quantization operation that does
        `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
        centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t quantizeBias;
      int   shlBits;
    };
    


    
    /*! maintains high-level summary of the build process */
    template<typename T, int D>
    struct CUBQL_ALIGN(16) BuildState {
      using vec_t = cuBQL::vec_t<T,D>;//float,3>;
      using box_t = cuBQL::box_t<T,D>;//float,3>;
      using bvh_t = cuBQL::BinaryBVH<T,D>;//float,3>;
      using atomic_box_t = gpuBuilder_impl::AtomicBox<box_t>;
      
      /*! number of nodes alloced so far */
      int numNodesAlloced;

      /*! number of *valid* prims that get put into the BVH; this will
        be computed by sarting with the input number of prims, and
        removing those that have invalid/empty bounds */
      int numValidPrims;
      
      /*! bounds of prim centers, relative to which we will computing
        morton codes */
      atomic_box_t a_centBounds;
      box_t        centBounds;
      Quantizer<T,D> quantizer;
    };

    template<typename T, int D>
    __global__
    void clearBuildState(BuildState<T,D> *buildState,
                         int          numPrims)
    {
      if (threadIdx.x != 0) return;
      
      buildState->a_centBounds.clear();
      // let's _start_ with the assumption that all are valid, and
      // subtract those later on that are not.
      buildState->numValidPrims   = numPrims;
      buildState->numNodesAlloced = 0;
    }
    
    template<typename T, int D>
    __global__
    void fillBuildState(BuildState<T,D>  *buildState,
                        const typename BuildState<T,D>::box_t *prims,
                        int          numPrims)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      __shared__ atomic_box_t l_centBounds;
      if (threadIdx.x == 0)
        l_centBounds.clear();
      
      // ------------------------------------------------------------------
      __syncthreads();
      // ------------------------------------------------------------------
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid < numPrims) {
        box_t prim = prims[tid];
        if (!prim.empty()) 
          atomic_grow(l_centBounds,prim.center());
      }
      
      // ------------------------------------------------------------------
      __syncthreads();
      // ------------------------------------------------------------------
      if (threadIdx.x == 0)
        atomic_grow(buildState->a_centBounds,l_centBounds);
    }

    template<typename T, int D>
    __global__
    void finishBuildState(BuildState<T,D>  *buildState)
    {
      using ctx_t = BuildState<float,D>;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      using box_t        = typename ctx_t::box_t;
      
      if (threadIdx.x != 0) return;
      
      box_t centBounds = buildState->a_centBounds.make_box();
      buildState->quantizer.init(centBounds);
    }

    /* morton code computation: how the bits shift for 21 input bits:

       desired final step:
       ___u.__t_:_s__.r__q:__p_._o__:n__m.__l_:_k__.j__i:__h_._g__:f__e.__d_:_c__.b__a:

       stage -1
       ___u.____:ts__.__rq:____.po__:__nm.____:lk__.__ji:____.hg__:__fe.____:dc__.__ba:
       mask:
       0000.0000:1000.0010:0000.1000:0010.0000:1000.0010:0000.1000:0010.0000:1000.0010
       move by 2
       hex    00:       82:       08:       20:       82:       08:       20:       82

       stage -2
       ___u.____:____.tsrq:____.____:ponm.____:____.lkji:____.____:hgfe.____:____.dcba:
       mask:
       0000.0000:0000.1100:0000.0000:1100.0000:0000.1100:0000.0000:1100.0000:0000.1100
       move by 4
       hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c

       stage -3
       ____.____:___u.tsrq:____.____:____.____:ponm.lkji:____.____:____.____:hgfe.dcba:
       mask:
       0000.0000:1111.0000:0000.0000:0000.0000:1111.0000:0000.0000:0000.0000:1111.0000
       move by 8
       hex    00:       f0:       00:       00:       f0:       00:       00:       f0

       stage -4
       ____.____:___u.tsrq:____.____:____.____:____.____:____.____:ponm.lkji:hgfe.dcba:
       mask:
       0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:1111.1111.0000:0000
       move by 16
       hex     00:      00:       00:       00:       00:       00:       ff:       00

       stage -5
       ____.____:____.____:____.____:____.____:____.____:___u.tsrq:ponm.lkji:hgfe.dcba:
       move:
       0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:0001.1111:0000.0000:0000.0000
       move by 32
       hex    00:       00:       00:       00:       00:       1f:       00:       00
    */
    inline __device__
    uint64_t shiftBits(uint64_t x, uint64_t maskOfBitstoMove, int howMuchToShift)
    { return ((x & maskOfBitstoMove)<<howMuchToShift) | (x & ~maskOfBitstoMove); }

    /*! insert 1 zero in-between every two successive bits of the
      input. ie, bit 0 stays at 0, bit 1 goes to 3, etc */
    inline __device__
    uint64_t bitInterleave11(uint64_t x)
    {
      // current x is this:                      FEDC.BAzy.xwvu.tsrq.ponm.lkji.hgfe.dcba

      // 0000.0000:0000.0000:0000.0000:0000.0000:FEDC.BAzy:xwvu.tsrq:ponm.lkji:hgfe.dcba

      x = shiftBits(x,0xffff0000ull,16);
      
      // 0000.0000:0000.0000:FEDC.BAzy:xwvu.tsrq:0000.0000:0000.0000:ponm.lkji:hgfe.dcba

      x = shiftBits(x,0b0000000000000000111111110000000000000000000000001111111100000000ull,8);
      
      // 0000.0000:FEDC.BAzy:0000.0000:xwvu.tsrq:0000.0000:ponm.lkji:0000.0000:hgfe.dcba

      x = shiftBits(x,0b0000000011110000000000001111000000000000111100000000000011110000ull,4);
      
      // 0000.FEDC:0000.BAzy:0000.xwvu:0000.tsrq:0000.ponm:0000.lkji:0000.hgfe:0000.dcba

      x = shiftBits(x,0b0000110000001100000011000000110000001100000011000000110000001100ull,2);

      // 00FE.00DC:00BA.00zy:00xw.00vu:00ts.00rq:00po.00nm:00lk.00ji:00hg.00fe:00dc.00ba
      
      x = shiftBits(x,0b0010001000100010001000100010001000100010001000100010001000100010ull,1);
      
      // 0F0E.0D0C:0B0A.0z0y:0x0w.0v0u:0t0s.0r0q:0p0o.0n0m:0l0k.0j0i:0h0g.0f0e:0d0c.0b0a
      return x;
    }
    
    /*! insert 2 zeroes in-between every two successive bits of the
      input. ie, bit 0 stays at 0, bit 1 goes to 3, etc */
    /*! insert 2 zeroes in-between every two successive bits of the
      input. ie, bit 0 stays at 0, bit 1 goes to 3, etc */
    inline __device__
    uint64_t bitInterleave21(uint64_t x)
    {
      //hex    00:       00:       00:       00:       00:       10:       00:       00
      x = shiftBits(x,0x00000000001f0000ull,32); 
      //hex     00:      00:       00:       00:       00:       00:       ff:       00
      x = shiftBits(x,0x000000000000ff00ull,16); 
      //hex    00:       f0:       00:       00:       f0:       00:       00:       f0
      x = shiftBits(x,0x00f00000f00000f0ull,8); 
      //hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c
      x = shiftBits(x,0x000c00c00c00c00cull,4); 
      //hex    00:       82:       08:       20:       82:       08:       20:       82
      x = shiftBits(x,0x0082082082082082ull,2);
      return x;
    }
    
    inline __device__
    uint64_t interleaveBits64(vec2ui coords)
    {
#if 0
      uint64_t result = 0;
      for (int i=0;i<31;i++) {
        uint64_t bx = (coords.x >> i) & 1;
        uint64_t by = (coords.y >> i) & 1;
        result |= (bx << (2*i+0));
        result |= (by << (2*i+1));
      }
      return result;
#else
      return
        (bitInterleave11(coords.x) << 0)
        |
        (bitInterleave11(coords.y) << 1);
#endif
    }
    
    inline __device__
    uint64_t interleaveBits64(vec3ui coords)
    {
      return
        (bitInterleave21(coords.x) << 0)
        |
        (bitInterleave21(coords.y) << 1)
        |
        (bitInterleave21(coords.z) << 2);
    }
    
    inline __device__
    uint64_t interleaveBits64(vec4ui coords)
    {
      uint32_t xz
        =
        (bitInterleave11(coords.x) << 0)
        |
        (bitInterleave11(coords.z) << 1);
      uint32_t yw
        =
        (bitInterleave11(coords.y) << 0)
        |
        (bitInterleave11(coords.w) << 1);
      return
        (bitInterleave11(xz) << 0)
        |
        (bitInterleave11(yw) << 1);
    }
    
    template<typename T, int D>
    inline __device__
    uint64_t computeMortonCode(typename BuildState<T,D>::vec_t P,
                               const Quantizer<T,D> quantizer)
    {
      return interleaveBits64(quantizer.quantize(P));
    }
    
    template<typename T, int D>
    __global__
    void computeUnsortedKeysAndPrimIDs(uint64_t    *mortonCodes,
                                       uint32_t    *primIDs,
                                       BuildState<T,D>  *buildState,
                                       const typename BuildState<T,D>::box_t *prims,
                                       int numPrims)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid >= numPrims) return;

      int primID = tid;
      box_t prim = prims[primID];
      while (prim.empty()) {
        primID = atomicAdd(&buildState->numValidPrims,-1)-1;
        if (tid >= primID) return;
        prim = prims[primID];
      }

      primIDs[tid] = primID;
      mortonCodes[tid]
        = computeMortonCode(prim.center(),buildState->quantizer);
    }

    struct TempNode {
      union {
        /*! nodes that have been opened by their parents, but have not
          yet been finished. such nodes descibe a list of
          primitives; the range of keys covered in this subtree -
          which can/will be used to determine where to split - is
          encoded in first and last key in that range */
        struct {
          uint32_t begin;
          uint32_t end;
        } open;
        /*! nodes that are finished and done */
        struct {
          uint32_t offset;
          uint32_t count;
        } finished;
        // force alignment to 8-byte values, so compiler can
        // read/write more efficiently
        uint64_t bits;
      };
    };


    inline __device__
    bool findSplit(int &split,
                   const uint64_t *__restrict__ keys,
                   int begin, int end,
                   int maxAllowedLeafSize)
    {
      uint64_t firstKey = keys[begin];
      uint64_t lastKey  = keys[end-1];
      
      if (firstKey == lastKey) {
        // same keys entire range - no split in there ....
#if 1
        if ((end-begin) > maxAllowedLeafSize) {
          // printf("range %i %i needs split %i\n",begin,end,maxAllowedLeafSize);
          split = (begin+end)/2;
          return true;
        }
#endif
        return false;
      }
      
      int numMatchingBits = __clzll(firstKey ^ lastKey);
      // the first key in the plane we're searching has
      // 'numMatchingBits+1' top bits of lastkey, and 0es otherwise
      const uint64_t searchKey = lastKey & (0xffffffffffffffffull<<(63-numMatchingBits));

      while (end > begin) {
        int mid = (begin+end)/2;
        if (keys[mid] < searchKey) {
          begin = mid+1;
        } else {
          end = mid;
        }
      }
      split = begin;
      return true;
    }

    template<typename T, int D>
    __global__
    void initNodes(BuildState<T,D> *buildState,
                   TempNode   *nodes,
                   int numValidPrims)
    {
      if (threadIdx.x != 0) return;
      
      buildState->numNodesAlloced = 2;
      TempNode n0, n1;
      n0.open.begin = 0;
      n0.open.end   = numValidPrims;
      n1.bits = 0;
      nodes[0] = n0;
      nodes[1] = n1;
    }

    template<typename T, int D>
    __global__
    void createNodes(BuildState<T,D> *buildState,
                     int leafThreshold,
                     int maxAllowedLeafSize,
                     TempNode *nodes,
                     int begin, int end,
                     const uint64_t *keys)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      __shared__ int l_allocOffset;
      
      if (threadIdx.x == 0)
        l_allocOffset = 0;
      // ==================================================================
      __syncthreads();
      // ==================================================================
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      int nodeID = begin + tid;
      bool validNode = (nodeID < end);
      int split   = -1;
      int childID = -1;
      TempNode node;
      
      if (validNode) {
        node = nodes[nodeID];
        int size = node.open.end - node.open.begin;
        if (size <= leafThreshold) {
          // we WANT to make a leaf
          node.finished.offset = node.open.begin;
          node.finished.count  = size;
        } else if (!findSplit(split,keys,node.open.begin,node.open.end,
                              leafThreshold/*maxAllowedLeafSize*/)) {
          // we HAVE TO make a leaf because we couldn't split
          node.finished.offset = node.open.begin;
          node.finished.count  = size;
        } else {
          // we COULD split - yay!
          childID = atomicAdd(&l_allocOffset,2);
        }
      }
      
      // ==================================================================
      __syncthreads();
      // ==================================================================
      if (threadIdx.x == 0)
        l_allocOffset = atomicAdd(&buildState->numNodesAlloced,l_allocOffset);
      // ==================================================================
      __syncthreads();
      // ==================================================================
      if (childID >= 0) {
        childID += l_allocOffset;
        TempNode c0, c1;
        c0.open.begin = node.open.begin;
        c0.open.end   = split;
        c1.open.begin = split;
        c1.open.end   = node.open.end;
        // we COULD actually write those as a int4 if we really wanted
        // to ...
        nodes[childID+0]     = c0;
        nodes[childID+1]     = c1;
        node.finished.offset = childID;
        node.finished.count  = 0;
      }
      if (validNode)
        nodes[nodeID] = node;
    }
                     
    template<typename T, int D>
    __global__
    void writeFinalNodes(typename bvh_t<T,D>::Node *finalNodes,
                         const TempNode *__restrict__ tempNodes,
                         int numNodes)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numNodes) return;
      bvh3f::Node node;
      TempNode tempNode = tempNodes[tid];
      node.admin.offset = tempNode.finished.offset;
      node.admin.count  = tempNode.finished.count;

      if (tid == 1)
        node.admin.offsetAndCountBits = 0;
      
      finalNodes[tid].admin.offsetAndCountBits = node.admin.offsetAndCountBits;
    }
    
    template<typename T, int D>
    void build(BinaryBVH<T,D>        &bvh,
               const typename BuildState<T,D>::box_t       *boxes,
               uint32_t           numPrims,
               BuildConfig        buildConfig,
               cudaStream_t       s,
               GpuMemoryResource &memResource)
    {
      const int makeLeafThreshold
        = (buildConfig.makeLeafThreshold > 0)
        ? min(buildConfig.makeLeafThreshold,buildConfig.maxAllowedLeafSize)
        : 1;

      // ==================================================================
      // first MAJOR step: compute buildstate's centBounds value,
      // which we need for computing morton codes.
      // ==================================================================
      /* step 1.1, init build state; in particular, clear the shared
         centbounds we need to atomically grow centroid bounds in next
         step */
      BuildState<T,D> *d_buildState = 0;
      _ALLOC(d_buildState,1,s,memResource);
      clearBuildState<<<32,1,0,s>>>
        (d_buildState,numPrims);
      /* step 1.2, compute the centbounds we need for morton codes; we
         do this by atomically growing this shared centBounds with
         each (non-invalid) input prim */
      fillBuildState<<<divRoundUp((int)numPrims,1024),1024,0,s>>>
        (d_buildState,boxes,numPrims);
      /* step 1.3, convert vom atomic_box to regular box, which is
         cheaper to digest for the following kernels */
      finishBuildState<<<32,1,0,s>>>
        (d_buildState);

      static BuildState<T,D> *h_buildState = 0;
      if (!h_buildState)
        CUBQL_CUDA_CALL(MallocHost((void**)&h_buildState,
                                   sizeof(*h_buildState)));

      
      cudaEvent_t stateDownloadedEvent;
      CUBQL_CUDA_CALL(EventCreate(&stateDownloadedEvent));
      
      CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,
                                  sizeof(*h_buildState),
                                  cudaMemcpyDeviceToHost,s));
      CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
      CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));

      const int numValidPrims = h_buildState->numValidPrims;

      // ==================================================================
      // second MAJOR step: compute morton codes and primIDs array,
      // and do key/value sort to get those pairs sorted by ascending
      // morton code
      // ==================================================================
      /* 2.1, allocate mem for _unsorted_ prim IDs and morton codes,
         then compute initial primID array (will already exclude prims
         that are invalid) and (unsorted) morton code array */
      uint64_t *d_primKeys_unsorted;
      uint32_t *d_primIDs_unsorted;
      _ALLOC(d_primKeys_unsorted,numPrims,s,memResource);
      _ALLOC(d_primIDs_unsorted,numPrims,s,memResource);
      computeUnsortedKeysAndPrimIDs
        <<<divRoundUp(numValidPrims,1024),1024,0,s>>>
        (d_primKeys_unsorted,d_primIDs_unsorted,
         d_buildState,boxes,numPrims);

      /* 2.2: ask cub radix sorter for how much temp mem it needs, and
         allocate */
      size_t cub_tempMemSize;
      uint64_t *d_primKeys_sorted = 0;
      uint32_t *d_primIDs_inMortonOrder = 0;
      // with tempMem ptr null this won't do anything but return reqd
      // temp size*/
      auto rc =
      cub::DeviceRadixSort::SortPairs
        (nullptr,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,64,s);
      
      // 2.3: allocate temp mem and output arrays
      void     *d_tempMem = 0;
      memResource.malloc(&d_tempMem,cub_tempMemSize,s);
      _ALLOC(d_primKeys_sorted,numValidPrims,s,memResource);
      _ALLOC(d_primIDs_inMortonOrder,numValidPrims,s,memResource);

      // 2.4: sort
      rc = 
      cub::DeviceRadixSort::SortPairs
        (d_tempMem,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,64,s);
      rc = rc;
      // 2.5 - cleanup after sort: no longer need tempmem, or unsorted inputs
      _FREE(d_primKeys_unsorted,s,memResource);
      _FREE(d_primIDs_unsorted,s,memResource);
      _FREE(d_tempMem,s,memResource);

      // ==================================================================
      // third MAJOR step: create temp-nodes from keys
      // ==================================================================
      /* 3.1: allocate nodes array (do this only onw so we can re-use
         just freed memory); and initialize node 0 to span entire
         range of prims */
      uint32_t upperBoundOnNumNodesToBeCreated = 2*numValidPrims;
      TempNode *nodes = 0;
      _ALLOC(nodes,upperBoundOnNumNodesToBeCreated,s,memResource);
      initNodes<<<32,1,0,s>>>(d_buildState,nodes,numValidPrims);

      /* 3.2 extract nodes until no more (temp-)nodes get created */
      int numNodesAlloced = 1; /*!< device actually things it's two,
                                 but we intentionally use 1 here to
                                 make first round start with right
                                 could of _valid_ nodes*/
      
      int numNodesDone    = 0;
      while (numNodesDone < numNodesAlloced) {
        int numNodesStillToDo = numNodesAlloced - numNodesDone;
        createNodes<<<divRoundUp(numNodesStillToDo,1024),1024,0,s>>>
          (d_buildState,makeLeafThreshold,
           buildConfig.maxAllowedLeafSize,
           nodes,numNodesDone,numNodesAlloced,
           d_primKeys_sorted);
        CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,sizeof(*h_buildState),
                                    cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
        CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));
        
        numNodesDone = numNodesAlloced;
        numNodesAlloced = h_buildState->numNodesAlloced;
      }
      
      // ==================================================================
      // step four: create actual ndoes - we now know how many, and
      // what they point to; let's just fillin topology and let refit
      // fill in the boxes later on
      // ==================================================================
      /* 4.1 - free keys, we no longer need them. */
      _FREE(d_primKeys_sorted,s,memResource);
      /* 4.2 - save morton-ordered prims in bvh - that's where the
         final nodes will be pointing into, so they are our primID
         array. */
      bvh.primIDs = d_primIDs_inMortonOrder;
      bvh.numPrims = numValidPrims;

      /* 4.3 alloc 'final' nodes; we now know exactly how many we
         have */
      bvh.numNodes = numNodesAlloced;
      _ALLOC(bvh.nodes,numNodesAlloced,s,memResource);
      writeFinalNodes<T,D><<<divRoundUp(numNodesAlloced,1024),1024,0,s>>>
        (bvh.nodes,nodes,numNodesAlloced);
      
      /* 4.4 cleanup - free temp nodes, free build state, and release event */
      CUBQL_CUDA_CALL(EventDestroy(stateDownloadedEvent));
      _FREE(nodes,s,memResource);
      _FREE(d_buildState,s,memResource);

      // ==================================================================
      // done. all we need to do now is refit the bboxes
      // ==================================================================
      cuBQL::cuda::refit(bvh,boxes,s,memResource);
    }
  }

  namespace cuda {
    template<typename T, int D>
    void radixBuilder(cuBQL::BinaryBVH<T,D>    &bvh,
                      const cuBQL::box_t<T,D>  *boxes,
                      uint32_t                  numPrims,
                      cuBQL::BuildConfig        buildConfig,
                      cudaStream_t              s,
                      cuBQL::GpuMemoryResource &memResource)
    {
      radixBuilder_impl::build(bvh,boxes,numPrims,buildConfig,s,memResource);
    }
  }
}

