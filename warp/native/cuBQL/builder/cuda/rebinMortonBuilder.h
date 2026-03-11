// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/sm_builder.h"
#include "cuBQL/builder/cuda/radix.h"

#define NO_BLOCK_PHASE 1

#ifdef CUBQL_GPU_BUILDER_IMPLEMENTATION
namespace cuBQL {
  namespace rebinRadixBuilder_impl {
    using gpuBuilder_impl::atomic_grow;
    using gpuBuilder_impl::_ALLOC;
    using gpuBuilder_impl::_FREE;

    template<typename T, int D> struct Quantizer;

    template<int D> struct numMortonBits;
    template<> struct numMortonBits<2> { enum { value = 15 }; };
    template<> struct numMortonBits<3> { enum { value = 10 }; };
    template<> struct numMortonBits<4> { enum { value =  7 }; };
    
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
      using vec_t = cuBQL::vec_t<T,D>;
      using box_t = cuBQL::box_t<T,D>;
      using bvh_t = cuBQL::BinaryBVH<T,D>;
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
      // box_t        centBounds;

      int numBroadPhaseRebinPrims;
      int numBroadPhaseRebinJobs;
      
      // /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
      //     quantization operation that does
      //     `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
      //     centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      // vec_t CUBQL_ALIGN(16) quantizeBias, CUBQL_ALIGN(16) quantizeScale;
      Quantizer<T,D> quantizer;
    };

    template<typename T, int D>
    __global__
    void clearBuildState(BuildState<T,D>  *buildState,
                         int          numPrims)
    {
      if (threadIdx.x != 0) return;
      
      buildState->a_centBounds.clear();
      // let's _start_ with the assumption that all are valid, and
      // subtract those later on that are not.
      buildState->numValidPrims           = numPrims;
      buildState->numNodesAlloced         = 0;
      buildState->numBroadPhaseRebinPrims = 0;
      buildState->numBroadPhaseRebinJobs  = 0;
    }
    
    template<typename T, int D>
    __global__
    void fillBuildState(BuildState<T,D>  *buildState,
                        const typename BuildState<T,D>::box_t *prims,
                        int          numPrims)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
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
    // template<typename T, int D>
    // __global__
    // void finishBuildState(BuildState<T,D>  *buildState);

    // template<>
    // __global__
    // void finishBuildState(BuildState<float,3>  *buildState)
    // {
    //   using ctx_t = BuildState<float,3>;
    //   using vec_t = typename ctx_t::vec_t;
    //   using box_t = typename ctx_t::box_t;
    //   using bvh_t = typename ctx_t::bvh_t;
    //   using atomic_box_t = typename ctx_t::atomic_box_t;
      
    //   if (threadIdx.x != 0) return;
      
    //   box_t centBounds = buildState->a_centBounds.make_box();
    //   // buildState->centBounds = centBounds;
    //   buildState->quantizer.init(centBounds);
    // }


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
    inline __device__
    uint32_t shiftBits(uint32_t x, uint32_t maskOfBitstoMove, int howMuchToShift)
    { return ((x & maskOfBitstoMove)<<howMuchToShift) | (x & ~maskOfBitstoMove); }
    
    // inline __device__
    // uint64_t bitInterleave21(uint64_t x)
    // {
    //   //hex    00:       00:       00:       00:       00:       10:       00:       00
    //   x = shiftBits(x,0x00000000001f0000ull,32); 
    //   //hex     00:      00:       00:       00:       00:       00:       ff:       00
    //   x = shiftBits(x,0x000000000000ff00ull,16); 
    //   //hex    00:       f0:       00:       00:       f0:       00:       00:       f0
    //   x = shiftBits(x,0x00f00000f00000f0ull,8); 
    //   //hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c
    //   x = shiftBits(x,0x000c00c00c00c00cull,4); 
    //   //hex    00:       82:       08:       20:       82:       08:       20:       82
    //   x = shiftBits(x,0x0082082082082082ull,2);
    //   return x;
    // }
    // inline __device__
    // uint32_t bitInterleave10(uint32_t x)
    // {
    //   //hex     00:      00:       00:       00:       00:       00:       ff:       00
    //   x = shiftBits(x,0x0000ff00,16); 
    //   //hex    00:       f0:       00:       00:       f0:       00:       00:       f0
    //   x = shiftBits(x,0xf00000f0,8); 
    //   //hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c
    //   x = shiftBits(x,0x0c00c00c,4); 
    //   //hex    00:       82:       04:       20:       82:       04:       20:       82
    //   x = shiftBits(x,0x82082082,2);
    //   return x;
    // }

    /*! insert 1 zero in-between every two successive bits of the
      input. ie, bit 0 stays at 0, bit 1 goes to 3, etc */
    inline __device__
    uint32_t bitInterleave11(uint32_t x)
    {
      // current x is this:                      FEDC.BAzy.xwvu.tsrq.ponm.lkji.hgfe.dcba

      // 0000.0000:0000.0000:0000.0000:0000.0000:FEDC.BAzy:xwvu.tsrq:ponm.lkji:hgfe.dcba

      // x = shiftBits(x,0xffff0000ull,16);
      
      // 0000.0000:0000.0000:FEDC.BAzy:xwvu.tsrq:0000.0000:0000.0000:ponm.lkji:hgfe.dcba

      x = shiftBits(x,0b00000000000000001111111100000000u,8);
      
      // 0000.0000:FEDC.BAzy:0000.0000:xwvu.tsrq:0000.0000:ponm.lkji:0000.0000:hgfe.dcba

      x = shiftBits(x,0b00000000111100000000000011110000u,4);
      
      // 0000.FEDC:0000.BAzy:0000.xwvu:0000.tsrq:0000.ponm:0000.lkji:0000.hgfe:0000.dcba

      x = shiftBits(x,0b00001100000011000000110000001100u,2);

      // 00FE.00DC:00BA.00zy:00xw.00vu:00ts.00rq:00po.00nm:00lk.00ji:00hg.00fe:00dc.00ba
      
      x = shiftBits(x,0b00100010001000100010001000100010u,1);
      
      // 0F0E.0D0C:0B0A.0z0y:0x0w.0v0u:0t0s.0r0q:0p0o.0n0m:0l0k.0j0i:0h0g.0f0e:0d0c.0b0a
      return x;
    }
    
    inline __device__ uint32_t interleaveBits(vec2ui bits)
    {
      return
        (bitInterleave11(bits.y) << 1)
        |
        bitInterleave11(bits.x);
    }
    
    inline __device__ uint32_t interleaveBits(vec4ui bits)
    {
      uint32_t ix = bits.x;
      uint32_t iy = bits.y;
      uint32_t iz = bits.z;
      uint32_t iw = bits.w;
      ix = ((ix & 0xF0) << 12) | (ix & ~0xF0);
      ix = ((ix & 0b000000000011000000000000001100)<<6) | (ix & ~0b000000000011000000000000001100);
      ix = ((ix & 0b000010000000100000001000000010)<<3) | (ix & ~0b000010000000100000001000000010);
      
      iy = ((iy & 0xF0) << 12) | (iy & ~0xF0);
      iy = ((iy & 0b000000000011000000000000001100)<<6) | (iy & ~0b000000000011000000000000001100);
      iy = ((iy & 0b000010000000100000001000000010)<<3) | (iy & ~0b000010000000100000001000000010);
      
      iz = ((iz & 0xF0) << 12) | (iz & ~0xF0);
      iz = ((iz & 0b000000000011000000000000001100)<<6) | (iz & ~0b000000000011000000000000001100);
      iz = ((iz & 0b000010000000100000001000000010)<<3) | (iz & ~0b000010000000100000001000000010);
    
      iw = ((iw & 0xF0) << 12) | (iw & ~0xF0);
      iw = ((iw & 0b000000000011000000000000001100)<<6) | (iw & ~0b000000000011000000000000001100);
      iw = ((iw & 0b000010000000100000001000000010)<<3) | (iw & ~0b000010000000100000001000000010);
    
      return (iw << 3) | (iz << 2) | (iy << 1) | ix;
    }
    
    
    inline __device__ uint32_t interleaveBits(vec3ui mortonCell)
    {
      int ix = mortonCell.x;
      int iy = mortonCell.y;
      int iz = mortonCell.z;
      
      ix = ((ix & 0b000000000000000000001100000000) << 16) | (ix & ~0b000000000000000000001100000000);
      ix = ((ix & 0b000000000000000000000011110000) <<  8) | (ix & ~0b000000000000000000000011110000);
      ix = ((ix & 0b000000000000001100000000001100) <<  4) | (ix & ~0b000000000000001100000000001100);
      ix = ((ix & 0b000010000010000010000010000010) <<  2) | (ix & ~0b000010000010000010000010000010);
      
      iy = ((iy & 0b000000000000000000001100000000) << 16) | (iy & ~0b000000000000000000001100000000);
      iy = ((iy & 0b000000000000000000000011110000) <<  8) | (iy & ~0b000000000000000000000011110000);
      iy = ((iy & 0b000000000000001100000000001100) <<  4) | (iy & ~0b000000000000001100000000001100);
      iy = ((iy & 0b000010000010000010000010000010) <<  2) | (iy & ~0b000010000010000010000010000010);
      
      iz = ((iz & 0b000000000000000000001100000000) << 16) | (iz & ~0b000000000000000000001100000000);
      iz = ((iz & 0b000000000000000000000011110000) <<  8) | (iz & ~0b000000000000000000000011110000);
      iz = ((iz & 0b000000000000001100000000001100) <<  4) | (iz & ~0b000000000000001100000000001100);
      iz = ((iz & 0b000010000010000010000010000010) <<  2) | (iz & ~0b000010000010000010000010000010);

      return (iz << 2) | (iy << 1) | (ix);
    }
    
    template<typename T, int D>
    inline __device__
    // uint32_t computeMortonCode(vec3f P, vec3f quantizeBias, vec3f quantizeScale)
    uint32_t computeMortonCode(typename BuildState<T,D>::vec_t P,
                               const Quantizer<T,D> quantizer)
    {
      return interleaveBits(quantizer.quantize(P));
      // // // quantizeScale = vec3f(reduce_min(quantizeScale));
      // // P = (P - quantizeBias) * quantizeScale;
      // // vec3i mortonCell = min(max(vec3i(0),vec3i(P)),vec3i((1<<10)-1));
      // vec3i mortonCell = quantizer.quantize(P);
      // //min(max(vec3i(0),vec3i(P)),vec3i((1<<10)-1));
      // int ix = mortonCell.x;
      // int iy = mortonCell.y;
      // int iz = mortonCell.z;
      
      // ix = ((ix & 0b000000000000000000001100000000) << 16) | (ix & ~0b000000000000000000001100000000);
      // ix = ((ix & 0b000000000000000000000011110000) <<  8) | (ix & ~0b000000000000000000000011110000);
      // ix = ((ix & 0b000000000000001100000000001100) <<  4) | (ix & ~0b000000000000001100000000001100);
      // ix = ((ix & 0b000010000010000010000010000010) <<  2) | (ix & ~0b000010000010000010000010000010);
      
      // iy = ((iy & 0b000000000000000000001100000000) << 16) | (iy & ~0b000000000000000000001100000000);
      // iy = ((iy & 0b000000000000000000000011110000) <<  8) | (iy & ~0b000000000000000000000011110000);
      // iy = ((iy & 0b000000000000001100000000001100) <<  4) | (iy & ~0b000000000000001100000000001100);
      // iy = ((iy & 0b000010000010000010000010000010) <<  2) | (iy & ~0b000010000010000010000010000010);
      
      // iz = ((iz & 0b000000000000000000001100000000) << 16) | (iz & ~0b000000000000000000001100000000);
      // iz = ((iz & 0b000000000000000000000011110000) <<  8) | (iz & ~0b000000000000000000000011110000);
      // iz = ((iz & 0b000000000000001100000000001100) <<  4) | (iz & ~0b000000000000001100000000001100);
      // iz = ((iz & 0b000010000010000010000010000010) <<  2) | (iz & ~0b000010000010000010000010000010);

      // return (iz << 2) | (iy << 1) | (ix);
    }

    template<typename T, int D>
    inline __device__
    uint32_t computeMortonCode(vec_t<T,D> P, box_t<T,D> domain)
    {
      Quantizer<T,D> quantizer;
      quantizer.init(domain);
      // vec3f quantizeBias = domain.lower;
      // vec3f quantizeScale
      //   = vec3f(1<<10)*rcp(max(vec3f(reduce_max(domain.size())),vec3f(1e-20f)));
      return computeMortonCode(P,quantizer);//quantizeBias,quantizeScale);
      // P = (P - quantizeBias) * quantizeScale;
      // vec3i mortonCell = min(vec3i(P),vec3i((1<<10)-1));
      // return
      //   (bitInterleave10(mortonCell.z) << 2) |
      //   (bitInterleave10(mortonCell.y) << 1) |
      //   (bitInterleave10(mortonCell.x) << 0);
    }
    
    // inline __device__
    // uint64_t computeMortonCode(vec3f P, vec3f quantizeBias, vec3f quantizeScale)
    // {
    //   P = (P - quantizeBias) * quantizeScale;
    //   vec3i mortonCell = min(vec3i(P),vec3i((1<<21)-1));
    //   return
    //     (bitInterleave21(mortonCell.z) << 2) |
    //     (bitInterleave21(mortonCell.y) << 1) |
    //     (bitInterleave21(mortonCell.x) << 0);
    // }
    
    template<typename T, int D>
    __global__
    void computeUnsortedKeysAndPrimIDs(uint32_t    *mortonCodes,
                                       uint32_t    *primIDs,
                                       BuildState<T,D>  *buildState,
                                       const typename BuildState<T,D>::box_t *prims,
                                       int numPrims)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
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
                            // buildState->quantizeBias,
                            // buildState->quantizeScale);
    }

    struct TempNode {
      union {
        /*! nodes that have been opened by their parents, but have not
          yet been finished. such nodes descibe a list of
          primitives; the range of keys covered in this subtree -
          which can/will be used to determine where to split - is
          encoded in first and last key in that range */
        struct {
          inline __cubql_both uint64_t size() const { return end - begin; }
          uint64_t begin:32;
          uint64_t end:31;
          uint64_t open:1;
        } open;
        /*! nodes that are finished and done */
        struct {
          uint64_t offset:32;
          uint64_t count:31;
          uint64_t open:1;
        } finished;
        struct {
          uint64_t unused0:32;
          uint64_t unused1:31;
          uint64_t open:1;
        } common;
        // force alignment to 8-byte values, so compiler can
        // read/write more efficiently
        uint64_t bits;
      };
    };


    // inline __device__
    // bool findSplit(int64_t &split,
    //                const uint64_t *__restrict__ keys,
    //                uint64_t begin, uint64_t end)
    // {
    //   uint64_t firstKey = keys[begin];
    //   uint64_t lastKey  = keys[end-1];
      
    //   if (firstKey == lastKey)
    //     // same keys entire range - no split in there ....
    //     return false;
      
    //   int numMatchingBits = __clzll(firstKey ^ lastKey);
    //   // the first key in the plane we're searching has
    //   // 'numMatchingBits+1' top bits of lastkey, and 0es otherwise
    //   const uint64_t searchKey = lastKey & (0xffffffffffffffffull<<(63-numMatchingBits));

    //   while (end > begin) {
    //     int mid = (begin+end)/2;
    //     if (keys[mid] < searchKey) {
    //       begin = mid+1;
    //     } else {
    //       end = mid;
    //     }
    //   }
    //   split = begin;
    //   return true;
    // }

    inline __device__
    bool findSplit(int64_t &split,
                   const uint32_t *__restrict__ keys,
                   uint64_t begin, uint64_t end)
    {
      uint32_t firstKey = keys[begin];
      uint32_t lastKey  = keys[end-1];
      
      if (firstKey == lastKey)
        // same keys entire range - no split in there ....
        return false;
      
      int numMatchingBits = __clz(firstKey ^ lastKey);
      // the first key in the plane we're searching has
      // 'numMatchingBits+1' top bits of lastkey, and 0es otherwise
      const uint32_t searchKey = lastKey & (0xffffffffull<<(31-numMatchingBits));

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

    inline __device__
    bool smallEnoughToLeaveBlockPhase(uint32_t size)
    {
      return size <= 32;
    }
    
    inline __device__
    bool smallEnoughToLeaveBroadPhase(uint64_t size)
    {
#if NO_BLOCK_PHASE
      return smallEnoughToLeaveBlockPhase(size);
#else
      return size <= 1024;
#endif
    }
    
    inline __device__
    bool smallEnoughToLeaveBroadPhase(TempNode node)
    { return smallEnoughToLeaveBroadPhase(node.open.size()); }
    

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
      n0.open.open  = 1;
      n1.bits = 0;
      nodes[0] = n0;
      nodes[1] = n1;
    }

    template<typename T, int D>
    __global__
    void broadPhaseCreateNodes(BuildState<T,D> *buildState,
                               int leafThreshold,
                               TempNode *nodes,
                               int pass_begin, int pass_end,
                               const uint32_t *keys)
    {

      __shared__ int l_allocOffset;
      
      if (threadIdx.x == 0)
        l_allocOffset = 0;
      // ==================================================================
      __syncthreads();
      // ==================================================================
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      int nodeID = pass_begin + tid;
      bool validNode = (nodeID != 1) && (nodeID < pass_end);
      int64_t split   = -1;
      int childID = -1;
      TempNode node;
      
      if (validNode) {
        node = nodes[nodeID];
        const uint64_t subtree_begin = node.open.begin;
        const uint64_t subtree_end   = node.open.end;
        const int subtree_size  = int(subtree_end - subtree_begin);
        if (!node.common.open) {
          // node wasn't even open - this should only ever happen if
          // we've entered a new phase, or re-started after
          // rebinning. either way that node is already done and
          // doesn't need anything done.
          validNode = false;
          nodeID = -1;
        } else if (subtree_size <= leafThreshold) {
          // we WANT to make a leaf
          node.finished.offset = subtree_begin;
          node.finished.count  = subtree_size;
          node.finished.open   = false;
        } else if (smallEnoughToLeaveBroadPhase(subtree_size)) {
          // we're small enough to leave broad phase - leave this
          // marked as open and forget about it for now
          validNode = false;
        } else if (!findSplit(split,keys,subtree_begin,subtree_end)) {
          // we couln't split, and need (broad-phase) rebinning. leave
          // marked as open and forget about it for now, but keep
          // track that we need to find this later on
          validNode = false;
          atomicAdd(&buildState->numBroadPhaseRebinJobs,1);
          atomicAdd(&buildState->numBroadPhaseRebinPrims,(int)subtree_size);
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
        const uint64_t subtree_begin = node.open.begin;
        const uint64_t subtree_end   = node.open.end;
        c0.open.begin = subtree_begin;
        c0.open.end   = split;
        c0.open.open  = true;
        c1.open.begin = split;
        c1.open.end   = subtree_end;
        c1.open.open  = true;
        // we COULD actually write those as a int4 if we really wanted
        // to ...
        nodes[childID+0]     = c0;
        nodes[childID+1]     = c1;
        node.finished.offset = childID;
        node.finished.count  = 0;
        node.finished.open   = 0;
      }
      if (validNode)
        nodes[nodeID] = node;
    }

    inline __device__
    void swap(uint32_t &a, uint32_t &b)
    {
      uint32_t c = a; a = b; b = c;
    }

    inline __device__
    void bubbleSort(uint32_t *keys, uint32_t *values,
                    int begin, int end)
    {
      for (int i=begin;i<end;i++)
        for (int j=i+1;j<end;j++) {
          if (keys[j] < keys[i]) {
            swap(keys[i],keys[j]);
            swap(values[i],values[j]);
          }
        }
        
    }

    template<typename T, int D>
    __global__
    void threadPhaseCreateNodes(BuildState<T,D> *buildState,
                                const box_t<T,D> *boxes,
                                int leafThreshold,
                                int maxAllowedLeafSize,
                                TempNode *nodes,
                                int pass_begin, int pass_end,
                                uint32_t *keys,
                                uint32_t *primIDs)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
      __shared__ int l_allocOffset;

      if (threadIdx.x == 0)
        l_allocOffset = 0;
      // ==================================================================
      __syncthreads();
      // ==================================================================
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      int nodeID = pass_begin + tid;
      bool validNode = (nodeID != 1) && (nodeID < pass_end);
      int64_t split   = -1;
      int childID = -1;
      TempNode node;
      
      if (validNode) {
        node = nodes[nodeID];
        const uint64_t subtree_begin = node.open.begin;
        const uint64_t subtree_end   = node.open.end;
        const uint64_t subtree_size  = subtree_end - subtree_begin;
        if (!node.common.open) {
          // node wasn't even open - this should only ever happen if
          // we've entered a new phase, or re-started after
          // rebinning. either way that node is already done and
          // doesn't need anything done.
          validNode = false;
        } else if (subtree_size <= leafThreshold) {
          // we WANT to make a leaf
          node.finished.offset = subtree_begin;
          node.finished.count  = subtree_size;
          node.finished.open   = false;
        } else if (!findSplit(split,keys,subtree_begin,subtree_end)) {
          // we couldn't split, and need rebinning - but since we're
          // in thread phase we have do to that here and now.
          box_t domain;
          domain.clear();
          for (int i=subtree_begin;i<subtree_end;i++)
            domain.grow(boxes[primIDs[i]].center());
          if (!(domain.lower == domain.upper)) {
            for (int i=subtree_begin;i<subtree_end;i++)
              keys[i] = computeMortonCode(boxes[primIDs[i]].center(),domain);
          } else if (0 && subtree_size < maxAllowedLeafSize) {
            // make a leaf, no matter what user desired
            node.finished.offset = subtree_begin;
            node.finished.count  = subtree_size;
            node.finished.open   = false;
          } else {
            for (int i=subtree_begin;i<subtree_end;i++)
              keys[i] = i-subtree_begin;
          }
          if (node.common.open) {
            bubbleSort(keys,primIDs,subtree_begin,subtree_end);
            
            if (!findSplit(split,keys,subtree_begin,subtree_end)) {
              // ugh - we generated new keys and STILL can't find a split!?
              node.finished.offset = subtree_begin;
              node.finished.count  = subtree_size;
              node.finished.open   = false;
            } else {
              childID = atomicAdd(&l_allocOffset,2);
            }
          }
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
        const uint64_t subtree_begin = node.open.begin;
        const uint64_t subtree_end   = node.open.end;
        c0.open.begin = subtree_begin;
        c0.open.end   = split;
        c0.open.open  = true;
        c1.open.begin = split;
        c1.open.end   = subtree_end;
        c1.open.open  = true;
        // we COULD actually write those as a int4 if we really wanted
        // to ...
        nodes[childID+0]     = c0;
        nodes[childID+1]     = c1;
        node.finished.offset = childID;
        node.finished.count  = 0;
        node.finished.open   = false;
      }
      if (validNode)
        nodes[nodeID] = node;
    }
    
    template<typename T, int D>
    __global__
    void writeFinalNodes(typename BinaryBVH<T,D>::Node *finalNodes,
                         const TempNode *__restrict__ tempNodes,
                         int numNodes)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numNodes) return;
      typename BinaryBVH<T,D>::Node node;
      TempNode tempNode = tempNodes[tid];
      node.admin.offset = tempNode.finished.offset;
      node.admin.count  = tempNode.finished.count;
      finalNodes[tid].admin.offsetAndCountBits = node.admin.offsetAndCountBits;
    }

    struct RebinRange {
      union {
        struct {
          uint32_t nodeID;
          // high bits that we want to sort by:
          uint32_t beginOfRebinPrimRange;
        };
        uint64_t bits;
      };
    };
    
    template<typename T, int D>
    struct RebinDomain {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
      atomic_box_t centBounds;
    };

    template<typename T, int D>
    __global__
    void rebinClearDomains(RebinDomain<T,D> *rebinDomains,
                           int numBroadPhaseRebinJobs)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numBroadPhaseRebinJobs) return;
      rebinDomains[tid].centBounds.clear();
    }

    inline __device__
    int findMatchingRebinRange(const RebinRange *ranges,
                               int numRanges,
                               int rebinPrimID)
    {
      int begin = 0;
      int end = numRanges;

      while ((end - begin) > 1) {
        int mid = (begin+end)/2;
        if (rebinPrimID >= ranges[mid].beginOfRebinPrimRange) {
          begin = mid;
        } else {
          end = mid;
        }
      }
      return begin;
    }
    
    // 3.7 fill out rebin domains - every logical rebin prim will
    // first figure out which rebin job it is (from which it can
    // compute its global prim ID), then read that global prim id,
    // read its box, and extend its jobs rebin domain
    template<typename T, int D>
    __global__
    void rebinGrowDomains(BuildState<T,D> *buildState,
                          TempNode *nodes,
                          // original inputs:
                          const box_t<T,D> *boxes,
                          const uint32_t *primIDs_inMortonOrder,
                          // the jobs that those rebin prims are in
                          RebinRange  *rebinRanges_sorted,
                          RebinDomain<T,D> *rebinDomains,
                          int numBroadPhaseRebinJobs,
                          // how many prims we even have to write:
                          int numBroadPhaseRebinPrims)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numBroadPhaseRebinPrims) return;

      int rangeID = findMatchingRebinRange
        (rebinRanges_sorted,numBroadPhaseRebinJobs,tid);
      RebinRange range   = rebinRanges_sorted[rangeID];
      int        localID = tid - range.beginOfRebinPrimRange;
      TempNode   node    = nodes[range.nodeID];
      int primID = primIDs_inMortonOrder[node.open.begin + localID];
      box_t box  = boxes[primID];
      atomic_grow(rebinDomains[rangeID].centBounds,box.center());
    }

    struct RebinKey {
      union {
        struct {
          uint32_t morton;
          uint32_t rangeID;
        };
        uint64_t lo32_morton_hi32_rangeID;
      };
    };

    template<typename T, int D>
    __global__
    void rebinWriteBackNewKeysAndPrimIDs(BuildState<T,D> *buildState,
                                         TempNode *nodes,
                                         uint32_t *primKeys_sorted,
                                         uint32_t *primIDs_inMortonOrder,
                                         const RebinRange *rebinRanges,
                                         // prims:
                                         const RebinKey  *rebinKeys_sorted,
                                         const uint32_t  *rebinPrimIDs_sorted,
                                         int numBroadPhaseRebinPrims)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numBroadPhaseRebinPrims) return;

      RebinKey   rebinKey   = rebinKeys_sorted[tid];
      uint32_t   rebinValue = rebinPrimIDs_sorted[tid];
      RebinRange range      = rebinRanges[rebinKey.rangeID];
      int        localID    = tid - range.beginOfRebinPrimRange;
      int        globalID   = nodes[range.nodeID].open.begin + localID;

      primKeys_sorted[globalID] = rebinKey.morton;
      primIDs_inMortonOrder[globalID]  = rebinValue;
    }
    
    template<typename T, int D>
    __global__
    void rebinCreateNewKeysAndPrimIDs(BuildState<T,D> *buildState,
                                      TempNode *nodes,
                                      // original inputs:
                                      const typename BuildState<T,D>::box_t *boxes,
                                      const uint32_t *primIDs_inMortonOrder,
                                      // the jobs that those rebin prims are in
                                      RebinRange  *rebinRanges_sorted,
                                      RebinDomain<T,D> *rebinDomains,
                                      int numBroadPhaseRebinJobs,
                                      // arrays we need to fill:
                                      RebinKey  *newKeys,
                                      uint32_t *newValues,
                                      // how many prims we even have to write:
                                      int numBroadPhaseRebinPrims)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numBroadPhaseRebinPrims) return;

      uint32_t rangeID = findMatchingRebinRange
        (rebinRanges_sorted,numBroadPhaseRebinJobs,tid);
      RebinRange range   = rebinRanges_sorted[rangeID];
      int        localID = tid - range.beginOfRebinPrimRange;
      TempNode   node    = nodes[range.nodeID];
      uint32_t primID = primIDs_inMortonOrder[node.open.begin + localID];
      box_t box  = boxes[primID];
      box_t domain = rebinDomains[rangeID].centBounds.make_box();
      uint32_t newMorton
        = (domain.lower == domain.upper)
        ? localID
        : computeMortonCode(box.center(),domain);
      RebinKey newKey;
      newKey.morton  = newMorton;
      newKey.rangeID = rangeID;
      newKeys[tid]   = newKey;
      newValues[tid] = primID;
    }
    

    inline
    void keyValueSort(const uint64_t *d_keysValues_in,
                      uint64_t *d_keysValues_out,
                      int N,
                      cudaStream_t s,
                      GpuMemoryResource &memResource)
    {
      size_t cub_tempMemSize = 0;
      // query temp mem size
      cub::DeviceRadixSort::SortKeys
        (nullptr,cub_tempMemSize,
         /*key+values in:*/   d_keysValues_in,
         /*key+values out:*/  d_keysValues_out,
         N,32,64,s);
      
      // allocate temp mem and output arrays
      void     *d_tempMem = 0;
      memResource.malloc(&d_tempMem,cub_tempMemSize,s);

      // sort
      cub::DeviceRadixSort::SortKeys
        (d_tempMem,cub_tempMemSize,
         /*key+values in:*/   d_keysValues_in,
         /*key+values out:*/  d_keysValues_out,
         N,32,64,s);
      
      _FREE(d_tempMem,s,memResource);
    }                      

    inline
    void keyValueSort(const uint64_t *d_keys_in,
                      uint64_t *d_keys_out,
                      const uint32_t *d_values_in,
                      uint32_t *d_values_out,
                      
                      int N,
                      cudaStream_t s,
                      GpuMemoryResource &memResource)
    {
      size_t cub_tempMemSize = 0;
      // query temp mem size
      cub::DeviceRadixSort::SortPairs
        (nullptr,cub_tempMemSize,
         /*keys in:*/    d_keys_in,
         /*keys out:*/   d_keys_out,
         /*values in:*/  d_values_in,
         /*values out:*/ d_values_out,
         N,0,64,s);
      
      // allocate temp mem and output arrays
      void     *d_tempMem = 0;
      memResource.malloc(&d_tempMem,cub_tempMemSize,s);

      // sort
      cub::DeviceRadixSort::SortPairs
        (d_tempMem,cub_tempMemSize,
         /*keys in:*/    d_keys_in,
         /*keys out:*/   d_keys_out,
         /*values in:*/  d_values_in,
         /*values out:*/ d_values_out,
         N,0,64,s);
      
      _FREE(d_tempMem,s,memResource);
    }                      

    // executed once for each alloced node; job is to find all those
    // nodes that need rebinning, which are all those that are
    // alloced, still open at the end of broad-phase node generation,
    // and not small enought to leave broad phase
    template<typename T, int D>
    __global__
    void rebinFindRanges(BuildState<T,D> *buildState,
                         RebinRange *rebinRanges,
                         TempNode   *nodes,
                         int         numNodes)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numNodes) return;

      TempNode node = nodes[tid];
      if (!node.common.open) return;

      if (smallEnoughToLeaveBroadPhase(node)) return;
      int rangeID = atomicAdd(&buildState->numBroadPhaseRebinJobs,1);
      int rangeBegin = atomicAdd(&buildState->numBroadPhaseRebinPrims,node.open.size());
      rebinRanges[rangeID].beginOfRebinPrimRange = rangeBegin;
      rebinRanges[rangeID].nodeID = tid;
    }

    template<typename T, int D>
    inline 
    void build(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               int                numPrims,
               BuildConfig        buildConfig,
               cudaStream_t       s,
               GpuMemoryResource &memResource)
    {
      using ctx_t = BuildState<T,D>;
      using vec_t = typename ctx_t::vec_t;
      using box_t = typename ctx_t::box_t;
      using bvh_t = typename ctx_t::bvh_t;
      using atomic_box_t = typename ctx_t::atomic_box_t;
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
      fillBuildState<<<divRoundUp(numPrims,1024),1024,0,s>>>
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
      uint32_t *d_primKeys_unsorted;
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
      uint32_t *d_primKeys_sorted = 0;
      uint32_t *d_primIDs_inMortonOrder = 0;
      // with tempMem ptr null this won't do anything but return reqd
      // temp size*/
      cub::DeviceRadixSort::SortPairs
        (nullptr,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,32,s);
      
      // 2.3: allocate temp mem and output arrays
      void     *d_tempMem = 0;
      memResource.malloc(&d_tempMem,cub_tempMemSize,s);
      _ALLOC(d_primKeys_sorted,numValidPrims,s,memResource);
      _ALLOC(d_primIDs_inMortonOrder,numValidPrims,s,memResource);

      // 2.4: sort
      cub::DeviceRadixSort::SortPairs
        (d_tempMem,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,32,s);

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

      /* 3.2 extract *broad phase* nodes until no more (temp-)nodes
         get created */
      int numNodesAlloced = 1;
      int numNodesDone    = 0;
      while (true) {
        while (numNodesDone < numNodesAlloced) {
          int numNodesStillToDo = numNodesAlloced - numNodesDone;
          broadPhaseCreateNodes
            <<<divRoundUp(numNodesStillToDo,1024),1024,0,s>>>
            (d_buildState,makeLeafThreshold,
             nodes,numNodesDone,numNodesAlloced,
             d_primKeys_sorted);
          CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,sizeof(*h_buildState),
                                      cudaMemcpyDeviceToHost,s));
          CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
          CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));
          
          numNodesDone    = numNodesAlloced;
          numNodesAlloced = h_buildState->numNodesAlloced;
        }

        // 3.3: we ran out of broad-phase nodes that we COULD split,
        // but there migth be some that need rebinning (and would
        // restart the process): check how many there are.
        int numBroadPhaseRebinJobs  = h_buildState->numBroadPhaseRebinJobs;
        int numBroadPhaseRebinPrims = h_buildState->numBroadPhaseRebinPrims;
        if (numBroadPhaseRebinJobs == 0)
          // no huge rebins - done.
          break;

        // 3.4: alloc descriptors and centroids for those rebin jobs
        RebinRange  *rebinRanges_unsorted    = 0;
        RebinRange  *rebinRanges_sorted    = 0;
        RebinDomain<T,D> *rebinDomains = 0;
        
        _ALLOC(rebinRanges_unsorted,numBroadPhaseRebinJobs,s,memResource);
        _ALLOC(rebinRanges_sorted,numBroadPhaseRebinJobs,s,memResource);
        _ALLOC(rebinDomains,numBroadPhaseRebinJobs,s,memResource);

        // 3.5 find rebin ranges
        h_buildState->numBroadPhaseRebinJobs  = 0;
        h_buildState->numBroadPhaseRebinPrims = 0;
        CUBQL_CUDA_CALL(MemcpyAsync(d_buildState,h_buildState,sizeof(*h_buildState),
                                    cudaMemcpyHostToDevice,s));
        rebinFindRanges<<<divRoundUp(numNodesAlloced,1024),1024,0,s>>>
          (d_buildState,rebinRanges_unsorted,
           // nodes to extract from:
           nodes,numNodesAlloced);

        // 3.6 sort rebin ranges, and clear rebin domains
        keyValueSort((const uint64_t*)rebinRanges_unsorted,
                     (uint64_t*)rebinRanges_sorted,
                     numBroadPhaseRebinJobs,s,memResource);
        _FREE(rebinRanges_unsorted,s,memResource);
        rebinClearDomains<<<divRoundUp(numBroadPhaseRebinJobs,1024),1024,0,s>>>
          (rebinDomains,numBroadPhaseRebinJobs);

        // 3.7 fill out rebin domains - every logical rebin prim will
        // first figure out which rebin job it is (from which it can
        // compute its global prim ID), then read that global prim id,
        // read its box, and extend its jobs rebin domain
        rebinGrowDomains<<<divRoundUp(numBroadPhaseRebinPrims,1024),1024,0,s>>>
          (d_buildState,nodes,boxes,d_primIDs_inMortonOrder,
           // jobs:
           rebinRanges_sorted,rebinDomains,numBroadPhaseRebinJobs,
           // prims:
           numBroadPhaseRebinPrims);

        // 3.8 allocate rebin key/value arrays, and fill them with new
        // keys and correponding (global) primIDs
        RebinKey  *rebinKeys_unsorted    = 0;
        uint32_t  *rebinPrimIDs_unsorted = 0;
        _ALLOC(rebinKeys_unsorted,numBroadPhaseRebinPrims,s,memResource);
        _ALLOC(rebinPrimIDs_unsorted,numBroadPhaseRebinPrims,s,memResource);
        rebinCreateNewKeysAndPrimIDs
          <<<divRoundUp(numBroadPhaseRebinPrims,1024),1024,0,s>>>
          (d_buildState,nodes,boxes,d_primIDs_inMortonOrder,
           // jobs:
           rebinRanges_sorted,rebinDomains,numBroadPhaseRebinJobs,
           // prims:
           rebinKeys_unsorted,rebinPrimIDs_unsorted,numBroadPhaseRebinPrims);
        // aaand: no longer need the domains now
        _FREE(rebinDomains,s,memResource);

        // 3.9 re-sort new array
        RebinKey  *rebinKeys_sorted    = 0;
        uint32_t  *rebinPrimIDs_sorted = 0;
        _ALLOC(rebinKeys_sorted,numBroadPhaseRebinPrims,s,memResource);
        _ALLOC(rebinPrimIDs_sorted,numBroadPhaseRebinPrims,s,memResource);
        keyValueSort((uint64_t*)rebinKeys_unsorted,
                     (uint64_t*)rebinKeys_sorted,
                     rebinPrimIDs_unsorted,
                     rebinPrimIDs_sorted,
                     numBroadPhaseRebinPrims,s,memResource);
        _FREE(rebinKeys_unsorted,s,memResource);
        _FREE(rebinPrimIDs_unsorted,s,memResource);
          
        // 3.10 write back the new sorted keys - each rebin prim will
        // first find the range it belongs to, from that get the local
        // range's key and primid, and write those back to the
        // corresponding global location
        rebinWriteBackNewKeysAndPrimIDs
          <<<divRoundUp(numBroadPhaseRebinPrims,1024),1024,0,s>>>
          (d_buildState,nodes,d_primKeys_sorted,d_primIDs_inMortonOrder,
           // jobs:
           rebinRanges_sorted,
           // prims:
           rebinKeys_sorted,rebinPrimIDs_sorted,numBroadPhaseRebinPrims);
        
        // free temp memories
        _FREE(rebinKeys_sorted,s,memResource);
        _FREE(rebinPrimIDs_sorted,s,memResource);
        _FREE(rebinRanges_sorted,s,memResource);

        // re-set broad phase counters for next iteration (in case any
        // of our restarted broad-phase subtrees will trigger any
        // additional rebinning later on)
        h_buildState->numBroadPhaseRebinJobs = 0;
        h_buildState->numBroadPhaseRebinPrims = 0;
        CUBQL_CUDA_CALL(MemcpyAsync(d_buildState,h_buildState,sizeof(*h_buildState),
                                    cudaMemcpyHostToDevice,s));

        // and re-start the block creation process by resetting 'numDone' to 0
        numNodesDone = 0;
      }

      // ##################################################################
      // END OF BROAD PHASE
      //
      // at this point, no node should be in broad phase any more; any
      // of the existing allocated nodes can be open and in block
      // phase or thread phase, but none should be in broad phase any
      // more.
      // ##################################################################

#if NO_BLOCK_PHASE
      // let's not do any - we directly go from broad to thread phase
#else
      TODO;
#endif
      // ##################################################################
      // END OF BLOCK PHASE
      //
      // at this point, no node should be in either broad or block
      // phase any more; any of the existing allocated nodes can be
      // open but none should be in broad phase any more.
      // ##################################################################
      numNodesDone = 0;
      while (numNodesDone < numNodesAlloced) {
        int numNodesStillToDo = numNodesAlloced - numNodesDone;
        threadPhaseCreateNodes
          <<<divRoundUp(numNodesStillToDo,1024),1024,0,s>>>
          (d_buildState,boxes,makeLeafThreshold,buildConfig.maxAllowedLeafSize,
           nodes,numNodesDone,numNodesAlloced,
           d_primKeys_sorted,d_primIDs_inMortonOrder);
        CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,sizeof(*h_buildState),
                                    cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
        CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));
        numNodesDone    = numNodesAlloced;
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
    void rebinRadixBuilder(BinaryBVH<T,D>    &bvh,
                           const box_t<T,D>  *boxes,
                           uint32_t           numPrims,
                           BuildConfig        buildConfig,
                           cudaStream_t       s,
                           GpuMemoryResource &memResource)
    {
      rebinRadixBuilder_impl::build<T,D>
        (bvh,boxes,numPrims,buildConfig,s,memResource);
    }
    
  } // ::cuBQL::cuda
} // ::cuBQL
#endif

