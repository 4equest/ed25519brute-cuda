#pragma once
#include <cstdint>
#include "sha512.cuh"
#include "config.h"

// PTX inline assembly for efficient 32x32->64 multiply-add
// mad.wide.s32 computes: d = a*b + c (64-bit result from 32-bit inputs)
__device__ __forceinline__ int64_t mad_wide_s32(int32_t a, int32_t b, int64_t c) {
    int64_t result;
    asm("mad.wide.s32 %0, %1, %2, %3;" : "=l"(result) : "r"(a), "r"(b), "l"(c));
    return result;
}

// Simple multiply: mul.wide.s32
__device__ __forceinline__ int64_t mul_wide_s32(int32_t a, int32_t b) {
    int64_t result;
    asm("mul.wide.s32 %0, %1, %2;" : "=l"(result) : "r"(a), "r"(b));
    return result;
}

// PTX optimized carry propagation for field element reduction
// Equivalent to: carry = (curr + bias) >> SHIFT; next += carry * MUL; curr -= carry << SHIFT
template <int SHIFT, int MUL = 1>
__device__ __forceinline__ void carry_pass_ptx(int64_t& curr, int64_t& next) {
    uint32_t c_lo = (uint32_t)curr;
    uint32_t c_hi = (uint32_t)(curr >> 32);
    uint32_t t_lo, t_hi;
    
    constexpr uint32_t BIAS = (1U << (SHIFT - 1));
    constexpr uint32_t MASK = (1U << SHIFT) - 1;
    
    // t = curr + bias (64-bit add with carry chain)
    asm("add.cc.u32 %0, %1, %2;" : "=r"(t_lo) : "r"(c_lo), "r"(BIAS));
    asm("addc.s32 %0, %1, 0;" : "=r"(t_hi) : "r"(c_hi));
    
    // carry = t >> SHIFT (using funnel shift for efficiency)
    uint32_t carry_lo, carry_hi;
    asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(carry_lo) : "r"(t_lo), "r"(t_hi), "r"(SHIFT));
    asm("shr.s32 %0, %1, %2;" : "=r"(carry_hi) : "r"(t_hi), "r"(SHIFT));
    
    // next += carry * MUL
    uint32_t n_lo = (uint32_t)next;
    uint32_t n_hi = (uint32_t)(next >> 32);
    
    if constexpr (MUL == 1) {
        asm("add.cc.u32 %0, %1, %2;" : "=r"(n_lo) : "r"(n_lo), "r"(carry_lo));
        asm("addc.s32 %0, %1, %2;" : "=r"(n_hi) : "r"(n_hi), "r"(carry_hi));
    } else {
        // Multiply carry by MUL (e.g., 19) and add to next
        uint32_t r_lo, r_hi;
        asm("mad.lo.cc.u32 %0, %1, %2, 0;" : "=r"(r_lo) : "r"(carry_lo), "r"(MUL));
        asm("madc.hi.u32 %0, %1, %2, 0;" : "=r"(r_hi) : "r"(carry_lo), "r"(MUL));
        asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r_hi) : "r"(carry_hi), "r"(MUL), "r"(r_hi));
        asm("add.cc.u32 %0, %1, %2;" : "=r"(n_lo) : "r"(n_lo), "r"(r_lo));
        asm("addc.s32 %0, %1, %2;" : "=r"(n_hi) : "r"(n_hi), "r"(r_hi));
    }
    next = ((int64_t)(int32_t)n_hi << 32) | n_lo;
    
    // curr = (t & mask) - bias (result fits in ~SHIFT bits, sign extend to 64-bit)
    uint32_t new_c_lo = (t_lo & MASK) - BIAS;
    curr = (int64_t)(int32_t)new_c_lo;
}

// Field element: 10 limbs of ~25.5 bits each
typedef int32_t fe[10];

// Group element representations
typedef struct { fe X, Y, Z, T; } ge_p3;
typedef struct { fe X, Y, Z; } ge_p2;
typedef struct { fe X, Y, Z, T; } ge_p1p1;
typedef struct { fe yplusx, yminusx, xy2d; int32_t pad; } ge_precomp;

// Precomputed base point tables for scalar multiplication
#include "precomp_5bit.h"
#include "precomp_7bit.h"
#include "precomp_8bit.h"
extern __device__ ge_precomp d_base_5bit[52][16]; 
extern __device__ ge_precomp d_base_7bit[37][64]; 
extern __device__ ge_precomp d_base_8bit[32][128]; // 8-bit window table


// Field operations
__device__ __forceinline__ void fe_0(fe h) {
    for (int i = 0; i < 10; i++) h[i] = 0;
}

__device__ __forceinline__ void fe_1(fe h) {
    h[0] = 1;
    for (int i = 1; i < 10; i++) h[i] = 0;
}

__device__ __forceinline__ void fe_copy(fe h, const fe f) {
    for (int i = 0; i < 10; i++) h[i] = f[i];
}

__device__ __forceinline__ void fe_add(fe h, const fe f, const fe g) {
    for (int i = 0; i < 10; i++) h[i] = f[i] + g[i];
}

__device__ __forceinline__ void fe_sub(fe h, const fe f, const fe g) {
    for (int i = 0; i < 10; i++) h[i] = f[i] - g[i];
}

__device__ __forceinline__ void fe_neg(fe h, const fe f) {
    for (int i = 0; i < 10; i++) h[i] = -f[i];
}

__device__ void fe_cmov(fe f, const fe g, unsigned int b) {
    b = (unsigned int)(-(int)b);
    for (int i = 0; i < 10; i++) {
        int32_t x = f[i] ^ g[i];
        x &= b;
        f[i] ^= x;
    }
}

__device__ __forceinline__ void fe_mul(fe h, const fe f, const fe g) {
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int32_t g0 = g[0], g1 = g[1], g2 = g[2], g3 = g[3], g4 = g[4];
    int32_t g5 = g[5], g6 = g[6], g7 = g[7], g8 = g[8], g9 = g[9];
    
    // Precompute multiplied values
    int32_t g1_19 = 19 * g1, g2_19 = 19 * g2, g3_19 = 19 * g3, g4_19 = 19 * g4;
    int32_t g5_19 = 19 * g5, g6_19 = 19 * g6, g7_19 = 19 * g7, g8_19 = 19 * g8, g9_19 = 19 * g9;
    int32_t f1_2 = 2 * f1, f3_2 = 2 * f3, f5_2 = 2 * f5, f7_2 = 2 * f7, f9_2 = 2 * f9;
    
    // Use PTX mad.wide.s32 for chained multiply-add operations
    // h0 = f0*g0 + f1_2*g9_19 + f2*g8_19 + f3_2*g7_19 + f4*g6_19 + f5_2*g5_19 + f6*g4_19 + f7_2*g3_19 + f8*g2_19 + f9_2*g1_19
    int64_t h0 = mul_wide_s32(f0, g0);
    h0 = mad_wide_s32(f1_2, g9_19, h0);
    h0 = mad_wide_s32(f2, g8_19, h0);
    h0 = mad_wide_s32(f3_2, g7_19, h0);
    h0 = mad_wide_s32(f4, g6_19, h0);
    h0 = mad_wide_s32(f5_2, g5_19, h0);
    h0 = mad_wide_s32(f6, g4_19, h0);
    h0 = mad_wide_s32(f7_2, g3_19, h0);
    h0 = mad_wide_s32(f8, g2_19, h0);
    h0 = mad_wide_s32(f9_2, g1_19, h0);
    
    int64_t h1 = mul_wide_s32(f0, g1);
    h1 = mad_wide_s32(f1, g0, h1);
    h1 = mad_wide_s32(f2, g9_19, h1);
    h1 = mad_wide_s32(f3, g8_19, h1);
    h1 = mad_wide_s32(f4, g7_19, h1);
    h1 = mad_wide_s32(f5, g6_19, h1);
    h1 = mad_wide_s32(f6, g5_19, h1);
    h1 = mad_wide_s32(f7, g4_19, h1);
    h1 = mad_wide_s32(f8, g3_19, h1);
    h1 = mad_wide_s32(f9, g2_19, h1);
    
    int64_t h2 = mul_wide_s32(f0, g2);
    h2 = mad_wide_s32(f1_2, g1, h2);
    h2 = mad_wide_s32(f2, g0, h2);
    h2 = mad_wide_s32(f3_2, g9_19, h2);
    h2 = mad_wide_s32(f4, g8_19, h2);
    h2 = mad_wide_s32(f5_2, g7_19, h2);
    h2 = mad_wide_s32(f6, g6_19, h2);
    h2 = mad_wide_s32(f7_2, g5_19, h2);
    h2 = mad_wide_s32(f8, g4_19, h2);
    h2 = mad_wide_s32(f9_2, g3_19, h2);
    
    int64_t h3 = mul_wide_s32(f0, g3);
    h3 = mad_wide_s32(f1, g2, h3);
    h3 = mad_wide_s32(f2, g1, h3);
    h3 = mad_wide_s32(f3, g0, h3);
    h3 = mad_wide_s32(f4, g9_19, h3);
    h3 = mad_wide_s32(f5, g8_19, h3);
    h3 = mad_wide_s32(f6, g7_19, h3);
    h3 = mad_wide_s32(f7, g6_19, h3);
    h3 = mad_wide_s32(f8, g5_19, h3);
    h3 = mad_wide_s32(f9, g4_19, h3);
    
    int64_t h4 = mul_wide_s32(f0, g4);
    h4 = mad_wide_s32(f1_2, g3, h4);
    h4 = mad_wide_s32(f2, g2, h4);
    h4 = mad_wide_s32(f3_2, g1, h4);
    h4 = mad_wide_s32(f4, g0, h4);
    h4 = mad_wide_s32(f5_2, g9_19, h4);
    h4 = mad_wide_s32(f6, g8_19, h4);
    h4 = mad_wide_s32(f7_2, g7_19, h4);
    h4 = mad_wide_s32(f8, g6_19, h4);
    h4 = mad_wide_s32(f9_2, g5_19, h4);
    
    int64_t h5 = mul_wide_s32(f0, g5);
    h5 = mad_wide_s32(f1, g4, h5);
    h5 = mad_wide_s32(f2, g3, h5);
    h5 = mad_wide_s32(f3, g2, h5);
    h5 = mad_wide_s32(f4, g1, h5);
    h5 = mad_wide_s32(f5, g0, h5);
    h5 = mad_wide_s32(f6, g9_19, h5);
    h5 = mad_wide_s32(f7, g8_19, h5);
    h5 = mad_wide_s32(f8, g7_19, h5);
    h5 = mad_wide_s32(f9, g6_19, h5);
    
    int64_t h6 = mul_wide_s32(f0, g6);
    h6 = mad_wide_s32(f1_2, g5, h6);
    h6 = mad_wide_s32(f2, g4, h6);
    h6 = mad_wide_s32(f3_2, g3, h6);
    h6 = mad_wide_s32(f4, g2, h6);
    h6 = mad_wide_s32(f5_2, g1, h6);
    h6 = mad_wide_s32(f6, g0, h6);
    h6 = mad_wide_s32(f7_2, g9_19, h6);
    h6 = mad_wide_s32(f8, g8_19, h6);
    h6 = mad_wide_s32(f9_2, g7_19, h6);
    
    int64_t h7 = mul_wide_s32(f0, g7);
    h7 = mad_wide_s32(f1, g6, h7);
    h7 = mad_wide_s32(f2, g5, h7);
    h7 = mad_wide_s32(f3, g4, h7);
    h7 = mad_wide_s32(f4, g3, h7);
    h7 = mad_wide_s32(f5, g2, h7);
    h7 = mad_wide_s32(f6, g1, h7);
    h7 = mad_wide_s32(f7, g0, h7);
    h7 = mad_wide_s32(f8, g9_19, h7);
    h7 = mad_wide_s32(f9, g8_19, h7);
    
    int64_t h8 = mul_wide_s32(f0, g8);
    h8 = mad_wide_s32(f1_2, g7, h8);
    h8 = mad_wide_s32(f2, g6, h8);
    h8 = mad_wide_s32(f3_2, g5, h8);
    h8 = mad_wide_s32(f4, g4, h8);
    h8 = mad_wide_s32(f5_2, g3, h8);
    h8 = mad_wide_s32(f6, g2, h8);
    h8 = mad_wide_s32(f7_2, g1, h8);
    h8 = mad_wide_s32(f8, g0, h8);
    h8 = mad_wide_s32(f9_2, g9_19, h8);
    
    int64_t h9 = mul_wide_s32(f0, g9);
    h9 = mad_wide_s32(f1, g8, h9);
    h9 = mad_wide_s32(f2, g7, h9);
    h9 = mad_wide_s32(f3, g6, h9);
    h9 = mad_wide_s32(f4, g5, h9);
    h9 = mad_wide_s32(f5, g4, h9);
    h9 = mad_wide_s32(f6, g3, h9);
    h9 = mad_wide_s32(f7, g2, h9);
    h9 = mad_wide_s32(f8, g1, h9);
    h9 = mad_wide_s32(f9, g0, h9);
    
    // Carry propagation using PTX optimized functions
    carry_pass_ptx<26>(h0, h1);
    carry_pass_ptx<26>(h4, h5);
    carry_pass_ptx<25>(h1, h2);
    carry_pass_ptx<25>(h5, h6);
    carry_pass_ptx<26>(h2, h3);
    carry_pass_ptx<26>(h6, h7);
    carry_pass_ptx<25>(h3, h4);
    carry_pass_ptx<25>(h7, h8);
    carry_pass_ptx<26>(h4, h5);
    carry_pass_ptx<26>(h8, h9);
    carry_pass_ptx<25, 19>(h9, h0);  // Multiply carry by 19 for mod p
    carry_pass_ptx<26>(h0, h1);
    
    h[0] = (int32_t)h0; h[1] = (int32_t)h1; h[2] = (int32_t)h2; h[3] = (int32_t)h3; h[4] = (int32_t)h4;
    h[5] = (int32_t)h5; h[6] = (int32_t)h6; h[7] = (int32_t)h7; h[8] = (int32_t)h8; h[9] = (int32_t)h9;
}

__device__ __forceinline__ void fe_sq(fe h, const fe f) {
    int32_t f0 = f[0], f1 = f[1], f2 = f[2], f3 = f[3], f4 = f[4];
    int32_t f5 = f[5], f6 = f[6], f7 = f[7], f8 = f[8], f9 = f[9];
    int32_t f0_2 = 2*f0, f1_2 = 2*f1, f2_2 = 2*f2, f3_2 = 2*f3, f4_2 = 2*f4;
    int32_t f5_2 = 2*f5, f6_2 = 2*f6, f7_2 = 2*f7;
    int32_t f5_38 = 38*f5, f6_19 = 19*f6, f7_38 = 38*f7, f8_19 = 19*f8, f9_38 = 38*f9;
    
    // h0 = f0*f0 + f1_2*f9_38 + f2_2*f8_19 + f3_2*f7_38 + f4_2*f6_19 + f5*f5_38
    int64_t h0 = mul_wide_s32(f0, f0);
    h0 = mad_wide_s32(f1_2, f9_38, h0);
    h0 = mad_wide_s32(f2_2, f8_19, h0);
    h0 = mad_wide_s32(f3_2, f7_38, h0);
    h0 = mad_wide_s32(f4_2, f6_19, h0);
    h0 = mad_wide_s32(f5, f5_38, h0);
    
    // h1 = f0_2*f1 + f2*f9_38 + f3_2*f8_19 + f4*f7_38 + f5_2*f6_19
    int64_t h1 = mul_wide_s32(f0_2, f1);
    h1 = mad_wide_s32(f2, f9_38, h1);
    h1 = mad_wide_s32(f3_2, f8_19, h1);
    h1 = mad_wide_s32(f4, f7_38, h1);
    h1 = mad_wide_s32(f5_2, f6_19, h1);
    
    // h2 = f0_2*f2 + f1_2*f1 + f3_2*f9_38 + f4_2*f8_19 + f5_2*f7_38 + f6*f6_19
    int64_t h2 = mul_wide_s32(f0_2, f2);
    h2 = mad_wide_s32(f1_2, f1, h2);
    h2 = mad_wide_s32(f3_2, f9_38, h2);
    h2 = mad_wide_s32(f4_2, f8_19, h2);
    h2 = mad_wide_s32(f5_2, f7_38, h2);
    h2 = mad_wide_s32(f6, f6_19, h2);
    
    // h3 = f0_2*f3 + f1_2*f2 + f4*f9_38 + f5_2*f8_19 + f6*f7_38
    int64_t h3 = mul_wide_s32(f0_2, f3);
    h3 = mad_wide_s32(f1_2, f2, h3);
    h3 = mad_wide_s32(f4, f9_38, h3);
    h3 = mad_wide_s32(f5_2, f8_19, h3);
    h3 = mad_wide_s32(f6, f7_38, h3);
    
    // h4 = f0_2*f4 + f1_2*f3_2 + f2*f2 + f5_2*f9_38 + f6_2*f8_19 + f7*f7_38
    int64_t h4 = mul_wide_s32(f0_2, f4);
    h4 = mad_wide_s32(f1_2, f3_2, h4);
    h4 = mad_wide_s32(f2, f2, h4);
    h4 = mad_wide_s32(f5_2, f9_38, h4);
    h4 = mad_wide_s32(f6_2, f8_19, h4);
    h4 = mad_wide_s32(f7, f7_38, h4);
    
    // h5 = f0_2*f5 + f1_2*f4 + f2_2*f3 + f6*f9_38 + f7_2*f8_19
    int64_t h5 = mul_wide_s32(f0_2, f5);
    h5 = mad_wide_s32(f1_2, f4, h5);
    h5 = mad_wide_s32(f2_2, f3, h5);
    h5 = mad_wide_s32(f6, f9_38, h5);
    h5 = mad_wide_s32(f7_2, f8_19, h5);
    
    // h6 = f0_2*f6 + f1_2*f5_2 + f2_2*f4 + f3_2*f3 + f7_2*f9_38 + f8*f8_19
    int64_t h6 = mul_wide_s32(f0_2, f6);
    h6 = mad_wide_s32(f1_2, f5_2, h6);
    h6 = mad_wide_s32(f2_2, f4, h6);
    h6 = mad_wide_s32(f3_2, f3, h6);
    h6 = mad_wide_s32(f7_2, f9_38, h6);
    h6 = mad_wide_s32(f8, f8_19, h6);
    
    // h7 = f0_2*f7 + f1_2*f6 + f2_2*f5 + f3_2*f4 + f8*f9_38
    int64_t h7 = mul_wide_s32(f0_2, f7);
    h7 = mad_wide_s32(f1_2, f6, h7);
    h7 = mad_wide_s32(f2_2, f5, h7);
    h7 = mad_wide_s32(f3_2, f4, h7);
    h7 = mad_wide_s32(f8, f9_38, h7);
    
    // h8 = f0_2*f8 + f1_2*f7_2 + f2_2*f6 + f3_2*f5_2 + f4*f4 + f9*f9_38
    int64_t h8 = mul_wide_s32(f0_2, f8);
    h8 = mad_wide_s32(f1_2, f7_2, h8);
    h8 = mad_wide_s32(f2_2, f6, h8);
    h8 = mad_wide_s32(f3_2, f5_2, h8);
    h8 = mad_wide_s32(f4, f4, h8);
    h8 = mad_wide_s32(f9, f9_38, h8);
    
    // h9 = f0_2*f9 + f1_2*f8 + f2_2*f7 + f3_2*f6 + f4_2*f5
    int64_t h9 = mul_wide_s32(f0_2, f9);
    h9 = mad_wide_s32(f1_2, f8, h9);
    h9 = mad_wide_s32(f2_2, f7, h9);
    h9 = mad_wide_s32(f3_2, f6, h9);
    h9 = mad_wide_s32(f4_2, f5, h9);
    
    // Carry propagation using PTX optimized functions
    carry_pass_ptx<26>(h0, h1);
    carry_pass_ptx<26>(h4, h5);
    carry_pass_ptx<25>(h1, h2);
    carry_pass_ptx<25>(h5, h6);
    carry_pass_ptx<26>(h2, h3);
    carry_pass_ptx<26>(h6, h7);
    carry_pass_ptx<25>(h3, h4);
    carry_pass_ptx<25>(h7, h8);
    carry_pass_ptx<26>(h4, h5);
    carry_pass_ptx<26>(h8, h9);
    carry_pass_ptx<25, 19>(h9, h0);  // Multiply carry by 19 for mod p
    carry_pass_ptx<26>(h0, h1);
    
    h[0] = (int32_t)h0; h[1] = (int32_t)h1; h[2] = (int32_t)h2; h[3] = (int32_t)h3; h[4] = (int32_t)h4;
    h[5] = (int32_t)h5; h[6] = (int32_t)h6; h[7] = (int32_t)h7; h[8] = (int32_t)h8; h[9] = (int32_t)h9;
}

__device__ void fe_sq2(fe h, const fe f) {
    fe_sq(h, f);
    fe_add(h, h, h);
}

__device__ void fe_invert(fe out, const fe z) {
    fe t0, t1, t2, t3;
    int i;
    fe_sq(t0, z);
    fe_sq(t1, t0); fe_sq(t1, t1);
    fe_mul(t1, z, t1);
    fe_mul(t0, t0, t1);
    fe_sq(t2, t0);
    fe_mul(t1, t1, t2);
    fe_sq(t2, t1);
    #pragma unroll
    for (i = 0; i < 4; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t2, t1);
    #pragma unroll
    for (i = 0; i < 9; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);
    fe_sq(t3, t2);
    #pragma unroll
    for (i = 0; i < 19; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);
    fe_sq(t2, t2);
    #pragma unroll
    for (i = 0; i < 9; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t2, t1);
    #pragma unroll
    for (i = 0; i < 49; i++) fe_sq(t2, t2);
    fe_mul(t2, t2, t1);
    fe_sq(t3, t2);
    #pragma unroll
    for (i = 0; i < 99; i++) fe_sq(t3, t3);
    fe_mul(t2, t3, t2);
    fe_sq(t2, t2);
    #pragma unroll
    for (i = 0; i < 49; i++) fe_sq(t2, t2);
    fe_mul(t1, t2, t1);
    fe_sq(t1, t1);
    #pragma unroll
    for (i = 0; i < 4; i++) fe_sq(t1, t1);
    fe_mul(out, t1, t0);
}

__device__ void fe_tobytes(unsigned char* s, const fe h) {
    int32_t h0 = h[0], h1 = h[1], h2 = h[2], h3 = h[3], h4 = h[4];
    int32_t h5 = h[5], h6 = h[6], h7 = h[7], h8 = h[8], h9 = h[9];
    int32_t q, carry;
    
    q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;
    
    h0 += 19 * q;
    
    carry = h0 >> 26; h1 += carry; h0 -= carry << 26;
    carry = h1 >> 25; h2 += carry; h1 -= carry << 25;
    carry = h2 >> 26; h3 += carry; h2 -= carry << 26;
    carry = h3 >> 25; h4 += carry; h3 -= carry << 25;
    carry = h4 >> 26; h5 += carry; h4 -= carry << 26;
    carry = h5 >> 25; h6 += carry; h5 -= carry << 25;
    carry = h6 >> 26; h7 += carry; h6 -= carry << 26;
    carry = h7 >> 25; h8 += carry; h7 -= carry << 25;
    carry = h8 >> 26; h9 += carry; h8 -= carry << 26;
    carry = h9 >> 25;              h9 -= carry << 25;
    
    s[0] = (unsigned char)(h0);
    s[1] = (unsigned char)(h0 >> 8);
    s[2] = (unsigned char)(h0 >> 16);
    s[3] = (unsigned char)((h0 >> 24) | (h1 << 2));
    s[4] = (unsigned char)(h1 >> 6);
    s[5] = (unsigned char)(h1 >> 14);
    s[6] = (unsigned char)((h1 >> 22) | (h2 << 3));
    s[7] = (unsigned char)(h2 >> 5);
    s[8] = (unsigned char)(h2 >> 13);
    s[9] = (unsigned char)((h2 >> 21) | (h3 << 5));
    s[10] = (unsigned char)(h3 >> 3);
    s[11] = (unsigned char)(h3 >> 11);
    s[12] = (unsigned char)((h3 >> 19) | (h4 << 6));
    s[13] = (unsigned char)(h4 >> 2);
    s[14] = (unsigned char)(h4 >> 10);
    s[15] = (unsigned char)(h4 >> 18);
    s[16] = (unsigned char)(h5);
    s[17] = (unsigned char)(h5 >> 8);
    s[18] = (unsigned char)(h5 >> 16);
    s[19] = (unsigned char)((h5 >> 24) | (h6 << 1));
    s[20] = (unsigned char)(h6 >> 7);
    s[21] = (unsigned char)(h6 >> 15);
    s[22] = (unsigned char)((h6 >> 23) | (h7 << 3));
    s[23] = (unsigned char)(h7 >> 5);
    s[24] = (unsigned char)(h7 >> 13);
    s[25] = (unsigned char)((h7 >> 21) | (h8 << 4));
    s[26] = (unsigned char)(h8 >> 4);
    s[27] = (unsigned char)(h8 >> 12);
    s[28] = (unsigned char)((h8 >> 20) | (h9 << 6));
    s[29] = (unsigned char)(h9 >> 2);
    s[30] = (unsigned char)(h9 >> 10);
    s[31] = (unsigned char)(h9 >> 18);
}

__device__ int fe_isnegative(const fe f) {
    unsigned char s[32];
    fe_tobytes(s, f);
    return s[0] & 1;
}

// Group operations
__device__ void ge_p3_0(ge_p3* h) {
    fe_0(h->X); fe_1(h->Y); fe_1(h->Z); fe_0(h->T);
}

__device__ __forceinline__ void ge_p3_to_p2(ge_p2* r, const ge_p3* p) {
    fe_copy(r->X, p->X);
    fe_copy(r->Y, p->Y);
    fe_copy(r->Z, p->Z);
}

__device__ __forceinline__ void ge_p1p1_to_p3(ge_p3* r, const ge_p1p1* p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
    fe_mul(r->T, p->X, p->Y);
}

__device__ void ge_p1p1_to_p2(ge_p2* r, const ge_p1p1* p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
}

__device__ __forceinline__ void ge_p2_dbl(ge_p1p1* r, const ge_p2* p) {
    fe t0;
    fe_sq(r->X, p->X);
    fe_sq(r->Z, p->Y);
    fe_sq2(r->T, p->Z);
    fe_add(r->Y, p->X, p->Y);
    fe_sq(t0, r->Y);
    fe_add(r->Y, r->Z, r->X);
    fe_sub(r->Z, r->Z, r->X);
    fe_sub(r->X, t0, r->Y);
    fe_sub(r->T, r->T, r->Z);
}

__device__ __forceinline__ void ge_p3_dbl(ge_p1p1* r, const ge_p3* p) {
    ge_p2 q;
    ge_p3_to_p2(&q, p);
    ge_p2_dbl(r, &q);
}

// r = p + q (mixed addition)
__device__ __forceinline__ void ge_madd(ge_p1p1* r, const ge_p3* p, const ge_precomp* q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yplusx);
    fe_mul(r->Y, r->Y, q->yminusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}

// Helper functions for ge_scalarmult_base
__device__ static unsigned char equal(signed char b, signed char c) {
    unsigned char x = (unsigned char)(b ^ c);
    uint64_t y = x;
    y -= 1;
    y >>= 63;
    return (unsigned char)y;
}

__device__ static unsigned char negative(signed char b) {
    uint64_t x = (uint64_t)(int64_t)b;
    x >>= 63;
    return (unsigned char)x;
}

__device__ static void cmov_precomp(ge_precomp* t, const ge_precomp* u, unsigned char b) {
    fe_cmov(t->yplusx, u->yplusx, b);
    fe_cmov(t->yminusx, u->yminusx, b);
    fe_cmov(t->xy2d, u->xy2d, b);
}

__device__ static void select_base_shared(ge_precomp* t, const ge_precomp* table, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        *t = table[babs - 1];
    }
    
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

__device__ static void select_5bit(ge_precomp* t, const ge_precomp* table, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        // Return neutral element for precomp
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        // table[i] = (i+1) * P, so table[babs-1] = babs * P
        *t = table[babs - 1];
    }
    
    // Negate if b was negative
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

__device__ void ge_scalarmult_base_5bit_simple(ge_p3* h, const unsigned char* a) {
    signed char e[52];
    int i;
    
    ge_p1p1 r;
    ge_precomp t;
    
    // === Step 1: Decompose scalar into 5-bit signed digits ===
    // 52 windows: windows 0-50 have 5 bits each, window 51 has remaining bits
    int carry = 0;
    
    for (i = 0; i < 51; i++) {
        int bit_pos = i * 5;
        int byte_idx = bit_pos >> 3;
        int bit_idx = bit_pos & 7;
        
        // Extract 5 raw bits (may span 2 bytes)
        int raw = (a[byte_idx] >> bit_idx);
        if (byte_idx + 1 < 32) {
            raw |= ((int)a[byte_idx + 1] << (8 - bit_idx));
        }
        raw = (raw & 31) + carry;  // Extract 5 bits, then add carry
        
        // Now raw is in [0, 32]
        // Convert to signed digit: if raw >= 16, subtract 32 and carry 1
        carry = (raw >= 16) ? 1 : 0;
        e[i] = (signed char)(raw - (carry << 5));  // raw - 32 if carry
    }
    
    // Window 51: remaining bit(s) plus carry
    // After clamping, bit 255 is 0 (h[31] &= 127), so just use carry
    e[51] = (signed char)carry;
    
    // === Step 2: Initialize h to neutral element ===
    ge_p3_0(h);
    
    // === Step 3: Process all 52 windows ===
    // h = sum_{i=0}^{51} e[i] * 2^(5*i) * B
    // where d_base_5bit[i][j] = (j+1) * 2^(5*i) * B
    
    for (i = 0; i < 52; i++) {
        select_5bit(&t, d_base_5bit[i], e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

__device__ static void select_5bit_shared(ge_precomp* t, const ge_precomp* table, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        *t = table[babs - 1];
    }
    
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

// 5-bit window with shared memory double buffering
// Requires kernel launch with >= 256 threads
__device__ void ge_scalarmult_base_5bit_shared(ge_p3* h, const unsigned char* a) {
    signed char e[52];
    int i;
    int tid = threadIdx.x;
    
    ge_p1p1 r;
    ge_precomp t;
    
    // === Step 1: Decompose scalar into 5-bit signed digits ===
    int carry = 0;
    
    for (i = 0; i < 51; i++) {
        int bit_pos = i * 5;
        int byte_idx = bit_pos >> 3;
        int bit_idx = bit_pos & 7;
        
        int raw = (a[byte_idx] >> bit_idx);
        if (byte_idx + 1 < 32) {
            raw |= ((int)a[byte_idx + 1] << (8 - bit_idx));
        }
        raw = (raw & 31) + carry;
        
        carry = (raw >= 16) ? 1 : 0;
        e[i] = (signed char)(raw - (carry << 5));
    }
    e[51] = (signed char)carry;
    
    // === Step 2: Initialize h ===
    ge_p3_0(h);
    
    // === Step 3: Process with shared memory double buffering ===
    // Each ge_precomp group has 16 entries, each entry is 31 ints = 496 ints total
    __shared__ int32_t s_cache[2][496];
    ge_precomp* cache_0 = (ge_precomp*)s_cache[0];
    ge_precomp* cache_1 = (ge_precomp*)s_cache[1];
    
    // Prologue: Load group 0
    {
        const int32_t* src = (const int32_t*)d_base_5bit[0];
        if (tid < 496) s_cache[0][tid] = src[tid];
        if (tid + 256 < 496) s_cache[0][tid + 256] = src[tid + 256];
    }
    __syncthreads();
    
    // Main loop with double buffering (groups 0-50)
    #pragma unroll 1
    for (i = 0; i < 51; i++) {
        int curr = i & 1;
        int next = 1 - curr;
        ge_precomp* current_cache = (curr == 0) ? cache_0 : cache_1;
        
        // Prefetch next group
        const int32_t* next_src = (const int32_t*)d_base_5bit[i + 1];
        if (tid < 496) s_cache[next][tid] = next_src[tid];
        if (tid + 256 < 496) s_cache[next][tid + 256] = next_src[tid + 256];
        
        // Compute using current group
        select_5bit_shared(&t, current_cache, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
        
        __syncthreads();
    }
    
    // Epilogue: Process group 51
    {
        ge_precomp* current_cache = (51 & 1) ? cache_1 : cache_0;
        select_5bit_shared(&t, current_cache, e[51]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

// h = a * B where B is the base point
// Using 5-bit window optimization (simple version is faster)
// Select from 7-bit precomputed table
__device__ static __forceinline__ void select_7bit_shared(ge_precomp* t, const ge_precomp* table, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        *t = table[babs - 1];
    }
    
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

// 7-bit window with shared memory double buffering
__device__ __forceinline__ void ge_scalarmult_base_7bit_shared(ge_p3* h, const unsigned char* a) {
    signed char e[37];
    int i;
    int tid = threadIdx.x;
    
    ge_p1p1 r;
    ge_precomp t;
    
    // === Step 1: Decompose scalar into 7-bit signed digits ===
    int carry = 0;
    
    for (i = 0; i < 36; i++) {
        int bit_pos = i * 7;
        int byte_idx = bit_pos >> 3;
        int bit_idx = bit_pos & 7;
        
        int raw = (a[byte_idx] >> bit_idx);
        if (byte_idx + 1 < 32) {
            raw |= ((int)a[byte_idx + 1] << (8 - bit_idx));
        }
        
        raw = (raw & 127) + carry;
        carry = (raw + 64) >> 7;
        e[i] = (signed char)(raw - (carry << 7));
    }
    e[36] = (signed char)((a[31] >> 4) + carry);
    
    // === Step 2: Initialize h ===
    ge_p3_0(h);
    
    // === Step 3: Double Buffering Loop ===
    // 37 steps. Each step 64 entries. Entry=31 ints. Total=1984 ints.
    __shared__ int32_t s_cache[2][1984];
    ge_precomp* cache_0 = (ge_precomp*)s_cache[0];
    ge_precomp* cache_1 = (ge_precomp*)s_cache[1];
    
    // Prologue: Load group 0
    {
        const int32_t* src = (const int32_t*)d_base_7bit[0];
        #pragma unroll
        for (int k = 0; k < 1984; k += 256) {
            if (tid + k < 1984) s_cache[0][tid + k] = src[tid + k];
        }
    }
    __syncthreads();
    
    // Main loop (groups 0-35)
    #pragma unroll 1
    for (i = 0; i < 36; i++) {
        int curr = i & 1;
        int next = 1 - curr;
        ge_precomp* current_cache = (curr == 0) ? cache_0 : cache_1;
        
        // Prefetch next group
        const int32_t* next_src = (const int32_t*)d_base_7bit[i + 1];
        #pragma unroll
        for (int k = 0; k < 1984; k += 256) {
            if (tid + k < 1984) s_cache[next][tid + k] = next_src[tid + k];
        }
        
        select_7bit_shared(&t, current_cache, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
        
        __syncthreads();
    }
    
    // Epilogue: Process group 36
    {
        ge_precomp* current_cache = (36 & 1) ? cache_1 : cache_0;
        select_7bit_shared(&t, current_cache, e[36]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

// Select from 8-bit precomputed table
__device__ static __forceinline__ void select_8bit_shared(ge_precomp* t, const ge_precomp* table, signed char b) {
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        *t = table[babs - 1];
    }
    
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

// Select from 8-bit precomputed table - LAST WINDOW SPECIAL CASE
// Handles e[31] = -128 as +128
__device__ static __forceinline__ void select_8bit_shared_last_window(ge_precomp* t, const ge_precomp* table, signed char b) {
    if (b == -128) {
        // Special case: scalar overflowed 127 in last byte.
        // It means we wanted +128.
        // table[127] is the point for +128.
        *t = table[127];
        return;
    }
    
    // Normal case (same as standard select)
    ge_precomp minust;
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);
    
    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
    } else {
        *t = table[babs - 1];
    }
    
    if (bnegative) {
        fe_copy(minust.yplusx, t->yminusx);
        fe_copy(minust.yminusx, t->yplusx);
        fe_neg(minust.xy2d, t->xy2d);
        *t = minust;
    }
}

// 8-bit window with shared memory double buffering
__device__ __forceinline__ void ge_scalarmult_base_8bit_shared(ge_p3* h, const unsigned char* a) {
    signed char e[32];
    int i;
    int tid = threadIdx.x;
    
    ge_p1p1 r;
    ge_precomp t;
    
    // === Step 1: Decompose scalar into 8-bit signed digits ===
    // 32 windows of 8 bits each (total 256 bits)
    int carry = 0;
    for (i = 0; i < 32; i++) {
        int raw = a[i] + carry;
        raw = raw + carry;
    }
    
    // Let's rewrite the loop properly for 8-bit (aligned bytes)
    carry = 0;
    #pragma unroll
    for (i = 0; i < 32; i++) {
        int raw = a[i] + carry;
        carry = (raw + 128) >> 8;
        e[i] = (signed char)(raw - (carry << 8));
    }
    
    // === Step 2: Initialize h ===
    ge_p3_0(h);
    
    // === Step 3: Double Buffering Loop ===
    // 32 steps. Each step 128 entries. Entry=31 ints. Total=3968 ints.
    // Double buffer size = 2 * 3968 = 7936 ints.
    __shared__ int32_t s_cache[2][3968];
    ge_precomp* cache_0 = (ge_precomp*)s_cache[0];
    ge_precomp* cache_1 = (ge_precomp*)s_cache[1];
    
    // Prologue: Load group 0
    {
        const int32_t* src = (const int32_t*)d_base_8bit[0];
        #pragma unroll
        for (int k = 0; k < 3968; k += 256) {
            if (tid + k < 3968) s_cache[0][tid + k] = src[tid + k];
        }
    }
    __syncthreads();
    
    // Main loop (groups 0-31)
    // Note: Actually 32 steps. e[0]..e[31].
    // Loop i from 0 to 31.
    // Prefetch i+1. If i=31, no prefetch.
    
    #pragma unroll 1
    for (i = 0; i < 31; i++) {
        int curr = i & 1;
        int next = 1 - curr;
        ge_precomp* current_cache = (curr == 0) ? cache_0 : cache_1;
        
        // Prefetch next group (i+1)
        const int32_t* next_src = (const int32_t*)d_base_8bit[i + 1];
        #pragma unroll
        for (int k = 0; k < 3968; k += 256) {
            if (tid + k < 3968) s_cache[next][tid + k] = next_src[tid + k];
        }
        
        select_8bit_shared(&t, current_cache, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
        
        __syncthreads();
    }
    
    // Epilogue: Process group 31 (SPECIAL HANDLING)
    {
        ge_precomp* current_cache = (31 & 1) ? cache_1 : cache_0;
        // Use special function for last window
        select_8bit_shared_last_window(&t, current_cache, e[31]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

__device__ void ge_scalarmult_base(ge_p3* h, const unsigned char* a) {
    ge_scalarmult_base_8bit_shared(h, a);
}




// Convert point to bytes (public key)
__device__ void ge_p3_tobytes(unsigned char* s, const ge_p3* h) {
    fe recip, x, y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

// Main Ed25519 key generation: seed (32 bytes) -> public key (32 bytes)
__device__ void ed25519_pubkey_from_seed(const uint8_t* seed, uint8_t* pubkey) {
    uint8_t h[64];
    sha512_32bytes(seed, h);
    
    // Clamp the hash
    h[0] &= 248;
    h[31] &= 127;
    h[31] |= 64;
    
    // Scalar multiplication
    ge_p3 A;
    ge_scalarmult_base(&A, h);
    ge_p3_tobytes(pubkey, &A);
}

// Batch Public Key Generation with Montgomery's Trick (Batch Inversion)
// Generates 4 public keys with only 1 fe_invert instead of 4
#ifndef BATCH_SIZE
#define BATCH_SIZE 4
#endif

// Generic Batch Public Key Generation using Template
// Allows experimenting with different batch sizes easily
template <int BATCH_N>
__device__ void ed25519_pubkey_batch(
    const uint8_t seeds[BATCH_N][32], 
    uint8_t pubkeys[BATCH_N][32]
) {
    ge_p3 A[BATCH_N];
    fe Z[BATCH_N];       // Store Z coordinates
    fe products[BATCH_N]; // Cumulative products for batch inversion
    fe inv_products[BATCH_N]; // Inverse products
    
    // Step 1: Generate all scalar multiplications and store Z coordinates
    for (int i = 0; i < BATCH_N; i++) {
        uint8_t h[64];
        sha512_32bytes(seeds[i], h);
        
        // Clamp the hash
        h[0] &= 248;
        h[31] &= 127;
        h[31] |= 64;
        
        // Scalar multiplication
        ge_scalarmult_base(&A[i], h);
        fe_copy(Z[i], A[i].Z);
    }
    
    // Step 2: Compute cumulative products: products[i] = Z[0] * Z[1] * ... * Z[i]
    fe_copy(products[0], Z[0]);
    for (int i = 1; i < BATCH_N; i++) {
        fe_mul(products[i], products[i-1], Z[i]);
    }
    
    // Step 3: Compute single inversion of the final product
    fe final_inv;
    fe_invert(final_inv, products[BATCH_N - 1]);
    
    // Step 4: Compute individual inverses using Montgomery's Trick
    // inv_products[i] = 1/Z[i]
    // Working backwards: 1/(Z[0]*...*Z[i]) * (Z[0]*...*Z[i-1]) = 1/Z[i]
    for (int i = BATCH_N - 1; i >= 1; i--) {
        fe_mul(inv_products[i], final_inv, products[i-1]); // 1/Z[i]
        fe_mul(final_inv, final_inv, Z[i]); // Update: 1/(Z[0]*...*Z[i-1])
    }
    fe_copy(inv_products[0], final_inv); // 1/Z[0]
    
    // Step 5: Convert all points to bytes using the computed inverses
    for (int i = 0; i < BATCH_N; i++) {
        fe x, y;
        fe_mul(x, A[i].X, inv_products[i]);
        fe_mul(y, A[i].Y, inv_products[i]);
        fe_tobytes(pubkeys[i], y);
        pubkeys[i][31] ^= fe_isnegative(x) << 7;
    }
}

// Backward compatibility wrapper and specific instantiation
__device__ void ed25519_pubkey_batch4(
    const uint8_t seeds[4][32], 
    uint8_t pubkeys[4][32]
) {
    ed25519_pubkey_batch<4>(seeds, pubkeys);
}

// Full Ed25519 key pair generation: seed (32 bytes) -> private key (64 bytes) + public key (32 bytes)
// Private key format: seed (32 bytes) || public key (32 bytes)
__device__ void ed25519_create_keypair(const uint8_t* seed, uint8_t* private_key, uint8_t* pubkey) {
    // Generate public key
    ed25519_pubkey_from_seed(seed, pubkey);
    
    // Private key = seed || pubkey
    for (int i = 0; i < 32; i++) {
        private_key[i] = seed[i];
        private_key[32 + i] = pubkey[i];
    }
}

// Generate SSH wire format for Ed25519 public key
// Output: 51 bytes (header 19 bytes + pubkey 32 bytes)
__device__ void ed25519_ssh_wire_format(const uint8_t* pubkey, uint8_t* wire_format) {
    // SSH wire format header for ed25519: "00 00 00 0b ssh-ed25519 00 00 00 20"
    static const uint8_t header[19] = {
        0x00, 0x00, 0x00, 0x0b,  // length of "ssh-ed25519" = 11
        0x73, 0x73, 0x68, 0x2d, 0x65, 0x64, 0x32, 0x35, 0x35, 0x31, 0x39,  // "ssh-ed25519"
        0x00, 0x00, 0x00, 0x20   // length of public key = 32
    };
    
    for (int i = 0; i < 19; i++) {
        wire_format[i] = header[i];
    }
    for (int i = 0; i < 32; i++) {
        wire_format[19 + i] = pubkey[i];
    }
}
