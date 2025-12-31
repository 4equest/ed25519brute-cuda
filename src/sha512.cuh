#pragma once
#include <cstdint>

// SHA512 implementation for Ed25519 key generation
// Input: 32 bytes (seed), Output: 64 bytes (hash)

// SHA512 round constants
__constant__ uint64_t d_K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// Rotate right using PTX shf.r.clamp.b32 (funnel shift right)
// Template version: N is known at compile time, eliminating runtime branch
template <int N>
__device__ __forceinline__ uint64_t ROR64(uint64_t x) {
    uint32_t lo = (uint32_t)x;
    uint32_t hi = (uint32_t)(x >> 32);
    uint32_t r_lo, r_hi;
    
    if constexpr (N == 0) {
        return x;
    } else if constexpr (N < 32) {
        // shf.r.clamp.b32 d, a, b, c: d = (b:a >> c)[31:0]
        // For rotate right by N: new_lo = (hi:lo >> N), new_hi = (lo:hi >> N)
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r_lo) : "r"(lo), "r"(hi), "r"(N));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r_hi) : "r"(hi), "r"(lo), "r"(N));
    } else {
        // N >= 32: swap hi/lo, then rotate by (N - 32)
        constexpr int M = N - 32;
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r_lo) : "r"(hi), "r"(lo), "r"(M));
        asm("shf.r.clamp.b32 %0, %1, %2, %3;" : "=r"(r_hi) : "r"(lo), "r"(hi), "r"(M));
    }
    
    return ((uint64_t)r_hi << 32) | r_lo;
}

// SHA512 functions
__device__ __forceinline__ uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) {
    return z ^ (x & (y ^ z));
}

__device__ __forceinline__ uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) {
    return ((x | y) & z) | (x & y);
}

__device__ __forceinline__ uint64_t Sigma0(uint64_t x) {
    return ROR64<28>(x) ^ ROR64<34>(x) ^ ROR64<39>(x);
}

__device__ __forceinline__ uint64_t Sigma1(uint64_t x) {
    return ROR64<14>(x) ^ ROR64<18>(x) ^ ROR64<41>(x);
}

__device__ __forceinline__ uint64_t Gamma0(uint64_t x) {
    return ROR64<1>(x) ^ ROR64<8>(x) ^ (x >> 7);
}

__device__ __forceinline__ uint64_t Gamma1(uint64_t x) {
    return ROR64<19>(x) ^ ROR64<61>(x) ^ (x >> 6);
}

// Load 64-bit big-endian
__device__ __forceinline__ uint64_t LOAD64H(const uint8_t* y) {
    return ((uint64_t)y[0] << 56) | ((uint64_t)y[1] << 48) |
           ((uint64_t)y[2] << 40) | ((uint64_t)y[3] << 32) |
           ((uint64_t)y[4] << 24) | ((uint64_t)y[5] << 16) |
           ((uint64_t)y[6] << 8)  | ((uint64_t)y[7]);
}

// Store 64-bit big-endian
__device__ __forceinline__ void STORE64H(uint64_t x, uint8_t* y) {
    y[0] = (uint8_t)(x >> 56);
    y[1] = (uint8_t)(x >> 48);
    y[2] = (uint8_t)(x >> 40);
    y[3] = (uint8_t)(x >> 32);
    y[4] = (uint8_t)(x >> 24);
    y[5] = (uint8_t)(x >> 16);
    y[6] = (uint8_t)(x >> 8);
    y[7] = (uint8_t)(x);
}

// SHA512 for 32-byte input (Ed25519 seed)
// Output: 64 bytes
// Optimized: W[16] rolling buffer instead of W[80] to reduce register pressure
__device__ void sha512_32bytes(const uint8_t* input, uint8_t* output) {
    // Initial hash values
    uint64_t S[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    
    // Message schedule W[0..15] - rolling buffer
    uint64_t W[16];
    
    // Load input (32 bytes = 4 words)
    W[0] = LOAD64H(input);
    W[1] = LOAD64H(input + 8);
    W[2] = LOAD64H(input + 16);
    W[3] = LOAD64H(input + 24);
    
    // Padding: 0x80 at byte 32
    W[4] = 0x8000000000000000ULL;
    
    // Zero padding
    W[5] = 0; W[6] = 0; W[7] = 0;
    W[8] = 0; W[9] = 0; W[10] = 0;
    W[11] = 0; W[12] = 0; W[13] = 0;
    W[14] = 0;
    
    // Length in bits (32 * 8 = 256)
    W[15] = 256;
    
    // Compression function with integrated message schedule expansion
    uint64_t a = S[0], b = S[1], c = S[2], d = S[3];
    uint64_t e = S[4], f = S[5], g = S[6], hh = S[7];
    
    // First 16 rounds - use W directly
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint64_t t0 = hh + Sigma1(e) + Ch(e, f, g) + d_K512[i] + W[i];
        uint64_t t1 = Sigma0(a) + Maj(a, b, c);
        hh = g; g = f; f = e; e = d + t0;
        d = c; c = b; b = a; a = t0 + t1;
    }
    
    // Rounds 16-79: compute W on-the-fly using rolling buffer
    // W[i%16] = Gamma1(W[(i-2)%16]) + W[(i-7)%16] + Gamma0(W[(i-15)%16]) + W[(i-16)%16]
    // Which simplifies to:
    // W[i%16] = Gamma1(W[(i+14)%16]) + W[(i+9)%16] + Gamma0(W[(i+1)%16]) + W[i%16]
    #pragma unroll
    for (int i = 16; i < 80; i++) {
        int idx = i & 15;
        W[idx] = Gamma1(W[(i + 14) & 15]) + W[(i + 9) & 15] + Gamma0(W[(i + 1) & 15]) + W[idx];
        
        uint64_t t0 = hh + Sigma1(e) + Ch(e, f, g) + d_K512[i] + W[idx];
        uint64_t t1 = Sigma0(a) + Maj(a, b, c);
        hh = g; g = f; f = e; e = d + t0;
        d = c; c = b; b = a; a = t0 + t1;
    }
    
    // Finalize
    S[0] += a; S[1] += b; S[2] += c; S[3] += d;
    S[4] += e; S[5] += f; S[6] += g; S[7] += hh;
    
    // Output
    for (int i = 0; i < 8; i++) {
        STORE64H(S[i], output + i * 8);
    }
}
