#pragma once
#include <cstdint>

// SHA256 constants
__constant__ uint32_t d_K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    // Use CUDA intrinsic for funnel shift (rotate)
    // This maps directly to hardware instructions
    return __funnelshift_r(x, x, n);
}

__device__ __forceinline__ uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sha256_sigma0(uint32_t x) {
    return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ __forceinline__ uint32_t sha256_sigma1(uint32_t x) {
    return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ __forceinline__ uint32_t sha256_gamma0(uint32_t x) {
    return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha256_gamma1(uint32_t x) {
    return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

// SHA256 for SSH fingerprint: input is 51 bytes (19 byte header + 32 byte pubkey)
// SSH wire format:
//   00 00 00 0b  (length of "ssh-ed25519" = 11)
//   73 73 68 2d 65 64 32 35 35 31 39  ("ssh-ed25519")
//   00 00 00 20  (length of public key = 32)
//   [32 bytes public key]
__device__ void sha256_ssh_fingerprint(const uint8_t* pubkey, uint8_t* hash) {
    // Build message directly in w[] array
    // Total: 51 bytes message + 1 byte 0x80 + padding + 8 bytes length = 64 bytes (1 block)
    uint32_t w[64];
    
    // SSH Wire Format for Ed25519 public key (51 bytes total):
    // Bytes  0- 3: 00 00 00 0b         (length of "ssh-ed25519" = 11)
    // Bytes  4-14: ssh-ed25519         (algorithm name)
    // Bytes 15-18: 00 00 00 20         (length of public key = 32)
    // Bytes 19-50: [32 bytes public key]
    
    // w[0]: bytes 0-3
    w[0] = 0x0000000b;
    
    // w[1]: bytes 4-7: "ssh-"
    w[1] = 0x7373682d;
    
    // w[2]: bytes 8-11: "ed25"
    w[2] = 0x65643235;
    
    // w[3]: bytes 12-15: "519" + 0x00 (start of length field)
    w[3] = 0x35313900;
    
    // w[4]: bytes 16-19: 0x00 0x00 0x20 pubkey[0]
    w[4] = 0x00002000 | pubkey[0];
    
    // w[5]: bytes 20-23: pubkey[1..4]
    w[5] = ((uint32_t)pubkey[1] << 24) | ((uint32_t)pubkey[2] << 16) | 
           ((uint32_t)pubkey[3] << 8) | pubkey[4];
    
    // w[6]: bytes 24-27: pubkey[5..8]
    w[6] = ((uint32_t)pubkey[5] << 24) | ((uint32_t)pubkey[6] << 16) | 
           ((uint32_t)pubkey[7] << 8) | pubkey[8];
    
    // w[7]: bytes 28-31: pubkey[9..12]
    w[7] = ((uint32_t)pubkey[9] << 24) | ((uint32_t)pubkey[10] << 16) | 
           ((uint32_t)pubkey[11] << 8) | pubkey[12];
    
    // w[8]: bytes 32-35: pubkey[13..16]
    w[8] = ((uint32_t)pubkey[13] << 24) | ((uint32_t)pubkey[14] << 16) | 
           ((uint32_t)pubkey[15] << 8) | pubkey[16];
    
    // w[9]: bytes 36-39: pubkey[17..20]
    w[9] = ((uint32_t)pubkey[17] << 24) | ((uint32_t)pubkey[18] << 16) | 
           ((uint32_t)pubkey[19] << 8) | pubkey[20];
    
    // w[10]: bytes 40-43: pubkey[21..24]
    w[10] = ((uint32_t)pubkey[21] << 24) | ((uint32_t)pubkey[22] << 16) | 
            ((uint32_t)pubkey[23] << 8) | pubkey[24];
    
    // w[11]: bytes 44-47: pubkey[25..28]
    w[11] = ((uint32_t)pubkey[25] << 24) | ((uint32_t)pubkey[26] << 16) | 
            ((uint32_t)pubkey[27] << 8) | pubkey[28];
    
    // w[12]: bytes 48-51: pubkey[29..31] + 0x80 padding
    w[12] = ((uint32_t)pubkey[29] << 24) | ((uint32_t)pubkey[30] << 16) | 
            ((uint32_t)pubkey[31] << 8) | 0x80;
    
    // Padding zeros
    w[13] = 0;
    w[14] = 0;
    
    // Length in bits: 51 * 8 = 408 = 0x198
    w[15] = 408;
    
    // Message schedule expansion
    // Note: w[0]..w[3] are constants used in expansion, keep them assigned above
    for (int i = 16; i < 64; i++) {
        w[i] = sha256_gamma1(w[i-2]) + w[i-7] + sha256_gamma0(w[i-15]) + w[i-16];
    }
    
    // Compression - OPTIMIZED: Start from round 4
    // Rounds 0-3 process only constant header bytes (w[0]..w[3])
    // Precomputed state after round 3:
    uint32_t a = 0xd0379364, b = 0x0cc892cb, c = 0xf1447237, d = 0xfc088858;
    uint32_t e = 0xc6d778ab, f = 0x5401156e, g = 0xd3c515c2, h = 0x98c7e2ad;
    
    #pragma unroll
    for (int i = 4; i < 64; i++) {
        uint32_t t1 = h + sha256_sigma1(e) + sha256_ch(e, f, g) + d_K256[i] + w[i];
        uint32_t t2 = sha256_sigma0(a) + sha256_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    
    // Add precomputed initial state (after rounds 0-3 were applied to IV)
    // Final state = IV + (compression result)
    a += 0x6a09e667; b += 0xbb67ae85; c += 0x3c6ef372; d += 0xa54ff53a;
    e += 0x510e527f; f += 0x9b05688c; g += 0x1f83d9ab; h += 0x5be0cd19;
    
    // Output as big-endian bytes
    hash[0]  = (uint8_t)(a >> 24); hash[1]  = (uint8_t)(a >> 16); hash[2]  = (uint8_t)(a >> 8); hash[3]  = (uint8_t)a;
    hash[4]  = (uint8_t)(b >> 24); hash[5]  = (uint8_t)(b >> 16); hash[6]  = (uint8_t)(b >> 8); hash[7]  = (uint8_t)b;
    hash[8]  = (uint8_t)(c >> 24); hash[9]  = (uint8_t)(c >> 16); hash[10] = (uint8_t)(c >> 8); hash[11] = (uint8_t)c;
    hash[12] = (uint8_t)(d >> 24); hash[13] = (uint8_t)(d >> 16); hash[14] = (uint8_t)(d >> 8); hash[15] = (uint8_t)d;
    hash[16] = (uint8_t)(e >> 24); hash[17] = (uint8_t)(e >> 16); hash[18] = (uint8_t)(e >> 8); hash[19] = (uint8_t)e;
    hash[20] = (uint8_t)(f >> 24); hash[21] = (uint8_t)(f >> 16); hash[22] = (uint8_t)(f >> 8); hash[23] = (uint8_t)f;
    hash[24] = (uint8_t)(g >> 24); hash[25] = (uint8_t)(g >> 16); hash[26] = (uint8_t)(g >> 8); hash[27] = (uint8_t)g;
    hash[28] = (uint8_t)(h >> 24); hash[29] = (uint8_t)(h >> 16); hash[30] = (uint8_t)(h >> 8); hash[31] = (uint8_t)h;
}
