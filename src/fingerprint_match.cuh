#pragma once
#include <cstdint>
#include <cstring>

// Base64 decoding table (CPU-side, regular array)
static const int8_t h_b64_decode_table[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
};

// ============================================================================
// PREFIX MATCHING
// ============================================================================
inline void decode_prefix_pattern(
    const char* pattern,
    uint8_t* binary,
    int* full_bytes,
    int* partial_bits,
    uint8_t* partial_mask,
    uint8_t* partial_value
) {
    int len = (int)strlen(pattern);
    int total_bits = len * 6;
    
    *full_bytes = total_bits / 8;
    *partial_bits = total_bits % 8;
    
    uint8_t bits[256] = {0};
    for (int i = 0; i < len; i++) {
        uint8_t val = (uint8_t)h_b64_decode_table[(uint8_t)pattern[i]];
        int bit_pos = i * 6;
        for (int b = 0; b < 6; b++) {
            int byte_idx = (bit_pos + b) / 8;
            int bit_idx = 7 - ((bit_pos + b) % 8);
            if (val & (1 << (5 - b))) {
                bits[byte_idx] |= (1 << bit_idx);
            }
        }
    }
    
    for (int i = 0; i < *full_bytes; i++) binary[i] = bits[i];
    
    if (*partial_bits > 0) {
        *partial_mask = (uint8_t)(0xFF << (8 - *partial_bits));
        *partial_value = bits[*full_bytes] & *partial_mask;
    } else {
        *partial_mask = 0;
        *partial_value = 0;
    }
}

__device__ __forceinline__ bool match_prefix(
    const uint8_t* hash,
    const uint8_t* prefix_bytes,
    int prefix_full_bytes,
    int prefix_partial_bits,
    uint8_t prefix_partial_mask,
    uint8_t prefix_partial_value
) {
    for (int i = 0; i < prefix_full_bytes; i++) {
        if (hash[i] != prefix_bytes[i]) return false;
    }
    if (prefix_partial_bits > 0) {
        if ((hash[prefix_full_bytes] & prefix_partial_mask) != prefix_partial_value) return false;
    }
    return true;
}


inline void decode_suffix_pattern_uniform(
    const char* pattern,
    int* start_offset,
    int* match_len,
    uint8_t* targets,
    uint8_t* masks
) {
    int len = (int)strlen(pattern);
    if (len == 0) {
        *match_len = 0;
        return;
    }
    
    // Bits in hash
    // Fingerprint chars 0..42 map to bits 0..257. Hash is 256 bits (0..255).
    // Suffix starts at char (43 - len).
    int start_bit = (43 - len) * 6;
    int end_bit = 256;
    
    int start_byte = start_bit / 8;
    int end_byte = (end_bit - 1) / 8;
    
    *start_offset = start_byte;
    *match_len = end_byte - start_byte + 1;
    
    // Temporary buffer to hold the bits of the suffix pattern
    // This buffer will be aligned to the start of the suffix bits (0-indexed relative to suffix)
    // We need to shift it to align with the hash bytes.
    uint8_t pattern_bits[64] = {0};
    
    for (int i = 0; i < len; i++) {
        uint8_t val = (uint8_t)h_b64_decode_table[(uint8_t)pattern[i]];
        int bit_pos = i * 6;
        for (int b = 0; b < 6; b++) {
            // Check if this bit corresponds to a valid hash bit
            if (start_bit + bit_pos + b < end_bit) {
                // Byte index: K / 8
                // Bit index: 7 - (K % 8)
                
                // Let's directly write to the targets buffer (aligned to hash)
                int absolute_bit = start_bit + bit_pos + b;
                int byte_idx_in_hash = absolute_bit / 8;
                int bit_idx_in_byte = 7 - (absolute_bit % 8);
                
                int target_idx = byte_idx_in_hash - start_byte;
                
                if (val & (1 << (5 - b))) {
                    targets[target_idx] |= (1 << bit_idx_in_byte);
                }
                
                // Set mask bit since this bit is part of the pattern
                masks[target_idx] |= (1 << bit_idx_in_byte);
            }
        }
    }
}

// Device function: Check if sha256 hash matches suffix pattern
// Early-exit optimization: First byte check before loop
__device__ __forceinline__ bool match_suffix_uniform(
    const uint8_t* hash,
    int start_offset,
    int len,
    const uint8_t* targets,
    const uint8_t* masks
) {
    for (int i = 0; i < len; i++) {
        if ((hash[start_offset + i] & masks[i]) != targets[i]) return false;
    }
    return true;
}

