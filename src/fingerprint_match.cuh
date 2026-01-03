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

// ============================================================================
// 32-BIT PREFIX MATCHING (Optimized)
// ============================================================================

// Host-side function: Decode Base64 pattern into uint32_t targets and masks
// Pattern is packed as big-endian into uint32_t words to match SHA256 state
inline void decode_prefix_pattern_32bit(
    const char* pattern,
    uint32_t* targets,
    uint32_t* masks,
    int* full_words,
    int* partial_bits
) {
    int len = (int)strlen(pattern);
    int total_bits = len * 6;
    
    *full_words = total_bits / 32;
    *partial_bits = total_bits % 32;
    
    // First decode to bytes (big-endian)
    uint8_t bytes[32] = {0};
    for (int i = 0; i < len; i++) {
        uint8_t val = (uint8_t)h_b64_decode_table[(uint8_t)pattern[i]];
        int bit_pos = i * 6;
        for (int b = 0; b < 6; b++) {
            int byte_idx = (bit_pos + b) / 8;
            int bit_idx = 7 - ((bit_pos + b) % 8);
            if (val & (1 << (5 - b))) {
                bytes[byte_idx] |= (1 << bit_idx);
            }
        }
    }
    
    // Pack bytes into uint32_t words (big-endian order to match SHA256 state)
    for (int i = 0; i < 8; i++) {
        targets[i] = ((uint32_t)bytes[i*4] << 24) | ((uint32_t)bytes[i*4+1] << 16) |
                     ((uint32_t)bytes[i*4+2] << 8) | bytes[i*4+3];
    }
    
    // Build masks: all 1s for bits covered by pattern
    uint8_t mask_bytes[32] = {0};
    int total_bytes = (total_bits + 7) / 8;
    for (int i = 0; i < total_bytes - 1; i++) {
        mask_bytes[i] = 0xFF;
    }
    if (total_bits % 8 != 0) {
        mask_bytes[total_bytes - 1] = (uint8_t)(0xFF << (8 - (total_bits % 8)));
    } else if (total_bytes > 0) {
        mask_bytes[total_bytes - 1] = 0xFF;
    }
    
    // Pack masks into uint32_t words
    for (int i = 0; i < 8; i++) {
        masks[i] = ((uint32_t)mask_bytes[i*4] << 24) | ((uint32_t)mask_bytes[i*4+1] << 16) |
                   ((uint32_t)mask_bytes[i*4+2] << 8) | mask_bytes[i*4+3];
    }
}

// Device function: 32-bit prefix matching
__device__ __forceinline__ bool match_prefix_32bit(
    const uint32_t* hash,
    const uint32_t* targets,
    const uint32_t* masks,
    int full_words,
    uint32_t partial_mask
) {
    // Check full words
    for (int i = 0; i < full_words; i++) {
        if (hash[i] != targets[i]) return false;
    }
    // Check partial word
    if (partial_mask != 0) {
        if ((hash[full_words] & partial_mask) != (targets[full_words] & partial_mask)) return false;
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

// ============================================================================
// 32-BIT SUFFIX MATCHING (Optimized)
// ============================================================================

// Host-side function: Decode suffix pattern into uint32_t targets and masks
inline void decode_suffix_pattern_32bit(
    const char* pattern,
    uint32_t* targets,
    uint32_t* masks,
    int* start_word,
    int* word_count
) {
    int len = (int)strlen(pattern);
    if (len == 0) {
        *word_count = 0;
        return;
    }
    
    // First decode to bytes using existing logic
    uint8_t byte_targets[32] = {0};
    uint8_t byte_masks[32] = {0};
    int start_offset, match_len;
    
    // Reuse byte-based decoding
    int start_bit = (43 - len) * 6;
    int end_bit = 256;
    int start_byte = start_bit / 8;
    int end_byte = (end_bit - 1) / 8;
    
    start_offset = start_byte;
    match_len = end_byte - start_byte + 1;
    
    for (int i = 0; i < len; i++) {
        uint8_t val = (uint8_t)h_b64_decode_table[(uint8_t)pattern[i]];
        int bit_pos = i * 6;
        for (int b = 0; b < 6; b++) {
            if (start_bit + bit_pos + b < end_bit) {
                int absolute_bit = start_bit + bit_pos + b;
                int byte_idx_in_hash = absolute_bit / 8;
                int bit_idx_in_byte = 7 - (absolute_bit % 8);
                int target_idx = byte_idx_in_hash - start_byte;
                
                if (val & (1 << (5 - b))) {
                    byte_targets[target_idx] |= (1 << bit_idx_in_byte);
                }
                byte_masks[target_idx] |= (1 << bit_idx_in_byte);
            }
        }
    }
    
    // Convert to word indices
    *start_word = start_byte / 4;
    int end_word_idx = (end_byte) / 4;
    *word_count = end_word_idx - *start_word + 1;
    
    // Pack bytes into full hash (32 bytes) then extract relevant words
    uint8_t full_targets[32] = {0};
    uint8_t full_masks[32] = {0};
    for (int i = 0; i < match_len; i++) {
        full_targets[start_offset + i] = byte_targets[i];
        full_masks[start_offset + i] = byte_masks[i];
    }
    
    // Pack into uint32_t words
    for (int i = 0; i < 8; i++) {
        targets[i] = ((uint32_t)full_targets[i*4] << 24) | ((uint32_t)full_targets[i*4+1] << 16) |
                     ((uint32_t)full_targets[i*4+2] << 8) | full_targets[i*4+3];
        masks[i] = ((uint32_t)full_masks[i*4] << 24) | ((uint32_t)full_masks[i*4+1] << 16) |
                   ((uint32_t)full_masks[i*4+2] << 8) | full_masks[i*4+3];
    }
}

// Device function: 32-bit suffix matching
__device__ __forceinline__ bool match_suffix_32bit(
    const uint32_t* hash,
    const uint32_t* targets,
    const uint32_t* masks,
    int start_word,
    int word_count
) {
    for (int i = 0; i < word_count; i++) {
        int idx = start_word + i;
        if ((hash[idx] & masks[idx]) != targets[idx]) return false;
    }
    return true;
}

// ============================================================================
// HELPER: Convert uint32_t hash to bytes (for output/display)
// ============================================================================
__device__ __forceinline__ void hash32_to_bytes(const uint32_t* hash32, uint8_t* bytes) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        bytes[i*4]   = (uint8_t)(hash32[i] >> 24);
        bytes[i*4+1] = (uint8_t)(hash32[i] >> 16);
        bytes[i*4+2] = (uint8_t)(hash32[i] >> 8);
        bytes[i*4+3] = (uint8_t)(hash32[i]);
    }
}
