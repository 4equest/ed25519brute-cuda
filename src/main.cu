#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <random>
#include <cuda_runtime.h>

#include "config.h"
#include "sha256.cuh"
#include "sha512.cuh"
#include "ed25519.cuh"
#include "fingerprint_match.cuh"
#include "openssh_format.h"

// Ed25519 Precomputed Tables (Definitions)
__device__ ge_precomp d_base_5bit[52][16]; 
__device__ ge_precomp d_base_7bit[37][64]; 
__device__ ge_precomp d_base_8bit[32][128]; 


// Match result structure - contains full key information
struct MatchResult {
    uint8_t seed[32];           // 32-byte random seed
    uint8_t private_key[64];    // 64-byte private key (seed || pubkey)
    uint8_t public_key[32];     // 32-byte public key
    uint8_t fingerprint[32];    // 32-byte SHA256 fingerprint
    int found;                  // Match flag
};

// Device-side PREFIX match parameters (32-bit optimized)
__constant__ uint32_t d_prefix_targets[8];
__constant__ uint32_t d_prefix_masks[8];
__constant__ int d_prefix_full_words;
__constant__ uint32_t d_prefix_partial_mask;

// Base seed for deterministic search (randomized at start)
__constant__ uint8_t d_base_seed[32];

// Device-side SUFFIX match parameters (32-bit optimized)
__constant__ uint32_t d_suffix_targets[8];
__constant__ uint32_t d_suffix_masks[8];
__constant__ int d_suffix_start_word;
__constant__ int d_suffix_word_count;

__constant__ int d_match_mode;  // 0=prefix only, 1=suffix only, 2=both

// CUDA Kernel for Brute Forcing (Counter-based) - Batch Inversion Optimized
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) search_kernel(
    MatchResult* result,
    uint64_t base_counter
) {
    if (result->found) return;  // Early exit if already found
    
    // Global thread ID
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Batch processing - BATCH_SIZE keys at a time
    uint8_t seeds[BATCH_SIZE][32];
    uint8_t pubkeys[BATCH_SIZE][32];
    uint32_t hash[8];  // 32-bit hash for optimized matching
    
    // Process ITERATIONS_PER_THREAD keys in batches
    for (int iter = 0; iter < ITERATIONS_PER_THREAD; iter += BATCH_SIZE) {
        // Prepare seeds
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; b++) {
            uint64_t current_idx = base_counter + ((uint64_t)thread_id * ITERATIONS_PER_THREAD) + iter + b;
            
            // Copy base seed
            #pragma unroll
            for(int i=0; i<32; i++) seeds[b][i] = d_base_seed[i];
            
            // XOR first 8 bytes with counter
            *((uint64_t*)&seeds[b][0]) ^= current_idx;
        }
        
        // Generate public keys with single fe_invert (Batch Inversion)
        ed25519_pubkey_batch<BATCH_SIZE>(seeds, pubkeys);
        
        // Check each key for match
        #pragma unroll
        for (int b = 0; b < BATCH_SIZE; b++) {
            // Compute SHA256 fingerprint (now outputs uint32_t[8])
            sha256_ssh_fingerprint(pubkeys[b], hash);
            
            // Check match based on mode (using 32-bit functions)
            bool matched = false;
            
            if (d_match_mode == 0) {
                matched = match_prefix_32bit(hash, d_prefix_targets, d_prefix_masks,
                                             d_prefix_full_words, d_prefix_partial_mask);
            } else if (d_match_mode == 1) {
                matched = match_suffix_32bit(hash, d_suffix_targets, d_suffix_masks,
                                             d_suffix_start_word, d_suffix_word_count);
            } else {
                matched = match_prefix_32bit(hash, d_prefix_targets, d_prefix_masks,
                                             d_prefix_full_words, d_prefix_partial_mask);
                if (matched) {
                    matched = match_suffix_32bit(hash, d_suffix_targets, d_suffix_masks,
                                                 d_suffix_start_word, d_suffix_word_count);
                }
            }
            
            if (matched) {
                if (atomicExch(&result->found, 1) == 0) {
                    memcpy(result->seed, seeds[b], 32);
                    memcpy(result->public_key, pubkeys[b], 32);
                    // Convert uint32_t hash to bytes for output
                    hash32_to_bytes(hash, result->fingerprint);
                    memcpy(result->private_key, seeds[b], 32);
                    memcpy(result->private_key + 32, pubkeys[b], 32);
                }
                return;
            }
        }
    }
}

// Host function to setup prefix matching parameters (32-bit optimized)
void setup_prefix_params(const char* prefix) {
    uint32_t targets[8] = {0};
    uint32_t masks[8] = {0};
    int full_words, partial_bits;
    
    decode_prefix_pattern_32bit(prefix, targets, masks, &full_words, &partial_bits);
    
    // Calculate partial mask for the word containing the partial bits
    uint32_t partial_mask = 0;
    if (partial_bits > 0) {
        partial_mask = masks[full_words];
    }
    
    cudaMemcpyToSymbol(d_prefix_targets, targets, sizeof(uint32_t) * 8);
    cudaMemcpyToSymbol(d_prefix_masks, masks, sizeof(uint32_t) * 8);
    cudaMemcpyToSymbol(d_prefix_full_words, &full_words, sizeof(int));
    cudaMemcpyToSymbol(d_prefix_partial_mask, &partial_mask, sizeof(uint32_t));
}

// Helper to setup suffix params (32-bit optimized)
void setup_suffix_params(const char* suffix) {
    uint32_t targets[8] = {0};
    uint32_t masks[8] = {0};
    int start_word, word_count;
    
    decode_suffix_pattern_32bit(suffix, targets, masks, &start_word, &word_count);
    
    cudaMemcpyToSymbol(d_suffix_targets, targets, sizeof(uint32_t) * 8);
    cudaMemcpyToSymbol(d_suffix_masks, masks, sizeof(uint32_t) * 8);
    cudaMemcpyToSymbol(d_suffix_start_word, &start_word, sizeof(int));
    cudaMemcpyToSymbol(d_suffix_word_count, &word_count, sizeof(int));
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --fingerprint-prefix <string>  Fingerprint prefix to search for\n");
    fprintf(stderr, "  --fingerprint-suffix <string>  Fingerprint suffix to search for\n");
    fprintf(stderr, "  --blocks <number>              Number of CUDA blocks (default: 256, max: 4096)\n");
    fprintf(stderr, "\nNote: At least one of prefix or suffix must be specified.\n");
}

// Forward declarations
void base64_encode(const uint8_t* data, int len, char* out);

// Write key information to file
void write_key_info(const MatchResult* result, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s for writing\n", filename);
        return;
    }
    
    // Compute fingerprint base64
    char fp_b64[45];
    base64_encode(result->fingerprint, 32, fp_b64);
    
    fprintf(f, "# Ed25519 SSH Key (Generated by ed25519brute_cuda)\n\n");
    
    fprintf(f, "## Seed (32 bytes, hex)\n");
    for (int i = 0; i < 32; i++) fprintf(f, "%02x", result->seed[i]);
    fprintf(f, "\n\n");
    
    fprintf(f, "## Private Key (64 bytes, hex)\n");
    for (int i = 0; i < 64; i++) fprintf(f, "%02x", result->private_key[i]);
    fprintf(f, "\n\n");
    
    fprintf(f, "## Public Key (32 bytes, hex)\n");
    for (int i = 0; i < 32; i++) fprintf(f, "%02x", result->public_key[i]);
    fprintf(f, "\n\n");
    
    fprintf(f, "## SSH Fingerprint\n");
    fprintf(f, "SHA256:%s\n", fp_b64);
    
    fclose(f);
    printf("Key information written to %s\n", filename);
}

// Simple CPU SHA256 for verification
void cpu_sha256_block(const uint8_t* msg, size_t len, uint8_t* hash);

// Base64 encode for display
const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void base64_encode(const uint8_t* data, int len, char* out) {
    int i, j;
    for (i = 0, j = 0; i < len; ) {
        uint32_t a = i < len ? data[i++] : 0;
        uint32_t b = i < len ? data[i++] : 0;
        uint32_t c = i < len ? data[i++] : 0;
        uint32_t triple = (a << 16) | (b << 8) | c;
        
        out[j++] = b64_chars[(triple >> 18) & 0x3F];
        out[j++] = b64_chars[(triple >> 12) & 0x3F];
        out[j++] = b64_chars[(triple >> 6) & 0x3F];
        out[j++] = b64_chars[triple & 0x3F];
    }
    // Remove padding for raw base64 (SHA256 fingerprint specific)
    // Calculate required characters: ceil(bits / 6)
    // 32 bytes * 8 = 256 bits. 256 / 6 = 42.66 -> 43 chars.
    int required_chars = (len * 8 + 5) / 6;
    out[required_chars] = '\0';
}

// Helper to check if string contains only valid Base64 characters for custom alphabet verification
bool is_valid_base64(const char* str) {
    if (!str) return true;
    while (*str) {
        char c = *str;
        bool valid = (c >= 'A' && c <= 'Z') || 
                     (c >= 'a' && c <= 'z') || 
                     (c >= '0' && c <= '9') || 
                     (c == '+') || (c == '/');
        if (!valid) return false;
        str++;
    }
    return true;
}

bool file_exists(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (f) {
        fclose(f);
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    // Initialize secure RNG
    // On Windows (MSVC), std::random_device calls RtlGenRandom (SystemFunction036)
    // which is cryptographically secure.
    std::random_device rd;
    std::uniform_int_distribution<unsigned short> dist(0, 255);
    
    // Generate random base seed on host
    // Use random_device directly to ensure CSPRNG quality
    uint8_t base_seed[32];
    for(int i=0; i<32; i++) base_seed[i] = (uint8_t)dist(rd);
    
    // Copy base seed to device constant memory
    cudaMemcpyToSymbol(d_base_seed, base_seed, 32);

    // Copy Ed25519 precomputed tables to device memory
    cudaMemcpyToSymbol(d_base_5bit, base_5bit, sizeof(ge_precomp) * 52 * 16);
    cudaMemcpyToSymbol(d_base_7bit, base_7bit, sizeof(ge_precomp) * 37 * 64);
    cudaMemcpyToSymbol(d_base_8bit, base_8bit, sizeof(ge_precomp) * 32 * 128);

    
    const char* prefix = nullptr;
    const char* suffix = nullptr;
    int blocks = DEFAULT_BLOCKS;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--fingerprint-prefix") == 0 && i + 1 < argc) {
            prefix = argv[++i];
        } else if (strcmp(argv[i], "--fingerprint-suffix") == 0 && i + 1 < argc) {
            suffix = argv[++i];
        } else if (strcmp(argv[i], "--blocks") == 0 && i + 1 < argc) {
            blocks = atoi(argv[++i]);
            if (blocks <= 0 || blocks > MAX_BLOCKS) {
                fprintf(stderr, "Error: blocks must be > 0 and <= %d\n", MAX_BLOCKS);
                return 1;
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!prefix && !suffix) {
        print_usage(argv[0]);
        return 1;
    }
    



    // Validate Base64 characters in prefix
    if (prefix && !is_valid_base64(prefix)) {
        fprintf(stderr, "Error: Prefix contains invalid Base64 characters.\n");
        fprintf(stderr, "Allowed characters are: A-Z, a-z, 0-9, +, /\n");
        return 1;
    }

    // Validate Base64 characters in suffix
    if (suffix) {
        if (!is_valid_base64(suffix)) {
            fprintf(stderr, "Error: Suffix contains invalid Base64 characters.\n");
            fprintf(stderr, "Allowed characters are: A-Z, a-z, 0-9, +, /\n");
            return 1;
        }

        // Validate suffix last character (must be one of "AEIMQUYcgkosw048")
        int slen = strlen(suffix);
        if (slen > 0) {
            const char* valid_last = "AEIMQUYcgkosw048";
            if (!strchr(valid_last, suffix[slen - 1])) {
                fprintf(stderr, "Error: Last character of suffix must be one of \"%s\"\n", valid_last);
                fprintf(stderr, "This is required for valid Base64 encoded SHA256 hashes.\n");
                return 1;
            }
        }
    }
    
    // Check for existing key files and prompt for overwrite
    const char* output_files[] = {"found_key.txt", "id_ed25519", "id_ed25519.pub"};
    bool any_exists = false;
    for (int i = 0; i < 3; i++) {
        if (file_exists(output_files[i])) {
            if (!any_exists) {
                printf("Warning: The following key files already exist:\n");
                any_exists = true;
            }
            printf("  - %s\n", output_files[i]);
        }
    }
    if (any_exists) {
        printf("Do you want to overwrite them? (y/n): ");
        fflush(stdout);
        int c = getchar();
        // Clear remaining input
        while (getchar() != '\n' && !feof(stdin));
        if (c != 'y' && c != 'Y') {
            printf("Aborted.\n");
            return 0;
        }
    }

    printf("SSH Key Fingerprint CUDA Brute Force\n");
    printf("=====================================\n");
    if (prefix) printf("Searching for prefix: %s\n", prefix);
    if (suffix) printf("Searching for suffix: %s\n", suffix);
    
    // Setup CUDA
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    // Set match mode
    int match_mode;
    if (prefix && suffix) {
        match_mode = 2;  // Both
    } else if (prefix) {
        match_mode = 0;  // Prefix only
    } else {
        match_mode = 1;  // Suffix only
    }
    cudaMemcpyToSymbol(d_match_mode, &match_mode, sizeof(int));
    
    // Setup prefix parameters
    if (prefix) {
        setup_prefix_params(prefix);
    }
    
    // Setup suffix parameters
    if (suffix) {
        setup_suffix_params(suffix);
    }
    
    // Allocate result buffers - double buffering
    // Using NUM_STREAMS from config.h
    cudaStream_t streams[NUM_STREAMS];
    MatchResult* d_results[NUM_STREAMS];
    MatchResult h_results[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_results[i], sizeof(MatchResult));
        h_results[i].found = 0;
        cudaMemcpy(d_results[i], &h_results[i], sizeof(MatchResult), cudaMemcpyHostToDevice);
    }
    
    printf("Starting search...\n");
    
    uint64_t total_keys = 0;
    uint64_t batch_counter = 0;
    const uint64_t keys_per_launch = (uint64_t)blocks * THREADS_PER_BLOCK * ITERATIONS_PER_THREAD;
        
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t last_report_keys = 0;
    
    bool found = false;
    int found_stream = -1;
    
    // Launch first batch on stream 0
    search_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[0]>>>(d_results[0], batch_counter);
    batch_counter += keys_per_launch;
    total_keys += keys_per_launch;
    
    // Main search loop - double buffered
    while (!found) {
        for (int s = 0; s < NUM_STREAMS && !found; s++) {
            int current_stream = s;
            int next_stream = (s + 1) % NUM_STREAMS;
            
            // Launch next kernel on next stream (if not first iteration)
            search_kernel<<<blocks, THREADS_PER_BLOCK, 0, streams[next_stream]>>>(d_results[next_stream], batch_counter);
            batch_counter += keys_per_launch;
            
            // Wait for current stream to complete
            cudaStreamSynchronize(streams[current_stream]);
            
            // Check for errors
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
                found = true;
                break;
            }
            
            // Check result (async copy already completed due to sync)
            cudaMemcpy(&h_results[current_stream], d_results[current_stream], sizeof(MatchResult), cudaMemcpyDeviceToHost);
            
            total_keys += keys_per_launch;
            
            if (h_results[current_stream].found) {
                found = true;
                found_stream = current_stream;
                
                // Wait for other stream to avoid resource leaks
                cudaStreamSynchronize(streams[next_stream]);
                break;
            }
            
            // Reset result for reuse
            h_results[current_stream].found = 0;
            cudaMemcpyAsync(d_results[current_stream], &h_results[current_stream], sizeof(MatchResult), cudaMemcpyHostToDevice, streams[current_stream]);
            
            // Progress report
            if (total_keys - last_report_keys > PROGRESS_REPORT_INTERVAL) {
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = current_time - start_time;
                double seconds = elapsed.count();
                double hashrate = (double)total_keys / seconds / 1000000.0;
                
                printf("\rChecked: %llu M keys | Speed: %.2f MKeys/s | Time: %.1fs", 
                       (unsigned long long)(total_keys / 1000000), hashrate, seconds);
                fflush(stdout);
                last_report_keys = total_keys;
            }
        }
    }
    
    if (found_stream >= 0 && h_results[found_stream].found) {
        printf("\n============================================\n");
        printf("Match found after checking ~%llu keys!\n", (unsigned long long)total_keys);
        printf("============================================\n\n");
        
        // Compute fingerprint base64
        char fp_b64[45];
        base64_encode(h_results[found_stream].fingerprint, 32, fp_b64);
        
        // Print seed
        printf("Seed (32 bytes):       ");
        for (int i = 0; i < 32; i++) printf("%02x", h_results[found_stream].seed[i]);
        printf("\n");
        
        // Print private key
        printf("Private Key (64 bytes): ");
        for (int i = 0; i < 64; i++) printf("%02x", h_results[found_stream].private_key[i]);
        printf("\n");
        
        // Print public key
        printf("Public Key (32 bytes):  ");
        for (int i = 0; i < 32; i++) printf("%02x", h_results[found_stream].public_key[i]);
        printf("\n");
        
        // Print fingerprint
        printf("Fingerprint:            SHA256:%s\n\n", fp_b64);
        
        // Write output to file (hex format for debugging)
        write_key_info(&h_results[found_stream], "found_key.txt");
        
        // Write OpenSSH format keys
        if (write_openssh_keys(h_results[found_stream].seed, 
                               h_results[found_stream].public_key,
                               "id_ed25519", "id_ed25519.pub")) {
            printf("OpenSSH private key written to: id_ed25519\n");
            printf("OpenSSH public key written to:  id_ed25519.pub\n");
        } else {
            fprintf(stderr, "Error: Failed to write OpenSSH keys\n");
        }
    }
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_results[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return (found_stream >= 0 && h_results[found_stream].found) ? 0 : 1;
}
