#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdarg>

#include "sha512.cuh"
#include "ed25519.cuh"

#include "sha256.cuh"
#include "fingerprint_match.cuh"

// Ed25519 Precomputed Tables (Definitions)
__device__ ge_precomp d_base_5bit[52][16]; 
__device__ ge_precomp d_base_7bit[37][64]; 
__device__ ge_precomp d_base_8bit[32][128]; 


struct TestVector {
    uint8_t seed[32];
    uint8_t sha512_hash[64];
    uint8_t private_key[64];     // seed || pubkey
    uint8_t public_key[32];
    uint8_t fingerprint_hash[32];
    char fingerprint_b64[44];
};

struct SHA256TestVector {
    uint8_t pubkey[32];
    uint8_t expected_hash[32];
};

struct MatchTestVector {
    uint8_t hash[32];
    char pattern[32];
    int expected;
};

// Helper for logging
FILE* log_file = NULL;

void log_msg(const char* fmt, ...) {
    va_list args;
    
    // Print to console
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    // Print to file
    if (log_file) {
        va_start(args, fmt);
        vfprintf(log_file, fmt, args);
        va_end(args);
        fflush(log_file);
    }
}

// Note: printf macro redefinition moved after device kernels to avoid conflicts

struct TestResults {
    int total;
    int passed;
    int failed;
};

// Device-side parameters for matching tests
__constant__ uint8_t d_test_prefix[32];
__constant__ int d_test_prefix_len;
__constant__ int d_test_prefix_partial_bits;
__constant__ uint8_t d_test_prefix_partial_mask;
__constant__ uint8_t d_test_prefix_partial_value;

__device__ void decode_prefix_pattern_dev(const char* pattern, uint8_t* binary, int* len, int* pbits, uint8_t* mask, uint8_t* val);

// Test kernel for prefix matching
__global__ void test_prefix_match_kernel(const uint8_t* hash, int* result) {
    *result = match_prefix(hash, d_test_prefix, d_test_prefix_len, 
                          d_test_prefix_partial_bits, d_test_prefix_partial_mask, 
                          d_test_prefix_partial_value);
}

// Device-side SUFFIX match parameters (NEW Uniform Masking)
__constant__ int d_test_suffix_start_offset;
__constant__ int d_test_suffix_len;
__constant__ uint8_t d_test_suffix_targets[32];
__constant__ uint8_t d_test_suffix_masks[32];

// Test kernel for suffix matching
__global__ void test_suffix_match_kernel(const uint8_t* hash, int* result) {
    *result = match_suffix_uniform(hash, 
                                   d_test_suffix_start_offset, d_test_suffix_len,
                                   d_test_suffix_targets, d_test_suffix_masks);
}

// CUDA kernels
__global__ void test_sha512_kernel(const uint8_t* seed, uint8_t* output) {
    sha512_32bytes(seed, output);
}

__global__ void test_ed25519_pubkey_kernel(const uint8_t* seed, uint8_t* pubkey) {
    uint8_t l_pub[32];
    ed25519_pubkey_from_seed(seed, l_pub);
    if (threadIdx.x == 0) {
        for(int i=0; i<32; i++) pubkey[i] = l_pub[i];
    }
}

__global__ void test_ed25519_keypair_kernel(const uint8_t* seed, uint8_t* private_key, uint8_t* pubkey) {
    uint8_t l_priv[64];
    uint8_t l_pub[32];
    ed25519_create_keypair(seed, l_priv, l_pub);
    if (threadIdx.x == 0) {
        for(int i=0; i<64; i++) private_key[i] = l_priv[i];
        for(int i=0; i<32; i++) pubkey[i] = l_pub[i];
    }
}

// Full pipeline: seed -> public key -> fingerprint hash
__global__ void test_full_pipeline_kernel(const uint8_t* seed, uint8_t* pubkey, uint8_t* fp_hash) {
    uint8_t l_pub[32];
    uint8_t l_hash[32];
    
    ed25519_pubkey_from_seed(seed, l_pub);
    sha256_ssh_fingerprint(l_pub, l_hash);
    
    if (threadIdx.x == 0) {
        for(int i=0; i<32; i++) pubkey[i] = l_pub[i];
        for(int i=0; i<32; i++) fp_hash[i] = l_hash[i];
    }
}

__global__ void test_only_sha256_kernel(const uint8_t* pubkey, uint8_t* hash) {
    sha256_ssh_fingerprint(pubkey, hash);
}

#define TEST_BATCH_SIZE 32

// === BATCH INVERSION TESTS ===
// Test kernel for batch inversion: compares batch vs single key generation
__global__ void test_batch_vs_single_kernel(
    const uint8_t seeds[TEST_BATCH_SIZE][32], 
    uint8_t batch_pubkeys[TEST_BATCH_SIZE][32], 
    uint8_t single_pubkeys[TEST_BATCH_SIZE][32]
) {
    // Generate keys using batch method
    uint8_t l_batch[TEST_BATCH_SIZE][32];
    ed25519_pubkey_batch<TEST_BATCH_SIZE>(seeds, l_batch);
    
    // Generate keys using single method
    uint8_t l_single[TEST_BATCH_SIZE][32];
    for (int i = 0; i < TEST_BATCH_SIZE; i++) {
        ed25519_pubkey_from_seed(seeds[i], l_single[i]);
    }
    
    // Copy results to output (thread 0 only)
    if (threadIdx.x == 0) {
        for (int i = 0; i < TEST_BATCH_SIZE; i++) {
            for (int j = 0; j < 32; j++) {
                batch_pubkeys[i][j] = l_batch[i][j];
                single_pubkeys[i][j] = l_single[i][j];
            }
        }
    }
}

// Test kernel for batch with specific edge-case seeds
__global__ void test_batch_edge_cases_kernel(
    uint8_t batch_pubkeys[TEST_BATCH_SIZE][32],
    uint8_t single_pubkeys[TEST_BATCH_SIZE][32]
) {
    // Edge case seeds
    uint8_t seeds[TEST_BATCH_SIZE][32];
    
    // Initialize seeds with patterns
    for(int b=0; b<TEST_BATCH_SIZE; b++) {
        for(int i=0; i<32; i++) {
            if(b==0) seeds[b][i] = 0x00; // All zeros
            else if(b==1) seeds[b][i] = 0xFF; // All ones
            else if(b==2) seeds[b][i] = (i%2==0) ? 0xAA : 0x55; // Alternating
            else if(b==3) seeds[b][i] = (uint8_t)i; // Sequential
            else seeds[b][i] = (uint8_t)(b*17 + i); // Others
        }
    }
    
    // Batch generation
    uint8_t l_batch[TEST_BATCH_SIZE][32];
    ed25519_pubkey_batch<TEST_BATCH_SIZE>(seeds, l_batch);
    
    // Single generation
    uint8_t l_single[TEST_BATCH_SIZE][32];
    for (int i = 0; i < TEST_BATCH_SIZE; i++) {
        ed25519_pubkey_from_seed(seeds[i], l_single[i]);
    }
    
    // Copy results
    if (threadIdx.x == 0) {
        for (int i = 0; i < TEST_BATCH_SIZE; i++) {
            for (int j = 0; j < 32; j++) {
                batch_pubkeys[i][j] = l_batch[i][j];
                single_pubkeys[i][j] = l_single[i][j];
            }
        }
    }
}

void print_hex(const char* label, const uint8_t* data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

bool compare_bytes(const uint8_t* a, const uint8_t* b, int len) {
    for (int i = 0; i < len; i++) if (a[i] != b[i]) return false;
    return true;
}

void base64_encode(const uint8_t* input, int len, char* output) {
    static const char t[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int i = 0, j = 0;
    while (i < len) {
        uint32_t a = i < len ? input[i++] : 0;
        uint32_t b = i < len ? input[i++] : 0;
        uint32_t c = i < len ? input[i++] : 0;
        uint32_t v = (a << 16) | (b << 8) | c;
        output[j++] = t[(v >> 18) & 0x3F];
        output[j++] = t[(v >> 12) & 0x3F];
        output[j++] = t[(v >> 6) & 0x3F];
        output[j++] = t[v & 0x3F];
    }
    int mod = len % 3;
    if (mod == 1) j -= 2;
    else if (mod == 2) j -= 1;
    output[j] = '\0';
}

TestVector* load_test_vectors(const char* filename, uint32_t* count) {
    char path[256];
    snprintf(path, sizeof(path), "testdata/%s", filename);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return nullptr;
    }
    
    fread(count, sizeof(uint32_t), 1, f);
    
    TestVector* v = (TestVector*)malloc(*count * sizeof(TestVector));
    for (uint32_t i = 0; i < *count; i++) {
        fread(v[i].seed, 1, 32, f);
        fread(v[i].sha512_hash, 1, 64, f);
        fread(v[i].private_key, 1, 64, f);  // New field
        fread(v[i].public_key, 1, 32, f);
        fread(v[i].fingerprint_hash, 1, 32, f);
        fread(v[i].fingerprint_b64, 1, 44, f);
    }
    fclose(f);
    return v;
}

SHA256TestVector* load_sha256_vectors(const char* filename, uint32_t* count) {
    char path[256];
    snprintf(path, sizeof(path), "testdata/%s", filename);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return nullptr;
    }
    fread(count, sizeof(uint32_t), 1, f);
    SHA256TestVector* v = (SHA256TestVector*)malloc(*count * sizeof(SHA256TestVector));
    for (uint32_t i = 0; i < *count; i++) {
        fread(v[i].pubkey, 1, 32, f);
        fread(v[i].expected_hash, 1, 32, f);
    }
    fclose(f);
    return v;
}

MatchTestVector* load_match_vectors(const char* filename, uint32_t* count) {
    char path[256];
    snprintf(path, sizeof(path), "testdata/%s", filename);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return nullptr;
    }
    fread(count, sizeof(uint32_t), 1, f);
    MatchTestVector* v = (MatchTestVector*)malloc(*count * sizeof(MatchTestVector));
    for (uint32_t i = 0; i < *count; i++) {
        fread(v[i].hash, 1, 32, f);
        fread(v[i].pattern, 1, 32, f);
        fread(&v[i].expected, sizeof(int), 1, f);
    }
    fclose(f);
    return v;
}

void run_sha512_tests(TestVector* v, uint32_t n, TestResults* r) {
    printf("\n=== SHA512 Tests (%d vectors) ===\n", n);
    
    uint8_t *d_seed, *d_out;
    uint8_t h_out[64];
    cudaMalloc(&d_seed, 32);
    cudaMalloc(&d_out, 64);
    
    int pass = 0, fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        r->total++;
        cudaMemcpy(d_seed, v[i].seed, 32, cudaMemcpyHostToDevice);
        test_sha512_kernel<<<1, 1>>>(d_seed, d_out);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 64, cudaMemcpyDeviceToHost);
        
        if (compare_bytes(h_out, v[i].sha512_hash, 64)) {
            pass++; r->passed++;
        } else {
            fail++; r->failed++;
            printf("[%d] FAILED\n", i+1);
            print_hex("  Expected", v[i].sha512_hash, 32);
            print_hex("  Got     ", h_out, 32);
        }
    }
    printf("Result: %d/%d passed\n", pass, n);
    
    cudaFree(d_seed);
    cudaFree(d_out);
}

void run_ed25519_pubkey_tests(TestVector* v, uint32_t n, TestResults* r) {
    printf("\n=== Ed25519 Public Key Tests (%d vectors) ===\n", n);
    
    uint8_t *d_seed, *d_pub;
    uint8_t h_pub[32];
    cudaMalloc(&d_seed, 32);
    cudaMalloc(&d_pub, 32);
    
    int pass = 0, fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        r->total++;
        cudaMemcpy(d_seed, v[i].seed, 32, cudaMemcpyHostToDevice);
        test_ed25519_pubkey_kernel<<<1, 256>>>(d_seed, d_pub);
        cudaDeviceSynchronize();
        cudaMemcpy(h_pub, d_pub, 32, cudaMemcpyDeviceToHost);
        
        if (compare_bytes(h_pub, v[i].public_key, 32)) {
            pass++; r->passed++;
        } else {
            fail++; r->failed++;
            printf("[%d] FAILED\n", i+1);
            print_hex("  Expected", v[i].public_key, 32);
            print_hex("  Got     ", h_pub, 32);
        }
    }
    printf("Result: %d/%d passed\n", pass, n);
    
    cudaFree(d_seed);
    cudaFree(d_pub);
}

void run_ed25519_keypair_tests(TestVector* v, uint32_t n, TestResults* r) {
    printf("\n=== Ed25519 Key Pair Tests (Private + Public) (%d vectors) ===\n", n);
    
    uint8_t *d_seed, *d_priv, *d_pub;
    uint8_t h_priv[64], h_pub[32];
    cudaMalloc(&d_seed, 32);
    cudaMalloc(&d_priv, 64);
    cudaMalloc(&d_pub, 32);
    
    int pass = 0, fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        r->total++;
        cudaMemcpy(d_seed, v[i].seed, 32, cudaMemcpyHostToDevice);
        test_ed25519_keypair_kernel<<<1, 256>>>(d_seed, d_priv, d_pub);
        cudaDeviceSynchronize();
        cudaMemcpy(h_priv, d_priv, 64, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pub, d_pub, 32, cudaMemcpyDeviceToHost);
        
        bool priv_ok = compare_bytes(h_priv, v[i].private_key, 64);
        bool pub_ok = compare_bytes(h_pub, v[i].public_key, 32);
        
        if (priv_ok && pub_ok) {
            pass++; r->passed++;
        } else {
            fail++; r->failed++;
            printf("[%d] FAILED", i+1);
            if (!priv_ok) printf(" (private key)");
            if (!pub_ok) printf(" (public key)");
            printf("\n");
            if (!priv_ok) {
                print_hex("  Expected priv", v[i].private_key, 64);
                print_hex("  Got priv     ", h_priv, 64);
            }
        }
    }
    printf("Result: %d/%d passed\n", pass, n);
    
    cudaFree(d_seed);
    cudaFree(d_priv);
    cudaFree(d_pub);
}

void run_full_pipeline_tests(TestVector* v, uint32_t n, TestResults* r) {
    printf("\n=== Full Pipeline Tests (Seed -> Pubkey -> Fingerprint) (%d vectors) ===\n", n);
    
    uint8_t *d_seed, *d_pub, *d_fp;
    uint8_t h_pub[32], h_fp[32];
    char fp_b64[45];
    cudaMalloc(&d_seed, 32);
    cudaMalloc(&d_pub, 32);
    cudaMalloc(&d_fp, 32);
    
    int pass = 0, fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        r->total++;
        cudaMemcpy(d_seed, v[i].seed, 32, cudaMemcpyHostToDevice);
        test_full_pipeline_kernel<<<1, 256>>>(d_seed, d_pub, d_fp);
        cudaDeviceSynchronize();
        cudaMemcpy(h_pub, d_pub, 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fp, d_fp, 32, cudaMemcpyDeviceToHost);
        
        base64_encode(h_fp, 32, fp_b64);
        
        bool pub_ok = compare_bytes(h_pub, v[i].public_key, 32);
        bool fp_ok = (strcmp(fp_b64, v[i].fingerprint_b64) == 0);
        
        if (pub_ok && fp_ok) {
            pass++; r->passed++;
        } else {
            fail++; r->failed++;
            printf("[%d] FAILED", i+1);
            if (!pub_ok) printf(" (pubkey)");
            if (!fp_ok) printf(" (fingerprint)");
            printf("\n");
            if (!fp_ok) {
                printf("  Expected FP: %s\n", v[i].fingerprint_b64);
                printf("  Got FP:      %s\n", fp_b64);
            }
        }
    }
    printf("Result: %d/%d passed\n", pass, n);
    
    cudaFree(d_seed);
    cudaFree(d_pub);
    cudaFree(d_fp);
}

// Debug kernel to check constants on device (uses CUDA printf directly)
__global__ void debug_check_constants_kernel() {
    // Note: Using CUDA's built-in printf for device code
    ::printf("--- DEBUG DEVICE MEMORY ---\n");
    ::printf("d_base_5bit[0][0].yplusx[0]: %d (Expected: 25967493)\n", d_base_5bit[0][0].yplusx[0]);
    ::printf("d_K256[0]: %08x (Expected: 428a2f98)\n", d_K256[0]);
    ::printf("---------------------------\n");
}

// Now define printf macro for host code
#define printf log_msg

// Helper to setup prefix params for tests (duplicated from main.cu for isolation)
void setup_test_prefix_params(const char* prefix) {
    uint8_t prefix_bytes[32] = {0};
    int full_bytes, partial_bits;
    uint8_t partial_mask, partial_value;
    
    decode_prefix_pattern(prefix, prefix_bytes, &full_bytes, &partial_bits,
                          &partial_mask, &partial_value);
    
    cudaMemcpyToSymbol(d_test_prefix, prefix_bytes, 32);
    cudaMemcpyToSymbol(d_test_prefix_len, &full_bytes, sizeof(int));
    cudaMemcpyToSymbol(d_test_prefix_partial_bits, &partial_bits, sizeof(int));
    cudaMemcpyToSymbol(d_test_prefix_partial_mask, &partial_mask, sizeof(uint8_t));
    cudaMemcpyToSymbol(d_test_prefix_partial_value, &partial_value, sizeof(uint8_t));
}

// Helper to setup suffix params for tests
void setup_test_suffix_params(const char* suffix) {
    int start_offset, match_len;
    uint8_t targets[32] = {0};
    uint8_t masks[32] = {0};
    
    decode_suffix_pattern_uniform(suffix, 
                                  &start_offset, &match_len,
                                  targets, masks);
                                  

    
    cudaMemcpyToSymbol(d_test_suffix_start_offset, &start_offset, sizeof(int));
    cudaMemcpyToSymbol(d_test_suffix_len, &match_len, sizeof(int));
    cudaMemcpyToSymbol(d_test_suffix_targets, targets, 32);
    cudaMemcpyToSymbol(d_test_suffix_masks, masks, 32);
}

void run_matching_tests(TestResults* r) {
    printf("\n=== Prefix/Suffix Matching Tests ===\n");
    
    uint8_t *d_hash;
    int *d_result;
    int h_result;
    cudaMalloc(&d_hash, 32);
    cudaMalloc(&d_result, sizeof(int));
    
    // Test Case 1: Suffix 'A' (last 4 bits = 0000)
    // Hash ending in 0x00 should match, 0x01 should not
    uint8_t h_hash[32] = {0};
    h_hash[31] = 0x00; // ...0000 0000 (Last 4 bits 0000) -> Matches A
    
    setup_test_suffix_params("A");
    
    cudaMemcpy(d_hash, h_hash, 32, cudaMemcpyHostToDevice);
    test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_result) {
        printf("[PASS] Suffix 'A' matches hash ending in 0x00\n");
        r->passed++;
    } else {
        printf("[FAIL] Suffix 'A' did NOT match hash ending in 0x00\n");
        r->failed++;
    }
    r->total++;
    
    // Test Case 2: Suffix 'A' should NOT match 0x01
    h_hash[31] = 0x01; // ...0000 0001 (Last 4 bits 0001) -> Matches E/F... not A
    cudaMemcpy(d_hash, h_hash, 32, cudaMemcpyHostToDevice);
    test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (!h_result) {
        printf("[PASS] Suffix 'A' does not match hash ending in 0x01\n");
        r->passed++;
    } else {
        printf("[FAIL] Suffix 'A' INCORRECTLY matched hash ending in 0x01\n");
        r->failed++;
    }
    r->total++;
    
    // Test Case 3: Suffix '8' (last 4 bits = 1111)
    // Hash ending in 0x0F matches "8" (val=60=111100 -> top 4 bits 1111)
    h_hash[31] = 0x0F; // ...0000 1111 (Last 4 bits 1111)
    setup_test_suffix_params("8");
    
    cudaMemcpy(d_hash, h_hash, 32, cudaMemcpyHostToDevice);
    test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_result) {
        printf("[PASS] Suffix '8' matches hash ending in 0x0F\n");
        r->passed++;
    } else {
        printf("[FAIL] Suffix '8' did NOT match hash ending in 0x0F\n");
        r->failed++;
    }
    r->total++;
    
    // Test Case 4: Suffix 'AE' (12 bits: ...0000 0001)
    // Hash bits 244-255. 
    // Byte 30 (lower 2 bits of 'A'): 00. (Mask 0x03).
    // Byte 31: 
    //   Upper 4 bits (lower 4 bits of 'A'): 0000. (Mask 0xF0).
    //   Lower 4 bits (upper 4 bits of 'E'): 0001. (Mask 0x0F).
    // Total Byte 31: 0000 0001 (0x01).
    h_hash[30] = 0xFC; // xxxx xx00 (Last 2 bits 00)
    h_hash[31] = 0x01; // 0000 0001
    setup_test_suffix_params("AE");
    
    cudaMemcpy(d_hash, h_hash, 32, cudaMemcpyHostToDevice);
    test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_result) {
        printf("[PASS] Suffix 'AE' matches hash ending in ...0_01\n");
        r->passed++;
    } else {
        printf("[FAIL] Suffix 'AE' did NOT match hash ending in ...0_01\n");
        r->failed++;
    }
    r->total++;
    
    // Test Case 5: Suffix 'AE' negative test
    h_hash[31] = 0x11; // 0001 0001 (Last bit differs)
    cudaMemcpy(d_hash, h_hash, 32, cudaMemcpyHostToDevice);
    test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (!h_result) {
        printf("[PASS] Suffix 'AE' does not match hash ending in ...11\n");
        r->passed++;
    } else {
        printf("[FAIL] Suffix 'AE' INCORRECTLY matched hash ending in ...11\n");
        r->failed++;
    }
    r->total++;

    cudaFree(d_hash);
    cudaFree(d_result);
}

// === BATCH INVERSION TESTS ===
void run_batch_inversion_tests(TestVector* v, uint32_t n, TestResults* r) {
    printf("\n=== Batch Inversion Tests (Batch Size: %d) ===\n", TEST_BATCH_SIZE);
    
    // Test 1: Compare batch vs single for test vectors
    printf("\n[Test 1] Batch vs Single key generation (test vectors)\n");
    
    uint8_t (*d_seeds)[32];
    uint8_t (*d_batch)[32];
    uint8_t (*d_single)[32];
    uint8_t h_batch[TEST_BATCH_SIZE][32], h_single[TEST_BATCH_SIZE][32];
    
    cudaMalloc(&d_seeds, TEST_BATCH_SIZE * 32);
    cudaMalloc(&d_batch, TEST_BATCH_SIZE * 32);
    cudaMalloc(&d_single, TEST_BATCH_SIZE * 32);
    
    // Test with available vectors
    int batch_count = (n >= TEST_BATCH_SIZE) ? 1 : 0;
    int pass = 0, fail = 0;
    
    for (int batch = 0; batch < batch_count; batch++) {
        // Copy seeds to device
        uint8_t seeds[TEST_BATCH_SIZE][32];
        for (int i = 0; i < TEST_BATCH_SIZE; i++) {
            memcpy(seeds[i], v[batch * TEST_BATCH_SIZE + i].seed, 32);
        }
        cudaMemcpy(d_seeds, seeds, TEST_BATCH_SIZE * 32, cudaMemcpyHostToDevice);
        
        // Run batch vs single kernel
        test_batch_vs_single_kernel<<<1, 256>>>(d_seeds, d_batch, d_single);
        cudaDeviceSynchronize();
        
        cudaMemcpy(h_batch, d_batch, TEST_BATCH_SIZE * 32, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_single, d_single, TEST_BATCH_SIZE * 32, cudaMemcpyDeviceToHost);
        
        // Compare results
        for (int i = 0; i < TEST_BATCH_SIZE; i++) {
            r->total++;
            bool match = compare_bytes(h_batch[i], h_single[i], 32);
            bool expected_match = compare_bytes(h_batch[i], v[batch * TEST_BATCH_SIZE + i].public_key, 32);
            
            if (match && expected_match) {
                pass++;
                r->passed++;
            } else {
                fail++;
                r->failed++;
                printf("[FAIL] Batch vs Single mismatch for vector %d\n", batch * TEST_BATCH_SIZE + i + 1);
            }
        }
    }
    printf("Batch vs Single (test vectors): %d/%d passed\n", pass, (int)(batch_count * TEST_BATCH_SIZE));
    
    // Test 2: Edge case seeds
    printf("\n[Test 2] Edge case seeds\n");
    
    test_batch_edge_cases_kernel<<<1, 256>>>(d_batch, d_single);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_batch, d_batch, TEST_BATCH_SIZE * 32, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_single, d_single, TEST_BATCH_SIZE * 32, cudaMemcpyDeviceToHost);
    
    pass = 0; fail = 0;
    
    for (int i = 0; i < TEST_BATCH_SIZE; i++) {
        r->total++;
        bool match = compare_bytes(h_batch[i], h_single[i], 32);
        
        if (match) {
            pass++;
            r->passed++;
        } else {
            printf("[FAIL] Edge Case %d: Batch does NOT match Single\n", i);
            fail++;
            r->failed++;
        }
    }
    printf("Edge case tests: %d/%d passed\n", pass, TEST_BATCH_SIZE);
    
    cudaFree(d_seeds);
    cudaFree(d_batch);
    cudaFree(d_single);
}

int main() {
    log_file = fopen("test_log.txt", "w");
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024 * 1024 * 64); // Increase printf buffer
    
    printf("================================================\n");
    printf("CUDA Kernel Tests\n");
    printf("================================================\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    
    // Initialize Ed25519 precomputed tables
    cudaError_t err;
    void* ptr;
    
    err = cudaGetSymbolAddress(&ptr, d_base_5bit);
    if (err == cudaSuccess) {
        err = cudaMemcpy(ptr, base_5bit, sizeof(ge_precomp) * 52 * 16, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("Error copying d_base_5bit content: %s\n", cudaGetErrorString(err));
    } else printf("Error getting symbol d_base_5bit: %s\n", cudaGetErrorString(err));

    err = cudaGetSymbolAddress(&ptr, d_base_7bit);
    if (err == cudaSuccess) {
        err = cudaMemcpy(ptr, base_7bit, sizeof(ge_precomp) * 37 * 64, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("Error copying d_base_7bit content: %s\n", cudaGetErrorString(err));
    } else printf("Error getting symbol d_base_7bit: %s\n", cudaGetErrorString(err));

    err = cudaGetSymbolAddress(&ptr, d_base_8bit);
    if (err == cudaSuccess) {
        err = cudaMemcpy(ptr, base_8bit, sizeof(ge_precomp) * 32 * 128, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) printf("Error copying d_base_8bit content: %s\n", cudaGetErrorString(err));
    } else printf("Error getting symbol d_base_8bit: %s\n", cudaGetErrorString(err));

    
    // Debug Check
    debug_check_constants_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    uint32_t count;
    TestVector* vectors = load_test_vectors("test_vectors.bin", &count);
    if (!vectors) return 1;
    printf("Loaded %d test vectors\n", count);
    
    TestResults results = {0, 0, 0};
    
    run_sha512_tests(vectors, count, &results);
    run_ed25519_pubkey_tests(vectors, count, &results);
    run_ed25519_keypair_tests(vectors, count, &results);
    run_full_pipeline_tests(vectors, count, &results);
    run_matching_tests(&results);
    
    // New Isolated SHA-256 Tests
    uint32_t sha256_count;
    SHA256TestVector* sha256_vectors = load_sha256_vectors("sha256_test_vectors.bin", &sha256_count);
    if (sha256_vectors) {
        printf("\n=== Isolated SHA-256 Tests (%d vectors) ===\n", sha256_count);
        
        uint8_t *d_pub, *d_hash;
        uint8_t h_hash[32];
        cudaMalloc(&d_pub, 32);
        cudaMalloc(&d_hash, 32);
        
        int pass = 0, fail = 0;
        for (uint32_t i = 0; i < sha256_count; i++) {
            results.total++;
            cudaMemcpy(d_pub, sha256_vectors[i].pubkey, 32, cudaMemcpyHostToDevice);
            test_only_sha256_kernel<<<1, 1>>>(d_pub, d_hash); // Single thread enough for unit test
            cudaDeviceSynchronize();
            cudaMemcpy(h_hash, d_hash, 32, cudaMemcpyDeviceToHost);
            
            if (compare_bytes(h_hash, sha256_vectors[i].expected_hash, 32)) {
                pass++; results.passed++;
            } else {
                fail++; results.failed++;
                if (fail <= 3) {
                     printf("[%d] FAILED\n", i+1);
                     print_hex("  Expected", sha256_vectors[i].expected_hash, 32);
                     print_hex("  Got     ", h_hash, 32);
                }
            }
        }
        printf("Result: %d/%d passed\n", pass, sha256_count);
        cudaFree(d_pub);
        cudaFree(d_hash);
        free(sha256_vectors);
    }

    // New Prefix Matching Tests (from file)
    uint32_t prefix_count;
    MatchTestVector* prefix_vectors = load_match_vectors("prefix_test_vectors.bin", &prefix_count);
    if (prefix_vectors) {
        printf("\n=== Prefix Matching Tests (File-based, %d vectors) ===\n", prefix_count);
        
        uint8_t *d_hash;
        int *d_result;
        int h_result;
        cudaMalloc(&d_hash, 32);
        cudaMalloc(&d_result, sizeof(int));
        
        int pass = 0, fail = 0;
        for (uint32_t i = 0; i < prefix_count; i++) {
            results.total++;

            char safe_pattern[33] = {0};
            memcpy(safe_pattern, prefix_vectors[i].pattern, 32);
            
            setup_test_prefix_params(safe_pattern);
            
            cudaMemcpy(d_hash, prefix_vectors[i].hash, 32, cudaMemcpyHostToDevice);
            test_prefix_match_kernel<<<1, 1>>>(d_hash, d_result);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
            
            // Normalize result to 0 or 1
            int match = h_result ? 1 : 0;
            if (match == prefix_vectors[i].expected) {
                pass++; results.passed++;
            } else {
                fail++; results.failed++;
                if (fail <= 5) printf("[%d] FAILED: Pattern='%s', Expected=%d, Got=%d\n", i+1, safe_pattern, prefix_vectors[i].expected, match);
            }
        }
        printf("Result: %d/%d passed\n", pass, prefix_count);
        cudaFree(d_hash);
        cudaFree(d_result);
        free(prefix_vectors);
    }

    // New Suffix Matching Tests (from file)
    uint32_t suffix_count;
    MatchTestVector* suffix_vectors = load_match_vectors("suffix_test_vectors.bin", &suffix_count);
    if (suffix_vectors) {
        printf("\n=== Suffix Matching Tests (File-based, %d vectors) ===\n", suffix_count);
        
        uint8_t *d_hash;
        int *d_result;
        int h_result;
        cudaMalloc(&d_hash, 32);
        cudaMalloc(&d_result, sizeof(int));
        
        int pass = 0, fail = 0;
        for (uint32_t i = 0; i < suffix_count; i++) {
            results.total++;
            char safe_pattern[33] = {0};
            memcpy(safe_pattern, suffix_vectors[i].pattern, 32);
            
            setup_test_suffix_params(safe_pattern);
            
            cudaMemcpy(d_hash, suffix_vectors[i].hash, 32, cudaMemcpyHostToDevice);
            test_suffix_match_kernel<<<1, 1>>>(d_hash, d_result);
            cudaDeviceSynchronize();
            cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
            
            int match = h_result ? 1 : 0;
            if (match == suffix_vectors[i].expected) {
                pass++; results.passed++;
            } else {
                fail++; results.failed++;
                if (fail <= 5) printf("[%d] FAILED: Pattern='%s', Expected=%d, Got=%d\n", i+1, safe_pattern, suffix_vectors[i].expected, match);
            }
        }
        printf("Result: %d/%d passed\n", pass, suffix_count);
        cudaFree(d_hash);
        cudaFree(d_result);
        free(suffix_vectors);
    }

    run_batch_inversion_tests(vectors, count, &results);
    
    free(vectors);
    
    printf("\n================================================\n");
    printf("SUMMARY: %d/%d tests passed\n", results.passed, results.total);
    printf("================================================\n");
    
    if (results.failed > 0) {
        printf("SOME TESTS FAILED!\n");
        if (log_file) fclose(log_file);
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    
    if (log_file) fclose(log_file);
    return 0;
}
