#ifndef OPENSSH_FORMAT_H
#define OPENSSH_FORMAT_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

// OpenSSH Key Format Constants
static const char* AUTH_MAGIC = "openssh-key-v1";
static const char* CIPHER_NONE = "none";
static const char* KDF_NONE = "none";
static const char* KEY_TYPE = "ssh-ed25519";

// Base64 encoding table
static const char B64_TABLE[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Write uint32 in big-endian format
inline void write_uint32_be(uint8_t* buf, uint32_t val) {
    buf[0] = (val >> 24) & 0xFF;
    buf[1] = (val >> 16) & 0xFF;
    buf[2] = (val >> 8) & 0xFF;
    buf[3] = val & 0xFF;
}

// Write SSH string (4-byte length + data)
inline size_t write_ssh_string(uint8_t* buf, const void* data, size_t len) {
    write_uint32_be(buf, (uint32_t)len);
    memcpy(buf + 4, data, len);
    return 4 + len;
}

// Base64 encode with line wrapping (70 chars per line)
inline void base64_encode_pem(const uint8_t* data, size_t len, FILE* f) {
    size_t line_pos = 0;
    size_t i = 0;
    
    while (i < len) {
        uint32_t a = (i < len) ? data[i++] : 0;
        uint32_t b = (i < len) ? data[i++] : 0;
        uint32_t c = (i < len) ? data[i++] : 0;
        
        uint32_t triple = (a << 16) | (b << 8) | c;
        
        size_t remaining = len - (i - 3);
        
        fputc(B64_TABLE[(triple >> 18) & 0x3F], f);
        fputc(B64_TABLE[(triple >> 12) & 0x3F], f);
        
        if (remaining > 1) {
            fputc(B64_TABLE[(triple >> 6) & 0x3F], f);
        } else {
            fputc('=', f);
        }
        
        if (remaining > 2) {
            fputc(B64_TABLE[triple & 0x3F], f);
        } else {
            fputc('=', f);
        }
        
        line_pos += 4;
        if (line_pos >= 70) {
            fputc('\n', f);
            line_pos = 0;
        }
    }
    
    if (line_pos > 0) {
        fputc('\n', f);
    }
}

// Base64 encode to string (no line wrapping, no padding for .pub)
inline void base64_encode_string(const uint8_t* data, size_t len, char* out) {
    size_t i = 0;
    size_t j = 0;
    
    while (i < len) {
        uint32_t a = (i < len) ? data[i++] : 0;
        uint32_t b = (i < len) ? data[i++] : 0;
        uint32_t c = (i < len) ? data[i++] : 0;
        
        uint32_t triple = (a << 16) | (b << 8) | c;
        
        out[j++] = B64_TABLE[(triple >> 18) & 0x3F];
        out[j++] = B64_TABLE[(triple >> 12) & 0x3F];
        out[j++] = B64_TABLE[(triple >> 6) & 0x3F];
        out[j++] = B64_TABLE[triple & 0x3F];
    }
    
    // Add padding
    size_t pad = (3 - (len % 3)) % 3;
    if (pad >= 1) out[j - 1] = '=';
    if (pad >= 2) out[j - 2] = '=';
    
    out[j] = '\0';
}

// Generate OpenSSH public key blob
// Returns size of blob
inline size_t generate_pubkey_blob(const uint8_t* pubkey, uint8_t* blob) {
    size_t offset = 0;
    
    // Key type: "ssh-ed25519"
    offset += write_ssh_string(blob + offset, KEY_TYPE, strlen(KEY_TYPE));
    
    // Public key (32 bytes)
    offset += write_ssh_string(blob + offset, pubkey, 32);
    
    return offset; // Should be 51 bytes
}

// Write OpenSSH public key file (.pub format)
inline bool write_openssh_pubkey(const uint8_t* pubkey, const char* filename, const char* comment = nullptr) {
    FILE* f = fopen(filename, "w");
    if (!f) return false;
    
    // Generate public key blob
    uint8_t blob[64];
    size_t blob_len = generate_pubkey_blob(pubkey, blob);
    
    // Base64 encode (4 * ceil(51/3) = 68 chars)
    char b64[128];
    base64_encode_string(blob, blob_len, b64);
    
    // Write: ssh-ed25519 <base64> [comment]
    fprintf(f, "ssh-ed25519 %s", b64);
    if (comment && comment[0]) {
        fprintf(f, " %s", comment);
    }
    fprintf(f, "\n");
    
    fclose(f);
    return true;
}

// Write OpenSSH private key file (PEM format)
inline bool write_openssh_privkey(const uint8_t* seed, const uint8_t* pubkey, 
                                  const char* filename, const char* comment = nullptr) {
    FILE* f = fopen(filename, "w");
    if (!f) return false;
    
    // Build the binary structure
    uint8_t buffer[512];
    size_t offset = 0;
    
    // 1. AUTH_MAGIC (15 bytes with null terminator)
    memcpy(buffer + offset, AUTH_MAGIC, 15);
    offset += 15;
    
    // 2. ciphername: "none"
    offset += write_ssh_string(buffer + offset, CIPHER_NONE, 4);
    
    // 3. kdfname: "none"
    offset += write_ssh_string(buffer + offset, KDF_NONE, 4);
    
    // 4. kdfoptions: empty string
    write_uint32_be(buffer + offset, 0);
    offset += 4;
    
    // 5. number of keys: 1
    write_uint32_be(buffer + offset, 1);
    offset += 4;
    
    // 6. public key blob (length-prefixed)
    uint8_t pubkey_blob[64];
    size_t pubkey_blob_len = generate_pubkey_blob(pubkey, pubkey_blob);
    offset += write_ssh_string(buffer + offset, pubkey_blob, pubkey_blob_len);
    
    // 7. Private key section (encrypted section, but we use "none" cipher)
    // Build private section first to calculate length
    uint8_t priv_section[256];
    size_t priv_offset = 0;
    
    // Generate random checkint
    std::random_device rd;
    uint32_t checkint = ((uint32_t)rd() << 16) | (uint32_t)rd();
    
    // checkint (repeated twice)
    write_uint32_be(priv_section + priv_offset, checkint);
    priv_offset += 4;
    write_uint32_be(priv_section + priv_offset, checkint);
    priv_offset += 4;
    
    // Key type: "ssh-ed25519"
    priv_offset += write_ssh_string(priv_section + priv_offset, KEY_TYPE, strlen(KEY_TYPE));
    
    // Public key (32 bytes)
    priv_offset += write_ssh_string(priv_section + priv_offset, pubkey, 32);
    
    // Private key: seed (32 bytes) || pubkey (32 bytes) = 64 bytes
    uint8_t privkey[64];
    memcpy(privkey, seed, 32);
    memcpy(privkey + 32, pubkey, 32);
    priv_offset += write_ssh_string(priv_section + priv_offset, privkey, 64);
    
    // Comment
    const char* cmt = comment ? comment : "";
    priv_offset += write_ssh_string(priv_section + priv_offset, cmt, strlen(cmt));
    
    // Padding: 1, 2, 3, ... until block size (8) aligned
    size_t padded_len = (priv_offset + 7) & ~7;
    for (size_t i = priv_offset, p = 1; i < padded_len; i++, p++) {
        priv_section[i] = (uint8_t)p;
    }
    priv_offset = padded_len;
    
    // Write private section with length prefix
    offset += write_ssh_string(buffer + offset, priv_section, priv_offset);
    
    // Write PEM format
    fprintf(f, "-----BEGIN OPENSSH PRIVATE KEY-----\n");
    base64_encode_pem(buffer, offset, f);
    fprintf(f, "-----END OPENSSH PRIVATE KEY-----\n");
    
    fclose(f);
    return true;
}

// Combined function to write both keys
inline bool write_openssh_keys(const uint8_t* seed, const uint8_t* pubkey,
                               const char* privkey_file, const char* pubkey_file,
                               const char* comment = nullptr) {
    bool ok = true;
    ok = ok && write_openssh_privkey(seed, pubkey, privkey_file, comment);
    ok = ok && write_openssh_pubkey(pubkey, pubkey_file, comment);
    return ok;
}

#endif // OPENSSH_FORMAT_H
