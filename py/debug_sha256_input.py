import struct
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def ssh_wire_format(pubkey: bytes) -> bytes:
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey

def main():
    seed = b'\x00' * 32
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key()
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    wire_format = ssh_wire_format(pub_bytes)
    
    print("=== Wire Format Details ===")
    print(f"Total length: {len(wire_format)} bytes\n")
    
    print("Raw bytes:")
    for i, b in enumerate(wire_format):
        print(f"{b:02x} ", end='')
        if (i + 1) % 16 == 0:
            print()
    print("\n")
    
    print("As 32-bit big-endian words (SHA256 w[] format):")
    padded_len = (len(wire_format) + 3) // 4 * 4
    padded = wire_format.ljust(padded_len, b'\x00')
    
    for i in range(0, padded_len, 4):
        w = struct.unpack(">I", padded[i:i+4])[0]
        end_idx = min(i+4, len(wire_format))
        byte_str = " ".join(f"{b:02x}" for b in wire_format[i:end_idx])
        print(f"w[{i//4:2d}] = 0x{w:08x}  // bytes {i:2d}-{i+3:2d}: {byte_str}")
        
    print("\n=== SHA256 Block (64 bytes with padding) ===")
    block = bytearray(64)
    block[:len(wire_format)] = wire_format
    block[51] = 0x80
    
    # Length in bits at the end (big-endian 64-bit)
    bit_len = 51 * 8
    struct.pack_into(">Q", block, 56, bit_len)
    
    print("Full block hex:")
    for i, b in enumerate(block):
        print(f"{b:02x} ", end='')
        if (i + 1) % 16 == 0:
            print()
    print()
    
    print("\nAs 32-bit words:")
    for i in range(0, 64, 4):
        w = struct.unpack(">I", block[i:i+4])[0]
        print(f"w[{i//4:2d}] = 0x{w:08x}")
        
    print("\n=== Expected SHA256 Fingerprint ===")
    hash_val = hashlib.sha256(wire_format).digest()
    for i, b in enumerate(hash_val):
        print(f"{b:02x} ", end='')
        if (i + 1) % 16 == 0:
            print()
    print()

if __name__ == "__main__":
    main()
