import struct
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def main():
    seed = b'\x00' * 32
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key()
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    wire_format = header + pub_bytes
    
    print(f"SSH Wire Format ({len(wire_format)} bytes):")
    print(f"Header ({len(header)} bytes):")
    for i, b in enumerate(header):
        print(f"{b:02x} ", end='')
        if (i + 1) % 4 == 0:
            print(f"  // bytes {i-3}-{i}")
            
    print(f"\nPublic Key ({len(pub_bytes)} bytes):")
    for i, b in enumerate(pub_bytes):
        print(f"{b:02x} ", end='')
        if (i + 1) % 4 == 0:
            print(f"  // pubkey[{i-3}-{i}]")
            
    print(f"\nFull Wire Format by 32-bit words (big-endian):")
    for i in range(0, len(wire_format), 4):
        chunk = wire_format[i:i+4]
        if len(chunk) == 4:
            w = struct.unpack(">I", chunk)[0]
            byte_str = " ".join(f"{b:02x}" for b in chunk)
            print(f"w[{i//4:2d}] = 0x{w:08x}  // bytes {i}-{i+3}: {byte_str}")
        else:
            byte_str = " ".join(f"{b:02x}" for b in chunk)
            print(f"Remaining bytes {i}-{len(wire_format)-1}: {byte_str}")
            
    print(f"\nAfter message (51 bytes), padding starts at byte 51:")
    print(f"w[12] should have pubkey[31] at position 0, then 0x80 at position 1")
    print(f"pubkey[31] = 0x{pub_bytes[31]:02x}")

if __name__ == "__main__":
    main()
