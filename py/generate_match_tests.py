import struct
import base64
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def ssh_wire_format(pubkey: bytes) -> bytes:
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey

def main():
    num_vectors = 100
    print(f"Generating match_test_vectors.bin ({num_vectors} vectors)...")

    with open("match_test_vectors.bin", "wb") as f:
        f.write(struct.pack("<I", num_vectors))
        
        for i in range(num_vectors):
            # Deterministic seed generation
            seed = bytes([(i * 37 + j * 17 + 123) & 0xff for j in range(32)])
            
            # Ed25519 Key
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
            public_key = private_key.public_key()
            pub_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            wire = ssh_wire_format(pub_bytes)
            fingerprint_hash = hashlib.sha256(wire).digest()
            fingerprint_b64 = base64.b64encode(fingerprint_hash).decode('ascii').replace('=', '')
            
            f.write(seed)
            f.write(fingerprint_hash)
            
            # Pad to 44 bytes
            fp_bytes = fingerprint_b64.encode('ascii')
            f.write(fp_bytes.ljust(44, b'\x00'))
            
    print(f"Generated {num_vectors} test vectors")

if __name__ == "__main__":
    main()
