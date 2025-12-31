import os
import base64
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def ssh_wire_format(pubkey: bytes) -> bytes:
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey

def main():
    print("Looking for fingerprint ending with 'A' (last 4 hash bits = 0000)...")
    
    for i in range(10000):
        # Deterministic generation for reproducibility
        seed = bytes([(i * 37 + j * 17 + 123) & 0xff for j in range(32)])
        
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        wire = ssh_wire_format(pub_bytes)
        hash_val = hashlib.sha256(wire).digest()
        fp_b64 = base64.b64encode(hash_val).decode('ascii').replace('=', '')
        
        if fp_b64.endswith('A'):
            print(f"Found at i={i}!")
            print(f"Fingerprint: SHA256:{fp_b64}")
            print(f"Hash hex: {hash_val.hex()}")
            
            last_byte = hash_val[31]
            print(f"Last byte: 0x{last_byte:02x} (binary: {last_byte:08b})")
            print(f"Last 4 bits: {last_byte & 0x0F:04b} = {last_byte & 0x0F}")
            print()
            
            print(f"Seed hex: {seed.hex()}")
            break
            
    print("\n=== Suffix 'A' Analysis ===")
    print("Base64 'A' = 0 (binary: 000000)")
    print("In a 43-char fingerprint, last char covers bits 252-257")
    print("- Bits 252-255 = last 4 bits of hash (byte 31, lower nibble)")
    print("- Bits 256-257 = padding zeros (always 00)")
    print("")
    print("So 'A' (000000) means:")
    print("- Last 4 hash bits = 0000")
    print("- hash[31] & 0x0F == 0")

if __name__ == "__main__":
    main()
