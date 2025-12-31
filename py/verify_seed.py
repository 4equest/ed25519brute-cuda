import sys
import base64
import hashlib
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def ssh_wire_format(pubkey: bytes) -> bytes:
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey

def verify_seed(seed_hex: str):
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError:
        print(f"Error decoding hex: {seed_hex}")
        return

    if len(seed) != 32:
        print(f"Seed must be 32 bytes (got {len(seed)})")
        return

    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key()
    pub_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    
    wire = ssh_wire_format(pub_bytes)
    hash_val = hashlib.sha256(wire).digest()
    fp_b64 = base64.b64encode(hash_val).decode('ascii').replace('=', '')
    
    print(f"Seed: {seed.hex()}")
    print(f"Fingerprint: {fp_b64}")
    print(f"Hash bytes (hex): {hash_val.hex()}")
    
    last_byte = hash_val[31]
    print(f"Last byte: 0x{last_byte:02x}")
    print(f"Last 4 bits: {last_byte & 0x0F:04b} ({last_byte & 0x0F})")
    
    match_msg = "DOES NOT MATCH suffix 'A'"
    if (last_byte & 0x0F) == 0:
        match_msg = "MATCHES suffix 'A'"
    print(match_msg)
    
    with open("verify_seed_log.txt", "w") as f:
        f.write(f"Seed: {seed.hex()}\n")
        f.write(f"Fingerprint: {fp_b64}\n")
        f.write(f"Hash bytes (hex): {hash_val.hex()}\n")
        f.write(f"Last byte: 0x{last_byte:02x}\n")
        f.write(f"Last 4 bits: {last_byte & 0x0F:04b} ({last_byte & 0x0F})\n")
        f.write(f"{match_msg}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_seed.py <seed_hex>")
    else:
        verify_seed(sys.argv[1])
