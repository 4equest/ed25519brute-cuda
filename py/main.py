from __future__ import annotations

import os
import sys
import argparse
import base64
import hashlib
import multiprocessing
import time
from typing import TYPE_CHECKING
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized
    from multiprocessing.synchronize import Event as EventType

# Constants
ENCODED_PUBLIC_KEY_LEN = 44
FINGERPRINT_LEN = 43
KEYS_PER_BATCH = 0x10000  # 65536 keys per batch

# Shared counter for progress tracking
key_counter: Synchronized[int]

def ssh_wire_format(pubkey: bytes) -> bytes:
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey

def worker_job(
    target_akey: bytes | None,
    target_fp_prefix: bytes | None,
    target_fp_suffix: bytes | None,
    found_event: EventType,
    result_queue: multiprocessing.Queue[bytes],
    counter: Synchronized[int],
) -> None:
    """
    Worker process to brute force keys.
    """
    # Pre-calculate lengths for checks to avoid repeated len() calls
    akey_len = len(target_akey) if target_akey else 0
    fp_prefix_len = len(target_fp_prefix) if target_fp_prefix else 0
    fp_suffix_len = len(target_fp_suffix) if target_fp_suffix else 0
    
    seed_buf = bytearray(32)
    
    while not found_event.is_set():
        # Generate new random seed base
        seed_buf[:] = os.urandom(32)
        
        # Optimize: iterate 65536 times changing only first 2 bytes
        for i in range(0x10000):
            if found_event.is_set():
                return

            seed_buf[0] = i & 0xff
            seed_buf[1] = (i >> 8) & 0xff
            
            # Generate Key
            try:
                private_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes(seed_buf))
            except Exception:
                continue

            public_key = private_key.public_key()
            pub_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            # Check Authorized Key Suffix
            if akey_len > 0:
                assert target_akey is not None
                input_bytes = bytes([seed_buf[31]]) + pub_bytes
                b64_key = base64.b64encode(input_bytes).decode('ascii')
                
                if not b64_key.endswith(target_akey.decode('ascii')):
                    if fp_prefix_len == 0 and fp_suffix_len == 0:
                        continue
            
            # Check Fingerprint
            if fp_prefix_len > 0 or fp_suffix_len > 0:
                wire = ssh_wire_format(pub_bytes)
                fp_hash = hashlib.sha256(wire).digest()
                fp_b64 = base64.b64encode(fp_hash).decode('ascii').replace('=', '')
                
                match = True
                if fp_prefix_len > 0:
                    assert target_fp_prefix is not None
                    if not fp_b64.startswith(target_fp_prefix.decode('ascii')):
                        match = False
                
                if match and fp_suffix_len > 0:
                    assert target_fp_suffix is not None
                    if not fp_b64.endswith(target_fp_suffix.decode('ascii')):
                        match = False
                        
                if match:
                    if akey_len > 0:
                        assert target_akey is not None
                        input_bytes = bytes([seed_buf[31]]) + pub_bytes
                        b64_key = base64.b64encode(input_bytes).decode('ascii')
                        if not b64_key.endswith(target_akey.decode('ascii')):
                            continue

                    # FOUND!
                    result_queue.put(bytes(seed_buf))
                    found_event.set()
                    return
            elif akey_len > 0:
                result_queue.put(bytes(seed_buf))
                found_event.set()
                return
        
        # Update shared counter after each batch
        with counter.get_lock():
            counter.value += KEYS_PER_BATCH

def main() -> None:
    parser = argparse.ArgumentParser(description="Ed25519 CPU Brute Forcer (Python Port)")
    _ = parser.add_argument("--authorized-key-suffix", type=str, default="", help="Suffix of the authorized key (base64)")
    _ = parser.add_argument("--fingerprint-prefix", type=str, default="", help="Prefix of the fingerprint (base64)")
    _ = parser.add_argument("--fingerprint-suffix", type=str, default="", help="Suffix of the fingerprint (base64)")
    
    args = parser.parse_args()
    
    akey_suffix: bytes = args.authorized_key_suffix.encode('ascii')
    fp_prefix: bytes = args.fingerprint_prefix.encode('ascii')
    fp_suffix: bytes = args.fingerprint_suffix.encode('ascii')
    
    if not akey_suffix and not fp_prefix and not fp_suffix:
        parser.print_help()
        sys.exit(1)
        
    # Validate fingerprint suffix last char
    if fp_suffix:
        valid_last = b"AEIMQUYcgkosw048"
        if fp_suffix[-1] not in valid_last:
            print("Error: the last character of fingerprint suffix must be one of \"AEIMQUYcgkosw048\"")
            sys.exit(1)

    # Print header
    print("SSH Key Fingerprint CPU Brute Force (Python)")
    print("=" * 45)
    if akey_suffix:
        print(f"Searching for authorized-key suffix: {akey_suffix.decode('ascii')}")
    if fp_prefix:
        print(f"Searching for fingerprint prefix: {fp_prefix.decode('ascii')}")
    if fp_suffix:
        print(f"Searching for fingerprint suffix: {fp_suffix.decode('ascii')}")
    
    num_cpu = os.cpu_count() or 1
    print(f"Using {num_cpu} CPU cores")
    print("Starting search...")
    print()
    
    found_event: EventType = multiprocessing.Event()
    result_queue: multiprocessing.Queue[bytes] = multiprocessing.Queue()
    counter: Synchronized[int] = multiprocessing.Value('Q', 0)  # unsigned long long
    workers: list[multiprocessing.Process] = []
    
    start_time = time.time()
    last_report_time = start_time
    last_report_keys = 0
    
    for _ in range(num_cpu):
        p = multiprocessing.Process(
            target=worker_job,
            args=(akey_suffix, fp_prefix, fp_suffix, found_event, result_queue, counter)
        )
        p.start()
        workers.append(p)
        
    try:
        # Progress monitoring loop
        while not found_event.is_set():
            time.sleep(0.5)
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            with counter.get_lock():
                total_keys = counter.value
            
            if current_time - last_report_time >= 1.0:
                keys_since_last = total_keys - last_report_keys
                time_since_last = current_time - last_report_time
                
                if time_since_last > 0:
                    hashrate = keys_since_last / time_since_last / 1000.0  # KKeys/s
                    
                    print(
                        f"\rChecked: {total_keys / 1_000_000:.1f} M keys | "
                        f"Speed: {hashrate:.2f} KKeys/s | "
                        f"Time: {elapsed:.1f}s",
                        end="",
                        flush=True,
                    )
                
                last_report_time = current_time
                last_report_keys = total_keys
            
            # Check if result is available
            if not result_queue.empty():
                break
        
        # Get result
        seed = result_queue.get(timeout=1)
        found_event.set()
        
        end_time = time.time()
        total_time = end_time - start_time
        with counter.get_lock():
            total_keys = counter.value
        
        # Terminate workers
        for p in workers:
            p.terminate()
            
        # Generate key info
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        public_key = private_key.public_key()
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        
        wire = ssh_wire_format(pub_bytes)
        fp_hash = hashlib.sha256(wire).digest()
        fp_b64 = base64.b64encode(fp_hash).decode('ascii').replace('=', '')
        
        # Print results
        print()
        print()
        print("=" * 45)
        print(f"Match found after checking ~{total_keys:,} keys!")
        print(f"Time elapsed: {total_time:.2f}s")
        print("=" * 45)
        print()
        
        print(f"Seed (32 bytes):        {seed.hex()}")
        print(f"Private Key (64 bytes): {seed.hex()}{pub_bytes.hex()}")
        print(f"Public Key (32 bytes):  {pub_bytes.hex()}")
        print(f"Fingerprint:            SHA256:{fp_b64}")
        print()
        
        # Write output files
        priv_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        pub_ssh = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH
        )
        
        with open("id_ed25519", "wb") as f:
            _ = f.write(priv_pem)
            
        with open("id_ed25519.pub", "wb") as f:
            _ = f.write(pub_ssh)
        
        print("Private key written to: id_ed25519")
        print("Public key written to:  id_ed25519.pub")
            
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        found_event.set()
        for p in workers:
            p.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        found_event.set()
        for p in workers:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows
    main()
