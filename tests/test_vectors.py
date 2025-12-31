import struct
import base64
import hashlib
import os
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


def ssh_wire_format(pubkey: bytes) -> bytes:
    """Generate SSH wire format for Ed25519 public key."""
    header = b'\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20'
    return header + pubkey


def load_test_vectors(filename: str):
    """Load test vectors from binary file."""
    if not os.path.exists(filename):
        pytest.skip(f"{filename} not found")
    
    with open(filename, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        vectors = []
        for _ in range(count):
            seed = f.read(32)
            sha512_hash = f.read(64)
            private_key = f.read(64)  # seed + pubkey
            public_key = f.read(32)
            fingerprint_hash = f.read(32)
            fingerprint_b64 = f.read(44).rstrip(b'\x00').decode('ascii')
            vectors.append({
                'seed': seed,
                'sha512_hash': sha512_hash,
                'private_key': private_key,
                'public_key': public_key,
                'fingerprint_hash': fingerprint_hash,
                'fingerprint_b64': fingerprint_b64,
            })
        return vectors


def load_sha256_vectors(filename: str):
    """Load SHA256 test vectors from binary file."""
    if not os.path.exists(filename):
        pytest.skip(f"{filename} not found")
    
    with open(filename, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        vectors = []
        for _ in range(count):
            pubkey = f.read(32)
            expected_hash = f.read(32)
            vectors.append({
                'pubkey': pubkey,
                'expected_hash': expected_hash,
            })
        return vectors


def load_match_vectors(filename: str):
    """Load match test vectors from binary file."""
    if not os.path.exists(filename):
        pytest.skip(f"{filename} not found")
    
    with open(filename, "rb") as f:
        count = struct.unpack("<I", f.read(4))[0]
        vectors = []
        for _ in range(count):
            seed = f.read(32)
            fingerprint_hash = f.read(32)
            fingerprint_b64 = f.read(44).rstrip(b'\x00').decode('ascii')
            vectors.append({
                'seed': seed,
                'fingerprint_hash': fingerprint_hash,
                'fingerprint_b64': fingerprint_b64,
            })
        return vectors


class TestPythonMatches:

    @pytest.fixture
    def test_vectors(self):
        return load_test_vectors("testdata/test_vectors.bin")
    
    @pytest.fixture
    def sha256_vectors(self):
        return load_sha256_vectors("testdata/sha256_test_vectors.bin")
    
    @pytest.fixture
    def match_vectors(self):
        return load_match_vectors("testdata/match_test_vectors.bin")
    
    def test_sha512_matches(self, test_vectors):
        for i, v in enumerate(test_vectors):
            py_hash = hashlib.sha512(v['seed']).digest()
            assert py_hash == v['sha512_hash'], f"SHA512 mismatch at vector {i}"
    
    def test_ed25519_pubkey_matches(self, test_vectors):
        for i, v in enumerate(test_vectors):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(v['seed'])
            public_key = private_key.public_key()
            py_pubkey = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            assert py_pubkey == v['public_key'], f"Pubkey mismatch at vector {i}"
    
    def test_private_key_format_matches(self, test_vectors):
        for i, v in enumerate(test_vectors):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(v['seed'])
            public_key = private_key.public_key()
            py_pubkey = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            py_privkey = v['seed'] + py_pubkey
            assert py_privkey == v['private_key'], f"Private key mismatch at vector {i}"
    
    def test_fingerprint_hash_matches(self, test_vectors):
        for i, v in enumerate(test_vectors):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(v['seed'])
            public_key = private_key.public_key()
            py_pubkey = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            wire = ssh_wire_format(py_pubkey)
            py_fp_hash = hashlib.sha256(wire).digest()
            assert py_fp_hash == v['fingerprint_hash'], f"Fingerprint hash mismatch at vector {i}"
    
    def test_fingerprint_b64_matches(self, test_vectors):
        for i, v in enumerate(test_vectors):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(v['seed'])
            public_key = private_key.public_key()
            py_pubkey = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            wire = ssh_wire_format(py_pubkey)
            py_fp_hash = hashlib.sha256(wire).digest()
            py_fp_b64 = base64.b64encode(py_fp_hash).decode('ascii').replace('=', '')
            assert py_fp_b64 == v['fingerprint_b64'], f"Fingerprint B64 mismatch at vector {i}"
    
    def test_sha256_isolated_matches(self, sha256_vectors):
        for i, v in enumerate(sha256_vectors):
            wire = ssh_wire_format(v['pubkey'])
            py_hash = hashlib.sha256(wire).digest()
            assert py_hash == v['expected_hash'], f"SHA256 mismatch at vector {i}"
    
    def test_match_vectors_fingerprint_matches(self, match_vectors):
        for i, v in enumerate(match_vectors):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(v['seed'])
            public_key = private_key.public_key()
            py_pubkey = public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            
            wire = ssh_wire_format(py_pubkey)
            py_fp_hash = hashlib.sha256(wire).digest()
            py_fp_b64 = base64.b64encode(py_fp_hash).decode('ascii').replace('=', '')
            
            assert py_fp_hash == v['fingerprint_hash'], f"Match vector {i} hash mismatch"
            assert py_fp_b64 == v['fingerprint_b64'], f"Match vector {i} B64 mismatch"
