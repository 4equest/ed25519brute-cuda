#!/usr/bin/env python3
"""
Comprehensive tests for gen_conditions.py

Tests cover:
- Base64 decoding accuracy
- Endianness (big-endian matching SHA256 state)
- Single and multiple pattern handling
- Prefix and suffix boundary cases
- Edge cases (1 char, max length, special Base64 chars)
- Validation (invalid last char for suffix)
- Grouping by length
- Multi-word variations
"""

import unittest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gen_conditions import (
    decode_base64_to_bits,
    bits_to_bytes,
    bytes_to_uint32_be,
    compute_prefix_pattern,
    compute_suffix_pattern,
    generate_prefix_condition,
    generate_suffix_condition,
    generate_single_prefix_condition,
    generate_single_suffix_condition,
    generate_suffix_group_condition,
    validate_suffix_last_char,
    B64_DECODE,
    VALID_SUFFIX_LAST_CHARS
)


class TestBase64Decoding(unittest.TestCase):
    """Test Base64 decoding matches C++ implementation."""
    
    def test_decode_table_size(self):
        """Verify all 64 Base64 characters are mapped."""
        self.assertEqual(len(B64_DECODE), 64)
    
    def test_decode_A(self):
        """'A' should decode to 0 (000000)."""
        bits = decode_base64_to_bits("A")
        self.assertEqual(bits, [0, 0, 0, 0, 0, 0])
    
    def test_decode_slash(self):
        """'/' should decode to 63 (111111)."""
        bits = decode_base64_to_bits("/")
        self.assertEqual(bits, [1, 1, 1, 1, 1, 1])
    
    def test_decode_z(self):
        """'z' should decode to 51."""
        bits = decode_base64_to_bits("z")
        expected = [1, 1, 0, 0, 1, 1]  # 51 = 110011
        self.assertEqual(bits, expected)
    
    def test_decode_4eq(self):
        """'4eq' should decode to specific bit pattern."""
        bits = decode_base64_to_bits("4eq")
        # '4' = 56 = 111000
        # 'e' = 30 = 011110
        # 'q' = 42 = 101010
        expected = [1,1,1,0,0,0, 0,1,1,1,1,0, 1,0,1,0,1,0]
        self.assertEqual(bits, expected)
    
    def test_invalid_char(self):
        """Invalid Base64 character should raise ValueError."""
        with self.assertRaises(ValueError):
            decode_base64_to_bits("A=")  # '=' is not in standard Base64


class TestBitConversions(unittest.TestCase):
    """Test bit-to-byte and byte-to-uint32 conversions."""
    
    def test_bits_to_bytes_simple(self):
        """8 zero bits should produce [0]."""
        bits = [0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(bits_to_bytes(bits), [0])
    
    def test_bits_to_bytes_all_ones(self):
        """8 one bits should produce [255]."""
        bits = [1, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(bits_to_bytes(bits), [255])
    
    def test_bits_to_bytes_msb_first(self):
        """Verify MSB-first ordering: 10000000 = 128."""
        bits = [1, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(bits_to_bytes(bits), [128])
    
    def test_bytes_to_uint32_be(self):
        """Verify big-endian uint32 conversion."""
        bytes_list = [0xE1, 0xEA, 0x80, 0x00]
        result = bytes_to_uint32_be(bytes_list)
        self.assertEqual(result[0], 0xE1EA8000)
    
    def test_bytes_to_uint32_be_multiple_words(self):
        """Verify multi-word conversion."""
        bytes_list = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        result = bytes_to_uint32_be(bytes_list)
        self.assertEqual(result[0], 0x01020304)
        self.assertEqual(result[1], 0x05060708)


class TestSuffixValidation(unittest.TestCase):
    """Test suffix last character validation."""
    
    def test_valid_last_chars_accepted(self):
        """All valid last characters should be accepted."""
        for char in VALID_SUFFIX_LAST_CHARS:
            try:
                validate_suffix_last_char(char)
            except ValueError:
                self.fail(f"Valid char '{char}' was rejected")
    
    def test_invalid_last_char_rejected(self):
        """Invalid last characters should raise ValueError."""
        invalid_chars = "BDFHJLNPRTVXZbdfhjlnprtv1235679+/"
        for char in invalid_chars:
            with self.assertRaises(ValueError, msg=f"Char '{char}' should be rejected"):
                validate_suffix_last_char(char)
    
    def test_empty_pattern_valid(self):
        """Empty pattern should pass validation."""
        validate_suffix_last_char("")  # Should not raise


class TestPrefixPatterns(unittest.TestCase):
    """Test prefix pattern decoding and condition generation."""
    
    def test_single_char_prefix_A(self):
        """Prefix 'A' = 6 bits of zeros."""
        targets, masks = compute_prefix_pattern("A")
        # 6 bits = partial word only, mask covers top 6 bits: 0xFC000000
        self.assertEqual(masks[0], 0xFC000000)
        self.assertEqual(targets[0] & masks[0], 0x00000000)
    
    def test_prefix_AAAA(self):
        """Prefix 'AAAA' = 24 bits of zeros."""
        targets, masks = compute_prefix_pattern("AAAA")
        # 24 bits = mask 0xFFFFFF00
        self.assertEqual(masks[0], 0xFFFFFF00)
        self.assertEqual(targets[0] & masks[0], 0x00000000)
    
    def test_prefix_4eq(self):
        """Prefix '4eq' = 18 bits."""
        targets, masks = compute_prefix_pattern("4eq")
        # '4eq' bits: 111000 011110 101010 = 0xE1EA80 (top 18 bits)
        # In 32-bit word: 0xE1EA8000 with mask 0xFFFFC000 (18 bits)
        self.assertEqual(masks[0], 0xFFFFC000)
        self.assertEqual(targets[0] & masks[0], 0xE1EA8000)
    
    def test_prefix_full_word(self):
        """Prefix with 32+ bits should have full word mask."""
        # 6 chars = 36 bits = 1 full word + 4 bits
        targets, masks = compute_prefix_pattern("AAAAAA")
        self.assertEqual(masks[0], 0xFFFFFFFF)  # Full word
        self.assertEqual(masks[1], 0xF0000000)  # 4 bits
    
    def test_generate_single_prefix(self):
        """Generated prefix code should contain correct hex values."""
        targets, masks = compute_prefix_pattern("4eq")
        code = generate_single_prefix_condition(targets, masks)
        self.assertIn("0xE1EA8000U", code)
        self.assertIn("0xFFFFC000U", code)


class TestSuffixPatterns(unittest.TestCase):
    """Test suffix pattern decoding and condition generation."""
    
    def test_single_char_suffix_A(self):
        """Suffix 'A' = last 4 bits of hash."""
        targets, masks, start_word, word_count = compute_suffix_pattern("A")
        # Should only affect word 7 (last word)
        self.assertEqual(start_word, 7)
        self.assertEqual(word_count, 1)
        # 'A' = 000000, last 4 bits used, mask = 0x0000000F
        self.assertEqual(masks[7], 0x0000000F)
        self.assertEqual(targets[7] & masks[7], 0x00000000)
    
    def test_suffix_AAAA(self):
        """Suffix 'AAAA' = 24 bits pattern, 22 in hash."""
        targets, masks, start_word, word_count = compute_suffix_pattern("AAAA")
        self.assertEqual(start_word, 7)
        self.assertEqual(word_count, 1)
        # mask = 0x003FFFFF (22 bits)
        self.assertEqual(masks[7], 0x003FFFFF)
    
    def test_suffix_long_pattern(self):
        """Long suffix should span multiple words."""
        # 10 chars = 60 bits, all in hash (starts at bit 198)
        targets, masks, start_word, word_count = compute_suffix_pattern("AAAAAAAAAA")
        self.assertEqual(start_word, 6)
        self.assertGreaterEqual(word_count, 2)
    
    def test_generate_single_suffix(self):
        """Generated suffix code should contain correct hex values."""
        targets, masks, start_word, word_count = compute_suffix_pattern("AAAA")
        code = generate_single_suffix_condition(targets, masks, start_word, word_count)
        self.assertIn("hash[7]", code)
        self.assertIn("0x003FFFFFU", code)


class TestMultiplePatterns(unittest.TestCase):
    """Test multiple pattern optimization."""
    
    def test_multiple_suffix_same_length_or(self):
        """Multiple suffixes of same length should generate OR condition."""
        code = generate_suffix_condition(["AAA0", "AAA4"])
        self.assertIn("||", code)
        # Both should have different target values
        self.assertIn("0x0000000DU", code)
        self.assertIn("0x0000000EU", code)
    
    def test_multiple_suffix_different_length(self):
        """Different length suffixes should generate grouped OR."""
        code = generate_suffix_condition(["A", "AAAA"])
        self.assertIn("||", code)
        # Both mask sizes should appear
        self.assertIn("0x0000000FU", code)
        self.assertIn("0x003FFFFFU", code)
    
    def test_multiple_prefix_common_optimization(self):
        """Multiple prefixes with common start should optimize."""
        code = generate_prefix_condition(["AAB", "AAC"])
        # Should have OR condition for the varying part
        self.assertIn("||", code)


class TestMultiWordVariation(unittest.TestCase):
    """Test handling of patterns that vary across multiple words."""
    
    def test_long_suffix_multi_word(self):
        """Long suffix spanning multiple words should generate correct code."""
        # Use patterns that differ in bits across word boundary
        # 12 chars = 72 bits, starts at bit 186, ends at 256
        # Bit 186 is in word 5 (186/32 = 5.8)
        targets, masks, start_word, word_count = compute_suffix_pattern("AAAAAAAAAAAA")
        self.assertEqual(start_word, 5)
        self.assertEqual(word_count, 3)
    
    def test_suffix_word_boundary_crossing(self):
        """Suffix that crosses word boundaries should check multiple words."""
        # 7 chars = 42 bits, starts at bit 216
        targets, masks, start_word, word_count = compute_suffix_pattern("AAAAAAA")
        self.assertEqual(start_word, 6)
        self.assertEqual(word_count, 2)
        # Both words should have non-zero masks
        self.assertNotEqual(masks[6], 0)
        self.assertNotEqual(masks[7], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_max_length_suffix(self):
        """Maximum practical suffix length."""
        pattern = "A" * 42  # Close to full fingerprint
        try:
            targets, masks, start_word, word_count = compute_suffix_pattern(pattern)
            self.assertGreater(word_count, 0)
        except Exception as e:
            self.fail(f"Max length suffix failed: {e}")
    
    def test_special_base64_chars(self):
        """Test + and / characters."""
        bits_plus = decode_base64_to_bits("+")
        bits_slash = decode_base64_to_bits("/")
        # '+' = 62, '/' = 63
        self.assertEqual(bits_plus, [1, 1, 1, 1, 1, 0])  # 62
        self.assertEqual(bits_slash, [1, 1, 1, 1, 1, 1])  # 63
    
    def test_empty_prefix_list(self):
        """Empty prefix list should return 'return true'."""
        code = generate_prefix_condition([])
        self.assertIn("return true", code)
    
    def test_empty_suffix_list(self):
        """Empty suffix list should return 'return true'."""
        code = generate_suffix_condition([])
        self.assertIn("return true", code)


class TestEndiannessConsistency(unittest.TestCase):
    """Test that endianness matches C++ SHA256 implementation."""
    
    def test_sha256_state_byte_order(self):
        """SHA256 state is big-endian: byte 0 is MSB of word 0."""
        # For prefix "AA", first 12 bits should be zeros
        targets, masks = compute_prefix_pattern("AA")
        # 12 bits = mask 0xFFF00000
        self.assertEqual(masks[0], 0xFFF00000)
        self.assertEqual(targets[0] & masks[0], 0x00000000)
    
    def test_suffix_bit_alignment(self):
        """Suffix bits should align correctly with hash end."""
        # Suffix "A" should check last 4 bits of hash (bits 252-255)
        targets, masks, start_word, word_count = compute_suffix_pattern("A")
        # Mask should be 0x0000000F (lower 4 bits of last word)
        self.assertEqual(masks[7], 0x0000000F)


class TestCodeGenerationCorrectness(unittest.TestCase):
    """Test that generated code is syntactically correct."""
    
    def test_generated_code_has_return(self):
        """All generated conditions should have return statement."""
        for pattern in ["A", "AAAA", "4eq"]:
            code = generate_prefix_condition([pattern])
            self.assertIn("return", code)
            
            if pattern[-1] in VALID_SUFFIX_LAST_CHARS:
                code = generate_suffix_condition([pattern])
                self.assertIn("return", code)
    
    def test_no_dangling_operators(self):
        """Generated code should not have dangling || or &&."""
        code = generate_suffix_condition(["AAA0", "AAA4"])
        # Check no dangling operators at end of lines
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped:
                self.assertFalse(stripped.endswith("||"), f"Dangling ||: {line}")
                self.assertFalse(stripped.endswith("&&"), f"Dangling &&: {line}")


    def test_check_ordering(self):
        """Test that checks are sorted by selectivity (popcount)."""
        # Prefix pattern where one word is fully masked and another is partial
        # '4eques' -> suffix, but let's test prefix logic
        # AAAAAA... -> A is 000000.
        # Let's manually construct a case.
        # If we have targets/masks:
        # Use indices 0 and 1 because prefix loop breaks at first 0 mask
        targets = [0] * 8
        masks = [0] * 8
        
        # Word 0: Partial match (few bits)
        targets[0] = 0x00000001
        masks[0] = 0x0000000F  # Popcount 4
        
        # Word 1: Full match (many bits)
        targets[1] = 0x12345678
        masks[1] = 0xFFFFFFFF  # Popcount 32
        
        code = generate_single_prefix_condition(targets, masks)
        
        # Verify that hash[1] check appears before hash[0]
        pos0 = code.find("hash[0]")
        pos1 = code.find("hash[1]")
        
        self.assertNotEqual(pos0, -1, f"hash[0] not found in code:\n{code}")
        self.assertNotEqual(pos1, -1, f"hash[1] not found in code:\n{code}")
        self.assertLess(pos1, pos0, "More selective check (hash[1]) should appear before hash[0]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
