#!/usr/bin/env python3
"""
Generate fixed compile-time matching conditions for CUDA fingerprint search.

This script generates optimized CUDA code for matching prefix/suffix patterns
at compile time, eliminating constant memory access overhead.

Usage:
    python gen_conditions.py --prefix AAA --suffix AAAA -o fixed_conditions.cuh
    python gen_conditions.py --suffix AAA0 --suffix AAA4 -o fixed_conditions.cuh
"""

import argparse
import sys
from typing import List, Tuple, Dict
from itertools import groupby

# Base64 decoding table (matches C++ h_b64_decode_table)
B64_DECODE: Dict[str, int] = {}
for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"):
    B64_DECODE[c] = i

# Valid last characters for fingerprint suffix (only these produce valid 256-bit hashes)
# These are Base64 chars where the lower 2 bits are 00 (since last char uses only 4 of 6 bits)
VALID_SUFFIX_LAST_CHARS = set("AEIMQUYcgkosw048")


def validate_suffix_last_char(pattern: str) -> None:
    """Validate that suffix pattern ends with a valid Base64 character for SHA256 hash."""
    if not pattern:
        return
    if len(pattern) > 43:
        raise ValueError(f"Suffix pattern too long: {len(pattern)} chars (max 43)")
    last_char = pattern[-1]
    if last_char not in VALID_SUFFIX_LAST_CHARS:
        raise ValueError(
            f"Invalid suffix last character '{last_char}'. "
            f"Must be one of: {sorted(VALID_SUFFIX_LAST_CHARS)}"
        )


def decode_base64_to_bits(pattern: str) -> List[int]:
    """Decode Base64 pattern to list of bits (MSB first)."""
    bits = []
    for c in pattern:
        if c not in B64_DECODE:
            raise ValueError(f"Invalid Base64 character: {c}")
        val = B64_DECODE[c]
        for i in range(5, -1, -1):  # 6 bits per character, MSB first
            bits.append((val >> i) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> List[int]:
    """Convert bit list to byte list (big-endian, MSB first)."""
    # Simply truncate to 256 bits if longer (though validation should prevent this)
    if len(bits) > 256:
        bits = bits[:256]
    
    # Pad to multiple of 8
    padded = bits + [0] * ((8 - len(bits) % 8) % 8)
    
    result = []
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        result.append(byte)
    return result


def bytes_to_uint32_be(byte_list: List[int]) -> List[int]:
    """Convert byte list to uint32 list (big-endian, matching SHA256 state)."""
    # Truncate to 32 bytes if longer
    if len(byte_list) > 32:
        byte_list = byte_list[:32]

    # Pad to 32 bytes (8 words)
    padded = byte_list + [0] * (32 - len(byte_list))
    
    result = []
    for i in range(0, 32, 4):
        word = (padded[i] << 24) | (padded[i+1] << 16) | (padded[i+2] << 8) | padded[i+3]
        result.append(word)
    return result


def compute_prefix_pattern(pattern: str) -> Tuple[List[int], List[int]]:
    """
    Compute targets and masks for a single prefix pattern.
    
    Returns:
        targets: List of 8 uint32 target values
        masks: List of 8 uint32 mask values
    """
    if len(pattern) > 43:
        raise ValueError(f"Prefix pattern too long: {len(pattern)} chars (max 43)")
        
    bits = decode_base64_to_bits(pattern)
    total_bits = len(bits)
    
    # Create target bits (256 bits, padded with zeros)
    target_bits = bits + [0] * (256 - len(bits))
    target_bytes = bits_to_bytes(target_bits)
    
    # Create mask bits (1 for bits we care about)
    mask_bits = [1] * total_bits + [0] * (256 - total_bits)
    mask_bytes = bits_to_bytes(mask_bits)
    
    targets = bytes_to_uint32_be(target_bytes)
    masks = bytes_to_uint32_be(mask_bytes)
    
    return targets, masks


def compute_suffix_pattern(pattern: str) -> Tuple[List[int], List[int], int, int]:
    """
    Compute targets and masks for a single suffix pattern.
    
    Returns:
        targets: List of 8 uint32 target values
        masks: List of 8 uint32 mask values
        start_word: First word index with non-zero mask
        word_count: Number of words with non-zero mask
    """
    bits = decode_base64_to_bits(pattern)
    pattern_len = len(pattern)
    
    # Suffix bits position in hash
    # Fingerprint has 43 chars = 258 bits, but hash is 256 bits
    # Suffix char index (43 - pattern_len) corresponds to bit (43 - pattern_len) * 6
    start_bit = (43 - pattern_len) * 6
    end_bit = 256  # Hash ends at bit 256
    
    # Create full 256-bit target and mask arrays
    target_bits = [0] * 256
    mask_bits = [0] * 256
    
    for i, bit in enumerate(bits):
        bit_pos = start_bit + i
        if 0 <= bit_pos < 256:
            target_bits[bit_pos] = bit
            mask_bits[bit_pos] = 1
    
    target_bytes = bits_to_bytes(target_bits)
    mask_bytes = bits_to_bytes(mask_bits)
    
    targets = bytes_to_uint32_be(target_bytes)
    masks = bytes_to_uint32_be(mask_bytes)
    
    # Find start_word and word_count
    start_word = 8
    end_word = 0
    for i in range(8):
        if masks[i] != 0:
            if i < start_word:
                start_word = i
            end_word = i + 1
    
    word_count = end_word - start_word if start_word < 8 else 0
    
    return targets, masks, start_word, word_count


def popcount(n: int) -> int:
    """Count number of set bits."""
    return bin(n).count('1')


def generate_single_prefix_condition(targets: List[int], masks: List[int]) -> str:
    """Generate condition for a single prefix pattern, sorted by selectivity."""
    checks = []
    
    for i in range(8):
        mask = masks[i]
        if mask == 0:
            break
        target = targets[i] & mask
        checks.append((mask, i, target))
    
    # Sort by popcount descending (high selectivity first)
    checks.sort(key=lambda x: popcount(x[0]), reverse=True)
    
    lines = []
    for mask, i, target in checks:
        if mask == 0xFFFFFFFF:
            lines.append(f"    if (hash[{i}] != 0x{target:08X}U) return false;")
        else:
            lines.append(f"    if ((hash[{i}] & 0x{mask:08X}U) != 0x{target:08X}U) return false;")
    
    lines.append("    return true;")
    return "\n".join(lines)


def generate_multi_prefix_condition(patterns: List[str]) -> str:
    """Generate condition for multiple prefix patterns (same length)."""
    if not patterns:
        return "    return true;"
    
    if len(patterns) == 1:
        targets, masks = compute_prefix_pattern(patterns[0])
        return generate_single_prefix_condition(targets, masks)
    
    # Verify all patterns have same length
    lengths = set(len(p) for p in patterns)
    if len(lengths) != 1:
        raise ValueError("All prefix patterns must have the same length")
    
    # Compute targets/masks for all patterns
    all_data = [compute_prefix_pattern(p) for p in patterns]
    reference_masks = all_data[0][1]
    
    lines = []
    common_checks = []
    varying_words = []
    
    # Find common and varying words
    for i in range(8):
        mask = reference_masks[i]
        if mask == 0:
            break
        
        targets_at_i = set(d[0][i] & mask for d in all_data)
        
        if len(targets_at_i) == 1:
            # Common value
            target = list(targets_at_i)[0]
            common_checks.append((mask, i, target))
        else:
            # Varying value
            varying_words.append((i, mask, targets_at_i))
    
    # Sort common checks by popcount descending
    common_checks.sort(key=lambda x: popcount(x[0]), reverse=True)
    
    for mask, i, target in common_checks:
        if mask == 0xFFFFFFFF:
            lines.append(f"    if (hash[{i}] != 0x{target:08X}U) return false;")
        else:
            lines.append(f"    if ((hash[{i}] & 0x{mask:08X}U) != 0x{target:08X}U) return false;")
    
    if not varying_words:
        lines.append("    return true;")
    elif len(varying_words) == 1:
        # Single varying word - simple OR
        i, mask, targets = varying_words[0]
        # Minimal CSE: Only mask if not full word
        if mask == 0xFFFFFFFF:
            or_parts = [f"hash[{i}] == 0x{t:08X}U" for t in sorted(targets)]
        else:
            # Masking is required, but let compiler handle CSE. 
            # Or use explicit var if we want to be safe, but adhering to "minimal registers"
            # we will just output raw checks and trust ptxas to fuse common (hash[i] & mask).
            # ACTUALLY, strict requirement: "Minimal registers". Repeate computation is better than long live var.
            # But here multiple compares follow immediately. So a short-lived var is optimal.
            # "uint32_t m = hash[i] & mask; if (m != A && m != B) return false" style
            # But the current structure is "return condition".
            # Let's use simple OR logic.
            or_parts = [f"((hash[{i}] & 0x{mask:08X}U) == 0x{t:08X}U)" for t in sorted(targets)]
            
        lines.append(f"    return {' || '.join(or_parts)};")
    else:
        # Multiple varying words
        # Sort varying words by popcount descending to group effectively
        # But here we need to match complete patterns.
        # Check order WITHIN the AND (&&) matters.
        
        # Determine sort order for words based on mask popcount
        varying_indices_sorted = sorted(varying_words, key=lambda x: popcount(x[1]), reverse=True)
        
        pattern_checks = []
        for targets, masks in all_data:
            word_checks = []
            # Check words in sorted order (most selective first)
            for i, mask, _ in varying_indices_sorted:
                target = targets[i] & mask
                if mask == 0xFFFFFFFF:
                    word_checks.append(f"hash[{i}] == 0x{target:08X}U")
                else:
                    word_checks.append(f"(hash[{i}] & 0x{mask:08X}U) == 0x{target:08X}U")
            pattern_checks.append("(" + " && ".join(word_checks) + ")")
        lines.append(f"    return {' || '.join(pattern_checks)};")
    
    return "\n".join(lines)


def generate_single_suffix_condition(targets: List[int], masks: List[int], 
                                      start_word: int, word_count: int) -> str:
    """Generate condition for a single suffix pattern, sorted by selectivity."""
    checks = []
    
    for i in range(start_word, start_word + word_count):
        mask = masks[i]
        if mask == 0:
            continue
        target = targets[i] & mask
        checks.append((mask, i, target))
        
    # Sort by popcount descending
    checks.sort(key=lambda x: popcount(x[0]), reverse=True)
    
    lines = []
    for mask, i, target in checks:
        if mask == 0xFFFFFFFF:
            lines.append(f"    if (hash[{i}] != 0x{target:08X}U) return false;")
        else:
            lines.append(f"    if ((hash[{i}] & 0x{mask:08X}U) != 0x{target:08X}U) return false;")
            
    lines.append("    return true;")
    return "\n".join(lines)


def generate_suffix_group_condition(patterns: List[str]) -> str:
    """Generate condition for a group of suffix patterns (same length), sorted."""
    if not patterns:
        return "    return true;"
    
    if len(patterns) == 1:
        targets, masks, start_word, word_count = compute_suffix_pattern(patterns[0])
        return generate_single_suffix_condition(targets, masks, start_word, word_count)
    
    all_data = [compute_suffix_pattern(p) for p in patterns]
    reference_masks = all_data[0][1]
    start_word = all_data[0][2]
    word_count = all_data[0][3]
    
    lines = []
    common_checks = []
    varying_words = []
    
    for i in range(start_word, start_word + word_count):
        mask = reference_masks[i]
        if mask == 0:
            continue
        
        targets_at_i = set(d[0][i] & mask for d in all_data)
        
        if len(targets_at_i) == 1:
            target = list(targets_at_i)[0]
            common_checks.append((mask, i, target))
        else:
            varying_words.append((i, mask, targets_at_i))
            
    # Sort common checks by popcount
    common_checks.sort(key=lambda x: popcount(x[0]), reverse=True)
    
    for mask, i, target in common_checks:
        if mask == 0xFFFFFFFF:
            lines.append(f"    if (hash[{i}] != 0x{target:08X}U) return false;")
        else:
            lines.append(f"    if ((hash[{i}] & 0x{mask:08X}U) != 0x{target:08X}U) return false;")
            
    if not varying_words:
        lines.append("    return true;")
    elif len(varying_words) == 1:
        i, mask, targets = varying_words[0]
        if mask == 0xFFFFFFFF:
            or_parts = [f"hash[{i}] == 0x{t:08X}U" for t in sorted(targets)]
        else:
            or_parts = [f"((hash[{i}] & 0x{mask:08X}U) == 0x{t:08X}U)" for t in sorted(targets)]
        lines.append(f"    return {' || '.join(or_parts)};")
    else:
        # Multiple varying words - Sort words by popcount
        varying_indices_sorted = sorted(varying_words, key=lambda x: popcount(x[1]), reverse=True)
        
        pattern_checks = []
        for targets, masks, _, _ in all_data:
            word_checks = []
            for i, mask, _ in varying_indices_sorted:
                target = targets[i] & mask
                if mask == 0xFFFFFFFF:
                    word_checks.append(f"hash[{i}] == 0x{target:08X}U")
                else:
                    word_checks.append(f"(hash[{i}] & 0x{mask:08X}U) == 0x{target:08X}U")
            pattern_checks.append("(" + " && ".join(word_checks) + ")")
        lines.append(f"    return {' || '.join(pattern_checks)};")
    
    return "\n".join(lines)


def generate_prefix_condition(patterns: List[str]) -> str:
    """Generate optimized prefix matching condition."""
    if not patterns:
        return "    return true;  // No prefix condition"
    
    # Group by length
    sorted_patterns = sorted(patterns, key=len)
    groups = [list(g) for _, g in groupby(sorted_patterns, key=len)]
    
    if len(groups) == 1:
        return generate_multi_prefix_condition(groups[0])
    
    # Multiple length groups - generate OR of group conditions
    # For prefix, we can't easily combine different lengths, so we generate
    # individual full checks for each group
    group_conditions = []
    for group in groups:
        # For each group, generate the condition as a lambda-like block
        cond = generate_multi_prefix_condition(group)
        # Wrap in a block that returns the result
        group_conditions.append(f"[&]() {{ {cond.replace(chr(10), ' ')} }}()")
    
    return f"    return {' || '.join(group_conditions)};"


def generate_suffix_condition(patterns: List[str]) -> str:
    """Generate optimized suffix matching condition."""
    if not patterns:
        return "    return true;  // No suffix condition"
    
    # Group by length (same length = same mask shape)
    sorted_patterns = sorted(patterns, key=len)
    groups = [list(g) for _, g in groupby(sorted_patterns, key=len)]
    
    if len(groups) == 1:
        return generate_suffix_group_condition(groups[0])
    
    # Multiple length groups - generate separate conditions for each group
    lines = []
    group_results = []
    
    for idx, group in enumerate(groups):
        # Generate condition for this group inline
        group_cond = generate_suffix_group_condition(group)
        # Create a helper variable for readability
        var_name = f"match_len{len(group[0])}"
        
        # Convert the multi-line condition to inline lambda
        inline_cond = group_cond.replace("\n", " ").replace("    ", "")
        group_results.append(f"[&]() {{ {inline_cond} }}()")
    
    return f"    return {' || '.join(group_results)};"


def generate_header(prefixes: List[str], suffixes: List[str]) -> str:
    """Generate the complete header file."""
    prefix_list = ", ".join(f'"{p}"' for p in prefixes) if prefixes else "none"
    suffix_list = ", ".join(f'"{s}"' for s in suffixes) if suffixes else "none"
    
    header = f"""#pragma once
// Auto-generated fixed matching conditions
// DO NOT EDIT - Generated by gen_conditions.py
//
// Prefix patterns: {prefix_list}
// Suffix patterns: {suffix_list}

"""
    
    # Generate prefix function
    if prefixes:
        prefix_code = generate_prefix_condition(prefixes)
        header += f"""__device__ __forceinline__ bool match_prefix_fixed(const uint32_t* hash) {{
{prefix_code}
}}

"""
    
    # Generate suffix function
    if suffixes:
        suffix_code = generate_suffix_condition(suffixes)
        header += f"""__device__ __forceinline__ bool match_suffix_fixed(const uint32_t* hash) {{
{suffix_code}
}}
"""
    
    return header


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed compile-time matching conditions for CUDA fingerprint search."
    )
    parser.add_argument("--prefix", action="append", default=[],
                        help="Prefix pattern (can be specified multiple times)")
    parser.add_argument("--suffix", action="append", default=[],
                        help="Suffix pattern (can be specified multiple times)")
    parser.add_argument("-o", "--output", default="fixed_conditions.cuh",
                        help="Output file path")
    
    args = parser.parse_args()
    
    if not args.prefix and not args.suffix:
        parser.error("At least one --prefix or --suffix must be specified")
    
    # Validate patterns
    for p in args.prefix:
        try:
            decode_base64_to_bits(p)
        except ValueError as e:
            parser.error(f"Invalid prefix pattern '{p}': {e}")
    
    for s in args.suffix:
        try:
            decode_base64_to_bits(s)
            validate_suffix_last_char(s)
        except ValueError as e:
            parser.error(f"Invalid suffix pattern '{s}': {e}")
    
    # Generate header
    header_content = generate_header(args.prefix, args.suffix)
    
    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(header_content)
    
    print(f"Generated: {args.output}")
    if args.prefix:
        print(f"  Prefix patterns: {args.prefix}")
    if args.suffix:
        print(f"  Suffix patterns: {args.suffix}")


if __name__ == "__main__":
    main()
