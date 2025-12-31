
import sys
import math

# Ed25519 constants
Q = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
d = -121665 * pow(121666, Q-2, Q) % Q
I = pow(2, (Q-1)//4, Q)

def inv(x):
    return pow(x, Q-2, Q)

def xrecover(y):
    xx = (y*y - 1) * inv(d*y*y + 1)
    x = pow(xx, (Q+3)//8, Q)
    if (x*x - xx) % Q != 0: x = (x*I) % Q
    if x % 2 != 0: x = Q-x
    return x

By = 4 * inv(5)
Bx = xrecover(By)
B = (Bx % Q, By % Q)

def edwards_add(P, Q_pt):
    x1, y1 = P
    x2, y2 = Q_pt
    x3 = (x1*y2 + x2*y1) * inv(1 + d*x1*x2*y1*y2)
    y3 = (y1*y2 + x1*x2) * inv(1 - d*x1*x2*y1*y2)
    return (x3 % Q, y3 % Q)

def edwards_double(P):
    x1, y1 = P
    x3 = (2*x1*y1) * inv(1 + d*x1*x1*y1*y1)
    y3 = (y1*y1 + x1*x1) * inv(1 - d*x1*x1*y1*y1)
    return (x3 % Q, y3 % Q)

def scalarmult(P, e):
    if e == 0: return (0, 1)
    Q_pt = (0, 1)
    for i in range(256):
        if (e >> i) & 1:
            Q_pt = edwards_add(Q_pt, P)
        P = edwards_double(P)
    return Q_pt

def to_limbs(x):
    # Convert large int to 10 limbs (26, 25, 26, 25...)
    x = x % Q
    limbs = []
    shifts = [26, 25, 26, 25, 26, 25, 26, 25, 26, 25]
    mask_26 = (1 << 26) - 1
    mask_25 = (1 << 25) - 1
    
    current = x
    for s in shifts:
        mask = (1 << s) - 1
        limbs.append(current & mask)
        current >>= s
        
    return limbs

def print_limbs(limbs):
    return "{ " + ", ".join(str(l) for l in limbs) + " }"

def generate(window_bits=7):
    filename = f"precomp_{window_bits}bit.h"
    entries_per_step = 1 << (window_bits - 1)
    num_steps = math.ceil(256 / window_bits)
    
    print(f"Generating {filename} with window_bits={window_bits}")
    print(f"Steps: {num_steps}, Entries per step: {entries_per_step}")
    
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write(f"// Auto-generated {window_bits}-bit window precomputed table\n")
        f.write(f"// Steps: {num_steps}. Entries: {entries_per_step} (signed window)\n")
        
        f.write(f"static const ge_precomp base_{window_bits}bit[{num_steps}][{entries_per_step}] = {{\n")
        
        # Multiplier = 2^window_bits * B
        Step_Multiplier = B
        for _ in range(window_bits):
            Step_Multiplier = edwards_double(Step_Multiplier)
            
        Current_Base = B
        
        for i in range(num_steps):
            f.write(f"    // Group {i} (2^{{{window_bits}*i}} * B)\n")
            f.write("    {\n")
            
            P = Current_Base
            Accumulator = P
            
            for j in range(entries_per_step): 
                # j=0 -> 1*Base, j=1 -> 2*Base, ...
                
                Rx, Ry = Accumulator
                yplusx = (Ry + Rx) % Q
                yminusx = (Ry - Rx) % Q
                xy2d = (Rx * Ry * 2 * d) % Q
                
                l1 = to_limbs(yplusx)
                l2 = to_limbs(yminusx)
                l3 = to_limbs(xy2d)
                
                f.write(f"        {{\n")
                f.write(f"            {print_limbs(l1)}, // y+x\n")
                f.write(f"            {print_limbs(l2)}, // y-x\n")
                f.write(f"            {print_limbs(l3)}  // 2dxy\n")
                f.write(f"        }},\n")
                
                if j < entries_per_step - 1:
                    Accumulator = edwards_add(Accumulator, P)
            
            f.write("    },\n")
            
            # Advance Base
            for _ in range(window_bits):
                Current_Base = edwards_double(Current_Base)
                
        f.write("};\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate(int(sys.argv[1]))
    else:
        generate(7) # Default to 7
