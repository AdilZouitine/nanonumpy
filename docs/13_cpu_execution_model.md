# What happens inside the CPU?

### 1. Memory load

The CPU loads contiguous `f32` values from memory into vector registers. The data may come from RAM, but usually it passes through cache levels first.

### 2. Vector registers

A vector register is a wide CPU register that stores multiple values.

- XMM: 128 bits, four `f32`
- YMM: 256 bits, eight `f32`
- NEON Q register: 128 bits, four `f32`

### 3. SIMD lane execution

```text
YMM register A:
[a0, a1, a2, a3, a4, a5, a6, a7]

YMM register B:
[b0, b1, b2, b3, b4, b5, b6, b7]

vaddps:
[a0+b0, a1+b1, a2+b2, a3+b3, a4+b4, a5+b5, a6+b6, a7+b7]
```

### 4. Store result

After computation, the vector register is stored back to the output array.

### 5. Loop mechanics

The SIMD loop advances by the vector width:

- AVX2 advances by 8 floats
- SSE advances by 4 floats
- NEON advances by 4 floats

Then the scalar tail handles leftover elements.

### 6. CPU pipeline, simplified

Modern CPUs do not simply execute one instruction from start to finish before the next. They have pipelines and execution units.

At a simplified level:

- load units bring data into registers
- floating-point/vector units execute SIMD arithmetic
- store units write results back
- branch prediction helps the loop continue efficiently

### 7. Bottlenecks

Performance can be limited by Python conversion overhead, memory bandwidth, cache misses, alignment, CPU frequency changes, thermal throttling, compiler optimizations, and small input sizes.

### 8. Why SIMD helps

SIMD reduces loop overhead and increases arithmetic throughput by doing multiple operations per instruction. It does not remove the cost of memory access, allocation, or Python/Rust conversion.
