// ============================================================================
// Apple Silicon Metal 3 Fused Kernels for mrl-index + EmbeddingGemma-2
// ============================================================================
// 1. `encode_paxblock64_fused`:
//    In-register bitplane transposition of 64 augmented 960-d embeddings
//    (`768` base + `128` FWHT mid + `64` folded FWHT tail) into a 4 KiB
//    `PaxBlock64` using hardware `simd_ballot()` and race-free disjoint
//    32-bit word indexing (`d * 2 + simdgroup_id`).
//
// 2. `fused_rerank_dot_768`:
//    Threadgroup SRAM query-pinned cosine similarity epilogue emitting only
//    1 scalar `float` score (4 B) per shortlist candidate.
// ============================================================================

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant uint MRL_PREFIX_DIMS = 64;
constant uint RABITQ_DIMS     = 128;
constant uint TAIL_FOLD_DIMS  = 64;
constant uint AUGMENTED_DIMS  = 960; // 768 + 128 + 64

struct VectorMetaPacked {
    half  scale;
    half  offset;
    half  tail_norm;
    uchar flags;
} __attribute__((packed));

// 1. Fused PaxBlock64 Bitplane Encoder (1 threadgroup = 64 threads = 2 SIMD-groups = 1 PaxBlock64)
kernel void encode_paxblock64_fused(
    device const float*   aug_vecs     [[buffer(0)]], // [B * 64, 960]
    device uchar*         pax_blocks   [[buffer(1)]], // [B, 4096] (16 KiB aligned per 4 blocks)
    device const uint*    active_counts[[buffer(2)]], // [B] active vectors per block (<= 64)
    uint tgid                          [[threadgroup_position_in_grid]],
    uint tid                           [[thread_index_in_threadgroup]],
    uint sg_id                         [[simdgroup_index_in_threadgroup]],
    uint lane_id                       [[thread_index_in_simdgroup]]
) {
    uint active_n = active_counts[tgid];
    bool is_active = (tid < active_n);

    device const float* v = aug_vecs + (tgid * 64 + tid) * AUGMENTED_DIMS;
    device uchar* block_base = pax_blocks + (tgid * 4096);

    // Byte offsets inside 4096-byte PaxBlock64:
    //   0..64:     BlockHeader (64 B)
    //   64..512:   meta: [VectorMeta; 64] (448 B, 7 B each)
    //   512..1024: stratum0_sign (512 B = 128 x uint32)
    //   1024..2048:stratum1_rabitq (1024 B = 256 x uint32)
    //   2048..2560:stratum2_msb1 (512 B = 128 x uint32)
    //   2560..3072:stratum3_msb2 (512 B = 128 x uint32)
    //   3072..3584:stratum4_mid (512 B = 128 x uint32)
    //   3584..4096:stratum5_lsb (512 B = 128 x uint32)
    device uint* s0_u32 = reinterpret_cast<device uint*>(block_base + 512);
    device uint* s1_u32 = reinterpret_cast<device uint*>(block_base + 1024);
    device uint* s2_u32 = reinterpret_cast<device uint*>(block_base + 2048);
    device uint* s3_u32 = reinterpret_cast<device uint*>(block_base + 2560);
    device uint* s4_u32 = reinterpret_cast<device uint*>(block_base + 3072);
    device uint* s5_u32 = reinterpret_cast<device uint*>(block_base + 3584);

    // Step A: Compute per-vector prefix min/max and mid/tail norms in registers
    float min_v = 1e30f;
    float max_v = -1e30f;
    if (is_active) {
        for (uint d = 0; d < MRL_PREFIX_DIMS; ++d) {
            float x = v[d];
            min_v = fmin(min_v, x);
            max_v = fmax(max_v, x);
        }
    } else {
        min_v = 0.0f;
        max_v = 0.0f;
    }

    float range = fmax(max_v - min_v, 1e-8f);
    float scale = range / 15.0f;
    float inv_scale = 1.0f / scale;
    float offset = min_v;

    float mid_sq = 0.0f;
    float high_sq = 0.0f;
    if (is_active) {
        for (uint d = 64; d < 192; ++d) {
            float x = v[d];
            mid_sq += x * x;
        }
        for (uint d = 192; d < 768; ++d) {
            float x = v[d];
            high_sq += x * x;
        }
    }
    float rest_norm = sqrt(mid_sq + high_sq);
    float high_norm = sqrt(high_sq);
    uchar rho_flag = (rest_norm > 1e-12f)
        ? uchar(clamp(rint((high_norm / rest_norm) * 255.0f), 0.0f, 255.0f))
        : 0;

    // Write 7-byte VectorMeta entry for thread `tid`
    device uchar* meta_ptr = block_base + 64 + tid * 7;
    ushort scale_bits = as_type<ushort>(half(is_active ? scale : 0.0f));
    ushort offset_bits = as_type<ushort>(half(is_active ? offset : 0.0f));
    ushort rest_bits = as_type<ushort>(half(is_active ? rest_norm : 0.0f));
    meta_ptr[0] = uchar(scale_bits & 0xff);
    meta_ptr[1] = uchar(scale_bits >> 8);
    meta_ptr[2] = uchar(offset_bits & 0xff);
    meta_ptr[3] = uchar(offset_bits >> 8);
    meta_ptr[4] = uchar(rest_bits & 0xff);
    meta_ptr[5] = uchar(rest_bits >> 8);
    meta_ptr[6] = is_active ? rho_flag : 0;

    // Step B: 4-Bit MRL Prefix Transposition via simd_ballot()
    // Disjoint 32-bit word index: `out_idx = d * 2 + sg_id` (zero race between sg0 and sg1).
    for (uint d = 0; d < MRL_PREFIX_DIMS; ++d) {
        uint q = 0;
        if (is_active) {
            float coord = v[d];
            q = uint(clamp(rint((coord - offset) * inv_scale), 0.0f, 15.0f));
        }
        uint b3 = (uint64_t)simd_ballot((q & 0x8u) != 0);
        uint b2 = (uint64_t)simd_ballot((q & 0x4u) != 0);
        uint b1 = (uint64_t)simd_ballot((q & 0x2u) != 0);
        uint b0 = (uint64_t)simd_ballot((q & 0x1u) != 0);

        if (lane_id == 0) {
            uint out_idx = d * 2 + sg_id;
            s0_u32[out_idx] = b3;
            s2_u32[out_idx] = b2;
            s3_u32[out_idx] = b1;
            s5_u32[out_idx] = b0;
        }
    }

    // Step C: Stratum-1 RaBitQ (128 pre-rotated dims at aug[768..896])
    for (uint d = 0; d < RABITQ_DIMS; ++d) {
        bool sign_pos = is_active && (v[768 + d] >= 0.0f);
        uint mask32 = (uint64_t)simd_ballot(sign_pos);
        if (lane_id == 0) {
            s1_u32[d * 2 + sg_id] = mask32;
        }
    }

    // Step D: Stratum-4 Folded Tail (64 pre-rotated dims at aug[896..960])
    for (uint d = 0; d < TAIL_FOLD_DIMS; ++d) {
        bool sign_pos = is_active && (v[768 + RABITQ_DIMS + d] >= 0.0f);
        uint mask32 = (uint64_t)simd_ballot(sign_pos);
        if (lane_id == 0) {
            s4_u32[d * 2 + sg_id] = mask32;
        }
    }
}

// 2. Fused Shortlist Rerank Epilogue (Pins query in 32 KiB threadgroup SRAM)
kernel void fused_rerank_dot_768(
    device const half*  cand_vecs [[buffer(0)]], // [K_cand, 768]
    device const float* query_vec [[buffer(1)]], // [768]
    device float*       out_scores[[buffer(2)]], // [K_cand]
    uint cand_idx                 [[threadgroup_position_in_grid]],
    uint lane_id                  [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_q[768];
    for (uint i = lane_id; i < 768; i += 32) {
        shared_q[i] = query_vec[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const half* cand = cand_vecs + cand_idx * 768;
    float dot_acc = 0.0f;
    float norm_acc = 0.0f;
    for (uint i = lane_id; i < 768; i += 32) {
        float c = float(cand[i]);
        dot_acc = fma(c, shared_q[i], dot_acc);
        norm_acc = fma(c, c, norm_acc);
    }

    float total_dot = simd_sum(dot_acc);
    float total_norm = sqrt(fmax(simd_sum(norm_acc), 1e-24f));
    if (lane_id == 0) {
        out_scores[cand_idx] = total_dot / total_norm;
    }
}
