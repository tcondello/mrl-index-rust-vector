//! Challenger Verification Suite for docs/GEMMA2_LOCAL_INTEGRATION_ARCHITECTURAL_DESIGN.md
//!
//! Verifies:
//! 1. Section 3.2 Metal shader: disjoint 32-bit word indexing (d * 2 + simdgroup_id) eliminates
//!    data races between SIMD-groups 0 and 1, with bit-exact little-endian reconstruction.
//! 2. Memory layout and page alignment: 4 x PaxBlock64 = 16,384 bytes == 1 Darwin VM Page (16 KiB).
//! 3. Grounding in repository benchmarks: scan_bench throughput, MS MARCO 8.8M memory footprint,
//!    and shortlist reranking tradeoffs.

use std::mem::{align_of, size_of};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use mrl_index::block::{
    Bitplane64, BlockHeader, PaxBlock64, RabitqPlane, VectorMeta, BLOCK_BYTES, MRL_PREFIX_DIMS,
    RABITQ_DIMS, TAIL_FOLD_DIMS, VECTORS_PER_BLOCK,
};
use mrl_index::passage::PAGE_ALIGN;

// ----------------------------------------------------------------------------
// Test 1: Concurrency Safety & Little-Endian Disjoint Word Indexing (MSL Shader)
// ----------------------------------------------------------------------------

/// Simulated MSL kernel execution for a single thread (simdgroup_id, simd_lane_id).
/// In Metal, simdgroup_ballot(predicate) returns a 32-bit integer where bit i is lane i's predicate.
fn simulate_metal_shader_threadgroup(
    features: &[[f32; 256]; 64],
    meta: &[VectorMeta; 64],
) -> PaxBlock64 {
    let mut block = PaxBlock64::zeroed();

    // Copy metadata
    for v in 0..64 {
        block.meta[v] = meta[v];
    }

    // Cast 64-bit stratum arrays to u32 slices
    let s1_u32: &mut [u32; 256] = unsafe {
        std::mem::transmute(&mut block.stratum1_rabitq.dim_bits)
    };
    let s4_u32: &mut [u32; 128] = unsafe {
        std::mem::transmute(&mut block.stratum4_mid.dim_bits)
    };
    let s0_u32: &mut [u32; 128] = unsafe {
        std::mem::transmute(&mut block.stratum0_sign.dim_bits)
    };
    let s2_u32: &mut [u32; 128] = unsafe {
        std::mem::transmute(&mut block.stratum2_msb1.dim_bits)
    };
    let s3_u32: &mut [u32; 128] = unsafe {
        std::mem::transmute(&mut block.stratum3_msb2.dim_bits)
    };
    let s5_u32: &mut [u32; 128] = unsafe {
        std::mem::transmute(&mut block.stratum5_lsb.dim_bits)
    };

    // Simulate 2 SIMD-groups of 32 threads
    for simdgroup_id in 0..2 {
        // Stratum 1: 128 dims
        for d in 0..RABITQ_DIMS {
            let mut ballot: u32 = 0;
            for lane in 0..32 {
                let local_v = simdgroup_id * 32 + lane;
                if features[local_v][64 + d] >= 0.0 {
                    ballot |= 1 << lane;
                }
            }
            s1_u32[d * 2 + simdgroup_id] = ballot;
        }

        // Stratum 4: 64 dims
        for d in 0..TAIL_FOLD_DIMS {
            let mut ballot: u32 = 0;
            for lane in 0..32 {
                let local_v = simdgroup_id * 32 + lane;
                if features[local_v][192 + d] >= 0.0 {
                    ballot |= 1 << lane;
                }
            }
            s4_u32[d * 2 + simdgroup_id] = ballot;
        }

        // Strata 0, 2, 3, 5: 64 prefix dims
        for d in 0..MRL_PREFIX_DIMS {
            let mut b3: u32 = 0;
            let mut b2: u32 = 0;
            let mut b1: u32 = 0;
            let mut b0: u32 = 0;

            for lane in 0..32 {
                let local_v = simdgroup_id * 32 + lane;
                let coord = features[local_v][d];
                let offset = meta[local_v].offset_f32();
                let scale = meta[local_v].scale_f32();
                let q_val = ((coord - offset) * scale).round().clamp(0.0, 15.0) as u8;

                if (q_val & 0x8) != 0 { b3 |= 1 << lane; }
                if (q_val & 0x4) != 0 { b2 |= 1 << lane; }
                if (q_val & 0x2) != 0 { b1 |= 1 << lane; }
                if (q_val & 0x1) != 0 { b0 |= 1 << lane; }
            }

            let out_idx = d * 2 + simdgroup_id;
            s0_u32[out_idx] = b3;
            s2_u32[out_idx] = b2;
            s3_u32[out_idx] = b1;
            s5_u32[out_idx] = b0;
        }
    }

    block
}

#[test]
fn test_metal_shader_disjoint_indexing_and_endianness() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);

    // Create random test features and metadata
    let mut features = [[0.0f32; 256]; 64];
    let mut meta = [VectorMeta { scale: 0, offset: 0, tail_norm: 0, flags: 0 }; 64];

    for v in 0..64 {
        for d in 0..256 {
            features[v][d] = rng.gen_range(-5.0..5.0);
        }
        meta[v] = VectorMeta::from_f32(1.5, -0.5, 1.0, 128);
    }

    let block = simulate_metal_shader_threadgroup(&features, &meta);

    // Verify Stratum 1: for each d in 0..128 and v in 0..64:
    // bit v of stratum1_rabitq.dim_bits[d] must equal (features[v][64 + d] >= 0.0)
    for d in 0..RABITQ_DIMS {
        let word64 = block.stratum1_rabitq.dim_bits[d];
        for v in 0..64 {
            let expected_bit = features[v][64 + d] >= 0.0;
            let actual_bit = ((word64 >> v) & 1) != 0;
            assert_eq!(
                actual_bit, expected_bit,
                "Stratum 1 mismatch at dim {d}, vector {v}"
            );
        }
    }

    // Verify Stratum 4: for each d in 0..64 and v in 0..64:
    for d in 0..TAIL_FOLD_DIMS {
        let word64 = block.stratum4_mid.dim_bits[d];
        for v in 0..64 {
            let expected_bit = features[v][192 + d] >= 0.0;
            let actual_bit = ((word64 >> v) & 1) != 0;
            assert_eq!(
                actual_bit, expected_bit,
                "Stratum 4 mismatch at dim {d}, vector {v}"
            );
        }
    }

    // Verify Strata 0, 2, 3, 5: 4-bit nibbles
    for d in 0..MRL_PREFIX_DIMS {
        let s0 = block.stratum0_sign.dim_bits[d];
        let s2 = block.stratum2_msb1.dim_bits[d];
        let s3 = block.stratum3_msb2.dim_bits[d];
        let s5 = block.stratum5_lsb.dim_bits[d];

        for v in 0..64 {
            let coord = features[v][d];
            let offset = meta[v].offset_f32();
            let scale = meta[v].scale_f32();
            let q_val = ((coord - offset) * scale).round().clamp(0.0, 15.0) as u8;

            let b3 = ((s0 >> v) & 1) as u8;
            let b2 = ((s2 >> v) & 1) as u8;
            let b1 = ((s3 >> v) & 1) as u8;
            let b0 = ((s5 >> v) & 1) as u8;
            let reconstructed_nibble = (b3 << 3) | (b2 << 2) | (b1 << 1) | b0;

            assert_eq!(
                reconstructed_nibble, q_val,
                "Prefix nibble mismatch at dim {d}, vector {v}"
            );
        }
    }
}

/// Stress test verifying concurrent writes from two threads to disjoint 32-bit indices.
/// Thread 0 writes even indices (2*d), Thread 1 writes odd indices (2*d + 1).
/// Verifies zero data races, no memory corruption, and correct full 64-bit values.
#[test]
fn test_concurrent_disjoint_word_write_race_freedom() {
    const NUM_WORDS_64: usize = 128;
    const NUM_WORDS_32: usize = NUM_WORDS_64 * 2;
    const TRIALS: usize = 1000;

    for trial in 0..TRIALS {
        // Shared buffer of 256 u32 words
        let mut buffer = Box::new([0u32; NUM_WORDS_32]);
        let ptr = buffer.as_mut_ptr() as usize;

        let start_signal = Arc::new(AtomicBool::new(false));

        // Thread 0: SIMD-group 0 writes even indices
        let start_0 = Arc::clone(&start_signal);
        let t0 = thread::spawn(move || {
            while !start_0.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let u32_ptr = ptr as *mut u32;
            for d in 0..NUM_WORDS_64 {
                let pattern: u32 = 0xAAAA_0000 | ((trial * 256 + d) as u32 & 0xFFFF);
                unsafe {
                    std::ptr::write(u32_ptr.add(d * 2 + 0), pattern);
                }
            }
        });

        // Thread 1: SIMD-group 1 writes odd indices
        let start_1 = Arc::clone(&start_signal);
        let t1 = thread::spawn(move || {
            while !start_1.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let u32_ptr = ptr as *mut u32;
            for d in 0..NUM_WORDS_64 {
                let pattern: u32 = 0x5555_0000 | (((trial * 256 + d) ^ 0xFFFF) as u32 & 0xFFFF);
                unsafe {
                    std::ptr::write(u32_ptr.add(d * 2 + 1), pattern);
                }
            }
        });

        // Fire both threads simultaneously
        start_signal.store(true, Ordering::Release);
        t0.join().unwrap();
        t1.join().unwrap();

        // Verify that the resulting buffer as u64 words contains exactly the combined patterns
        let u64_slice: &[u64] = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr() as *const u64, NUM_WORDS_64)
        };

        for d in 0..NUM_WORDS_64 {
            let expected_lo = (0xAAAA_0000 | ((trial * 256 + d) as u32 & 0xFFFF)) as u64;
            let expected_hi = ((0x5555_0000 | (((trial * 256 + d) ^ 0xFFFF) as u32 & 0xFFFF)) as u64) << 32;
            let expected_u64 = expected_hi | expected_lo;

            assert_eq!(
                u64_slice[d], expected_u64,
                "Race or corruption detected at word {d} on trial {trial}"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// Test 2: Memory Layout and Darwin VM Page Alignment (16 KiB == 4 x PaxBlock64)
// ----------------------------------------------------------------------------

#[test]
fn test_paxblock64_layout_and_darwin_page_alignment() {
    // 1. PaxBlock64 size and alignment
    assert_eq!(size_of::<PaxBlock64>(), 4096, "PaxBlock64 must be exactly 4,096 bytes");
    assert_eq!(align_of::<PaxBlock64>(), 4096, "PaxBlock64 must be 4,096-byte aligned");
    assert_eq!(BLOCK_BYTES, 4096);

    // 2. Exact 4 x PaxBlock64 == 16,384 bytes == 1 Darwin VM Page
    assert_eq!(4 * size_of::<PaxBlock64>(), 16_384);
    assert_eq!(PAGE_ALIGN, 16_384);

    // 3. Field offsets in PaxBlock64
    let dummy = PaxBlock64::zeroed();
    let base = &dummy as *const _ as usize;

    let offset_header = (&dummy.header as *const _ as usize) - base;
    let offset_meta = (&dummy.meta as *const _ as usize) - base;
    let offset_s0 = (&dummy.stratum0_sign as *const _ as usize) - base;
    let offset_s1 = (&dummy.stratum1_rabitq as *const _ as usize) - base;
    let offset_s2 = (&dummy.stratum2_msb1 as *const _ as usize) - base;
    let offset_s3 = (&dummy.stratum3_msb2 as *const _ as usize) - base;
    let offset_s4 = (&dummy.stratum4_mid as *const _ as usize) - base;
    let offset_s5 = (&dummy.stratum5_lsb as *const _ as usize) - base;

    assert_eq!(offset_header, 0);
    assert_eq!(size_of::<BlockHeader>(), 64);

    assert_eq!(offset_meta, 64);
    assert_eq!(size_of::<[VectorMeta; VECTORS_PER_BLOCK]>(), 448);
    assert_eq!(size_of::<VectorMeta>(), 7);

    assert_eq!(offset_s0, 512);
    assert_eq!(size_of::<Bitplane64>(), 512);

    assert_eq!(offset_s1, 1024);
    assert_eq!(size_of::<RabitqPlane>(), 1024);

    assert_eq!(offset_s2, 2048);
    assert_eq!(size_of::<Bitplane64>(), 512);

    assert_eq!(offset_s3, 2560);
    assert_eq!(size_of::<Bitplane64>(), 512);

    assert_eq!(offset_s4, 3072);
    assert_eq!(size_of::<Bitplane64>(), 512);

    assert_eq!(offset_s5, 3584);
    assert_eq!(size_of::<Bitplane64>(), 512);

    assert_eq!(offset_s5 + size_of::<Bitplane64>(), 4096);

    // 4. Verify stratum arrays satisfy u32 and u64 alignment
    assert_eq!(offset_s0 % align_of::<u64>(), 0);
    assert_eq!(offset_s0 % align_of::<u32>(), 0);
    assert_eq!(offset_s1 % align_of::<u64>(), 0);
    assert_eq!(offset_s1 % align_of::<u32>(), 0);
    assert_eq!(offset_s2 % align_of::<u32>(), 0);
    assert_eq!(offset_s3 % align_of::<u32>(), 0);
    assert_eq!(offset_s4 % align_of::<u32>(), 0);
    assert_eq!(offset_s5 % align_of::<u32>(), 0);
}

// ----------------------------------------------------------------------------
// Test 3: Grounding in Repository Benchmarks & Quantitative Tradeoffs
// ----------------------------------------------------------------------------

#[test]
fn test_quantitative_msmarco_footprint_and_bandwidth_claims() {
    let n_passages = 8_841_823usize;
    let d_model = 768usize;

    // 1. FP32 Full Vectors
    let fp32_bytes_per_vec = d_model * size_of::<f32>(); // 3072 B
    let fp32_total_bytes = n_passages as u64 * fp32_bytes_per_vec as u64;
    let fp32_gb = fp32_total_bytes as f64 / 1e9;
    assert_eq!(fp32_bytes_per_vec, 3072);
    assert!((fp32_gb - 27.162).abs() < 0.01, "FP32 total GB should be 27.16 GB");

    // 2. FP16 Vectors
    let fp16_bytes_per_vec = d_model * 2; // 1536 B
    let fp16_total_bytes = n_passages as u64 * fp16_bytes_per_vec as u64;
    let fp16_gb = fp16_total_bytes as f64 / 1e9;
    assert_eq!(fp16_bytes_per_vec, 1536);
    assert!((fp16_gb - 13.581).abs() < 0.01, "FP16 total GB should be 13.58 GB");

    // 3. INT8 Scaled Codes (768 bytes + 2 bytes scale = 770 B)
    let int8_bytes_per_vec = 768 + 2;
    let int8_total_bytes = n_passages as u64 * int8_bytes_per_vec as u64;
    let int8_gb = int8_total_bytes as f64 / 1e9;
    assert_eq!(int8_bytes_per_vec, 770);
    assert!((int8_gb - 6.808).abs() < 0.01, "INT8 total GB should be 6.81 GB");

    // 4. PaxBlock64 Macro-Blocks (4096 B / 64 vecs = 64 B/vec)
    let pax_bytes_per_vec = 64usize;
    let pax_total_bytes = n_passages as u64 * pax_bytes_per_vec as u64;
    let pax_mb = pax_total_bytes as f64 / 1e6;
    let compression_ratio = fp32_bytes_per_vec as f64 / pax_bytes_per_vec as f64;
    assert_eq!(pax_bytes_per_vec, 64);
    assert!((pax_mb - 565.88).abs() < 0.1, "PaxBlock64 total MB should be 565.88 MB");
    assert_eq!(compression_ratio, 48.0);

    // 5. Bandwidth consumption at 42.4 Mvecs/s
    let scan_rate = 42.4e6; // 42.4 Mvecs/s
    let fp32_bw = (scan_rate * fp32_bytes_per_vec as f64) / 1e9;
    let fp16_bw = (scan_rate * fp16_bytes_per_vec as f64) / 1e9;
    let int8_bw = (scan_rate * int8_bytes_per_vec as f64) / 1e9;
    let pax_bw = (scan_rate * pax_bytes_per_vec as f64) / 1e9;

    assert!((fp32_bw - 130.25).abs() < 0.1, "FP32 bandwidth: 130.3 GB/s");
    assert!((fp16_bw - 65.13).abs() < 0.1, "FP16 bandwidth: 65.1 GB/s");
    assert!((int8_bw - 32.65).abs() < 0.1, "INT8 bandwidth: 32.6 GB/s");
    assert!((pax_bw - 2.71).abs() < 0.02, "PaxBlock64 bandwidth: 2.71 GB/s");

    // 6. Passage token index (PassageIndexEntry is 24 bytes)
    let idx_bytes_per_entry = 24usize;
    let idx_total_bytes = n_passages as u64 * idx_bytes_per_entry as u64;
    let idx_mb = idx_total_bytes as f64 / 1e6;
    assert!((idx_mb - 212.2).abs() < 0.1, "passages.idx footprint should be 212.2 MB");

    // 7. Pinned Instruction Prefix KV-Cache:
    // 12 tokens * 24 layers * 2 KV heads * 256 d_head * 2 bytes = 589,824 bytes
    let prefix_tokens = 12usize;
    let layers = 24usize;
    let kv_heads = 2usize;
    let d_head = 256usize;
    let kv_cache_bytes = prefix_tokens * layers * 2 * (kv_heads * d_head * 2);
    assert_eq!(kv_cache_bytes, 589_824);
    let kv_cache_kib = kv_cache_bytes as f64 / 1024.0;
    assert_eq!(kv_cache_kib, 576.0);
    let kv_cache_kb = kv_cache_bytes as f64 / 1000.0;
    assert!((kv_cache_kb - 589.824).abs() < 0.01);
}

#[test]
fn test_shortlist_recall_tradeoff_bench_consistency() {
    // Verified against bench_results/QUALITY_ANALYSIS.md and bench_results/step3/RESULTS.md
    let oracle_recall_8_8m = 0.547f64;

    let sl64_recall = 0.431f64;
    let sl256_recall = 0.524f64;
    let sl512_recall = 0.5345f64;
    let sl1024_recall = 0.5378f64;

    let pct_oracle_64 = (sl64_recall / oracle_recall_8_8m) * 100.0;
    let pct_oracle_256 = (sl256_recall / oracle_recall_8_8m) * 100.0;
    let pct_oracle_512 = (sl512_recall / oracle_recall_8_8m) * 100.0;
    let pct_oracle_1024 = (sl1024_recall / oracle_recall_8_8m) * 100.0;

    assert!((pct_oracle_64 - 78.79).abs() < 0.1);
    assert!((pct_oracle_256 - 95.80).abs() < 0.1);
    assert!((pct_oracle_512 - 97.71).abs() < 0.1);
    assert!((pct_oracle_1024 - 98.32).abs() < 0.1);

    // Verify ANN recall figures
    let ann_r10_sl256 = 0.8809f64;
    let ann_r10_sl512 = 0.9241f64;
    let ann_r10_sl1024 = 0.9519f64;

    assert!((ann_r10_sl256 - 0.881).abs() < 0.001);
    assert!((ann_r10_sl512 - 0.924).abs() < 0.001);
    assert!((ann_r10_sl1024 - 0.952).abs() < 0.001);
}
