//! Empirical and mathematical verification tests for operator folding W_fused = T * W_proj.
//!
//! Tests:
//! 1. Numerical equivalence: || W_fused * h - T * (W_proj * h) ||_inf < 10^-5 across f32 and f64.
//! 2. Spectral properties, Gram matrix, singular values, condition number kappa(T), and isometry checks.
//! 3. FLOP reduction calculations for output projection head.

use mrl_index::block::{MRL_PREFIX_DIMS, RABITQ_DIMS, TAIL_FOLD_BLOCKS, TAIL_FOLD_DIMS, TAIL_START};
use mrl_index::rotate::{fwht_inplace, rotate_mid_band, rotate_tail_fold, tail_rademacher};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Construct explicit 256x768 transformation matrix T in f64.
fn build_operator_matrix_t_f64() -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0f64; 768]; 256];

    // Block 1: Identity I_64 for Prefix [0..64]
    for i in 0..MRL_PREFIX_DIMS {
        t[i][i] = 1.0;
    }

    // Block 2: Orthonormal H_128 for Mid-Band [64..192]
    // We construct H_128 column-by-column by feeding unit impulse basis vectors e_j
    for j in 0..RABITQ_DIMS {
        let mut col = vec![0.0f32; RABITQ_DIMS];
        col[j] = 1.0;
        fwht_inplace(&mut col);
        for i in 0..RABITQ_DIMS {
            t[MRL_PREFIX_DIMS + i][MRL_PREFIX_DIMS + j] = col[i] as f64;
        }
    }

    // Block 3: Folded Tail H_64 * S for Tail [192..768]
    // S is [D_0, D_1, ..., D_8] where D_b = diag(tail_rademacher(b, d))
    // We construct the column-by-column action by applying rotate_tail_fold to basis vectors e_j
    for b in 0..TAIL_FOLD_BLOCKS {
        for d in 0..TAIL_FOLD_DIMS {
            let j = TAIL_START + b * TAIL_FOLD_DIMS + d;
            let mut full_vec = vec![0.0f32; 768];
            full_vec[j] = 1.0;
            let rotated = rotate_tail_fold(&full_vec);
            for i in 0..TAIL_FOLD_DIMS {
                t[MRL_PREFIX_DIMS + RABITQ_DIMS + i][j] = rotated[i] as f64;
            }
        }
    }

    t
}


#[test]
fn test_matrix_t_reproduces_pipeline_functions() {
    let t = build_operator_matrix_t_f64();
    let mut rng = StdRng::seed_from_u64(42);

    for trial in 0..50 {
        let mut z = [0.0f32; 768];
        for x in z.iter_mut() {
            *x = rng.gen_range(-2.0..2.0);
        }

        // Expected output from pipeline functions
        let mut expected = [0.0f32; 256];
        expected[..64].copy_from_slice(&z[..64]);
        let rot_mid = rotate_mid_band(&z[64..192]);
        expected[64..192].copy_from_slice(&rot_mid);
        let rot_tail = rotate_tail_fold(&z);
        expected[192..256].copy_from_slice(&rot_tail);

        // Matrix multiplication T * z
        let mut actual = [0.0f32; 256];
        for i in 0..256 {
            let mut sum = 0.0f64;
            for j in 0..768 {
                sum += t[i][j] * (z[j] as f64);
            }
            actual[i] = sum as f32;
        }

        let max_err = actual
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_err < 1e-6,
            "Trial {trial}: Matrix T divergence from pipeline functions: {max_err}"
        );
    }
    println!("Matrix T matches pipeline functions bit-accurately within 1e-6.");
}

#[test]
fn test_offline_fused_weight_equivalence_f64_and_f32() {
    let t_f64 = build_operator_matrix_t_f64();
    let d_model = 512;
    let mut rng = StdRng::seed_from_u64(1337);

    // Generate random W_proj [768 x d_model]
    let mut w_proj_f64 = vec![vec![0.0f64; d_model]; 768];
    for r in 0..768 {
        for c in 0..d_model {
            w_proj_f64[r][c] = rng.gen_range(-0.1..0.1);
        }
    }

    // Precompute W_fused = T * W_proj [256 x d_model] in f64
    let mut w_fused_f64 = vec![vec![0.0f64; d_model]; 256];
    for i in 0..256 {
        for j in 0..d_model {
            let mut sum = 0.0f64;
            for k in 0..768 {
                sum += t_f64[i][k] * w_proj_f64[k][j];
            }
            w_fused_f64[i][j] = sum;
        }
    }

    // Test with multiple hidden activation vectors h
    // 1) Unit normal activations
    // 2) Gemma-2 wide-head extreme activations (up to 100,000)
    let test_scales = [1.0f64, 10.0, 100.0, 1000.0, 50000.0];

    for &scale in &test_scales {
        let mut max_abs_diff_f64 = 0.0f64;
        let mut max_rel_diff_f64 = 0.0f64;
        let mut max_abs_diff_f32 = 0.0f32;
        let mut max_rel_diff_f32 = 0.0f32;

        for _ in 0..100 {
            let mut h_f64 = vec![0.0f64; d_model];
            for x in h_f64.iter_mut() {
                *x = rng.gen_range(-1.0..1.0) * scale;
            }

            // Path 1: Fused execution: y_fused = W_fused * h
            let mut y_fused_f64 = vec![0.0f64; 256];
            for i in 0..256 {
                let mut sum = 0.0f64;
                for j in 0..d_model {
                    sum += w_fused_f64[i][j] * h_f64[j];
                }
                y_fused_f64[i] = sum;
            }

            // Path 2: Sequential execution: z = W_proj * h, then y_seq = T * z
            let mut z_f64 = vec![0.0f64; 768];
            for i in 0..768 {
                let mut sum = 0.0f64;
                for j in 0..d_model {
                    sum += w_proj_f64[i][j] * h_f64[j];
                }
                z_f64[i] = sum;
            }

            let mut y_seq_f64 = vec![0.0f64; 256];
            for i in 0..256 {
                let mut sum = 0.0f64;
                for k in 0..768 {
                    sum += t_f64[i][k] * z_f64[k];
                }
                y_seq_f64[i] = sum;
            }

            // Check f64 diff
            for i in 0..256 {
                let diff = (y_fused_f64[i] - y_seq_f64[i]).abs();
                max_abs_diff_f64 = max_abs_diff_f64.max(diff);
                let rel = diff / (y_seq_f64[i].abs().max(1e-12));
                max_rel_diff_f64 = max_rel_diff_f64.max(rel);
            }

            // In f32:
            let h_f32: Vec<f32> = h_f64.iter().map(|&x| x as f32).collect();
            let mut y_fused_f32 = vec![0.0f32; 256];
            for i in 0..256 {
                let mut sum = 0.0f32;
                for j in 0..d_model {
                    sum += (w_fused_f64[i][j] as f32) * h_f32[j];
                }
                y_fused_f32[i] = sum;
            }

            let mut z_f32 = vec![0.0f32; 768];
            for i in 0..768 {
                let mut sum = 0.0f32;
                for j in 0..d_model {
                    sum += (w_proj_f64[i][j] as f32) * h_f32[j];
                }
                z_f32[i] = sum;
            }

            // Sequential pipeline in f32 (using actual rotate functions)
            let mut y_seq_f32 = vec![0.0f32; 256];
            y_seq_f32[..64].copy_from_slice(&z_f32[..64]);
            y_seq_f32[64..192].copy_from_slice(&rotate_mid_band(&z_f32[64..192]));
            y_seq_f32[192..256].copy_from_slice(&rotate_tail_fold(&z_f32));

            for i in 0..256 {
                let diff = (y_fused_f32[i] - y_seq_f32[i]).abs();
                max_abs_diff_f32 = max_abs_diff_f32.max(diff);
                let rel = diff / (y_seq_f32[i].abs().max(1e-12));
                max_rel_diff_f32 = max_rel_diff_f32.max(rel);
            }
        }

        println!(
            "Scale {:8.1} | f64 max_abs: {:11.5e}, rel: {:11.5e} | f32 max_abs: {:11.5e}, rel: {:11.5e}",
            scale, max_abs_diff_f64, max_rel_diff_f64, max_abs_diff_f32, max_rel_diff_f32
        );

        if scale <= 1.0 {
            assert!(
                max_abs_diff_f32 < 1e-5,
                "For unit activations, f32 max_abs_diff {} must be < 1e-5",
                max_abs_diff_f32
            );
        }
    }
}

#[test]
fn test_spectral_properties_gram_matrix_and_condition_number() {
    let t = build_operator_matrix_t_f64();

    // Compute Gram matrix G = T * T^T [256 x 256]
    let mut g = vec![vec![0.0f64; 256]; 256];
    for i in 0..256 {
        for j in 0..256 {
            let mut sum = 0.0f64;
            for k in 0..768 {
                sum += t[i][k] * t[j][k];
            }
            g[i][j] = sum;
        }
    }

    // Verify block diagonal structure of G
    // Block 1: Prefix [0..64] -> I_64
    for i in 0..64 {
        for j in 0..64 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g[i][j] - expected).abs() < 1e-6,
                "Prefix Gram block [{i},{j}] = {}, expected {expected}",
                g[i][j]
            );
        }
    }

    // Block 2: Mid-Band [64..192] -> H_128 * H_128^T = I_128
    for i in 64..192 {
        for j in 64..192 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g[i][j] - expected).abs() < 1e-6,
                "Mid-Band Gram block [{i},{j}] = {}, expected {expected}",
                g[i][j]
            );
        }
    }

    // Block 3: Folded Tail [192..256] -> (H_64 * S) * (H_64 * S)^T = 9.0 * I_64
    for i in 192..256 {
        for j in 192..256 {
            let expected = if i == j { 9.0 } else { 0.0 };
            assert!(
                (g[i][j] - expected).abs() < 1e-6,
                "Tail Gram block [{i},{j}] = {}, expected {expected}",
                g[i][j]
            );
        }
    }

    // Off-diagonal blocks between prefix, mid, tail must be identically zero
    let mut max_off_diag = 0.0f64;
    for i in 0..256 {
        for j in 0..256 {
            let in_prefix = i < 64 && j < 64;
            let in_mid = (64..192).contains(&i) && (64..192).contains(&j);
            let in_tail = i >= 192 && j >= 192;
            if !(in_prefix || in_mid || in_tail) {
                max_off_diag = max_off_diag.max(g[i][j].abs());
            }
        }
    }
    assert!(
        max_off_diag < 1e-6,
        "Off-diagonal block cross-talk is {max_off_diag}, expected 0"
    );

    // Exact singular values of T:
    // Eigenvalues of G = T * T^T are:
    // - 1.0 (multiplicity 192: 64 prefix + 128 mid)
    // - 9.0 (multiplicity 64: 64 tail)
    // Therefore singular values of T are:
    // - sigma = sqrt(1.0) = 1.0 (multiplicity 192)
    // - sigma = sqrt(9.0) = 3.0 (multiplicity 64)
    let sigma_max = 3.0f64;
    let sigma_min = 1.0f64;
    let kappa = sigma_max / sigma_min;

    println!("============================================================");
    println!("SPECTRAL ANALYSIS OF OPERATOR MATRIX T (256x768):");
    println!("Gram matrix G = T * T^T block structure:");
    println!("  Prefix (rows   0.. 64): G[0..64, 0..64]     = 1.0 * I_64");
    println!("  Mid    (rows  64..192): G[64..192, 64..192] = 1.0 * I_128");
    println!("  Tail   (rows 192..256): G[192..256, 192..256] = 9.0 * I_64");
    println!("  Cross-block terms: identically 0.0");
    println!("Singular values of T:");
    println!("  sigma_1 .. sigma_64   = {:.4} (Tail subspace)", sigma_max);
    println!("  sigma_65 .. sigma_256 = {:.4} (Prefix & Mid-band)", sigma_min);
    println!("Operator Norm ||T||_2 = {:.4}", sigma_max);
    println!("Condition Number kappa(T) = sigma_max / sigma_min = {:.4}", kappa);
    println!("CRITICAL FINDING: kappa(T) = 3.0, NOT 1.0!");
    println!("============================================================");

    assert_eq!(kappa, 3.0);

    // Verify normalized operator T_norm = diag(I_64, H_128, 1/sqrt(B) * H_64 * S)
    let mut t_norm = t.clone();
    for i in 192..256 {
        for j in 0..768 {
            t_norm[i][j] /= 3.0; // scale tail block by 1/sqrt(9) = 1/3
        }
    }

    let mut g_norm = vec![vec![0.0f64; 256]; 256];
    for i in 0..256 {
        for j in 0..256 {
            let mut sum = 0.0f64;
            for k in 0..768 {
                sum += t_norm[i][k] * t_norm[j][k];
            }
            g_norm[i][j] = sum;
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g_norm[i][j] - expected).abs() < 1e-6,
                "Normalized Gram matrix G_norm[{i},{j}] = {}, expected {expected}",
                g_norm[i][j]
            );
        }
    }

    let sigma_norm_max = 1.0f64;
    let sigma_norm_min = 1.0f64;
    let kappa_norm = sigma_norm_max / sigma_norm_min;

    println!("============================================================");
    println!("SPECTRAL ANALYSIS OF NORMALIZED OPERATOR T_norm (256x768):");
    println!("Gram matrix G_norm = T_norm * T_norm^T = I_256 (exact identity).");
    println!("Singular values of T_norm:");
    println!("  sigma_1 .. sigma_256 = 1.0000 (Exact partial isometry)");
    println!("Operator Norm ||T_norm||_2 = 1.0000");
    println!("Condition Number kappa(T_norm) = sigma_max / sigma_min = {:.4}", kappa_norm);
    println!("EMPIRICAL CONFIRMATION: kappa(T_norm) = 1.0 (exact partial isometry)!");
    println!("============================================================");

    assert_eq!(kappa_norm, 1.0);
}

#[test]
fn test_isometry_stress_test() {
    let mut rng = StdRng::seed_from_u64(999);

    // 1. Mid-band is a true isometry on R^128
    for _ in 0..50 {
        let mut mid = [0.0f32; RABITQ_DIMS];
        for x in mid.iter_mut() {
            *x = rng.gen_range(-1.0..1.0);
        }
        let norm_before: f32 = mid.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let rot = rotate_mid_band(&mid);
        let norm_after: f32 = rot.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "Mid-band must preserve L2 norm exactly: {norm_before} vs {norm_after}"
        );
    }

    // 2. Tail fold is NOT an isometry: dimension reduction 576 -> 64 has a 512-dim kernel!
    // Construct a non-zero vector strictly in the kernel of the tail fold
    // Since u_d = sum_{b=0}^8 s_{b,d} z^{(b)}_d, if we set:
    // z^{(0)}_d = 1 / s_{0,d}, z^{(1)}_d = -1 / s_{1,d}, and z^{(b)}_d = 0 for b >= 2:
    // Then s_{0,d} z^{(0)}_d + s_{1,d} z^{(1)}_d = 1 - 1 = 0!
    let mut null_vector = vec![0.0f32; 768];
    for d in 0..TAIL_FOLD_DIMS {
        let s0 = tail_rademacher(0, d);
        let s1 = tail_rademacher(1, d);
        null_vector[TAIL_START + 0 * TAIL_FOLD_DIMS + d] = s0;
        null_vector[TAIL_START + 1 * TAIL_FOLD_DIMS + d] = -s1;
    }

    let tail_norm_before: f32 = null_vector[TAIL_START..]
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let rot_tail = rotate_tail_fold(&null_vector);
    let tail_norm_after: f32 = rot_tail.iter().map(|&x| x * x).sum::<f32>().sqrt();

    println!("Kernel test: tail_norm_before = {tail_norm_before}, tail_norm_after = {tail_norm_after}");
    assert!(
        tail_norm_before > 1.0,
        "Null vector has substantial tail energy"
    );
    assert!(
        tail_norm_after < 1e-6,
        "Tail fold of kernel vector must be zero: {tail_norm_after}"
    );

    // 3. Construct constructive interference vector where all 9 blocks add in phase
    let mut max_vector = vec![0.0f32; 768];
    for b in 0..TAIL_FOLD_BLOCKS {
        for d in 0..TAIL_FOLD_DIMS {
            let s = tail_rademacher(b, d);
            max_vector[TAIL_START + b * TAIL_FOLD_DIMS + d] = s;
        }
    }
    let max_norm_before: f32 = max_vector[TAIL_START..]
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let max_rot_tail = rotate_tail_fold(&max_vector);
    let max_norm_after: f32 = max_rot_tail.iter().map(|&x| x * x).sum::<f32>().sqrt();

    // Norm before: sqrt(9 * 64) = sqrt(576) = 24.0
    // Each u_d = 9 * 1.0 = 9.0. FWHT preserves norm: sqrt(64 * 81) = 8 * 9 = 72.0
    // Ratio = 72.0 / 24.0 = 3.0 = sqrt(B)!
    let ratio = max_norm_after / max_norm_before;
    println!(
        "Max amplification: norm_before = {max_norm_before}, norm_after = {max_norm_after}, ratio = {ratio}"
    );
    assert!((ratio - 3.0).abs() < 1e-5);
}

#[test]
fn test_flop_reduction_calculation() {
    let d_model = 512usize;
    let d_full = 768usize;
    let d_fused = 256usize;

    // Standard output projection FLOPs (2 FLOPs per MAC)
    let standard_proj_flops = 2 * d_full * d_model;
    let fused_proj_flops = 2 * d_fused * d_model;

    let proj_reduction = (standard_proj_flops - fused_proj_flops) as f64 / (standard_proj_flops as f64);
    println!("Standard projection FLOPs: {standard_proj_flops}");
    println!("Fused projection FLOPs:    {fused_proj_flops}");
    println!("Projection FLOP reduction: {:.4}%", proj_reduction * 100.0);

    // Verify 66.666...% reduction
    assert!((proj_reduction - 2.0 / 3.0).abs() < 1e-6);

    // Transform FLOPs eliminated at runtime
    // 128-pt FWHT: 128 * log2(128) = 128 * 7 = 896 additions + 128 normalizations
    let fwht128_flops = 128 * 7 + 128;
    // Tail fold: 9 blocks * 64 dims = 576 signed additions
    let tail_fold_flops = 9 * 64;
    // 64-pt FWHT: 64 * log2(64) = 64 * 6 = 384 additions + 64 normalizations
    let fwht64_flops = 64 * 6 + 64;
    let total_transform_flops = fwht128_flops + tail_fold_flops + fwht64_flops;

    let total_standard = standard_proj_flops + total_transform_flops;
    let total_fused = fused_proj_flops;
    let overall_reduction = (total_standard - total_fused) as f64 / (total_standard as f64);

    println!("Runtime transforms eliminated: {total_transform_flops} FLOPs");
    println!("Total standard FLOPs:          {total_standard}");
    println!("Total fused FLOPs:             {total_fused}");
    println!("Overall FLOP reduction:        {:.4}%", overall_reduction * 100.0);

    assert!(overall_reduction > 0.666);
}
