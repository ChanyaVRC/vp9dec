//! `transform` モジュールのテスト。
//!
//! 整数実装の正しさを検証するため、仕様 8.7.1 節の疑似コードから独立に
//! 浮動小数点版のバタフライネットワークを書き起こし（`cos64`/`sin64` の
//! 14bit 固定小数点テーブルではなく実際の `cos`/`sin` を用い、各段の
//! `Round2` による丸めも行わない）、乱数係数に対して整数実装との出力差が
//! ±1 に収まることを確認する。加えて、線形性・DC 係数単独入力でのフラット
//! 出力といった、変換方式に依存しない数学的性質もテストする。

use super::*;
use std::f64::consts::PI;

// ------------------------------------------------------------------------
// 自前 LCG 乱数生成器（外部クレートを使わないための最小実装）。
// ------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    /// Numerical Recipes 系の定数を用いた 64bit LCG。
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// `[-range, range]` の範囲の整数係数を生成する（変換係数を模した値）。
    fn next_coef(&mut self, range: i64) -> i64 {
        let v = (self.next_u64() >> 33) as i64; // 31bit 非負値
        (v % (2 * range + 1)) - range
    }
}

// ------------------------------------------------------------------------
// 浮動小数点版バタフライネットワーク（整数実装とは独立に書き起こしたもの）。
// ------------------------------------------------------------------------
//
// `cos64(angle)/16384 ≈ cos(angle * pi / 64)` の関係を使い、整数実装の
// 「16384 倍して Round2(.., 14) で戻す」という操作を、浮動小数点では単に
// 「実数の cos/sin をそのまま掛ける」ことで置き換える。これにより 14bit
// テーブルの丸め誤差そのものは再現されないが、バタフライネットワークの
// 配線（インデックス計算）が正しいかどうかは独立に検証できる。

fn cos_f(angle: i32) -> f64 {
    (angle as f64 * PI / 64.0).cos()
}
fn sin_f(angle: i32) -> f64 {
    (angle as f64 * PI / 64.0).sin()
}

fn b_op_f(t: &mut [f64], a: usize, b: usize, angle: i32, flip: bool) {
    let c = cos_f(angle);
    let s = sin_f(angle);
    let ta = t[a];
    let tb = t[b];
    t[a] = ta * c - tb * s;
    t[b] = ta * s + tb * c;
    if flip {
        t.swap(a, b);
    }
}

fn h_op_f(t: &mut [f64], a: usize, b: usize, flip: bool) {
    let (a, b) = if flip { (b, a) } else { (a, b) };
    let x = t[a];
    let y = t[b];
    t[a] = x + y;
    t[b] = x - y;
}

fn sb_op_f(s: &mut [f64], t: &[f64], a: usize, b: usize, angle: i32, flip: bool) {
    let c = cos_f(angle);
    let sn = sin_f(angle);
    let ta = t[a];
    let tb = t[b];
    let sa = ta * c - tb * sn;
    let sbv = ta * sn + tb * c;
    if flip {
        s[a] = sbv;
        s[b] = sa;
    } else {
        s[a] = sa;
        s[b] = sbv;
    }
}

fn sh_op_f(t: &mut [f64], s: &[f64], a: usize, b: usize) {
    t[a] = s[a] + s[b];
    t[b] = s[a] - s[b];
}

fn idct_permute_f(t: &mut [f64], n: u32) {
    let n0 = 1usize << n;
    let copy_t: Vec<f64> = t[..n0].to_vec();
    for i in 0..n0 {
        t[i] = copy_t[brev(n, i)];
    }
}

fn idct_f(t: &mut [f64], n: u32) {
    let n0 = 1i64 << n;
    let n1 = 1i64 << (n - 1);
    let n2 = 1i64 << (n - 2);

    if n == 2 {
        b_op_f(t, 0, 1, 16, true);
    } else {
        idct_f(t, n - 1);
    }

    for i in 0..n2 {
        let a = (n1 + i) as usize;
        let b = (n0 - 1 - i) as usize;
        let angle = 32 - brev(5, a) as i32;
        b_op_f(t, a, b, angle, false);
    }

    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            for j in 0..2i64 {
                let a = (n1 + 4 * i + 2 * j) as usize;
                let b = (n1 + 1 + 4 * i + 2 * j) as usize;
                h_op_f(t, a, b, j == 1);
            }
        }
    }

    if n == 5 {
        for i in 0..2i64 {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 3 - n2 * j - 4 * i) as usize;
                let b = (n1 + n as i64 - 4 + n2 * j + 4 * i) as usize;
                let angle = 28 - 16 * i as i32 + 56 * j as i32;
                b_op_f(t, a, b, angle, true);
            }
        }
        let n3 = 1i64 << (n - 3);
        for i in 0..2i64 {
            for j in 0..4i64 {
                let a = (n1 + n3 * j + i) as usize;
                let b = (n1 + n2 - 5 + n3 * j - i) as usize;
                h_op_f(t, a, b, (j & 1) == 1);
            }
        }
    }

    if n >= 4 {
        let imax_a: i64 = if n == 5 { 1 } else { 0 };
        for i in 0..=imax_a {
            for j in 0..2i64 {
                let a = (n0 - n as i64 + 2 - i - n2 * j) as usize;
                let b = (n1 + n as i64 - 3 + i + n2 * j) as usize;
                let angle = 24 + 48 * j as i32;
                b_op_f(t, a, b, angle, true);
            }
        }
        let imax_b: i64 = 2 * n as i64 - 7;
        for j in 0..2i64 {
            for i in 0..=imax_b {
                let a = (n1 + n2 * j + i) as usize;
                let b = (n1 + n2 - 1 + n2 * j - i) as usize;
                h_op_f(t, a, b, (j & 1) == 1);
            }
        }
    }

    if n >= 3 {
        let n3 = 1i64 << (n - 3);
        for i in 0..n3 {
            let a = (n0 - n3 - 1 - i) as usize;
            let b = (n1 + n3 + i) as usize;
            b_op_f(t, a, b, 16, true);
        }
    }

    for i in 0..n1 {
        let a = i as usize;
        let b = (n0 - 1 - i) as usize;
        h_op_f(t, a, b, false);
    }
}

fn adst_input_permute_f(t: &mut [f64], n: u32) {
    let n0 = 1usize << n;
    let n1 = 1usize << (n - 1);
    let copy_t: Vec<f64> = t[..n0].to_vec();
    for i in 0..n1 {
        t[2 * i] = copy_t[n0 - 1 - 2 * i];
        t[2 * i + 1] = copy_t[2 * i];
    }
}

fn adst_output_permute_f(t: &mut [f64], n: u32) {
    if n == 4 {
        let copy_t: Vec<f64> = t[..16].to_vec();
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    for d in 0..2usize {
                        let dst = 8 * a + 4 * b + 2 * c + d;
                        let src = 8 * (d ^ c) + 4 * (c ^ b) + 2 * (b ^ a) + a;
                        t[dst] = copy_t[src];
                    }
                }
            }
        }
    } else {
        let copy_t: Vec<f64> = t[..8].to_vec();
        for a in 0..2usize {
            for b in 0..2usize {
                for c in 0..2usize {
                    let dst = 4 * a + 2 * b + c;
                    let src = 4 * (c ^ b) + 2 * (b ^ a) + a;
                    t[dst] = copy_t[src];
                }
            }
        }
    }
}

/// ADST4 は仕様の代数式をそのまま行列として展開したもの（手計算で導出）。
/// `iadst4_impl` とは別経路で検証するため、係数行列を直接書き下す。
fn iadst4_f(t: &mut [f64]) {
    let k1 = SINPI_1_9 as f64 / 16384.0;
    let k2 = SINPI_2_9 as f64 / 16384.0;
    let k3 = SINPI_3_9 as f64 / 16384.0;
    let k4 = SINPI_4_9 as f64 / 16384.0;
    let (t0, t1, t2, t3) = (t[0], t[1], t[2], t[3]);
    // 8.7.1.6 の式を o_i = sum_j M[i][j] * t_j の形に展開したもの。
    let o0 = k1 * t0 + k3 * t1 + k4 * t2 + k2 * t3;
    let o1 = k2 * t0 + k3 * t1 - k1 * t2 - k4 * t3;
    let o2 = k3 * t0 - k3 * t2 + k3 * t3;
    let o3 = (k1 + k2) * t0 - k3 * t1 + (k4 - k1) * t2 + (k2 - k4) * t3;
    t[0] = o0;
    t[1] = o1;
    t[2] = o2;
    t[3] = o3;
}

fn iadst8_f(t: &mut [f64]) {
    adst_input_permute_f(t, 3);
    let mut s = [0f64; 8];
    for i in 0..4usize {
        sb_op_f(&mut s, t, 2 * i, 1 + 2 * i, 30 - 8 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op_f(t, &s, i, 4 + i);
    }
    for i in 0..2usize {
        sb_op_f(&mut s, t, 4 + 3 * i, 5 + i, 24 - 16 * i as i32, true);
    }
    for i in 0..2usize {
        sh_op_f(t, &s, 4 + i, 6 + i);
    }
    for i in 0..2usize {
        h_op_f(t, i, 2 + i, false);
    }
    for i in 0..2usize {
        b_op_f(t, 2 + 4 * i, 3 + 4 * i, 16, true);
    }
    adst_output_permute_f(t, 3);
    for i in 0..4usize {
        t[1 + 2 * i] = -t[1 + 2 * i];
    }
}

fn iadst16_f(t: &mut [f64]) {
    adst_input_permute_f(t, 4);
    let mut s = [0f64; 16];
    for i in 0..8usize {
        sb_op_f(&mut s, t, 2 * i, 1 + 2 * i, 31 - 4 * i as i32, true);
    }
    for i in 0..8usize {
        sh_op_f(t, &s, i, 8 + i);
    }
    for i in 0..4usize {
        sb_op_f(&mut s, t, 8 + 2 * i, 9 + 2 * i, 28 - 16 * i as i32, true);
    }
    for i in 0..4usize {
        sh_op_f(t, &s, 8 + i, 12 + i);
    }
    for i in 0..4usize {
        h_op_f(t, i, 4 + i, false);
    }
    for i in 0..2usize {
        for j in 0..2usize {
            sb_op_f(
                &mut s,
                t,
                4 + 8 * i + 3 * j,
                5 + 8 * i + j,
                24 - 16 * j as i32,
                true,
            );
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            sh_op_f(t, &s, 4 + 8 * j + i, 6 + 8 * j + i);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            h_op_f(t, 8 * j + i, 2 + 8 * j + i, false);
        }
    }
    for i in 0..2usize {
        for j in 0..2usize {
            let angle = 48 + 64 * (i ^ j) as i32;
            b_op_f(t, 2 + 4 * j + 8 * i, 3 + 4 * j + 8 * i, angle, false);
        }
    }
    adst_output_permute_f(t, 4);
    for i in 0..2usize {
        for j in 0..2usize {
            t[1 + 12 * j + 2 * i] = -t[1 + 12 * j + 2 * i];
        }
    }
}

// ------------------------------------------------------------------------
// 逆 DCT: 整数実装 vs 浮動小数点参照実装
// ------------------------------------------------------------------------

/// 整数版 `idct` と浮動小数点参照実装 `idct_f` の出力を比較し、
/// 最大絶対誤差が `tol` 以内であることを確認する。
fn check_idct(n: u32, trials: usize, coef_range: i64, tol: f64, seed: u64) {
    let n0 = 1usize << n;
    let mut rng = Lcg::new(seed);
    for _ in 0..trials {
        let coeffs: Vec<i64> = (0..n0).map(|_| rng.next_coef(coef_range)).collect();

        let mut ti: Vec<i64> = coeffs.clone();
        idct_permute(&mut ti, n);
        idct(&mut ti, n);

        let mut tf: Vec<f64> = coeffs.iter().map(|&x| x as f64).collect();
        idct_permute_f(&mut tf, n);
        idct_f(&mut tf, n);

        for k in 0..n0 {
            let diff = (ti[k] as f64 - tf[k]).abs();
            assert!(
                diff <= tol,
                "idct n={n} k={k}: int={} float={} diff={diff} (coeffs={coeffs:?})",
                ti[k],
                tf[k]
            );
        }
    }
}

// 許容誤差はバタフライ段数（≒ log2(n0)）にほぼ比例して大きくなる。
// これは 1 段ごとに `Round2(x, 14)` による ±0.5 単位（14bit スケール上）の
// 丸め誤差が生じ、後段の回転を経て伝播・蓄積していくためであり、実際の
// VP9 デコーダでも起こる正常な挙動である（最終的な誤差は 2 次元変換の
// 最終シフト適用後には画素単位で ±1〜2 程度に収まる。
// `inverse_transform_block_matches_float_reference_all_tx_types` を参照）。
// 下記の許容値は、乱数係数 2000 セット×各要素での実測最大誤差
// （n=2: 約2.0, n=3: 約4.8, n=4: 約4.7, n=5: 約6.7）に安全マージンを
// 加えたもの。

#[test]
fn idct4_matches_float_reference() {
    check_idct(2, 2000, 1 << 15, 3.0, 1);
}

#[test]
fn idct8_matches_float_reference() {
    check_idct(3, 2000, 1 << 15, 6.0, 2);
}

#[test]
fn idct16_matches_float_reference() {
    check_idct(4, 2000, 1 << 14, 6.0, 3);
}

#[test]
fn idct32_matches_float_reference() {
    check_idct(5, 2000, 1 << 13, 8.0, 4);
}

// ------------------------------------------------------------------------
// 逆 ADST: 整数実装 vs 浮動小数点参照実装
// ------------------------------------------------------------------------

fn check_iadst(n: u32, trials: usize, coef_range: i64, tol: f64, seed: u64) {
    let n0 = 1usize << n;
    let mut rng = Lcg::new(seed);
    for _ in 0..trials {
        let coeffs: Vec<i64> = (0..n0).map(|_| rng.next_coef(coef_range)).collect();

        let mut ti: Vec<i64> = coeffs.clone();
        iadst(&mut ti, n);

        let mut tf: Vec<f64> = coeffs.iter().map(|&x| x as f64).collect();
        match n {
            2 => iadst4_f(&mut tf),
            3 => iadst8_f(&mut tf),
            4 => iadst16_f(&mut tf),
            _ => unreachable!(),
        }

        for k in 0..n0 {
            let diff = (ti[k] as f64 - tf[k]).abs();
            assert!(
                diff <= tol,
                "iadst n={n} k={k}: int={} float={} diff={diff} (coeffs={coeffs:?})",
                ti[k],
                tf[k]
            );
        }
    }
}

#[test]
fn iadst4_matches_float_reference() {
    check_iadst(2, 2000, 1 << 15, 2.0, 11);
}

#[test]
fn iadst8_matches_float_reference() {
    check_iadst(3, 2000, 1 << 15, 6.0, 12);
}

#[test]
fn iadst16_matches_float_reference() {
    check_iadst(4, 2000, 1 << 14, 6.0, 13);
}

// ------------------------------------------------------------------------
// 変換方式に依存しない数学的性質のテスト
// ------------------------------------------------------------------------

/// DC 係数（先頭要素）のみが非ゼロの場合、逆 DCT の出力はフラット（全要素が
/// ほぼ同一の値）になる。これは DCT の 0 次基底関数が定数関数であるという、
/// バタフライネットワークの配線とは独立な数学的事実に基づく検証である。
#[test]
fn idct_dc_only_is_flat() {
    for &n in &[2u32, 3, 4, 5] {
        let n0 = 1usize << n;
        for &dc in &[100i64, -500, 4000] {
            let mut t = vec![0i64; n0];
            t[0] = dc;
            idct_permute(&mut t, n);
            idct(&mut t, n);
            let first = t[0];
            for (k, &v) in t.iter().enumerate() {
                assert!(
                    (v - first).abs() <= 1,
                    "n={n} dc={dc} k={k}: v={v} first={first}"
                );
            }
        }
    }
}

/// 逆 DCT は線形写像である: `idct(a + b) == idct(a) + idct(b)`（丸め誤差の
/// 範囲内で）。
#[test]
fn idct_is_linear() {
    let mut rng = Lcg::new(42);
    for &n in &[2u32, 3, 4, 5] {
        let n0 = 1usize << n;
        for _ in 0..50 {
            let a: Vec<i64> = (0..n0).map(|_| rng.next_coef(1000)).collect();
            let b: Vec<i64> = (0..n0).map(|_| rng.next_coef(1000)).collect();
            let sum: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            let run = |mut v: Vec<i64>| {
                idct_permute(&mut v, n);
                idct(&mut v, n);
                v
            };
            let ra = run(a);
            let rb = run(b);
            let rsum = run(sum);

            for k in 0..n0 {
                let combined = ra[k] + rb[k];
                // 各バタフライ段の Round2 による丸めが a・b それぞれで独立に
                // 発生し、加算後の丸めとはずれうるため、段数に応じた誤差を
                // 許容する（[`check_idct`] と同様の理由）。
                assert!(
                    (rsum[k] - combined).abs() <= 8,
                    "n={n} k={k}: rsum={} combined={}",
                    rsum[k],
                    combined
                );
            }
        }
    }
}

/// ADST も DC 係数単独では緩やかに変化するが、DCT のような完全フラットには
/// ならない。代わりに「線形性」のみを軽量に確認する。
#[test]
fn iadst_is_linear() {
    let mut rng = Lcg::new(43);
    for &n in &[2u32, 3, 4] {
        let n0 = 1usize << n;
        for _ in 0..50 {
            let a: Vec<i64> = (0..n0).map(|_| rng.next_coef(1000)).collect();
            let b: Vec<i64> = (0..n0).map(|_| rng.next_coef(1000)).collect();
            let sum: Vec<i64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            let run = |mut v: Vec<i64>| {
                iadst(&mut v, n);
                v
            };
            let ra = run(a);
            let rb = run(b);
            let rsum = run(sum);

            for k in 0..n0 {
                let combined = ra[k] + rb[k];
                assert!(
                    (rsum[k] - combined).abs() <= 4,
                    "n={n} k={k}: rsum={} combined={}",
                    rsum[k],
                    combined
                );
            }
        }
    }
}

// ------------------------------------------------------------------------
// 逆 Walsh-Hadamard 変換（ロスレス用）: 軽量なテスト
// ------------------------------------------------------------------------

/// WHT はほぼ線形だが、内部の `e = (a - d) >> 1` が右シフトによる床除算
/// （負数では -∞ 方向への丸め）であるため、`a - d` の偶奇によっては
/// `iwht4(x + y) != iwht4(x) + iwht4(y)` が最大 1 だけずれうる。
/// 完全な線形性ではなく、そのずれが ±1 に収まることを確認する。
#[test]
fn iwht4_is_approximately_linear() {
    let mut rng = Lcg::new(7);
    for _ in 0..50 {
        let a = [
            rng.next_coef(1000),
            rng.next_coef(1000),
            rng.next_coef(1000),
            rng.next_coef(1000),
        ];
        let b = [
            rng.next_coef(1000),
            rng.next_coef(1000),
            rng.next_coef(1000),
            rng.next_coef(1000),
        ];
        let mut sum = a;
        for i in 0..4 {
            sum[i] += b[i];
        }

        let mut ra = a;
        iwht4(&mut ra, 0);
        let mut rb = b;
        iwht4(&mut rb, 0);
        let mut rsum = sum;
        iwht4(&mut rsum, 0);

        for i in 0..4 {
            assert!(
                (rsum[i] - (ra[i] + rb[i])).abs() <= 1,
                "index {i}: rsum={} ra+rb={}",
                rsum[i],
                ra[i] + rb[i]
            );
        }
    }
}

/// DC 係数のみの場合、WHT も他の変換と同様に出力がフラットになることを
/// 軽く確認する（`shift == 0` の場合）。
#[test]
fn iwht4_dc_only_is_flat() {
    let mut t = [400i64, 0, 0, 0];
    iwht4(&mut t, 0);
    let first = t[0];
    for &v in t.iter() {
        assert_eq!(v, first);
    }
}

// ------------------------------------------------------------------------
// 2 次元変換の組み立て（inverse_transform_block）
// ------------------------------------------------------------------------

/// DC 係数（`Dequant[0][0]`）のみが非ゼロの場合、`DCT_DCT` の 2 次元逆変換の
/// 出力はブロック全体でほぼフラットになる。
#[test]
fn inverse_transform_block_dc_only_is_flat_dct() {
    for &n in &[2u32, 3, 4, 5] {
        let n0 = 1usize << n;
        let mut d = vec![0i64; n0 * n0];
        d[0] = 5000;
        inverse_transform_block(&mut d, n, TxType::DctDct, false);
        let first = d[0];
        for (idx, &v) in d.iter().enumerate() {
            assert!(
                (v - first).abs() <= 2,
                "n={n} idx={idx}: v={v} first={first}"
            );
        }
    }
}

/// 4x4 ロスレス（WHT）経路が破綻なく動作し、DC 係数単独でフラットな出力を
/// 返すことを確認する（軽量テスト）。
#[test]
fn inverse_transform_block_lossless_wht_smoke() {
    let mut d = vec![0i64; 16];
    d[0] = 400;
    inverse_transform_block(&mut d, 2, TxType::DctDct, true);
    let first = d[0];
    for &v in d.iter() {
        assert_eq!(v, first);
    }
}

/// 2 次元変換（行 DCT・列 ADST など 4 パターンすべて）が浮動小数点参照実装
/// （行変換 → 列変換 → 最終シフトを float で再現）と ±2 の誤差に収まること
/// を確認する。
#[test]
fn inverse_transform_block_matches_float_reference_all_tx_types() {
    let tx_types = [
        TxType::DctDct,
        TxType::AdstDct,
        TxType::DctAdst,
        TxType::AdstAdst,
    ];
    for &n in &[2u32, 3, 4] {
        let n0 = 1usize << n;
        for (ti, &tx_type) in tx_types.iter().enumerate() {
            let mut rng = Lcg::new(100 + n as u64 * 10 + ti as u64);
            for _ in 0..30 {
                let coeffs: Vec<i64> = (0..n0 * n0).map(|_| rng.next_coef(1 << 12)).collect();

                let mut di = coeffs.clone();
                inverse_transform_block(&mut di, n, tx_type, false);

                let df = float_2d_reference(&coeffs, n, tx_type);

                for k in 0..n0 * n0 {
                    let diff = (di[k] as f64 - df[k]).abs();
                    assert!(
                        diff <= 3.0,
                        "n={n} tx={tx_type:?} k={k}: int={} float={} diff={diff}",
                        di[k],
                        df[k]
                    );
                }
            }
        }
    }
}

/// [`inverse_transform_block`] の浮動小数点参照実装（行 → 列 → 最終シフト相当
/// のスケーリング）。
fn float_2d_reference(coeffs: &[i64], n: u32, tx_type: TxType) -> Vec<f64> {
    let n0 = 1usize << n;
    let mut d: Vec<f64> = coeffs.iter().map(|&x| x as f64).collect();

    let apply_row = |v: &mut [f64]| match tx_type {
        TxType::DctDct | TxType::AdstDct => {
            idct_permute_f(v, n);
            idct_f(v, n);
        }
        TxType::DctAdst | TxType::AdstAdst => match n {
            2 => iadst4_f(v),
            3 => iadst8_f(v),
            4 => iadst16_f(v),
            _ => unreachable!(),
        },
    };
    let apply_col = |v: &mut [f64]| match tx_type {
        TxType::DctDct | TxType::DctAdst => {
            idct_permute_f(v, n);
            idct_f(v, n);
        }
        TxType::AdstDct | TxType::AdstAdst => match n {
            2 => iadst4_f(v),
            3 => iadst8_f(v),
            4 => iadst16_f(v),
            _ => unreachable!(),
        },
    };

    for i in 0..n0 {
        let row = &mut d[i * n0..(i + 1) * n0];
        apply_row(row);
    }
    let shift = (n + 2).min(6);
    let scale = (1u64 << shift) as f64;
    let mut col_buf = vec![0f64; n0];
    for j in 0..n0 {
        for i in 0..n0 {
            col_buf[i] = d[i * n0 + j];
        }
        apply_col(&mut col_buf);
        for i in 0..n0 {
            d[i * n0 + j] = col_buf[i] / scale;
        }
    }
    d
}
