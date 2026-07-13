//! Kernel micro-benchmarks across formats and shapes.
//!
//!     cargo bench -p undertow-quant
//!
//! Shapes mirror real projections: a decode matvec, an expert FFN matrix,
//! and a wide lm-head row block.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use undertow_quant::{QTensor, QuantFormat};

fn bench_matmul(c: &mut Criterion) {
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("decode-matvec", 1, 2048, 768),
        ("expert-ffn", 8, 2048, 1536),
        ("lm-head-block", 1, 2048, 8192),
    ];
    for &(name, seq, in_dim, out_dim) in shapes {
        let mut group = c.benchmark_group(name);
        let macs = (seq * in_dim * out_dim) as u64;
        group.throughput(Throughput::Elements(macs));
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|k| ((k as f32) * 0.13).sin() * 0.1)
            .collect();
        let x: Vec<f32> = (0..seq * in_dim)
            .map(|k| ((k as f32) * 0.31).cos())
            .collect();
        for fmt in [QuantFormat::F32, QuantFormat::Int8, QuantFormat::Int4] {
            let t = QTensor::quantize(&w, out_dim, in_dim, fmt).unwrap();
            let mut out = vec![0f32; seq * out_dim];
            group.bench_function(BenchmarkId::new("simd", fmt.name()), |b| {
                b.iter(|| t.matmul(std::hint::black_box(&mut out), &x, seq))
            });
            group.bench_function(BenchmarkId::new("scalar", fmt.name()), |b| {
                b.iter(|| t.matmul_scalar(std::hint::black_box(&mut out), &x, seq))
            });
        }
        // Opt-in integer path, int8 only.
        undertow_quant::set_fast_int8(true);
        let t = QTensor::quantize(&w, out_dim, in_dim, QuantFormat::Int8).unwrap();
        let mut out = vec![0f32; seq * out_dim];
        group.bench_function(BenchmarkId::new("fast-int8", "int8"), |b| {
            b.iter(|| t.matmul(std::hint::black_box(&mut out), &x, seq))
        });
        undertow_quant::set_fast_int8(false);
        group.finish();
    }
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
