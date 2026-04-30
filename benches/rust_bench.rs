use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nano_numpy_simd::ops::{elementwise_naive_rust, elementwise_simd, Op};

fn bench_add(c: &mut Criterion) {
    let a = vec![1.0_f32; 1_000_000];
    let b = vec![2.0_f32; 1_000_000];

    c.bench_function("naive rust add", |bench| {
        bench.iter(|| {
            let result = elementwise_naive_rust(black_box(&a), black_box(&b), Op::Add).unwrap();
            black_box(result);
        });
    });

    c.bench_function("simd dispatched add", |bench| {
        bench.iter(|| {
            let result = elementwise_simd(black_box(&a), black_box(&b), Op::Add).unwrap();
            black_box(result);
        });
    });
}

criterion_group!(benches, bench_add);
criterion_main!(benches);
