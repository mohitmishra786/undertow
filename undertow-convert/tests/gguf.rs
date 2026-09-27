use std::fs::File;
use std::io::{Seek, Write};
use tempfile::tempdir;
use undertow_convert::{convert_gguf, is_gguf_file, ConvertOptions, GgufReader, TensorSource};
use undertow_io::ShardedModelReader;

fn write_string<W: Write>(w: &mut W, s: &str) {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
    w.write_all(bytes).unwrap();
}

fn write_meta_u64<W: Write>(w: &mut W, key: &str, val: u64) {
    write_string(w, key);
    w.write_all(&10u32.to_le_bytes()).unwrap(); // UINT64
    w.write_all(&val.to_le_bytes()).unwrap();
}

fn write_meta_str<W: Write>(w: &mut W, key: &str, val: &str) {
    write_string(w, key);
    w.write_all(&8u32.to_le_bytes()).unwrap(); // STRING
    write_string(w, val);
}

fn write_synthetic_gguf(path: &std::path::Path) {
    let mut file = File::create(path).unwrap();

    // 1. Header
    file.write_all(&0x46554747u32.to_le_bytes()).unwrap(); // "GGUF" magic
    file.write_all(&3u32.to_le_bytes()).unwrap(); // Version 3
    file.write_all(&3u64.to_le_bytes()).unwrap(); // 3 tensors
    file.write_all(&4u64.to_le_bytes()).unwrap(); // 4 metadata entries

    // 2. Metadata
    write_meta_str(&mut file, "general.architecture", "deepseek2");
    write_meta_u64(&mut file, "general.alignment", 32);
    write_meta_u64(&mut file, "deepseek2.block_count", 1);
    write_meta_u64(&mut file, "deepseek2.embedding_length", 32);

    // 3. Tensor infos
    // Tensor 0: blk.0.attn_q.weight: 2-D [in_dim=32, out_dim=32], dtype F16 (type 1)
    // Size = 32 * 32 * 2 = 2048 bytes
    write_string(&mut file, "blk.0.attn_q.weight");
    file.write_all(&2u32.to_le_bytes()).unwrap(); // 2 dims
    file.write_all(&32u64.to_le_bytes()).unwrap(); // dim[0] = in_dim
    file.write_all(&32u64.to_le_bytes()).unwrap(); // dim[1] = out_dim
    file.write_all(&1u32.to_le_bytes()).unwrap(); // F16
    let offset_t0 = 0u64;
    file.write_all(&offset_t0.to_le_bytes()).unwrap();

    // Tensor 1: blk.0.ffn_gate_exps.weight: 3-D [in_dim=32, out_dim=64, num_experts=2], dtype Q8_0 (type 8)
    // In Q8_0: 32 elements per block = 34 bytes.
    // Row bytes = (32 / 32) * 34 = 34 bytes.
    // Expert bytes = 64 * 34 = 2176 bytes. Total = 2 * 2176 = 4352 bytes.
    let len_t0 = 32 * 32 * 2;
    let offset_t1 = len_t0 as u64;
    write_string(&mut file, "blk.0.ffn_gate_exps.weight");
    file.write_all(&3u32.to_le_bytes()).unwrap(); // 3 dims
    file.write_all(&32u64.to_le_bytes()).unwrap(); // in_dim
    file.write_all(&64u64.to_le_bytes()).unwrap(); // out_dim
    file.write_all(&2u64.to_le_bytes()).unwrap(); // num_experts
    file.write_all(&8u32.to_le_bytes()).unwrap(); // Q8_0
    file.write_all(&offset_t1.to_le_bytes()).unwrap();

    // Tensor 2: blk.0.ffn_up_exps.weight: 3-D [in_dim=256, out_dim=32, num_experts=2], dtype Q4_K (type 12)
    // In Q4_K: 256 elements per block = 144 bytes.
    // Row bytes = (256 / 256) * 144 = 144 bytes.
    // Expert bytes = 32 * 144 = 4608 bytes. Total = 2 * 4608 = 9216 bytes.
    let len_t1 = 4352;
    let offset_t2 = offset_t1 + len_t1 as u64;
    write_string(&mut file, "blk.0.ffn_up_exps.weight");
    file.write_all(&3u32.to_le_bytes()).unwrap(); // 3 dims
    file.write_all(&256u64.to_le_bytes()).unwrap(); // in_dim
    file.write_all(&32u64.to_le_bytes()).unwrap(); // out_dim
    file.write_all(&2u64.to_le_bytes()).unwrap(); // num_experts
    file.write_all(&12u32.to_le_bytes()).unwrap(); // Q4_K
    file.write_all(&offset_t2.to_le_bytes()).unwrap();

    // 4. Align to 32 bytes
    let cur = file.stream_position().unwrap();
    let pad = (32 - (cur % 32)) % 32;
    if pad > 0 {
        file.write_all(&vec![0u8; pad as usize]).unwrap();
    }

    // 5. Write Data
    // T0: F16 data (1024 floats)
    for i in 0..1024 {
        let f = (i as f32) * 0.01;
        let h = half::f16::from_f32(f);
        file.write_all(&h.to_le_bytes()).unwrap();
    }

    // T1: Q8_0 data (2 experts * 64 rows = 128 blocks of 34 bytes)
    for _ in 0..128 {
        // d: f16 scale = 0.5
        let d = half::f16::from_f32(0.5);
        file.write_all(&d.to_le_bytes()).unwrap();
        // 32 quants: i8
        for j in 0..32 {
            let q = (j as i8) - 16;
            file.write_all(&[q as u8]).unwrap();
        }
    }

    // T2: Q4_K data (2 experts * 32 rows = 64 blocks of 144 bytes)
    for _ in 0..64 {
        // d = 1.0 (f16)
        let d = half::f16::from_f32(1.0);
        file.write_all(&d.to_le_bytes()).unwrap();
        // dmin = 0.0 (f16)
        let dmin = half::f16::from_f32(0.0);
        file.write_all(&dmin.to_le_bytes()).unwrap();
        // scales (12 bytes)
        file.write_all(&[1u8; 12]).unwrap();
        // qs (128 bytes)
        file.write_all(&[0x21u8; 128]).unwrap(); // nibbles 1 and 2
    }
}

#[test]
fn test_gguf_reader_and_conversion() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");
    write_synthetic_gguf(&gguf_path);

    assert!(is_gguf_file(&gguf_path));

    let reader = GgufReader::open(&gguf_path).expect("open gguf");
    assert_eq!(
        reader
            .metadata
            .get("general.architecture")
            .unwrap()
            .as_str(),
        Some("deepseek2")
    );

    let names = reader.tensor_names();
    assert!(names.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.0.gate_proj.weight".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.1.gate_proj.weight".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.0.up_proj.weight".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.1.up_proj.weight".to_string()));

    let info_q = reader
        .info("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(info_q.shape, vec![32, 32]);

    let q_tensor = reader
        .read_f32("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_tensor.shape, vec![32, 32]);
    assert!((q_tensor.data[0] - 0.0).abs() < 1e-4);
    assert!((q_tensor.data[1] - 0.01).abs() < 1e-4);

    let gate_tensor = reader
        .read_f32("model.layers.0.mlp.experts.0.gate_proj.weight")
        .unwrap();
    assert_eq!(gate_tensor.shape, vec![64, 32]);
    // First element: -16 * 0.5 = -8.0
    assert!((gate_tensor.data[0] - (-8.0)).abs() < 1e-4);

    // Test convert_gguf
    let out_dir = dir.path().join("undertow_out");
    let opts = ConvertOptions {
        row_chunk: 16,
        force: true,
        ..Default::default()
    };

    let report = convert_gguf(
        &gguf_path,
        &out_dir,
        |_dst| Ok(Box::new(undertow_deepseek_moe::classify_tensor)),
        &opts,
    )
    .expect("convert gguf");

    assert_eq!(report.tensors, 5);
    assert!(out_dir.join("config.json").exists());
    assert!(out_dir.join("dense.safetensors").exists());
    assert!(out_dir.join("experts-00000.safetensors").exists());
    assert!(out_dir.join("model.safetensors.index.json").exists());

    // Verify roundtrip read
    let model_reader = ShardedModelReader::open(&out_dir).expect("open converted model");
    let read_q = undertow_io::read_qtensor(
        &model_reader,
        "model.layers.0.self_attn.q_proj.weight",
        32,
        32,
    )
    .expect("read converted dense qtensor");
    assert_eq!(read_q.out_dim(), 32);
    assert_eq!(read_q.in_dim(), 32);
    let recon_q = read_q.dequantize();
    assert_eq!(recon_q.len(), 32 * 32);

    let read_expert = undertow_io::read_qtensor(
        &model_reader,
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        64,
        32,
    )
    .expect("read converted expert qtensor");
    assert_eq!(read_expert.out_dim(), 64);
    assert_eq!(read_expert.in_dim(), 32);
    let recon_exp = read_expert.dequantize();
    assert_eq!(recon_exp.len(), 64 * 32);
    // The first row original values were [-8.0, ...], check that reconstructed values are close
    assert!((recon_exp[0] - (-8.0)).abs() < 0.2);
}
