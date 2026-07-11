use std::io::{BufWriter, Write};
use std::path::Path;

use engine_core::{Result, Tensor};

/// Write tensors (f32) to a single `.safetensors` file.
///
/// Used by the oracle-model generator and, later, the converter. Tensors
/// are written in the order given; names must be unique.
pub fn write_safetensors(path: impl AsRef<Path>, tensors: &[(String, &Tensor)]) -> Result<()> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, t) in tensors {
        let nbytes = (t.numel() * 4) as u64;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": t.shape,
                "data_offsets": [offset, offset + nbytes],
            }),
        );
        offset += nbytes;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .expect("header serialization cannot fail");
    // Spec: pad header with spaces to 8-byte alignment.
    while !header_bytes.len().is_multiple_of(8) {
        header_bytes.push(b' ');
    }

    let mut w = BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    w.write_all(&header_bytes)?;
    for (_, t) in tensors {
        for v in &t.data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SafetensorsReader;

    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::new(vec![4], vec![-1., 0., 0.5, 2.0]);
        write_safetensors(&path, &[("a".into(), &a), ("b".into(), &b)]).unwrap();

        let r = SafetensorsReader::open(&path).unwrap();
        assert_eq!(r.read_f32("a").unwrap(), a);
        assert_eq!(r.read_f32("b").unwrap(), b);
        assert!(r.read_f32("missing").is_err());
    }
}
