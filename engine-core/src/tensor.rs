//! Minimal owned f32 tensor.
//!
//! Deliberately tiny for the scalar reference phase: row-major, f32-only,
//! no views, no strides. Quantized storage formats live in `engine-quant`
//! and will get their own representation; this type is the *dequantized /
//! reference* currency between components.

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Row-major. A weight matrix applied as `y = W x` is stored `[out, in]`
    /// (the safetensors / PyTorch `nn.Linear` convention).
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.0; n],
        }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Rows of a 2-D tensor (`shape[0]`).
    pub fn dim0(&self) -> usize {
        self.shape[0]
    }

    /// Columns of a 2-D tensor (`shape[1]`).
    pub fn dim1(&self) -> usize {
        self.shape[1]
    }

    /// Borrow row `r` of a 2-D tensor.
    pub fn row(&self, r: usize) -> &[f32] {
        let cols = self.dim1();
        &self.data[r * cols..(r + 1) * cols]
    }
}
