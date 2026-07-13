//! [`ModelAdapter`] implementation for the DeepSeek-MoE family.

use std::ops::Range;

use undertow_core::adapter::{
    AttentionKind, ExpertLayout, ModelAdapter, MtpHeadSpec, RopeKind, RouterAdapter,
};

use crate::config::DeepseekConfig;
use crate::router::DeepseekSigmoidRouter;

pub struct DeepseekMoeAdapter {
    cfg: DeepseekConfig,
}

impl DeepseekMoeAdapter {
    pub fn new(cfg: DeepseekConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &DeepseekConfig {
        &self.cfg
    }
}

impl ModelAdapter for DeepseekMoeAdapter {
    fn architecture(&self) -> &'static str {
        "deepseek_moe"
    }

    fn num_layers(&self) -> usize {
        self.cfg.num_hidden_layers
    }

    fn dense_layer_range(&self) -> Range<usize> {
        0..self.cfg.first_k_dense_replace
    }

    fn attention(&self) -> AttentionKind {
        AttentionKind::Mla {
            num_heads: self.cfg.num_attention_heads,
            q_lora_rank: self.cfg.q_lora_rank,
            kv_lora_rank: self.cfg.kv_lora_rank,
            qk_nope_head_dim: self.cfg.qk_nope_head_dim,
            qk_rope_head_dim: self.cfg.qk_rope_head_dim,
            v_head_dim: self.cfg.v_head_dim,
            rope: RopeKind::InterleavedPartial {
                theta: self.cfg.rope_theta(),
            },
        }
    }

    fn router(&self) -> Box<dyn RouterAdapter> {
        Box::new(DeepseekSigmoidRouter::from_config(&self.cfg))
    }

    fn expert_layout(&self) -> ExpertLayout {
        ExpertLayout {
            num_routed_experts: self.cfg.n_routed_experts,
            num_experts_per_token: self.cfg.num_experts_per_tok,
            num_shared_experts: self.cfg.n_shared_experts,
            moe_intermediate_size: self.cfg.moe_intermediate_size,
        }
    }

    fn mtp_head(&self) -> Option<MtpHeadSpec> {
        // GLM-5.2 / DeepSeek-V3 ship an MTP layer; wiring it up is Phase 2.
        None
    }
}
