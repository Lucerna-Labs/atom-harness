//! Causal-glyph and experience memory for the Atom architecture.
//!
//! The upstream Atom DB retrieval membrane remains a lexical evidence system.
//! This crate reuses only its immutable atom/bond/cell substrate and adds
//! structural memories whose queries are typed graph motifs rather than words,
//! chunks, embeddings, or vector similarity.

mod experience;
mod experience_json;
mod experience_model;
mod experience_wire;
mod json;
mod memory;
mod model;
mod wire;

pub use experience::ExperienceMemory;
pub use experience_model::{
    ExperienceBatch, ExperienceBatchInventory, ExperienceFeedbackAdjustment,
    ExperienceFeedbackReport, ExperienceHit, ExperienceIngestReport, ExperienceInventoryItem,
    ExperienceInventoryReport, ExperienceOutcomeReport, ExperienceQuery, ExperienceRecallReport,
    ExperienceSpec, ROOT_PRIMITIVES,
};
pub use experience_wire::{parse_experience_batch, parse_experience_query};
pub use memory::CausalMemory;
pub use model::{
    FeatureSpec, FeedbackAdjustment, FeedbackReport, GlyphInventory, GlyphSpec, ImportReport,
    InventoryReport, Manifest, MatchedMotif, QueryFeature, QueryReport, StructuralHit,
    StructuralQuery,
};
pub use wire::{parse_manifest, parse_query};

pub const MANIFEST_RUNTIME: &str = "atom-causal-memory-manifest-v1";
pub const QUERY_RUNTIME: &str = "atom-causal-memory-query-v1";
pub const MEMORY_RUNTIME: &str = "atom-causal-memory-v1";
pub const EXPERIENCE_BATCH_RUNTIME: &str = "atom-causal-experience-batch-v1";
pub const EXPERIENCE_QUERY_RUNTIME: &str = "atom-causal-experience-query-v1";
pub const EXPERIENCE_MEMORY_RUNTIME: &str = "atom-causal-experience-v1";
