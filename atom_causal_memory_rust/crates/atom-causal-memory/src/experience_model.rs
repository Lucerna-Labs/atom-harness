use crate::{FeatureSpec, MatchedMotif, QueryFeature};
use atom_db::Digest;
use std::collections::BTreeSet;

pub const ROOT_PRIMITIVES: [&str; 7] = [
    "attraction_repulsion",
    "conservation",
    "decay",
    "dissipation",
    "gravitation",
    "nucleation",
    "radiation",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceSpec {
    pub experience_id: String,
    pub features: Vec<FeatureSpec>,
}

impl ExperienceSpec {
    pub fn feature_values(&self, role: &str) -> Vec<&str> {
        self.features
            .iter()
            .filter(|feature| feature.role == role)
            .map(|feature| feature.value.as_str())
            .collect()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceBatch {
    pub source_artifact_hash: String,
    pub batch_id: String,
    pub experiences: Vec<ExperienceSpec>,
    pub raw: String,
}

impl ExperienceBatch {
    pub fn validate(&self) -> Result<(), String> {
        validate_hash("source artifact hash", &self.source_artifact_hash)?;
        validate_text("batch id", &self.batch_id, 1024)?;
        if self.experiences.is_empty() || self.experiences.len() > 100_000 {
            return Err("experience batch must contain 1..=100000 records".into());
        }
        let roots = ROOT_PRIMITIVES.into_iter().collect::<BTreeSet<_>>();
        let mut ids = BTreeSet::new();
        for experience in &self.experiences {
            validate_text("experience id", &experience.experience_id, 1024)?;
            if !ids.insert(experience.experience_id.clone()) {
                return Err(format!(
                    "duplicate experience id in batch: {}",
                    experience.experience_id
                ));
            }
            if experience.features.is_empty() || experience.features.len() > 512 {
                return Err(format!(
                    "experience {} must contain 1..=512 features",
                    experience.experience_id
                ));
            }
            let mut features = BTreeSet::new();
            for feature in &experience.features {
                validate_text("experience feature role", &feature.role, 512)?;
                validate_text("experience feature value", &feature.value, 16 * 1024)?;
                if !features.insert(feature.clone()) {
                    return Err(format!(
                        "duplicate feature on {}: {}={}",
                        experience.experience_id, feature.role, feature.value
                    ));
                }
                if feature.role.starts_with("root/") && !roots.contains(feature.value.as_str()) {
                    return Err(format!(
                        "experience {} uses an unknown root primitive {}",
                        experience.experience_id, feature.value
                    ));
                }
            }
            for required in ["kind", "status", "domain", "cause", "effect", "direction"] {
                if experience.feature_values(required).len() != 1 {
                    return Err(format!(
                        "experience {} must have exactly one {required} feature",
                        experience.experience_id
                    ));
                }
            }
            let kind = experience.feature_values("kind")[0];
            if !matches!(kind, "observation" | "law") {
                return Err(format!(
                    "experience {} has unsupported kind {kind}",
                    experience.experience_id
                ));
            }
            let status = experience.feature_values("status")[0];
            if !matches!(
                status,
                "observed" | "hypothesis" | "crystallized" | "retired"
            ) {
                return Err(format!(
                    "experience {} has unsupported status {status}",
                    experience.experience_id
                ));
            }
            let direction = experience.feature_values("direction")[0];
            if !matches!(direction, "-1" | "0" | "+1") {
                return Err(format!(
                    "experience {} has invalid direction {direction}",
                    experience.experience_id
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceQuery {
    pub query_id: String,
    pub minimum_support: usize,
    pub minimum_coverage_per_million: u32,
    pub limit: usize,
    pub features: Vec<QueryFeature>,
    pub excluded_experiences: BTreeSet<String>,
}

impl ExperienceQuery {
    pub fn validate(&self) -> Result<(), String> {
        validate_text("experience query id", &self.query_id, 1024)?;
        if self.features.is_empty() || self.features.len() > 4096 {
            return Err("experience query must contain 1..=4096 features".into());
        }
        if self.minimum_support == 0 || self.minimum_support > self.features.len() {
            return Err("experience minimum support is outside the feature count".into());
        }
        if self.minimum_coverage_per_million > 1_000_000 {
            return Err("experience minimum coverage cannot exceed 1000000".into());
        }
        if self.limit == 0 || self.limit > 1024 {
            return Err("experience result limit must be within 1..=1024".into());
        }
        let mut seen = BTreeSet::new();
        for feature in &self.features {
            validate_text("experience query role", &feature.role, 512)?;
            validate_text("experience query value", &feature.value, 16 * 1024)?;
            if !seen.insert((feature.role.clone(), feature.value.clone())) {
                return Err(format!(
                    "duplicate experience query feature: {}={}",
                    feature.role, feature.value
                ));
            }
        }
        for experience_id in &self.excluded_experiences {
            validate_text("excluded experience id", experience_id, 1024)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceHit {
    pub experience_id: String,
    pub score: u64,
    pub matched_support: usize,
    pub query_support: usize,
    pub coverage_per_million: u32,
    pub motifs: Vec<MatchedMotif>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceRecallReport {
    pub runtime: &'static str,
    pub catalog_identity: Digest,
    pub snapshot_sequence: u64,
    pub query_id: String,
    pub answerable: bool,
    pub insufficient_evidence: bool,
    pub hits: Vec<ExperienceHit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceIngestReport {
    pub runtime: &'static str,
    pub source_artifact_hash: String,
    pub batch_id: String,
    pub batch_identity: Digest,
    pub committed: bool,
    pub ingested_experiences: usize,
    pub ingested_motifs: usize,
    pub total_batches: usize,
    pub total_experiences: usize,
    pub snapshot_sequence: u64,
    pub durable_atoms: u64,
    pub durable_bonds: u64,
    pub durable_cells: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceBatchInventory {
    pub batch_id: String,
    pub source_artifact_hash: String,
    pub experience_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceInventoryItem {
    pub experience_id: String,
    pub kind: String,
    pub status: String,
    pub domain: String,
    pub features: Vec<FeatureSpec>,
    pub feature_count: usize,
    pub strengthened_motifs: usize,
    pub weakened_motifs: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceInventoryReport {
    pub runtime: &'static str,
    pub catalog_identity: Digest,
    pub snapshot_sequence: u64,
    pub batches: Vec<ExperienceBatchInventory>,
    pub experiences: Vec<ExperienceInventoryItem>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceFeedbackAdjustment {
    pub experience_id: String,
    pub role: String,
    pub value: String,
    pub motif: Digest,
    pub polarity: String,
    pub count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceFeedbackReport {
    pub runtime: &'static str,
    pub query_id: String,
    pub expected_experience: String,
    pub selected_experience: String,
    pub prediction_correct: bool,
    pub event_identity: Digest,
    pub cell_identity: Digest,
    pub snapshot_sequence: u64,
    pub adjustments: Vec<ExperienceFeedbackAdjustment>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperienceOutcomeReport {
    pub runtime: &'static str,
    pub outcome_key: String,
    pub committed: bool,
    pub query_id: String,
    pub expected_experience: String,
    pub selected_experience: String,
    pub prediction_correct: bool,
    pub event_identity: Digest,
    pub cell_identity: Option<Digest>,
    pub snapshot_sequence: u64,
    pub adjustments: Vec<ExperienceFeedbackAdjustment>,
}

fn validate_hash(name: &str, value: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{name} must be 64 hexadecimal characters"));
    }
    Ok(())
}

fn validate_text(name: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.contains('\0') {
        return Err(format!("{name} must be non-empty, bounded, and NUL-free"));
    }
    Ok(())
}
