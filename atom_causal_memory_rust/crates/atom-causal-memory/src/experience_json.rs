use crate::{
    ExperienceFeedbackReport, ExperienceIngestReport, ExperienceInventoryReport,
    ExperienceOutcomeReport, ExperienceRecallReport,
};
use std::fmt::Write;

impl ExperienceIngestReport {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"source_artifact_hash\":{},",
                "\"batch_id\":{},\"batch_identity\":{},\"committed\":{},",
                "\"ingested_experiences\":{},\"ingested_motifs\":{},",
                "\"total_batches\":{},\"total_experiences\":{},",
                "\"snapshot_sequence\":{},\"durable_atoms\":{},",
                "\"durable_bonds\":{},\"durable_cells\":{}}}"
            ),
            json(self.runtime),
            json(&self.source_artifact_hash),
            json(&self.batch_id),
            json(&self.batch_identity.to_string()),
            self.committed,
            self.ingested_experiences,
            self.ingested_motifs,
            self.total_batches,
            self.total_experiences,
            self.snapshot_sequence,
            self.durable_atoms,
            self.durable_bonds,
            self.durable_cells,
        )
    }
}

impl ExperienceInventoryReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"catalog_identity\":{},",
                "\"snapshot_sequence\":{},\"batches\":["
            ),
            json(self.runtime),
            json(&self.catalog_identity.to_string()),
            self.snapshot_sequence,
        );
        for (index, batch) in self.batches.iter().enumerate() {
            comma(&mut out, index);
            let _ = write!(
                out,
                concat!(
                    "{{\"batch_id\":{},\"source_artifact_hash\":{},",
                    "\"experience_count\":{}}}"
                ),
                json(&batch.batch_id),
                json(&batch.source_artifact_hash),
                batch.experience_count,
            );
        }
        out.push_str("],\"experiences\":[");
        for (index, experience) in self.experiences.iter().enumerate() {
            comma(&mut out, index);
            let _ = write!(
                out,
                concat!(
                    "{{\"experience_id\":{},\"kind\":{},\"status\":{},",
                    "\"domain\":{},\"features\":["
                ),
                json(&experience.experience_id),
                json(&experience.kind),
                json(&experience.status),
                json(&experience.domain),
            );
            for (feature_index, feature) in experience.features.iter().enumerate() {
                comma(&mut out, feature_index);
                let _ = write!(
                    out,
                    "{{\"role\":{},\"value\":{}}}",
                    json(&feature.role),
                    json(&feature.value),
                );
            }
            let _ = write!(
                out,
                concat!(
                    "],\"feature_count\":{},\"strengthened_motifs\":{},",
                    "\"weakened_motifs\":{}}}"
                ),
                experience.feature_count,
                experience.strengthened_motifs,
                experience.weakened_motifs,
            );
        }
        out.push_str("]}");
        out
    }
}

impl ExperienceRecallReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"catalog_identity\":{},",
                "\"snapshot_sequence\":{},\"query_id\":{},\"answerable\":{},",
                "\"insufficient_evidence\":{},\"hits\":["
            ),
            json(self.runtime),
            json(&self.catalog_identity.to_string()),
            self.snapshot_sequence,
            json(&self.query_id),
            self.answerable,
            self.insufficient_evidence,
        );
        for (hit_index, hit) in self.hits.iter().enumerate() {
            comma(&mut out, hit_index);
            let _ = write!(
                out,
                concat!(
                    "{{\"experience_id\":{},\"score\":{},\"matched_support\":{},",
                    "\"query_support\":{},\"coverage_per_million\":{},\"motifs\":["
                ),
                json(&hit.experience_id),
                hit.score,
                hit.matched_support,
                hit.query_support,
                hit.coverage_per_million,
            );
            for (motif_index, motif) in hit.motifs.iter().enumerate() {
                comma(&mut out, motif_index);
                let _ = write!(
                    out,
                    concat!(
                        "{{\"role\":{},\"value\":{},\"motif\":{},",
                        "\"base_weight\":{},\"conductance_per_mille\":{},",
                        "\"strengthen_count\":{},\"weaken_count\":{},",
                        "\"contribution\":{}}}"
                    ),
                    json(&motif.role),
                    json(&motif.value),
                    json(&motif.motif.to_string()),
                    motif.base_weight,
                    motif.conductance_per_mille,
                    motif.strengthen_count,
                    motif.weaken_count,
                    motif.contribution,
                );
            }
            out.push_str("]}");
        }
        out.push_str("]}");
        out
    }
}

impl ExperienceFeedbackReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"query_id\":{},",
                "\"expected_experience\":{},\"selected_experience\":{},",
                "\"prediction_correct\":{},\"event_identity\":{},",
                "\"cell_identity\":{},\"snapshot_sequence\":{},\"adjustments\":["
            ),
            json(self.runtime),
            json(&self.query_id),
            json(&self.expected_experience),
            json(&self.selected_experience),
            self.prediction_correct,
            json(&self.event_identity.to_string()),
            json(&self.cell_identity.to_string()),
            self.snapshot_sequence,
        );
        for (index, adjustment) in self.adjustments.iter().enumerate() {
            comma(&mut out, index);
            let _ = write!(
                out,
                concat!(
                    "{{\"experience_id\":{},\"role\":{},\"value\":{},",
                    "\"motif\":{},\"polarity\":{},\"count\":{}}}"
                ),
                json(&adjustment.experience_id),
                json(&adjustment.role),
                json(&adjustment.value),
                json(&adjustment.motif.to_string()),
                json(&adjustment.polarity),
                adjustment.count,
            );
        }
        out.push_str("]}");
        out
    }
}

impl ExperienceOutcomeReport {
    pub fn to_json(&self) -> String {
        let cell_identity = self
            .cell_identity
            .map(|identity| json(&identity.to_string()))
            .unwrap_or_else(|| "null".to_string());
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"outcome_key\":{},",
                "\"committed\":{},\"query_id\":{},",
                "\"expected_experience\":{},\"selected_experience\":{},",
                "\"prediction_correct\":{},\"event_identity\":{},",
                "\"cell_identity\":{},\"snapshot_sequence\":{},\"adjustments\":["
            ),
            json(self.runtime),
            json(&self.outcome_key),
            self.committed,
            json(&self.query_id),
            json(&self.expected_experience),
            json(&self.selected_experience),
            self.prediction_correct,
            json(&self.event_identity.to_string()),
            cell_identity,
            self.snapshot_sequence,
        );
        for (index, adjustment) in self.adjustments.iter().enumerate() {
            comma(&mut out, index);
            let _ = write!(
                out,
                concat!(
                    "{{\"experience_id\":{},\"role\":{},\"value\":{},",
                    "\"motif\":{},\"polarity\":{},\"count\":{}}}"
                ),
                json(&adjustment.experience_id),
                json(&adjustment.role),
                json(&adjustment.value),
                json(&adjustment.motif.to_string()),
                json(&adjustment.polarity),
                adjustment.count,
            );
        }
        out.push_str("]}");
        out
    }
}

fn comma(out: &mut String, index: usize) {
    if index != 0 {
        out.push(',');
    }
}

fn json(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if control <= '\u{1f}' => {
                let _ = write!(out, "\\u{:04x}", control as u32);
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
