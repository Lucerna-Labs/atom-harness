use crate::{FeedbackReport, ImportReport, InventoryReport, QueryReport};
use std::fmt::Write;

impl ImportReport {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"source_graph_hash\":{},",
                "\"manifest_hash\":{},\"catalog_identity\":{},\"cell_identity\":{},",
                "\"committed\":{},\"glyph_count\":{},\"root_count\":{},",
                "\"motif_count\":{},\"snapshot_sequence\":{},",
                "\"root_history_versions\":{},\"durable_atoms\":{},",
                "\"durable_bonds\":{},\"durable_cells\":{}}}"
            ),
            json(self.runtime),
            json(&self.source_graph_hash),
            json(&self.manifest_hash),
            json(&self.catalog_identity.to_string()),
            json(&self.cell_identity.to_string()),
            self.committed,
            self.glyph_count,
            self.root_count,
            self.motif_count,
            self.snapshot_sequence,
            self.root_history_versions,
            self.durable_atoms,
            self.durable_bonds,
            self.durable_cells,
        )
    }
}

impl InventoryReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"source_graph_hash\":{},",
                "\"catalog_identity\":{},\"snapshot_sequence\":{},\"glyphs\":["
            ),
            json(self.runtime),
            json(&self.source_graph_hash),
            json(&self.catalog_identity.to_string()),
            self.snapshot_sequence,
        );
        for (index, glyph) in self.glyphs.iter().enumerate() {
            comma(&mut out, index);
            let _ = write!(
                out,
                concat!(
                    "{{\"primitive_id\":{},\"root\":{},\"status\":{},",
                    "\"feature_count\":{},\"strengthened_motifs\":{},",
                    "\"weakened_motifs\":{}}}"
                ),
                json(&glyph.primitive_id),
                glyph.root,
                json(&glyph.status),
                glyph.feature_count,
                glyph.strengthened_motifs,
                glyph.weakened_motifs,
            );
        }
        out.push_str("]}");
        out
    }
}

impl QueryReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"source_graph_hash\":{},",
                "\"snapshot_sequence\":{},\"query_id\":{},\"answerable\":{},",
                "\"insufficient_evidence\":{},\"hits\":["
            ),
            json(self.runtime),
            json(&self.source_graph_hash),
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
                    "{{\"primitive_id\":{},\"score\":{},\"matched_support\":{},",
                    "\"query_support\":{},\"coverage_per_million\":{},\"motifs\":["
                ),
                json(&hit.primitive_id),
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

impl FeedbackReport {
    pub fn to_json(&self) -> String {
        let mut out = format!(
            concat!(
                "{{\"schema\":1,\"runtime\":{},\"source_graph_hash\":{},",
                "\"query_id\":{},\"expected_glyph\":{},\"selected_glyph\":{},",
                "\"prediction_correct\":{},\"event_identity\":{},",
                "\"cell_identity\":{},\"snapshot_sequence\":{},\"adjustments\":["
            ),
            json(self.runtime),
            json(&self.source_graph_hash),
            json(&self.query_id),
            json(&self.expected_glyph),
            json(&self.selected_glyph),
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
                    "{{\"primitive_id\":{},\"role\":{},\"value\":{},",
                    "\"motif\":{},\"polarity\":{},\"count\":{}}}"
                ),
                json(&adjustment.primitive_id),
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

#[cfg(test)]
mod tests {
    use super::json;

    #[test]
    fn json_escapes_control_characters() {
        assert_eq!(json("a\n\"b\\c"), "\"a\\n\\\"b\\\\c\"");
    }
}
