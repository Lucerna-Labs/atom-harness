use atom_db::Digest;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FeatureSpec {
    pub role: String,
    pub value: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlyphSpec {
    pub primitive_id: String,
    pub features: Vec<FeatureSpec>,
}

impl GlyphSpec {
    pub fn feature_values(&self, role: &str) -> Vec<&str> {
        self.features
            .iter()
            .filter(|feature| feature.role == role)
            .map(|feature| feature.value.as_str())
            .collect()
    }

    pub fn is_root(&self) -> bool {
        self.features
            .iter()
            .any(|feature| feature.role == "kind" && feature.value == "root")
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    pub source_graph_hash: String,
    pub glyphs: Vec<GlyphSpec>,
    pub raw: String,
}

impl Manifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.source_graph_hash.len() != 64
            || !self
                .source_graph_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("source graph hash must be 64 hexadecimal characters".into());
        }
        if self.glyphs.is_empty() || self.glyphs.len() > 1_000_000 {
            return Err("causal glyph inventory must contain 1..=1000000 records".into());
        }
        let mut ids = BTreeSet::new();
        for glyph in &self.glyphs {
            validate_text("primitive id", &glyph.primitive_id, 512)?;
            if !ids.insert(glyph.primitive_id.clone()) {
                return Err(format!("duplicate causal glyph id: {}", glyph.primitive_id));
            }
            if glyph.features.is_empty() || glyph.features.len() > 65_536 {
                return Err(format!(
                    "glyph {} must contain 1..=65536 structural features",
                    glyph.primitive_id
                ));
            }
            let mut features = BTreeSet::new();
            for feature in &glyph.features {
                validate_text("feature role", &feature.role, 512)?;
                validate_text("feature value", &feature.value, 16 * 1024)?;
                if !features.insert(feature.clone()) {
                    return Err(format!(
                        "duplicate feature on {}: {}={}",
                        glyph.primitive_id, feature.role, feature.value
                    ));
                }
            }
            for required in [
                "kind",
                "status",
                "domain",
                "output/kind",
                "output/dimension",
            ] {
                if glyph.feature_values(required).len() != 1 {
                    return Err(format!(
                        "glyph {} must have exactly one {required} feature",
                        glyph.primitive_id
                    ));
                }
            }
            if glyph.is_root() {
                if !glyph.feature_values("recipe/mode").is_empty()
                    || glyph
                        .features
                        .iter()
                        .any(|feature| feature.role.starts_with("component/"))
                {
                    return Err(format!(
                        "immutable root {} cannot contain a composition recipe",
                        glyph.primitive_id
                    ));
                }
            } else if glyph.feature_values("recipe/mode").len() != 1
                || !glyph
                    .features
                    .iter()
                    .any(|feature| feature.role.starts_with("component/"))
            {
                return Err(format!(
                    "derived glyph {} requires a recipe mode and components",
                    glyph.primitive_id
                ));
            }
        }

        let expected_roots = BTreeSet::from([
            "attraction_repulsion".to_string(),
            "conservation".to_string(),
            "decay".to_string(),
            "dissipation".to_string(),
            "gravitation".to_string(),
            "nucleation".to_string(),
            "radiation".to_string(),
        ]);
        let roots = self
            .glyphs
            .iter()
            .filter(|glyph| glyph.is_root())
            .map(|glyph| glyph.primitive_id.clone())
            .collect::<BTreeSet<_>>();
        if roots != expected_roots {
            return Err("causal memory requires exactly the seven immutable Atom roots".into());
        }

        let by_id = self
            .glyphs
            .iter()
            .map(|glyph| (glyph.primitive_id.as_str(), glyph))
            .collect::<BTreeMap<_, _>>();
        for glyph in &self.glyphs {
            for feature in glyph
                .features
                .iter()
                .filter(|feature| feature.role.starts_with("component/"))
            {
                if !by_id.contains_key(feature.value.as_str()) {
                    return Err(format!(
                        "glyph {} references unknown component {}",
                        glyph.primitive_id, feature.value
                    ));
                }
            }
            for feature in glyph
                .features
                .iter()
                .filter(|feature| feature.role.starts_with("root-expansion/"))
            {
                if !roots.contains(&feature.value) {
                    return Err(format!(
                        "glyph {} has a non-root expansion member {}",
                        glyph.primitive_id, feature.value
                    ));
                }
            }
        }
        validate_acyclic(&by_id)?;
        Ok(())
    }
}

fn validate_text(name: &str, value: &str, maximum: usize) -> Result<(), String> {
    if value.is_empty() || value.len() > maximum || value.contains('\0') {
        return Err(format!("{name} must be non-empty, bounded, and NUL-free"));
    }
    Ok(())
}

fn validate_acyclic(by_id: &BTreeMap<&str, &GlyphSpec>) -> Result<(), String> {
    fn visit<'a>(
        id: &'a str,
        by_id: &BTreeMap<&'a str, &'a GlyphSpec>,
        states: &mut BTreeMap<&'a str, u8>,
    ) -> Result<(), String> {
        match states.get(id).copied().unwrap_or(0) {
            1 => return Err(format!("causal glyph composition cycle reaches {id}")),
            2 => return Ok(()),
            _ => {}
        }
        states.insert(id, 1);
        let glyph = by_id
            .get(id)
            .ok_or_else(|| format!("unknown causal glyph {id}"))?;
        for component in glyph
            .features
            .iter()
            .filter(|feature| feature.role.starts_with("component/"))
        {
            visit(component.value.as_str(), by_id, states)?;
        }
        states.insert(id, 2);
        Ok(())
    }

    let mut states = BTreeMap::new();
    for id in by_id.keys().copied() {
        visit(id, by_id, &mut states)?;
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryFeature {
    pub role: String,
    pub value: String,
    pub required: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralQuery {
    pub query_id: String,
    pub minimum_support: usize,
    pub minimum_coverage_per_million: u32,
    pub limit: usize,
    pub features: Vec<QueryFeature>,
    pub excluded_glyphs: BTreeSet<String>,
}

impl StructuralQuery {
    pub fn validate(&self) -> Result<(), String> {
        validate_text("query id", &self.query_id, 1024)?;
        if self.features.is_empty() || self.features.len() > 4096 {
            return Err("structural query must contain 1..=4096 features".into());
        }
        if self.minimum_support == 0 || self.minimum_support > self.features.len() {
            return Err("minimum support must be within the query feature count".into());
        }
        if self.minimum_coverage_per_million > 1_000_000 {
            return Err("minimum coverage cannot exceed 1000000".into());
        }
        if self.limit == 0 || self.limit > 1024 {
            return Err("query result limit must be within 1..=1024".into());
        }
        let mut seen = BTreeSet::new();
        for feature in &self.features {
            validate_text("query feature role", &feature.role, 512)?;
            validate_text("query feature value", &feature.value, 16 * 1024)?;
            if !seen.insert((feature.role.clone(), feature.value.clone())) {
                return Err(format!(
                    "duplicate query feature: {}={}",
                    feature.role, feature.value
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchedMotif {
    pub role: String,
    pub value: String,
    pub motif: Digest,
    pub base_weight: u64,
    pub conductance_per_mille: u16,
    pub strengthen_count: u64,
    pub weaken_count: u64,
    pub contribution: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuralHit {
    pub primitive_id: String,
    pub score: u64,
    pub matched_support: usize,
    pub query_support: usize,
    pub coverage_per_million: u32,
    pub motifs: Vec<MatchedMotif>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryReport {
    pub runtime: &'static str,
    pub source_graph_hash: String,
    pub snapshot_sequence: u64,
    pub query_id: String,
    pub answerable: bool,
    pub insufficient_evidence: bool,
    pub hits: Vec<StructuralHit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportReport {
    pub runtime: &'static str,
    pub source_graph_hash: String,
    pub manifest_hash: String,
    pub catalog_identity: Digest,
    pub cell_identity: Digest,
    pub committed: bool,
    pub glyph_count: usize,
    pub root_count: usize,
    pub motif_count: usize,
    pub snapshot_sequence: u64,
    pub root_history_versions: usize,
    pub durable_atoms: u64,
    pub durable_bonds: u64,
    pub durable_cells: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlyphInventory {
    pub primitive_id: String,
    pub root: bool,
    pub status: String,
    pub feature_count: usize,
    pub strengthened_motifs: usize,
    pub weakened_motifs: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InventoryReport {
    pub runtime: &'static str,
    pub source_graph_hash: String,
    pub catalog_identity: Digest,
    pub snapshot_sequence: u64,
    pub glyphs: Vec<GlyphInventory>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeedbackAdjustment {
    pub primitive_id: String,
    pub role: String,
    pub value: String,
    pub motif: Digest,
    pub polarity: String,
    pub count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FeedbackReport {
    pub runtime: &'static str,
    pub source_graph_hash: String,
    pub query_id: String,
    pub expected_glyph: String,
    pub selected_glyph: String,
    pub prediction_correct: bool,
    pub event_identity: Digest,
    pub cell_identity: Digest,
    pub snapshot_sequence: u64,
    pub adjustments: Vec<FeedbackAdjustment>,
}
