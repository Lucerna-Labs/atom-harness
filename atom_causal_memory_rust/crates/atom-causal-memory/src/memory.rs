use crate::{
    FeedbackAdjustment, FeedbackReport, GlyphInventory, ImportReport, InventoryReport,
    MEMORY_RUNTIME, Manifest, MatchedMotif, QueryFeature, QueryReport, StructuralHit,
    StructuralQuery,
};
use atom_db::{AtomDb, Bond, Cell, Digest, digest};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const CURRENT_CATALOG_ROOT: &[u8] = b"atom-causal-memory/catalog/current/v1";
const CATALOG_PREFIX: &[u8] = b"atom-causal-memory/catalog/v1\0";
const GLYPH_PREFIX: &[u8] = b"atom-causal-memory/glyph/v1\0";
const MOTIF_PREFIX: &[u8] = b"atom-causal-memory/motif/v1\0";
const ROLE_PREFIX: &[u8] = b"atom-causal-memory/role/v1\0";
const VALUE_PREFIX: &[u8] = b"atom-causal-memory/value/v1\0";
const MANIFEST_PREFIX: &[u8] = b"atom-causal-memory/manifest/v1\0";
const EVENT_PREFIX: &[u8] = b"atom-causal-memory/prediction-event/v1\0";
const HAS_MOTIF: &[u8] = b"atom-causal-memory/has-motif/v1";
const MOTIF_ROLE: &[u8] = b"atom-causal-memory/motif-role/v1";
const MOTIF_VALUE: &[u8] = b"atom-causal-memory/motif-value/v1";
const CATALOG_MEMBER: &[u8] = b"atom-causal-memory/catalog-member/v1";
const CATALOG_MANIFEST: &[u8] = b"atom-causal-memory/catalog-manifest/v1";
const EVENT_EXPECTED: &[u8] = b"atom-causal-memory/event-expected/v1";
const EVENT_SELECTED: &[u8] = b"atom-causal-memory/event-selected/v1";
const EVENT_ADJUSTED: &[u8] = b"atom-causal-memory/event-adjusted/v1";
const FEEDBACK_PREFIX: &[u8] = b"atom-causal-memory/feedback/v1/";
const FEEDBACK_MARKER: &[u8] = b"atom-causal-memory/feedback-marker/v1";
const STRENGTHEN_DELTA: i128 = 200;
const WEAKEN_DELTA: i128 = 350;
const BASE_CONDUCTANCE: i128 = 1_000;
const MAX_CONDUCTANCE: i128 = 4_000;

#[derive(Clone, Debug)]
struct IndexedMotif {
    identity: Digest,
    role: String,
    value: String,
    strengthen: u64,
    weaken: u64,
}

#[derive(Clone, Debug)]
struct IndexedGlyph {
    identity: Digest,
    primitive_id: String,
    motifs: BTreeMap<(String, String), IndexedMotif>,
}

#[derive(Clone, Debug)]
struct MemoryIndex {
    source_graph_hash: String,
    catalog_identity: Digest,
    snapshot_sequence: u64,
    glyphs: BTreeMap<String, IndexedGlyph>,
}

pub struct CausalMemory;

impl CausalMemory {
    pub fn import(path: impl AsRef<Path>, manifest: &Manifest) -> Result<ImportReport, String> {
        manifest.validate()?;
        let mut db = AtomDb::open_writer(path).map_err(error)?;
        let mut cell = db.begin_cell();
        let current_root = cell.put_atom(CURRENT_CATALOG_ROOT);
        let catalog_identity = cell.put_atom(prefixed(CATALOG_PREFIX, &manifest.source_graph_hash));
        let manifest_identity = cell.put_atom(prefixed(MANIFEST_PREFIX, &manifest.raw));
        let has_motif = cell.put_atom(HAS_MOTIF);
        let motif_role = cell.put_atom(MOTIF_ROLE);
        let motif_value = cell.put_atom(MOTIF_VALUE);
        let catalog_member = cell.put_atom(CATALOG_MEMBER);
        let catalog_manifest = cell.put_atom(CATALOG_MANIFEST);

        cell.put_bond(Bond {
            source: catalog_identity,
            relation: catalog_manifest,
            target: manifest_identity,
        });

        let glyph_ids = manifest
            .glyphs
            .iter()
            .map(|glyph| {
                let identity = cell.put_atom(prefixed(GLYPH_PREFIX, &glyph.primitive_id));
                (glyph.primitive_id.clone(), identity)
            })
            .collect::<BTreeMap<_, _>>();

        let mut motif_count = 0usize;
        for glyph in &manifest.glyphs {
            let glyph_identity = glyph_ids[&glyph.primitive_id];
            cell.put_bond(Bond {
                source: catalog_identity,
                relation: catalog_member,
                target: glyph_identity,
            });
            for feature in &glyph.features {
                let motif_identity = cell.put_atom(motif_bytes(
                    &glyph.primitive_id,
                    &feature.role,
                    &feature.value,
                ));
                let role_identity = cell.put_atom(prefixed(ROLE_PREFIX, &feature.role));
                let value_identity = cell.put_atom(prefixed(VALUE_PREFIX, &feature.value));
                cell.put_bond(Bond {
                    source: glyph_identity,
                    relation: has_motif,
                    target: motif_identity,
                });
                cell.put_bond(Bond {
                    source: motif_identity,
                    relation: motif_role,
                    target: role_identity,
                });
                cell.put_bond(Bond {
                    source: motif_identity,
                    relation: motif_value,
                    target: value_identity,
                });
                if feature.role.starts_with("component/")
                    || feature.role.starts_with("root-expansion/")
                {
                    let referenced = glyph_ids.get(&feature.value).ok_or_else(|| {
                        format!(
                            "structural relation {} references unknown glyph {}",
                            feature.role, feature.value
                        )
                    })?;
                    cell.put_bond(Bond {
                        source: glyph_identity,
                        relation: role_identity,
                        target: *referenced,
                    });
                }
                motif_count += 1;
            }
        }
        cell.set_root(current_root, catalog_identity);
        let receipt = db.commit_cell(cell).map_err(error)?;
        db.sync().map_err(error)?;
        let stats = db.stats().map_err(error)?;
        let root_history_versions = db.root_history(current_root).len();
        let manifest_hash = digest(&[
            b"atom-causal-memory/manifest-hash/v1\0",
            manifest.raw.as_bytes(),
        ])
        .to_string();
        Ok(ImportReport {
            runtime: MEMORY_RUNTIME,
            source_graph_hash: manifest.source_graph_hash.clone(),
            manifest_hash,
            catalog_identity,
            cell_identity: receipt.identity,
            committed: receipt.committed,
            glyph_count: manifest.glyphs.len(),
            root_count: manifest
                .glyphs
                .iter()
                .filter(|glyph| glyph.is_root())
                .count(),
            motif_count,
            snapshot_sequence: db.snapshot().sequence(),
            root_history_versions,
            durable_atoms: stats.atoms,
            durable_bonds: stats.bonds,
            durable_cells: stats.cells,
        })
    }

    pub fn query(path: impl AsRef<Path>, query: &StructuralQuery) -> Result<QueryReport, String> {
        query.validate()?;
        validate_structural_roles(query)?;
        let mut db = AtomDb::open_read_only(path).map_err(error)?;
        let index = build_index(&mut db)?;
        Ok(query_index(&index, query))
    }

    pub fn inventory(path: impl AsRef<Path>) -> Result<InventoryReport, String> {
        let mut db = AtomDb::open_read_only(path).map_err(error)?;
        let index = build_index(&mut db)?;
        let glyphs = index
            .glyphs
            .values()
            .map(|glyph| {
                let status = glyph
                    .motifs
                    .get(&("status".to_string(), status_value(glyph)))
                    .map(|motif| motif.value.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                GlyphInventory {
                    primitive_id: glyph.primitive_id.clone(),
                    root: glyph
                        .motifs
                        .contains_key(&("kind".to_string(), "root".to_string())),
                    status,
                    feature_count: glyph.motifs.len(),
                    strengthened_motifs: glyph
                        .motifs
                        .values()
                        .filter(|motif| motif.strengthen > 0)
                        .count(),
                    weakened_motifs: glyph
                        .motifs
                        .values()
                        .filter(|motif| motif.weaken > 0)
                        .count(),
                }
            })
            .collect();
        Ok(InventoryReport {
            runtime: MEMORY_RUNTIME,
            source_graph_hash: index.source_graph_hash,
            catalog_identity: index.catalog_identity,
            snapshot_sequence: index.snapshot_sequence,
            glyphs,
        })
    }

    pub fn observe_prediction(
        path: impl AsRef<Path>,
        query: &StructuralQuery,
        expected_glyph: &str,
        selected_glyph: &str,
    ) -> Result<FeedbackReport, String> {
        query.validate()?;
        validate_structural_roles(query)?;
        let mut db = AtomDb::open_writer(path).map_err(error)?;
        let index = build_index(&mut db)?;
        let expected = index.glyphs.get(expected_glyph).ok_or_else(|| {
            format!("expected glyph is absent from the active catalog: {expected_glyph}")
        })?;
        let selected = index.glyphs.get(selected_glyph).ok_or_else(|| {
            format!("selected glyph is absent from the active catalog: {selected_glyph}")
        })?;
        let expected_hit = match_glyph(expected, query, false)?;
        let selected_hit = match_glyph(selected, query, false)?;
        if expected_hit.motifs.is_empty() {
            return Err(
                "expected glyph shares no structural motif with the observation query".into(),
            );
        }
        if selected_hit.motifs.is_empty() {
            return Err(
                "selected glyph shares no structural motif with the observation query".into(),
            );
        }

        let prediction_correct = expected_glyph == selected_glyph;
        let mut intended = Vec::<(&str, &MatchedMotif, &str)>::new();
        if prediction_correct {
            for motif in &expected_hit.motifs {
                intended.push((expected_glyph, motif, "strengthen"));
            }
        } else {
            for motif in &selected_hit.motifs {
                intended.push((selected_glyph, motif, "weaken"));
            }
            for motif in &expected_hit.motifs {
                intended.push((expected_glyph, motif, "strengthen"));
            }
        }
        if intended.len() > 8192 {
            return Err("prediction observation exceeds the bounded adjustment budget".into());
        }

        let mut cell = db.begin_cell();
        let event_identity = cell.put_atom(event_bytes(
            index.snapshot_sequence,
            &query.query_id,
            expected_glyph,
            selected_glyph,
        ));
        let expected_identity = cell.put_atom(prefixed(GLYPH_PREFIX, expected_glyph));
        let selected_identity = cell.put_atom(prefixed(GLYPH_PREFIX, selected_glyph));
        let expected_relation = cell.put_atom(EVENT_EXPECTED);
        let selected_relation = cell.put_atom(EVENT_SELECTED);
        let adjusted_relation = cell.put_atom(EVENT_ADJUSTED);
        let feedback_marker = cell.put_atom(FEEDBACK_MARKER);
        cell.put_bond(Bond {
            source: event_identity,
            relation: expected_relation,
            target: expected_identity,
        });
        cell.put_bond(Bond {
            source: event_identity,
            relation: selected_relation,
            target: selected_identity,
        });

        let mut adjustments = Vec::with_capacity(intended.len());
        for (primitive_id, motif, polarity) in intended {
            let count = match polarity {
                "strengthen" => motif.strengthen_count.saturating_add(1),
                "weaken" => motif.weaken_count.saturating_add(1),
                _ => return Err("internal feedback polarity is invalid".into()),
            };
            let relation_identity = cell.put_atom(feedback_bytes(polarity, count));
            cell.put_bond(Bond {
                source: motif.motif,
                relation: relation_identity,
                target: feedback_marker,
            });
            cell.put_bond(Bond {
                source: event_identity,
                relation: adjusted_relation,
                target: motif.motif,
            });
            adjustments.push(FeedbackAdjustment {
                primitive_id: primitive_id.to_string(),
                role: motif.role.clone(),
                value: motif.value.clone(),
                motif: motif.motif,
                polarity: polarity.to_string(),
                count,
            });
        }
        let receipt = db.commit_cell(cell).map_err(error)?;
        db.sync().map_err(error)?;
        Ok(FeedbackReport {
            runtime: MEMORY_RUNTIME,
            source_graph_hash: index.source_graph_hash,
            query_id: query.query_id.clone(),
            expected_glyph: expected_glyph.to_string(),
            selected_glyph: selected_glyph.to_string(),
            prediction_correct,
            event_identity,
            cell_identity: receipt.identity,
            snapshot_sequence: db.snapshot().sequence(),
            adjustments,
        })
    }
}

fn build_index(db: &mut AtomDb) -> Result<MemoryIndex, String> {
    let current_root = atom_identity(CURRENT_CATALOG_ROOT);
    let catalog_identity = db
        .root(current_root)
        .ok_or_else(|| "causal-memory store has no active catalog root".to_string())?;
    let catalog_bytes = required_atom(db, catalog_identity)?;
    let source_graph_hash = parse_prefixed(&catalog_bytes, CATALOG_PREFIX)
        .ok_or_else(|| "active catalog identity has an invalid domain".to_string())?;
    if source_graph_hash.len() != 64 {
        return Err("active catalog source hash is invalid".into());
    }

    let catalog_member = atom_identity(CATALOG_MEMBER);
    let has_motif = atom_identity(HAS_MOTIF);
    let motif_role = atom_identity(MOTIF_ROLE);
    let motif_value = atom_identity(MOTIF_VALUE);
    let feedback_marker = atom_identity(FEEDBACK_MARKER);
    let mut glyphs = BTreeMap::new();
    for (_, membership) in db
        .bonds_from(catalog_identity)
        .into_iter()
        .filter(|(_, bond)| bond.relation == catalog_member)
    {
        let glyph_bytes = required_atom(db, membership.target)?;
        let primitive_id = parse_prefixed(&glyph_bytes, GLYPH_PREFIX)
            .ok_or_else(|| "catalog member is not a causal glyph atom".to_string())?;
        if glyphs.contains_key(&primitive_id) {
            return Err(format!(
                "active catalog contains duplicate glyph {primitive_id}"
            ));
        }
        glyphs.insert(
            primitive_id.clone(),
            IndexedGlyph {
                identity: membership.target,
                primitive_id,
                motifs: BTreeMap::new(),
            },
        );
    }
    if glyphs.is_empty() {
        return Err("active causal-memory catalog contains no glyphs".into());
    }

    for glyph in glyphs.values_mut() {
        let motif_ids = db
            .bonds_from(glyph.identity)
            .into_iter()
            .filter(|(_, bond)| bond.relation == has_motif)
            .map(|(_, bond)| bond.target)
            .collect::<BTreeSet<_>>();
        for motif_identity in motif_ids {
            let mut role = None;
            let mut value = None;
            let mut strengthen = 0u64;
            let mut weaken = 0u64;
            for (_, bond) in db.bonds_from(motif_identity) {
                if bond.relation == motif_role {
                    let bytes = required_atom(db, bond.target)?;
                    let parsed = parse_prefixed(&bytes, ROLE_PREFIX)
                        .ok_or_else(|| "motif role atom has an invalid domain".to_string())?;
                    assign_once(&mut role, parsed, "motif role")?;
                } else if bond.relation == motif_value {
                    let bytes = required_atom(db, bond.target)?;
                    let parsed = parse_prefixed(&bytes, VALUE_PREFIX)
                        .ok_or_else(|| "motif value atom has an invalid domain".to_string())?;
                    assign_once(&mut value, parsed, "motif value")?;
                } else if bond.target == feedback_marker {
                    let bytes = required_atom(db, bond.relation)?;
                    if let Some((polarity, count)) = parse_feedback(&bytes)? {
                        match polarity {
                            "strengthen" => strengthen = strengthen.max(count),
                            "weaken" => weaken = weaken.max(count),
                            _ => return Err("feedback polarity is invalid".into()),
                        }
                    }
                }
            }
            let role = role.ok_or_else(|| "causal motif has no role".to_string())?;
            let value = value.ok_or_else(|| "causal motif has no value".to_string())?;
            let expected = atom_identity(&motif_bytes(&glyph.primitive_id, &role, &value));
            if expected != motif_identity {
                return Err(format!(
                    "causal motif identity is detached from {} {}={}",
                    glyph.primitive_id, role, value
                ));
            }
            let key = (role.clone(), value.clone());
            if glyph
                .motifs
                .insert(
                    key,
                    IndexedMotif {
                        identity: motif_identity,
                        role,
                        value,
                        strengthen,
                        weaken,
                    },
                )
                .is_some()
            {
                return Err(format!(
                    "glyph {} contains a duplicate motif",
                    glyph.primitive_id
                ));
            }
        }
    }

    let root_count = glyphs
        .values()
        .filter(|glyph| {
            glyph
                .motifs
                .contains_key(&("kind".to_string(), "root".to_string()))
        })
        .count();
    if root_count != 7 {
        return Err("active causal-memory catalog does not preserve seven roots".into());
    }
    Ok(MemoryIndex {
        source_graph_hash,
        catalog_identity,
        snapshot_sequence: db.snapshot().sequence(),
        glyphs,
    })
}

fn query_index(index: &MemoryIndex, query: &StructuralQuery) -> QueryReport {
    let mut hits = index
        .glyphs
        .values()
        .filter(|glyph| !query.excluded_glyphs.contains(&glyph.primitive_id))
        .filter_map(|glyph| match_glyph(glyph, query, true).ok())
        .filter(|hit| hit.matched_support >= query.minimum_support)
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| right.matched_support.cmp(&left.matched_support))
            .then_with(|| left.primitive_id.cmp(&right.primitive_id))
    });
    hits.truncate(query.limit);
    let answerable = hits.first().is_some_and(|hit| {
        hit.coverage_per_million >= query.minimum_coverage_per_million
            && hit.matched_support >= query.minimum_support
    });
    QueryReport {
        runtime: MEMORY_RUNTIME,
        source_graph_hash: index.source_graph_hash.clone(),
        snapshot_sequence: index.snapshot_sequence,
        query_id: query.query_id.clone(),
        answerable,
        insufficient_evidence: !answerable,
        hits,
    }
}

fn match_glyph(
    glyph: &IndexedGlyph,
    query: &StructuralQuery,
    respect_required: bool,
) -> Result<StructuralHit, String> {
    let mut motifs = Vec::new();
    let mut total_weight = 0u64;
    let mut matched_weight = 0u64;
    for feature in &query.features {
        let base_weight = role_weight(&feature.role)
            .ok_or_else(|| format!("query role is not structural: {}", feature.role))?;
        total_weight = total_weight.saturating_add(base_weight);
        match glyph
            .motifs
            .get(&(feature.role.clone(), feature.value.clone()))
        {
            Some(motif) => {
                let conductance = conductance(motif.strengthen, motif.weaken);
                let contribution = base_weight.saturating_mul(conductance as u64);
                matched_weight = matched_weight.saturating_add(base_weight);
                motifs.push(MatchedMotif {
                    role: motif.role.clone(),
                    value: motif.value.clone(),
                    motif: motif.identity,
                    base_weight,
                    conductance_per_mille: conductance,
                    strengthen_count: motif.strengthen,
                    weaken_count: motif.weaken,
                    contribution,
                });
            }
            None if respect_required && feature.required => {
                return Ok(StructuralHit {
                    primitive_id: glyph.primitive_id.clone(),
                    score: 0,
                    matched_support: 0,
                    query_support: query.features.len(),
                    coverage_per_million: 0,
                    motifs: Vec::new(),
                });
            }
            None => {}
        }
    }
    let score = motifs
        .iter()
        .fold(0u64, |sum, motif| sum.saturating_add(motif.contribution));
    let coverage_per_million = if total_weight == 0 {
        0
    } else {
        ((matched_weight as u128 * 1_000_000) / total_weight as u128) as u32
    };
    Ok(StructuralHit {
        primitive_id: glyph.primitive_id.clone(),
        score,
        matched_support: motifs.len(),
        query_support: query.features.len(),
        coverage_per_million,
        motifs,
    })
}

fn validate_structural_roles(query: &StructuralQuery) -> Result<(), String> {
    for QueryFeature { role, .. } in &query.features {
        if role_weight(role).is_none() {
            return Err(format!(
                "query role is not structural and cannot enter causal resonance: {role}"
            ));
        }
    }
    Ok(())
}

fn role_weight(role: &str) -> Option<u64> {
    if role.starts_with("component/") {
        Some(1_800)
    } else if role.starts_with("root-expansion/") {
        Some(1_400)
    } else if role.starts_with("input/") || role.starts_with("output/") {
        Some(1_100)
    } else {
        match role {
            "recipe/mode" => Some(900),
            "domain" => Some(800),
            "invariant" => Some(450),
            "symmetry" => Some(300),
            "boundary" => Some(150),
            "scale" => Some(100),
            "kind" => Some(75),
            "status" => Some(50),
            _ => None,
        }
    }
}

fn conductance(strengthen: u64, weaken: u64) -> u16 {
    let positive = i128::from(strengthen.min(i64::MAX as u64)) * STRENGTHEN_DELTA;
    let negative = i128::from(weaken.min(i64::MAX as u64)) * WEAKEN_DELTA;
    (BASE_CONDUCTANCE + positive - negative).clamp(1, MAX_CONDUCTANCE) as u16
}

fn status_value(glyph: &IndexedGlyph) -> String {
    glyph
        .motifs
        .values()
        .find(|motif| motif.role == "status")
        .map(|motif| motif.value.clone())
        .unwrap_or_else(|| "unknown".to_string())
}

fn atom_identity(bytes: &[u8]) -> Digest {
    let mut cell = Cell::new();
    cell.put_atom(bytes)
}

fn required_atom(db: &mut AtomDb, identity: Digest) -> Result<Vec<u8>, String> {
    db.get_atom(identity)
        .map_err(error)?
        .ok_or_else(|| format!("missing atom {identity}"))
}

fn assign_once(target: &mut Option<String>, value: String, name: &str) -> Result<(), String> {
    if target.replace(value).is_some() {
        return Err(format!("causal motif contains more than one {name}"));
    }
    Ok(())
}

fn parse_feedback(bytes: &[u8]) -> Result<Option<(&str, u64)>, String> {
    let Some(tail) = bytes.strip_prefix(FEEDBACK_PREFIX) else {
        return Ok(None);
    };
    let text = std::str::from_utf8(tail).map_err(|_| "feedback relation is not UTF-8")?;
    let (polarity, count) = text
        .split_once('/')
        .ok_or_else(|| "feedback relation lacks a count".to_string())?;
    if !matches!(polarity, "strengthen" | "weaken") {
        return Err("feedback relation has an unknown polarity".into());
    }
    let count = count
        .parse::<u64>()
        .map_err(|_| "feedback relation count is invalid".to_string())?;
    if count == 0 {
        return Err("feedback relation count must be positive".into());
    }
    Ok(Some((polarity, count)))
}

fn prefixed(prefix: &[u8], value: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(prefix.len() + value.len());
    bytes.extend_from_slice(prefix);
    bytes.extend_from_slice(value.as_bytes());
    bytes
}

fn motif_bytes(primitive_id: &str, role: &str, value: &str) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(MOTIF_PREFIX.len() + primitive_id.len() + role.len() + value.len() + 2);
    bytes.extend_from_slice(MOTIF_PREFIX);
    bytes.extend_from_slice(primitive_id.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(role.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(value.as_bytes());
    bytes
}

fn feedback_bytes(polarity: &str, count: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FEEDBACK_PREFIX.len() + polarity.len() + 24);
    bytes.extend_from_slice(FEEDBACK_PREFIX);
    bytes.extend_from_slice(polarity.as_bytes());
    bytes.push(b'/');
    bytes.extend_from_slice(count.to_string().as_bytes());
    bytes
}

fn event_bytes(snapshot: u64, query_id: &str, expected: &str, selected: &str) -> Vec<u8> {
    let value = format!("{snapshot}\0{query_id}\0{expected}\0{selected}");
    prefixed(EVENT_PREFIX, &value)
}

fn parse_prefixed(bytes: &[u8], prefix: &[u8]) -> Option<String> {
    let value = bytes.strip_prefix(prefix)?;
    std::str::from_utf8(value).ok().map(ToOwned::to_owned)
}

fn error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::encode_text;
    use crate::{parse_manifest, parse_query};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn feature(role: &str, value: &str) -> String {
        format!("F\t{}\t{}\n", encode_text(role), encode_text(value))
    }

    fn glyph(id: &str, root: bool, components: &[&str], roots: &[&str]) -> String {
        let mut out = format!("G\t{}\n", encode_text(id));
        out.push_str(&feature("kind", if root { "root" } else { "derived" }));
        out.push_str(&feature(
            "status",
            if root {
                "immutable_root"
            } else {
                "crystallized"
            },
        ));
        out.push_str(&feature("domain", "mathematical_scalar_field"));
        out.push_str(&feature("input/000/kind", "bounded_scalar_field"));
        out.push_str(&feature("input/000/dimension", "state^1"));
        out.push_str(&feature("output/kind", "bounded_scalar_field"));
        out.push_str(&feature("output/dimension", "state^1"));
        if !root {
            out.push_str(&feature("recipe/mode", "serial"));
            for (index, component) in components.iter().enumerate() {
                out.push_str(&feature(&format!("component/{index:04}"), component));
            }
            for (index, root_id) in roots.iter().enumerate() {
                out.push_str(&feature(&format!("root-expansion/{index:04}"), root_id));
            }
        }
        out
    }

    fn test_manifest() -> Manifest {
        let mut raw = format!("{}\nH\t{}\n", crate::MANIFEST_RUNTIME, "a".repeat(64));
        for root in [
            "attraction_repulsion",
            "conservation",
            "decay",
            "dissipation",
            "gravitation",
            "nucleation",
            "radiation",
        ] {
            raw.push_str(&glyph(root, true, &[], &[]));
        }
        raw.push_str(&glyph(
            "primitive-a",
            false,
            &["radiation", "dissipation"],
            &["radiation", "dissipation"],
        ));
        raw.push_str(&glyph(
            "primitive-b",
            false,
            &["radiation", "decay"],
            &["radiation", "decay"],
        ));
        parse_manifest(&raw).unwrap()
    }

    fn query() -> StructuralQuery {
        let mut raw = format!(
            "{}\nQ\t{}\t5\t8\t900000\n",
            crate::QUERY_RUNTIME,
            encode_text("structural-test")
        );
        for (role, value, required) in [
            ("domain", "mathematical_scalar_field", true),
            ("recipe/mode", "serial", true),
            ("component/0000", "radiation", true),
            ("component/0001", "dissipation", true),
            ("root-expansion/0000", "radiation", false),
            ("root-expansion/0001", "dissipation", false),
        ] {
            raw.push_str(&format!(
                "F\t{}\t{}\t{}\n",
                encode_text(role),
                encode_text(value),
                if required { 1 } else { 0 }
            ));
        }
        parse_query(&raw).unwrap()
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("atom-causal-memory-{label}-{nonce}.atomdb"))
    }

    #[test]
    fn structural_query_uses_motifs_and_reopens() {
        let path = temp_path("query");
        let manifest = test_manifest();
        let imported = CausalMemory::import(&path, &manifest).unwrap();
        assert_eq!(imported.glyph_count, 9);
        assert_eq!(imported.root_count, 7);
        let before = CausalMemory::query(&path, &query()).unwrap();
        assert!(before.answerable);
        assert_eq!(before.hits[0].primitive_id, "primitive-a");
        assert_eq!(before.hits[0].coverage_per_million, 1_000_000);
        let feedback =
            CausalMemory::observe_prediction(&path, &query(), "primitive-a", "primitive-a")
                .unwrap();
        assert!(feedback.prediction_correct);
        let after = CausalMemory::query(&path, &query()).unwrap();
        assert!(after.hits[0].score > before.hits[0].score);
        assert!(
            after.hits[0]
                .motifs
                .iter()
                .all(|motif| motif.strengthen_count == 1)
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn wrong_prediction_strengthens_expected_and_weakens_selected() {
        let path = temp_path("feedback");
        CausalMemory::import(&path, &test_manifest()).unwrap();
        let feedback =
            CausalMemory::observe_prediction(&path, &query(), "primitive-a", "primitive-b")
                .unwrap();
        assert!(!feedback.prediction_correct);
        assert!(
            feedback
                .adjustments
                .iter()
                .any(|item| item.primitive_id == "primitive-a" && item.polarity == "strengthen")
        );
        assert!(
            feedback
                .adjustments
                .iter()
                .any(|item| item.primitive_id == "primitive-b" && item.polarity == "weaken")
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn unknown_required_motif_returns_insufficient_evidence() {
        let path = temp_path("unknown");
        CausalMemory::import(&path, &test_manifest()).unwrap();
        let mut unknown = query();
        unknown.features.push(QueryFeature {
            role: "component/0002".to_string(),
            value: "not-observed".to_string(),
            required: true,
        });
        let report = CausalMemory::query(&path, &unknown).unwrap();
        assert!(report.insufficient_evidence);
        assert!(report.hits.is_empty());
        fs::remove_file(path).unwrap();
    }
}
