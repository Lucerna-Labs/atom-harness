use crate::{
    EXPERIENCE_MEMORY_RUNTIME, ExperienceBatch, ExperienceBatchInventory,
    ExperienceFeedbackAdjustment, ExperienceFeedbackReport, ExperienceHit, ExperienceIngestReport,
    ExperienceInventoryItem, ExperienceInventoryReport, ExperienceOutcomeReport, ExperienceQuery,
    ExperienceRecallReport, FeatureSpec, MatchedMotif, QueryFeature,
};
use atom_db::{AtomDb, Bond, Cell, Digest};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const EXPERIENCE_HEAD_ROOT: &[u8] = b"atom-causal-experience/head/v1";
const CATALOG_BYTES: &[u8] = b"atom-causal-experience/catalog/v1";
const BATCH_PREFIX: &[u8] = b"atom-causal-experience/batch/v1\0";
const BATCH_MANIFEST_PREFIX: &[u8] = b"atom-causal-experience/batch-manifest/v1\0";
const EXPERIENCE_PREFIX: &[u8] = b"atom-causal-experience/record/v1\0";
const MOTIF_PREFIX: &[u8] = b"atom-causal-experience/motif/v1\0";
const ROLE_PREFIX: &[u8] = b"atom-causal-experience/role/v1\0";
const VALUE_PREFIX: &[u8] = b"atom-causal-experience/value/v1\0";
const EVENT_PREFIX: &[u8] = b"atom-causal-experience/outcome-event/v1\0";
const OUTCOME_KEY_PREFIX: &[u8] = b"atom-causal-experience/outcome-key/v1\0";
const IDEMPOTENT_EVENT_PREFIX: &[u8] = b"atom-causal-experience/idempotent-outcome/v1\0";
const OUTCOME_KEY_EVENT: &[u8] = b"atom-causal-experience/outcome-key-event/v1";
const CATALOG_BATCH: &[u8] = b"atom-causal-experience/catalog-batch/v1";
const CATALOG_EXPERIENCE: &[u8] = b"atom-causal-experience/catalog-experience/v1";
const BATCH_EXPERIENCE: &[u8] = b"atom-causal-experience/batch-experience/v1";
const BATCH_MANIFEST: &[u8] = b"atom-causal-experience/batch-manifest-link/v1";
const BATCH_PREVIOUS: &[u8] = b"atom-causal-experience/batch-previous/v1";
const HAS_MOTIF: &[u8] = b"atom-causal-experience/has-motif/v1";
const MOTIF_ROLE: &[u8] = b"atom-causal-experience/motif-role/v1";
const MOTIF_VALUE: &[u8] = b"atom-causal-experience/motif-value/v1";
const EVIDENCE_LINK: &[u8] = b"atom-causal-experience/evidence-link/v1";
const EVENT_EXPECTED: &[u8] = b"atom-causal-experience/event-expected/v1";
const EVENT_SELECTED: &[u8] = b"atom-causal-experience/event-selected/v1";
const EVENT_ADJUSTED: &[u8] = b"atom-causal-experience/event-adjusted/v1";
const FEEDBACK_PREFIX: &[u8] = b"atom-causal-experience/feedback/v1/";
const FEEDBACK_MARKER: &[u8] = b"atom-causal-experience/feedback-marker/v1";
const STRENGTHEN_DELTA: i128 = 200;
const WEAKEN_DELTA: i128 = 350;
const BASE_CONDUCTANCE: i128 = 1_000;
const MAX_CONDUCTANCE: i128 = 4_000;

#[derive(Clone, Debug)]
struct IndexedExperienceMotif {
    identity: Digest,
    role: String,
    value: String,
    strengthen: u64,
    weaken: u64,
}

#[derive(Clone, Debug)]
struct IndexedExperience {
    identity: Digest,
    experience_id: String,
    motifs: BTreeMap<(String, String), IndexedExperienceMotif>,
}

#[derive(Clone, Debug)]
struct IndexedBatch {
    source_artifact_hash: String,
    batch_id: String,
    experience_count: usize,
}

#[derive(Clone, Debug)]
struct ExperienceIndex {
    catalog_identity: Digest,
    snapshot_sequence: u64,
    batches: Vec<IndexedBatch>,
    experiences: BTreeMap<String, IndexedExperience>,
}

pub struct ExperienceMemory;

impl ExperienceMemory {
    pub fn ingest(
        path: impl AsRef<Path>,
        batch: &ExperienceBatch,
    ) -> Result<ExperienceIngestReport, String> {
        batch.validate()?;
        let mut db = AtomDb::open_writer(path).map_err(error)?;
        let catalog_identity = atom_identity(CATALOG_BYTES);
        let catalog_batch = atom_identity(CATALOG_BATCH);
        let batch_identity =
            atom_identity(&batch_bytes(&batch.source_artifact_hash, &batch.batch_id));

        let already_present = db
            .bonds_from(catalog_identity)
            .into_iter()
            .any(|(_, bond)| bond.relation == catalog_batch && bond.target == batch_identity);
        if already_present {
            let index = build_index(&mut db)?;
            let stats = db.stats().map_err(error)?;
            return Ok(ExperienceIngestReport {
                runtime: EXPERIENCE_MEMORY_RUNTIME,
                source_artifact_hash: batch.source_artifact_hash.clone(),
                batch_id: batch.batch_id.clone(),
                batch_identity,
                committed: false,
                ingested_experiences: batch.experiences.len(),
                ingested_motifs: batch
                    .experiences
                    .iter()
                    .map(|experience| experience.features.len())
                    .sum(),
                total_batches: index.batches.len(),
                total_experiences: index.experiences.len(),
                snapshot_sequence: index.snapshot_sequence,
                durable_atoms: stats.atoms,
                durable_bonds: stats.bonds,
                durable_cells: stats.cells,
            });
        }

        let head_root = atom_identity(EXPERIENCE_HEAD_ROOT);
        let previous_head = db.root(head_root);
        let mut cell = db.begin_cell();
        let head_root = cell.put_atom(EXPERIENCE_HEAD_ROOT);
        let catalog_identity = cell.put_atom(CATALOG_BYTES);
        let batch_identity =
            cell.put_atom(batch_bytes(&batch.source_artifact_hash, &batch.batch_id));
        let batch_manifest_identity = cell.put_atom(prefixed(BATCH_MANIFEST_PREFIX, &batch.raw));
        let catalog_batch = cell.put_atom(CATALOG_BATCH);
        let catalog_experience = cell.put_atom(CATALOG_EXPERIENCE);
        let batch_experience = cell.put_atom(BATCH_EXPERIENCE);
        let batch_manifest = cell.put_atom(BATCH_MANIFEST);
        let batch_previous = cell.put_atom(BATCH_PREVIOUS);
        let has_motif = cell.put_atom(HAS_MOTIF);
        let motif_role = cell.put_atom(MOTIF_ROLE);
        let motif_value = cell.put_atom(MOTIF_VALUE);
        let evidence_link = cell.put_atom(EVIDENCE_LINK);

        cell.put_bond(Bond {
            source: catalog_identity,
            relation: catalog_batch,
            target: batch_identity,
        });
        cell.put_bond(Bond {
            source: batch_identity,
            relation: batch_manifest,
            target: batch_manifest_identity,
        });
        if let Some(previous) = previous_head {
            cell.put_bond(Bond {
                source: batch_identity,
                relation: batch_previous,
                target: previous,
            });
        }

        let mut motif_count = 0usize;
        for experience in &batch.experiences {
            let experience_identity =
                cell.put_atom(prefixed(EXPERIENCE_PREFIX, &experience.experience_id));
            cell.put_bond(Bond {
                source: catalog_identity,
                relation: catalog_experience,
                target: experience_identity,
            });
            cell.put_bond(Bond {
                source: batch_identity,
                relation: batch_experience,
                target: experience_identity,
            });
            for feature in &experience.features {
                let motif_identity = cell.put_atom(motif_bytes(
                    &experience.experience_id,
                    &feature.role,
                    &feature.value,
                ));
                let role_identity = cell.put_atom(prefixed(ROLE_PREFIX, &feature.role));
                let value_identity = cell.put_atom(prefixed(VALUE_PREFIX, &feature.value));
                cell.put_bond(Bond {
                    source: experience_identity,
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
                if feature.role.starts_with("evidence/") {
                    let target = cell.put_atom(prefixed(EXPERIENCE_PREFIX, &feature.value));
                    cell.put_bond(Bond {
                        source: experience_identity,
                        relation: evidence_link,
                        target,
                    });
                }
                motif_count += 1;
            }
        }
        cell.set_root(head_root, batch_identity);
        let receipt = db.commit_cell(cell).map_err(error)?;
        db.sync().map_err(error)?;
        let index = build_index(&mut db)?;
        let stats = db.stats().map_err(error)?;
        Ok(ExperienceIngestReport {
            runtime: EXPERIENCE_MEMORY_RUNTIME,
            source_artifact_hash: batch.source_artifact_hash.clone(),
            batch_id: batch.batch_id.clone(),
            batch_identity,
            committed: receipt.committed,
            ingested_experiences: batch.experiences.len(),
            ingested_motifs: motif_count,
            total_batches: index.batches.len(),
            total_experiences: index.experiences.len(),
            snapshot_sequence: index.snapshot_sequence,
            durable_atoms: stats.atoms,
            durable_bonds: stats.bonds,
            durable_cells: stats.cells,
        })
    }

    pub fn inventory(path: impl AsRef<Path>) -> Result<ExperienceInventoryReport, String> {
        let mut db = AtomDb::open_read_only(path).map_err(error)?;
        let index = build_index(&mut db)?;
        Ok(inventory_report(&index))
    }

    pub fn recall(
        path: impl AsRef<Path>,
        query: &ExperienceQuery,
    ) -> Result<ExperienceRecallReport, String> {
        query.validate()?;
        validate_query_roles(query)?;
        let mut db = AtomDb::open_read_only(path).map_err(error)?;
        let index = build_index(&mut db)?;
        Ok(recall_index(&index, query))
    }

    pub fn observe_outcome(
        path: impl AsRef<Path>,
        query: &ExperienceQuery,
        expected_experience: &str,
        selected_experience: &str,
    ) -> Result<ExperienceFeedbackReport, String> {
        query.validate()?;
        validate_query_roles(query)?;
        let mut db = AtomDb::open_writer(path).map_err(error)?;
        let index = build_index(&mut db)?;
        let expected = index
            .experiences
            .get(expected_experience)
            .ok_or_else(|| format!("expected experience is absent: {expected_experience}"))?;
        let selected = index
            .experiences
            .get(selected_experience)
            .ok_or_else(|| format!("selected experience is absent: {selected_experience}"))?;
        let expected_hit = match_experience(expected, query, false)?;
        let selected_hit = match_experience(selected, query, false)?;
        if expected_hit.motifs.is_empty() {
            return Err("expected experience shares no motif with the outcome query".into());
        }
        if selected_hit.motifs.is_empty() {
            return Err("selected experience shares no motif with the outcome query".into());
        }
        let prediction_correct = expected_experience == selected_experience;
        let mut intended = Vec::<(&str, &MatchedMotif, &str)>::new();
        if prediction_correct {
            for motif in &expected_hit.motifs {
                intended.push((expected_experience, motif, "strengthen"));
            }
        } else {
            for motif in &selected_hit.motifs {
                intended.push((selected_experience, motif, "weaken"));
            }
            for motif in &expected_hit.motifs {
                intended.push((expected_experience, motif, "strengthen"));
            }
        }
        if intended.len() > 8192 {
            return Err("outcome observation exceeds the adjustment budget".into());
        }

        let mut cell = db.begin_cell();
        let event_identity = cell.put_atom(event_bytes(
            index.snapshot_sequence,
            &query.query_id,
            expected_experience,
            selected_experience,
        ));
        let expected_identity = cell.put_atom(prefixed(EXPERIENCE_PREFIX, expected_experience));
        let selected_identity = cell.put_atom(prefixed(EXPERIENCE_PREFIX, selected_experience));
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
        for (experience_id, motif, polarity) in intended {
            let count = match polarity {
                "strengthen" => motif.strengthen_count.saturating_add(1),
                "weaken" => motif.weaken_count.saturating_add(1),
                _ => return Err("internal outcome polarity is invalid".into()),
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
            adjustments.push(ExperienceFeedbackAdjustment {
                experience_id: experience_id.to_string(),
                role: motif.role.clone(),
                value: motif.value.clone(),
                motif: motif.motif,
                polarity: polarity.to_string(),
                count,
            });
        }
        let receipt = db.commit_cell(cell).map_err(error)?;
        db.sync().map_err(error)?;
        Ok(ExperienceFeedbackReport {
            runtime: EXPERIENCE_MEMORY_RUNTIME,
            query_id: query.query_id.clone(),
            expected_experience: expected_experience.to_string(),
            selected_experience: selected_experience.to_string(),
            prediction_correct,
            event_identity,
            cell_identity: receipt.identity,
            snapshot_sequence: db.snapshot().sequence(),
            adjustments,
        })
    }

    pub fn observe_outcome_once(
        path: impl AsRef<Path>,
        query: &ExperienceQuery,
        outcome_key: &str,
        expected_experience: &str,
        selected_experience: &str,
    ) -> Result<ExperienceOutcomeReport, String> {
        query.validate()?;
        validate_query_roles(query)?;
        if outcome_key.len() != 64
            || !outcome_key
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("outcome key must be lowercase SHA-256".into());
        }
        let mut db = AtomDb::open_writer(path).map_err(error)?;
        let index = build_index(&mut db)?;
        let expected = index
            .experiences
            .get(expected_experience)
            .ok_or_else(|| format!("expected experience is absent: {expected_experience}"))?;
        let selected = index
            .experiences
            .get(selected_experience)
            .ok_or_else(|| format!("selected experience is absent: {selected_experience}"))?;
        let expected_hit = match_experience(expected, query, false)?;
        let selected_hit = match_experience(selected, query, false)?;
        if expected_hit.motifs.is_empty() {
            return Err("expected experience shares no motif with the outcome query".into());
        }
        if selected_hit.motifs.is_empty() {
            return Err("selected experience shares no motif with the outcome query".into());
        }

        let key_identity = atom_identity(&outcome_key_bytes(outcome_key));
        let key_relation = atom_identity(OUTCOME_KEY_EVENT);
        let targets = db
            .bonds_from(key_identity)
            .into_iter()
            .filter(|(_, bond)| bond.relation == key_relation)
            .map(|(_, bond)| bond.target)
            .collect::<BTreeSet<_>>();
        if targets.len() > 1 {
            return Err("outcome key resolves to multiple events".into());
        }
        if let Some(event_identity) = targets.first().copied() {
            let bytes = required_atom(&mut db, event_identity)?;
            let stored = parse_idempotent_event(&bytes)?;
            let requested = (
                outcome_key,
                query.query_id.as_str(),
                expected_experience,
                selected_experience,
            );
            if stored != requested {
                return Err("outcome key conflicts with an existing event".into());
            }
            return Ok(ExperienceOutcomeReport {
                runtime: EXPERIENCE_MEMORY_RUNTIME,
                outcome_key: outcome_key.to_string(),
                committed: false,
                query_id: query.query_id.clone(),
                expected_experience: expected_experience.to_string(),
                selected_experience: selected_experience.to_string(),
                prediction_correct: expected_experience == selected_experience,
                event_identity,
                cell_identity: None,
                snapshot_sequence: index.snapshot_sequence,
                adjustments: Vec::new(),
            });
        }

        let prediction_correct = expected_experience == selected_experience;
        let mut intended = Vec::<(&str, &MatchedMotif, &str)>::new();
        if prediction_correct {
            for motif in &expected_hit.motifs {
                intended.push((expected_experience, motif, "strengthen"));
            }
        } else {
            for motif in &selected_hit.motifs {
                intended.push((selected_experience, motif, "weaken"));
            }
            for motif in &expected_hit.motifs {
                intended.push((expected_experience, motif, "strengthen"));
            }
        }
        if intended.len() > 8192 {
            return Err("outcome observation exceeds the adjustment budget".into());
        }

        let mut cell = db.begin_cell();
        let durable_key = cell.put_atom(outcome_key_bytes(outcome_key));
        let event_identity = cell.put_atom(idempotent_event_bytes(
            outcome_key,
            &query.query_id,
            expected_experience,
            selected_experience,
        ));
        let key_relation = cell.put_atom(OUTCOME_KEY_EVENT);
        cell.put_bond(Bond {
            source: durable_key,
            relation: key_relation,
            target: event_identity,
        });
        let expected_identity = cell.put_atom(prefixed(EXPERIENCE_PREFIX, expected_experience));
        let selected_identity = cell.put_atom(prefixed(EXPERIENCE_PREFIX, selected_experience));
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
        for (experience_id, motif, polarity) in intended {
            let count = match polarity {
                "strengthen" => motif.strengthen_count.saturating_add(1),
                "weaken" => motif.weaken_count.saturating_add(1),
                _ => return Err("internal outcome polarity is invalid".into()),
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
            adjustments.push(ExperienceFeedbackAdjustment {
                experience_id: experience_id.to_string(),
                role: motif.role.clone(),
                value: motif.value.clone(),
                motif: motif.motif,
                polarity: polarity.to_string(),
                count,
            });
        }
        let receipt = db.commit_cell(cell).map_err(error)?;
        if !receipt.committed {
            return Err("new idempotent outcome cell was not committed".into());
        }
        db.sync().map_err(error)?;
        Ok(ExperienceOutcomeReport {
            runtime: EXPERIENCE_MEMORY_RUNTIME,
            outcome_key: outcome_key.to_string(),
            committed: true,
            query_id: query.query_id.clone(),
            expected_experience: expected_experience.to_string(),
            selected_experience: selected_experience.to_string(),
            prediction_correct,
            event_identity,
            cell_identity: Some(receipt.identity),
            snapshot_sequence: db.snapshot().sequence(),
            adjustments,
        })
    }
}

fn build_index(db: &mut AtomDb) -> Result<ExperienceIndex, String> {
    let head_root = atom_identity(EXPERIENCE_HEAD_ROOT);
    if db.root(head_root).is_none() {
        return Err("experience memory has no durable head".into());
    }
    let catalog_identity = atom_identity(CATALOG_BYTES);
    required_atom(db, catalog_identity)?;

    let catalog_batch = atom_identity(CATALOG_BATCH);
    let catalog_experience = atom_identity(CATALOG_EXPERIENCE);
    let batch_experience = atom_identity(BATCH_EXPERIENCE);
    let has_motif = atom_identity(HAS_MOTIF);
    let motif_role = atom_identity(MOTIF_ROLE);
    let motif_value = atom_identity(MOTIF_VALUE);
    let feedback_marker = atom_identity(FEEDBACK_MARKER);

    let mut batches = Vec::new();
    for (_, membership) in db
        .bonds_from(catalog_identity)
        .into_iter()
        .filter(|(_, bond)| bond.relation == catalog_batch)
    {
        let bytes = required_atom(db, membership.target)?;
        let (source_artifact_hash, batch_id) = parse_batch(&bytes)?;
        let experience_count = db
            .bonds_from(membership.target)
            .into_iter()
            .filter(|(_, bond)| bond.relation == batch_experience)
            .map(|(_, bond)| bond.target)
            .collect::<BTreeSet<_>>()
            .len();
        batches.push(IndexedBatch {
            source_artifact_hash,
            batch_id,
            experience_count,
        });
    }
    batches.sort_by(|left, right| left.batch_id.cmp(&right.batch_id));
    if batches.is_empty() {
        return Err("experience catalog contains no batches".into());
    }

    let mut experiences = BTreeMap::new();
    for (_, membership) in db
        .bonds_from(catalog_identity)
        .into_iter()
        .filter(|(_, bond)| bond.relation == catalog_experience)
    {
        let bytes = required_atom(db, membership.target)?;
        let experience_id = parse_prefixed(&bytes, EXPERIENCE_PREFIX)
            .ok_or_else(|| "catalog member is not an experience atom".to_string())?;
        if experiences.contains_key(&experience_id) {
            continue;
        }
        experiences.insert(
            experience_id.clone(),
            IndexedExperience {
                identity: membership.target,
                experience_id,
                motifs: BTreeMap::new(),
            },
        );
    }
    if experiences.is_empty() {
        return Err("experience catalog contains no records".into());
    }

    for experience in experiences.values_mut() {
        let motif_ids = db
            .bonds_from(experience.identity)
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
                        .ok_or_else(|| "experience motif role has an invalid domain".to_string())?;
                    assign_once(&mut role, parsed, "experience motif role")?;
                } else if bond.relation == motif_value {
                    let bytes = required_atom(db, bond.target)?;
                    let parsed = parse_prefixed(&bytes, VALUE_PREFIX).ok_or_else(|| {
                        "experience motif value has an invalid domain".to_string()
                    })?;
                    assign_once(&mut value, parsed, "experience motif value")?;
                } else if bond.target == feedback_marker {
                    let bytes = required_atom(db, bond.relation)?;
                    if let Some((polarity, count)) = parse_feedback(&bytes)? {
                        match polarity {
                            "strengthen" => strengthen = strengthen.max(count),
                            "weaken" => weaken = weaken.max(count),
                            _ => return Err("experience feedback polarity is invalid".into()),
                        }
                    }
                }
            }
            let role = role.ok_or_else(|| "experience motif has no role".to_string())?;
            let value = value.ok_or_else(|| "experience motif has no value".to_string())?;
            let expected = atom_identity(&motif_bytes(&experience.experience_id, &role, &value));
            if expected != motif_identity {
                return Err(format!(
                    "experience motif is detached from {} {}={}",
                    experience.experience_id, role, value
                ));
            }
            let key = (role.clone(), value.clone());
            if experience
                .motifs
                .insert(
                    key,
                    IndexedExperienceMotif {
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
                    "experience {} contains a duplicate motif",
                    experience.experience_id
                ));
            }
        }
        for required in ["kind", "status", "domain", "cause", "effect", "direction"] {
            if count_role(&experience.motifs, required) != 1 {
                return Err(format!(
                    "experience {} has an invalid {required} cardinality",
                    experience.experience_id
                ));
            }
        }
    }

    Ok(ExperienceIndex {
        catalog_identity,
        snapshot_sequence: db.snapshot().sequence(),
        batches,
        experiences,
    })
}

fn inventory_report(index: &ExperienceIndex) -> ExperienceInventoryReport {
    ExperienceInventoryReport {
        runtime: EXPERIENCE_MEMORY_RUNTIME,
        catalog_identity: index.catalog_identity,
        snapshot_sequence: index.snapshot_sequence,
        batches: index
            .batches
            .iter()
            .map(|batch| ExperienceBatchInventory {
                batch_id: batch.batch_id.clone(),
                source_artifact_hash: batch.source_artifact_hash.clone(),
                experience_count: batch.experience_count,
            })
            .collect(),
        experiences: index
            .experiences
            .values()
            .map(|experience| ExperienceInventoryItem {
                experience_id: experience.experience_id.clone(),
                kind: feature_value(experience, "kind"),
                status: feature_value(experience, "status"),
                domain: feature_value(experience, "domain"),
                features: experience
                    .motifs
                    .values()
                    .map(|motif| FeatureSpec {
                        role: motif.role.clone(),
                        value: motif.value.clone(),
                    })
                    .collect(),
                feature_count: experience.motifs.len(),
                strengthened_motifs: experience
                    .motifs
                    .values()
                    .filter(|motif| motif.strengthen > 0)
                    .count(),
                weakened_motifs: experience
                    .motifs
                    .values()
                    .filter(|motif| motif.weaken > 0)
                    .count(),
            })
            .collect(),
    }
}

fn recall_index(index: &ExperienceIndex, query: &ExperienceQuery) -> ExperienceRecallReport {
    let mut hits = index
        .experiences
        .values()
        .filter(|experience| {
            !query
                .excluded_experiences
                .contains(&experience.experience_id)
        })
        .filter_map(|experience| match_experience(experience, query, true).ok())
        .filter(|hit| hit.matched_support >= query.minimum_support)
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| right.matched_support.cmp(&left.matched_support))
            .then_with(|| left.experience_id.cmp(&right.experience_id))
    });
    hits.truncate(query.limit);
    let answerable = hits.first().is_some_and(|hit| {
        hit.coverage_per_million >= query.minimum_coverage_per_million
            && hit.matched_support >= query.minimum_support
    });
    ExperienceRecallReport {
        runtime: EXPERIENCE_MEMORY_RUNTIME,
        catalog_identity: index.catalog_identity,
        snapshot_sequence: index.snapshot_sequence,
        query_id: query.query_id.clone(),
        answerable,
        insufficient_evidence: !answerable,
        hits,
    }
}

fn match_experience(
    experience: &IndexedExperience,
    query: &ExperienceQuery,
    respect_required: bool,
) -> Result<ExperienceHit, String> {
    let mut motifs = Vec::new();
    let mut total_weight = 0u64;
    let mut matched_weight = 0u64;
    for feature in &query.features {
        let base_weight = role_weight(&feature.role)
            .ok_or_else(|| format!("experience query role is not structural: {}", feature.role))?;
        total_weight = total_weight.saturating_add(base_weight);
        match experience
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
                return Ok(ExperienceHit {
                    experience_id: experience.experience_id.clone(),
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
    Ok(ExperienceHit {
        experience_id: experience.experience_id.clone(),
        score,
        matched_support: motifs.len(),
        query_support: query.features.len(),
        coverage_per_million,
        motifs,
    })
}

fn validate_query_roles(query: &ExperienceQuery) -> Result<(), String> {
    for QueryFeature { role, .. } in &query.features {
        if role_weight(role).is_none() {
            return Err(format!(
                "experience role cannot enter structural recall: {role}"
            ));
        }
    }
    Ok(())
}

fn role_weight(role: &str) -> Option<u64> {
    if role == "cause" || role == "effect" {
        Some(1_800)
    } else if role.starts_with("root/") {
        Some(1_500)
    } else if role.starts_with("evidence/") {
        Some(1_300)
    } else {
        match role {
            "context" => Some(1_100),
            "domain" => Some(1_000),
            "direction" => Some(900),
            "kind" => Some(700),
            "status" => Some(600),
            "delay" => Some(500),
            "magnitude" | "invariant" => Some(250),
            "support" | "confidence" | "contradiction" => Some(150),
            _ => None,
        }
    }
}

fn conductance(strengthen: u64, weaken: u64) -> u16 {
    let positive = i128::from(strengthen.min(i64::MAX as u64)) * STRENGTHEN_DELTA;
    let negative = i128::from(weaken.min(i64::MAX as u64)) * WEAKEN_DELTA;
    (BASE_CONDUCTANCE + positive - negative).clamp(1, MAX_CONDUCTANCE) as u16
}

fn count_role(motifs: &BTreeMap<(String, String), IndexedExperienceMotif>, role: &str) -> usize {
    motifs.values().filter(|motif| motif.role == role).count()
}

fn feature_value(experience: &IndexedExperience, role: &str) -> String {
    experience
        .motifs
        .values()
        .find(|motif| motif.role == role)
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
        return Err(format!("experience motif contains more than one {name}"));
    }
    Ok(())
}

fn batch_bytes(source_artifact_hash: &str, batch_id: &str) -> Vec<u8> {
    let value = format!("{source_artifact_hash}\0{batch_id}");
    prefixed(BATCH_PREFIX, &value)
}

fn parse_batch(bytes: &[u8]) -> Result<(String, String), String> {
    let value = parse_prefixed(bytes, BATCH_PREFIX)
        .ok_or_else(|| "experience batch atom has an invalid domain".to_string())?;
    let (source_artifact_hash, batch_id) = value
        .split_once('\0')
        .ok_or_else(|| "experience batch atom has no separator".to_string())?;
    if source_artifact_hash.len() != 64
        || !source_artifact_hash
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || batch_id.is_empty()
    {
        return Err("experience batch atom is malformed".into());
    }
    Ok((source_artifact_hash.to_string(), batch_id.to_string()))
}

fn prefixed(prefix: &[u8], value: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(prefix.len() + value.len());
    bytes.extend_from_slice(prefix);
    bytes.extend_from_slice(value.as_bytes());
    bytes
}

fn motif_bytes(experience_id: &str, role: &str, value: &str) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(MOTIF_PREFIX.len() + experience_id.len() + role.len() + value.len() + 2);
    bytes.extend_from_slice(MOTIF_PREFIX);
    bytes.extend_from_slice(experience_id.as_bytes());
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

fn parse_feedback(bytes: &[u8]) -> Result<Option<(&str, u64)>, String> {
    let Some(tail) = bytes.strip_prefix(FEEDBACK_PREFIX) else {
        return Ok(None);
    };
    let text =
        std::str::from_utf8(tail).map_err(|_| "experience feedback relation is not UTF-8")?;
    let (polarity, count) = text
        .split_once('/')
        .ok_or_else(|| "experience feedback relation lacks a count".to_string())?;
    if !matches!(polarity, "strengthen" | "weaken") {
        return Err("experience feedback relation has an unknown polarity".into());
    }
    let count = count
        .parse::<u64>()
        .map_err(|_| "experience feedback count is invalid".to_string())?;
    if count == 0 {
        return Err("experience feedback count must be positive".into());
    }
    Ok(Some((polarity, count)))
}

fn event_bytes(snapshot: u64, query_id: &str, expected: &str, selected: &str) -> Vec<u8> {
    let value = format!("{snapshot}\0{query_id}\0{expected}\0{selected}");
    prefixed(EVENT_PREFIX, &value)
}

fn outcome_key_bytes(outcome_key: &str) -> Vec<u8> {
    prefixed(OUTCOME_KEY_PREFIX, outcome_key)
}

fn idempotent_event_bytes(
    outcome_key: &str,
    query_id: &str,
    expected: &str,
    selected: &str,
) -> Vec<u8> {
    let value = format!("{outcome_key}\0{query_id}\0{expected}\0{selected}");
    prefixed(IDEMPOTENT_EVENT_PREFIX, &value)
}

fn parse_idempotent_event(bytes: &[u8]) -> Result<(&str, &str, &str, &str), String> {
    let value = bytes
        .strip_prefix(IDEMPOTENT_EVENT_PREFIX)
        .ok_or_else(|| "idempotent outcome event has an invalid domain".to_string())?;
    let text = std::str::from_utf8(value)
        .map_err(|_| "idempotent outcome event is not UTF-8".to_string())?;
    let mut fields = text.split('\0');
    let parsed = (
        fields
            .next()
            .ok_or_else(|| "idempotent outcome event has no key".to_string())?,
        fields
            .next()
            .ok_or_else(|| "idempotent outcome event has no query".to_string())?,
        fields
            .next()
            .ok_or_else(|| "idempotent outcome event has no expected experience".to_string())?,
        fields
            .next()
            .ok_or_else(|| "idempotent outcome event has no selected experience".to_string())?,
    );
    if fields.next().is_some() || parsed.0.is_empty() || parsed.1.is_empty() {
        return Err("idempotent outcome event fields are invalid".into());
    }
    Ok(parsed)
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
    use crate::experience_wire::encode_text;
    use crate::{parse_experience_batch, parse_experience_query};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "atom-causal-experience-{label}-{}-{nonce}.atomdb",
            std::process::id()
        ))
    }

    fn batch(batch_id: &str, law: bool) -> ExperienceBatch {
        let kind = if law { "law" } else { "observation" };
        let status = if law { "crystallized" } else { "observed" };
        let id = if law {
            "experience:law:a"
        } else {
            "experience:observation:a"
        };
        let mut raw = format!(
            "{}\nB\t{}\t{}\nE\t{}\n",
            crate::EXPERIENCE_BATCH_RUNTIME,
            "a".repeat(64),
            encode_text(batch_id),
            encode_text(id)
        );
        for (role, value) in [
            ("kind", kind),
            ("status", status),
            ("domain", "physical"),
            ("cause", "energy"),
            ("effect", "temperature"),
            ("direction", "+1"),
            ("root/0000", "conservation"),
        ] {
            raw.push_str(&format!(
                "F\t{}\t{}\n",
                encode_text(role),
                encode_text(value)
            ));
        }
        parse_experience_batch(&raw).unwrap()
    }

    fn query() -> ExperienceQuery {
        let mut raw = format!(
            "{}\nQ\t{}\t2\t8\t500000\n",
            crate::EXPERIENCE_QUERY_RUNTIME,
            encode_text("energy-temperature")
        );
        for (role, value, required) in [
            ("domain", "physical", "1"),
            ("cause", "energy", "1"),
            ("effect", "temperature", "0"),
            ("direction", "+1", "0"),
        ] {
            raw.push_str(&format!(
                "F\t{}\t{}\t{}\n",
                encode_text(role),
                encode_text(value),
                required
            ));
        }
        parse_experience_query(&raw).unwrap()
    }

    #[test]
    fn batches_append_and_duplicate_ingest_is_idempotent() {
        let path = temp_path("append");
        let first = ExperienceMemory::ingest(&path, &batch("observations", false)).unwrap();
        assert!(first.committed);
        let second = ExperienceMemory::ingest(&path, &batch("laws", true)).unwrap();
        assert!(second.committed);
        assert_eq!(second.total_batches, 2);
        assert_eq!(second.total_experiences, 2);
        let duplicate = ExperienceMemory::ingest(&path, &batch("laws", true)).unwrap();
        assert!(!duplicate.committed);
        assert_eq!(duplicate.snapshot_sequence, second.snapshot_sequence);
        let inventory = ExperienceMemory::inventory(&path).unwrap();
        assert_eq!(inventory.batches.len(), 2);
        assert_eq!(inventory.experiences.len(), 2);
        assert!(
            inventory.experiences[0]
                .features
                .iter()
                .any(|feature| feature.role == "effect" && feature.value == "temperature")
        );
        drop(fs::remove_file(path));
    }

    #[test]
    fn recall_feedback_and_reopen_are_structural() {
        let path = temp_path("recall");
        ExperienceMemory::ingest(&path, &batch("observations", false)).unwrap();
        ExperienceMemory::ingest(&path, &batch("laws", true)).unwrap();
        let before = ExperienceMemory::recall(&path, &query()).unwrap();
        assert!(before.answerable);
        assert_eq!(before.hits.len(), 2);
        let expected = "experience:law:a";
        let selected = "experience:observation:a";
        let wrong = ExperienceMemory::observe_outcome(&path, &query(), expected, selected).unwrap();
        assert!(!wrong.prediction_correct);
        assert!(
            wrong
                .adjustments
                .iter()
                .any(|item| { item.experience_id == expected && item.polarity == "strengthen" })
        );
        assert!(
            wrong
                .adjustments
                .iter()
                .any(|item| { item.experience_id == selected && item.polarity == "weaken" })
        );
        let reopened = ExperienceMemory::recall(&path, &query()).unwrap();
        assert_eq!(reopened.hits[0].experience_id, expected);
        drop(fs::remove_file(path));
    }

    #[test]
    fn outcome_key_is_idempotent_and_conflicts_fail_closed() {
        let path = temp_path("outcome-once");
        ExperienceMemory::ingest(&path, &batch("observations", false)).unwrap();
        ExperienceMemory::ingest(&path, &batch("laws", true)).unwrap();
        let outcome_key = &"b".repeat(64);
        let expected = "experience:law:a";
        let selected = "experience:observation:a";
        let first = ExperienceMemory::observe_outcome_once(
            &path,
            &query(),
            outcome_key,
            expected,
            selected,
        )
        .unwrap();
        assert!(first.committed);
        assert!(first.cell_identity.is_some());
        assert!(!first.adjustments.is_empty());
        let replay = ExperienceMemory::observe_outcome_once(
            &path,
            &query(),
            outcome_key,
            expected,
            selected,
        )
        .unwrap();
        assert!(!replay.committed);
        assert!(replay.cell_identity.is_none());
        assert!(replay.adjustments.is_empty());
        assert_eq!(replay.event_identity, first.event_identity);
        assert_eq!(replay.snapshot_sequence, first.snapshot_sequence);
        let mut conflicting = query();
        conflicting.query_id = "conflicting-outcome".to_string();
        let error = ExperienceMemory::observe_outcome_once(
            &path,
            &conflicting,
            outcome_key,
            expected,
            selected,
        )
        .unwrap_err();
        assert!(error.contains("outcome key conflicts"));
        drop(fs::remove_file(path));
    }

    #[test]
    fn unknown_required_structure_abstains() {
        let path = temp_path("unknown");
        ExperienceMemory::ingest(&path, &batch("observations", false)).unwrap();
        let mut unknown = query();
        unknown.features.push(QueryFeature {
            role: "effect".to_string(),
            value: "impossible".to_string(),
            required: true,
        });
        unknown.minimum_support = 2;
        let report = ExperienceMemory::recall(&path, &unknown).unwrap();
        assert!(report.insufficient_evidence);
        assert!(report.hits.is_empty());
        drop(fs::remove_file(path));
    }
}
