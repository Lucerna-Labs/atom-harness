use crate::{
    EXPERIENCE_BATCH_RUNTIME, EXPERIENCE_QUERY_RUNTIME, ExperienceBatch, ExperienceQuery,
    ExperienceSpec, FeatureSpec, QueryFeature,
};
use std::collections::BTreeSet;

pub fn parse_experience_batch(raw: &str) -> Result<ExperienceBatch, String> {
    let normalized = normalize(raw)?;
    let mut lines = normalized.lines();
    if lines.next() != Some(EXPERIENCE_BATCH_RUNTIME) {
        return Err("experience batch runtime marker is invalid".into());
    }
    let header = lines
        .next()
        .ok_or_else(|| "experience batch header is missing".to_string())?;
    let parts = header.split('\t').collect::<Vec<_>>();
    let ["B", source_artifact_hash, encoded_batch_id] = parts.as_slice() else {
        return Err("experience batch header is invalid".into());
    };
    let mut experiences = Vec::new();
    let mut current: Option<ExperienceSpec> = None;
    for (offset, line) in lines.enumerate() {
        let line_number = offset + 3;
        let parts = line.split('\t').collect::<Vec<_>>();
        match parts.as_slice() {
            ["E", encoded_id] => {
                if let Some(experience) = current.take() {
                    experiences.push(experience);
                }
                current = Some(ExperienceSpec {
                    experience_id: decode_text(encoded_id, line_number)?,
                    features: Vec::new(),
                });
            }
            ["F", encoded_role, encoded_value] => {
                let experience = current.as_mut().ok_or_else(|| {
                    format!("feature appears before an experience at line {line_number}")
                })?;
                experience.features.push(FeatureSpec {
                    role: decode_text(encoded_role, line_number)?,
                    value: decode_text(encoded_value, line_number)?,
                });
            }
            _ => return Err(format!("invalid experience record at line {line_number}")),
        }
    }
    if let Some(experience) = current {
        experiences.push(experience);
    }
    let batch = ExperienceBatch {
        source_artifact_hash: (*source_artifact_hash).to_string(),
        batch_id: decode_text(encoded_batch_id, 2)?,
        experiences,
        raw: normalized,
    };
    batch.validate()?;
    Ok(batch)
}

pub fn parse_experience_query(raw: &str) -> Result<ExperienceQuery, String> {
    let normalized = normalize(raw)?;
    let mut lines = normalized.lines();
    if lines.next() != Some(EXPERIENCE_QUERY_RUNTIME) {
        return Err("experience query runtime marker is invalid".into());
    }
    let header = lines
        .next()
        .ok_or_else(|| "experience query header is missing".to_string())?;
    let parts = header.split('\t').collect::<Vec<_>>();
    let ["Q", encoded_id, minimum_support, limit, minimum_coverage] = parts.as_slice() else {
        return Err("experience query header is invalid".into());
    };
    let mut query = ExperienceQuery {
        query_id: decode_text(encoded_id, 2)?,
        minimum_support: parse_usize(minimum_support, "minimum support")?,
        minimum_coverage_per_million: minimum_coverage
            .parse::<u32>()
            .map_err(|_| "minimum coverage is invalid".to_string())?,
        limit: parse_usize(limit, "result limit")?,
        features: Vec::new(),
        excluded_experiences: BTreeSet::new(),
    };
    for (offset, line) in lines.enumerate() {
        let line_number = offset + 3;
        let parts = line.split('\t').collect::<Vec<_>>();
        match parts.as_slice() {
            ["F", encoded_role, encoded_value, required] => {
                let required = match *required {
                    "0" => false,
                    "1" => true,
                    _ => {
                        return Err(format!(
                            "required flag must be 0 or 1 at line {line_number}"
                        ));
                    }
                };
                query.features.push(QueryFeature {
                    role: decode_text(encoded_role, line_number)?,
                    value: decode_text(encoded_value, line_number)?,
                    required,
                });
            }
            ["X", encoded_id] => {
                query
                    .excluded_experiences
                    .insert(decode_text(encoded_id, line_number)?);
            }
            _ => {
                return Err(format!(
                    "invalid experience query record at line {line_number}"
                ));
            }
        }
    }
    query.validate()?;
    Ok(query)
}

fn normalize(raw: &str) -> Result<String, String> {
    if raw.is_empty() || raw.len() > 256 * 1024 * 1024 {
        return Err("wire payload must contain 1..=268435456 bytes".into());
    }
    if raw.contains('\0') {
        return Err("wire payload cannot contain NUL bytes".into());
    }
    let normalized = raw.replace("\r\n", "\n");
    if normalized.contains('\r') {
        return Err("wire payload contains a bare carriage return".into());
    }
    Ok(normalized.trim_end_matches('\n').to_string())
}

fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("{name} is invalid"))
}

fn decode_text(encoded: &str, line_number: usize) -> Result<String, String> {
    if encoded.is_empty() || !encoded.len().is_multiple_of(2) {
        return Err(format!("hex text is invalid at line {line_number}"));
    }
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = nibble(pair[0])
            .ok_or_else(|| format!("hex text contains an invalid digit at line {line_number}"))?;
        let low = nibble(pair[1])
            .ok_or_else(|| format!("hex text contains an invalid digit at line {line_number}"))?;
        decoded.push((high << 4) | low);
    }
    String::from_utf8(decoded)
        .map_err(|_| format!("hex text is not valid UTF-8 at line {line_number}"))
}

fn nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn encode_text(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value.as_bytes() {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}
