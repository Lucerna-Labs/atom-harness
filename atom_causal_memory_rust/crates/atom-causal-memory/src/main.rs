use atom_causal_memory::{
    CausalMemory, ExperienceMemory, parse_experience_batch, parse_experience_query, parse_manifest,
    parse_query,
};
use std::io::{self, Read};

fn main() {
    if let Err(message) = run() {
        eprintln!("atom-causal-memory: {message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command, store] if command == "import" => {
            let input = read_stdin()?;
            let manifest = parse_manifest(&input)?;
            let report = CausalMemory::import(store, &manifest)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store] if command == "inventory" => {
            let report = CausalMemory::inventory(store)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store] if command == "query" => {
            let input = read_stdin()?;
            let query = parse_query(&input)?;
            let report = CausalMemory::query(store, &query)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store, expected, selected] if command == "observe" => {
            let input = read_stdin()?;
            let query = parse_query(&input)?;
            let report = CausalMemory::observe_prediction(store, &query, expected, selected)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store] if command == "ingest-experiences" => {
            let input = read_stdin()?;
            let batch = parse_experience_batch(&input)?;
            let report = ExperienceMemory::ingest(store, &batch)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store] if command == "experience-inventory" => {
            let report = ExperienceMemory::inventory(store)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store] if command == "recall-experiences" => {
            let input = read_stdin()?;
            let query = parse_experience_query(&input)?;
            let report = ExperienceMemory::recall(store, &query)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store, expected, selected] if command == "observe-experience" => {
            let input = read_stdin()?;
            let query = parse_experience_query(&input)?;
            let report = ExperienceMemory::observe_outcome(store, &query, expected, selected)?;
            println!("{}", report.to_json());
            Ok(())
        }
        [command, store, outcome_key, expected, selected]
            if command == "observe-experience-once" =>
        {
            let input = read_stdin()?;
            let query = parse_experience_query(&input)?;
            let report = ExperienceMemory::observe_outcome_once(
                store,
                &query,
                outcome_key,
                expected,
                selected,
            )?;
            println!("{}", report.to_json());
            Ok(())
        }
        _ => Err(usage().to_string()),
    }
}

fn read_stdin() -> Result<String, String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("could not read standard input: {error}"))?;
    if input.is_empty() {
        return Err("standard input payload is empty".into());
    }
    Ok(input)
}

fn usage() -> &'static str {
    "usage:
  atom-causal-memory import <store> < manifest
  atom-causal-memory inventory <store>
  atom-causal-memory query <store> < query
  atom-causal-memory observe <store> <expected-glyph> <selected-glyph> < query
  atom-causal-memory ingest-experiences <store> < batch
  atom-causal-memory experience-inventory <store>
  atom-causal-memory recall-experiences <store> < query
  atom-causal-memory observe-experience <store> <expected> <selected> < query
  atom-causal-memory observe-experience-once <store> <outcome-key> <expected> <selected> < query"
}
