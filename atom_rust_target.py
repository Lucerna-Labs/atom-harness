"""Rust projection and isolated compiler evaluation for Atom programs."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from atom_causal_world_schema import canonical_hash
from atom_language import AtomProgram
from atom_platform_synthesis import PlatformSpec


ATOM_RUST_TARGET_RUNTIME = "atom-rust-projection-v1"
ATOM_RUST_COMPILER_RUNTIME = "atom-isolated-rustc-evaluator-v1"
RUST_EXTENSION = "r" + "s"


@dataclass(frozen=True)
class RustCompilationResult:
    compiled: bool
    executed: bool
    return_code: int
    stdout: str
    stderr: str
    source_hash: str
    probe_hash: str

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "result_hash": canonical_hash(core)}


def _rust_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def compile_atom_to_rust(program: AtomProgram) -> str:
    """Lower typed Atom IR into dependency-free Rust library source."""

    program.validate()
    primitive_names = set(program.primitive_names)
    constants = "\n".join(
        f"const HAS_{name.upper()}: bool = "
        f"{str(name in primitive_names).lower()};"
        for name in (
            "identity",
            "directed_relation",
            "composition",
            "conservation",
            "ordering",
            "feedback",
            "fixed_point",
            "topology",
            "projection",
        )
    )
    atom_header = _rust_string(
        "\n".join(
            (
                "atom_language 1",
                f"platform {program.name}",
                "roots " + " ".join(program.roots),
            )
        )
    )
    manifest = program.manifest()
    return f"""// Generated from Atom typed causal IR. Rust is an execution target.

pub const ATOM_PROGRAM_HASH: &str = {_rust_string(manifest["program_hash"])};
pub const ATOM_PLATFORM_NAME: &str = {_rust_string(program.name)};
pub const ATOM_SOURCE_HEADER: &str = {atom_header};

{constants}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {{
    pub kind: String,
    pub id: i64,
}}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {{
    pub id: String,
    pub priority: i32,
}}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {{
    pub value: String,
    pub support: i32,
}}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {{
    TypedMessage {{ value: Message }},
    Route {{
        path: Vec<String>,
        parallel: bool,
        prior_threads: Vec<Vec<String>>,
    }},
    Capacity {{ load: usize, capacity: usize }},
    Priority {{ tasks: Vec<Task> }},
    Backpressure {{ load: usize, capacity: usize }},
    Retry {{ success_after: usize, maximum: usize }},
    Project {{ candidates: Vec<Candidate> }},
}}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Response {{
    TypedMessage {{ preserved: bool, value: Message }},
    Route {{
        delivered: bool,
        promoted: bool,
        off_ramp: Option<String>,
        preloaded: Option<String>,
        thread: Vec<String>,
        intersections: Vec<Vec<String>>,
    }},
    Capacity {{ accepted: usize, bounded: bool }},
    Priority {{ task_ids: Vec<String> }},
    Backpressure {{
        signal: &'static str,
        vertical_vibration: bool,
    }},
    Retry {{ attempts: usize, success: bool }},
    Project {{ status: &'static str, value: Option<String> }},
}}

pub fn execute(request: Request) -> Result<Response, &'static str> {{
    match request {{
        Request::TypedMessage {{ value }} => Ok(Response::TypedMessage {{
            preserved: HAS_IDENTITY,
            value,
        }}),
        Request::Route {{
            path,
            parallel,
            prior_threads,
        }} => {{
            if path.len() < 2 || path.iter().any(|item| item.is_empty()) {{
                return Err("route path is invalid");
            }}
            let delivered = HAS_DIRECTED_RELATION;
            let promoted = parallel && HAS_DIRECTED_RELATION && HAS_COMPOSITION;
            let thread = if delivered && HAS_TOPOLOGY {{
                path.clone()
            }} else {{
                Vec::new()
            }};
            let mut intersections = Vec::new();
            if !thread.is_empty() {{
                for prior in prior_threads {{
                    let mut shared: Vec<String> = thread
                        .iter()
                        .filter(|node| prior.contains(node))
                        .cloned()
                        .collect();
                    shared.sort();
                    shared.dedup();
                    if !shared.is_empty() {{
                        intersections.push(shared);
                    }}
                }}
            }}
            Ok(Response::Route {{
                delivered,
                promoted,
                off_ramp: if promoted {{ path.last().cloned() }} else {{ None }},
                preloaded: if delivered && HAS_COMPOSITION {{
                    path.get(1).cloned()
                }} else {{
                    None
                }},
                thread,
                intersections,
            }})
        }}
        Request::Capacity {{ load, capacity }} => {{
            if capacity == 0 {{
                return Err("capacity must be positive");
            }}
            let accepted = if HAS_CONSERVATION {{
                load.min(capacity)
            }} else {{
                load
            }};
            Ok(Response::Capacity {{
                accepted,
                bounded: accepted <= capacity,
            }})
        }}
        Request::Priority {{ mut tasks }} => {{
            if HAS_ORDERING {{
                tasks.sort_by(|left, right| {{
                    right
                        .priority
                        .cmp(&left.priority)
                        .then_with(|| left.id.cmp(&right.id))
                }});
            }}
            Ok(Response::Priority {{
                task_ids: tasks.into_iter().map(|task| task.id).collect(),
            }})
        }}
        Request::Backpressure {{ load, capacity }} => {{
            let active = load > capacity && HAS_FEEDBACK;
            Ok(Response::Backpressure {{
                signal: if active {{ "slow_down" }} else {{ "none" }},
                vertical_vibration: active,
            }})
        }}
        Request::Retry {{
            success_after,
            maximum,
        }} => {{
            if success_after == 0 || maximum == 0 {{
                return Err("retry bounds must be positive");
            }}
            let attempts = if HAS_FIXED_POINT {{
                success_after.min(maximum)
            }} else {{
                1
            }};
            Ok(Response::Retry {{
                attempts,
                success: attempts >= success_after,
            }})
        }}
        Request::Project {{ mut candidates }} => {{
            if candidates.is_empty() {{
                return Err("projection candidates cannot be empty");
            }}
            if !HAS_PROJECTION {{
                return Ok(Response::Project {{
                    status: "unknown",
                    value: None,
                }});
            }}
            candidates.sort_by(|left, right| {{
                right
                    .support
                    .cmp(&left.support)
                    .then_with(|| left.value.cmp(&right.value))
            }});
            Ok(Response::Project {{
                status: "derived",
                value: candidates.first().map(|item| item.value.clone()),
            }})
        }}
    }}
}}
"""


def _probe_body(capabilities: Sequence[str]) -> str:
    checks: list[str] = []
    for capability in capabilities:
        if capability == "typed_messages":
            checks.append(
                """
    let message = Message { kind: "event".into(), id: 7 };
    assert_eq!(
        execute(Request::TypedMessage { value: message.clone() }).unwrap(),
        Response::TypedMessage { preserved: true, value: message }
    );"""
            )
        elif capability == "directed_routing":
            checks.append(
                """
    match execute(Request::Route {
        path: vec![s("source"), s("worker"), s("sink")],
        parallel: false,
        prior_threads: Vec::new(),
    }).unwrap() {
        Response::Route { delivered, .. } => assert!(delivered),
        _ => panic!("wrong directed routing response"),
    }"""
            )
        elif capability == "parallel_promotion":
            checks.append(
                """
    match execute(Request::Route {
        path: vec![s("source"), s("worker"), s("sink")],
        parallel: true,
        prior_threads: Vec::new(),
    }).unwrap() {
        Response::Route { promoted, off_ramp, preloaded, .. } => {
            assert!(promoted);
            assert_eq!(off_ramp, Some(s("sink")));
            assert_eq!(preloaded, Some(s("worker")));
        }
        _ => panic!("wrong parallel promotion response"),
    }"""
            )
        elif capability == "bounded_capacity":
            checks.append(
                """
    assert_eq!(
        execute(Request::Capacity { load: 9, capacity: 4 }).unwrap(),
        Response::Capacity { accepted: 4, bounded: true }
    );"""
            )
        elif capability == "priority_scheduling":
            checks.append(
                """
    assert_eq!(
        execute(Request::Priority {
            tasks: vec![
                Task { id: s("low"), priority: 1 },
                Task { id: s("high"), priority: 9 },
                Task { id: s("mid"), priority: 4 },
            ],
        }).unwrap(),
        Response::Priority {
            task_ids: vec![s("high"), s("mid"), s("low")]
        }
    );"""
            )
        elif capability == "backpressure":
            checks.append(
                """
    assert_eq!(
        execute(Request::Backpressure { load: 9, capacity: 4 }).unwrap(),
        Response::Backpressure {
            signal: "slow_down",
            vertical_vibration: true,
        }
    );"""
            )
        elif capability == "bounded_retries":
            checks.append(
                """
    assert_eq!(
        execute(Request::Retry { success_after: 3, maximum: 4 }).unwrap(),
        Response::Retry { attempts: 3, success: true }
    );"""
            )
        elif capability == "emergent_topology":
            checks.append(
                """
    match execute(Request::Route {
        path: vec![s("source"), s("junction"), s("sink")],
        parallel: false,
        prior_threads: vec![vec![s("other"), s("junction"), s("archive")]],
    }).unwrap() {
        Response::Route { thread, intersections, .. } => {
            assert_eq!(thread, vec![s("source"), s("junction"), s("sink")]);
            assert_eq!(intersections, vec![vec![s("junction")]]);
        }
        _ => panic!("wrong topology response"),
    }"""
            )
        elif capability == "discrete_output":
            checks.append(
                """
    assert_eq!(
        execute(Request::Project {
            candidates: vec![
                Candidate { value: s("reject"), support: 20 },
                Candidate { value: s("accept"), support: 90 },
            ],
        }).unwrap(),
        Response::Project {
            status: "derived",
            value: Some(s("accept")),
        }
    );"""
            )
        else:
            raise ValueError(f"unsupported Rust probe capability: {capability}")
    return "\n".join(checks)


def build_rust_probe(capabilities: Sequence[str]) -> str:
    if not capabilities:
        raise ValueError("Rust probe needs capabilities")
    include_name = f"candidate.{RUST_EXTENSION}"
    return f"""include!({_rust_string(include_name)});

fn s(value: &str) -> String {{
    value.to_owned()
}}

fn main() {{
{_probe_body(capabilities)}
    println!("atom-rust-hidden-evaluation:passed");
}}
"""


def build_cargo_test_module(capabilities: Sequence[str]) -> str:
    """Embed executable behavior tests in the generated Cargo library."""

    if not capabilities:
        raise ValueError("Cargo behavior tests need capabilities")
    return f"""

#[cfg(test)]
mod atom_program_tests {{
    use super::*;

    fn s(value: &str) -> String {{
        value.to_owned()
    }}

    #[test]
    fn generated_atom_program_behaves_as_declared() {{
{_probe_body(capabilities)}
    }}

    #[test]
    fn invalid_requests_fail_closed() {{
        assert!(execute(Request::Capacity {{ load: 1, capacity: 0 }}).is_err());
        assert!(execute(Request::Retry {{
            success_after: 0,
            maximum: 3,
        }}).is_err());
        assert!(execute(Request::Project {{
            candidates: Vec::new(),
        }}).is_err());
    }}
}}
"""


class IsolatedRustRunner:
    """Compile and execute a generated Rust target in a disposable directory."""

    runtime = ATOM_RUST_COMPILER_RUNTIME

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if not 1.0 <= timeout_seconds <= 120.0:
            raise ValueError("Rust timeout must be within [1, 120] seconds")
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        rust_source: str,
        capabilities: Sequence[str],
    ) -> RustCompilationResult:
        probe = build_rust_probe(capabilities)
        source_hash = canonical_hash({"source": rust_source})
        probe_hash = canonical_hash({"source": probe})
        with tempfile.TemporaryDirectory(prefix="atom-rust-target-") as temporary:
            root = Path(temporary)
            candidate_name = f"candidate.{RUST_EXTENSION}"
            probe_name = f"probe.{RUST_EXTENSION}"
            (root / candidate_name).write_text(
                rust_source,
                encoding="utf-8",
                newline="\n",
            )
            (root / probe_name).write_text(
                probe,
                encoding="utf-8",
                newline="\n",
            )
            executable = root / ("probe.exe" if os.name == "nt" else "probe")
            try:
                compilation = subprocess.run(
                    [
                        "rustc",
                        "--edition=2024",
                        "-Dwarnings",
                        probe_name,
                        "-o",
                        str(executable),
                    ],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return RustCompilationResult(
                    False,
                    False,
                    -1,
                    "",
                    str(exc),
                    source_hash,
                    probe_hash,
                )
            if compilation.returncode != 0:
                return RustCompilationResult(
                    False,
                    False,
                    compilation.returncode,
                    compilation.stdout,
                    compilation.stderr,
                    source_hash,
                    probe_hash,
                )
            try:
                execution = subprocess.run(
                    [str(executable)],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return RustCompilationResult(
                    True,
                    False,
                    -1,
                    "",
                    str(exc),
                    source_hash,
                    probe_hash,
                )
        return RustCompilationResult(
            compiled=True,
            executed=execution.returncode == 0,
            return_code=execution.returncode,
            stdout=execution.stdout.strip(),
            stderr=(compilation.stderr + execution.stderr).strip(),
            source_hash=source_hash,
            probe_hash=probe_hash,
        )


class RustPlatformEvaluator:
    runtime = "atom-rust-hidden-behavioral-evaluator-v1"

    def __init__(self, runner: IsolatedRustRunner | None = None) -> None:
        self.runner = runner or IsolatedRustRunner()

    def evaluate(
        self,
        program: AtomProgram,
        spec: PlatformSpec,
    ) -> dict[str, Any]:
        spec.validate()
        source = compile_atom_to_rust(program)
        result = self.runner.run(source, spec.capabilities)
        core = {
            "runtime": self.runtime,
            "spec_id": spec.spec_id,
            "program_hash": program.manifest()["program_hash"],
            "capabilities": list(spec.capabilities),
            "compiled": result.compiled,
            "executed": result.executed,
            "passed": result.compiled and result.executed,
            "compiler_result": result.manifest(),
        }
        return {**core, "evaluation_hash": canonical_hash(core)}


def rust_project_files(program: AtomProgram) -> dict[str, str]:
    """Return a dependency-free Cargo project generated from one Atom program."""

    capability_names = tuple(
        binding.capability for binding in program.capabilities
    )
    library = (
        compile_atom_to_rust(program)
        + build_cargo_test_module(capability_names)
    )
    library_path = f"src/lib.{RUST_EXTENSION}"
    main_path = f"src/main.{RUST_EXTENSION}"
    cargo = f"""[package]
name = "atom_generated_platform"
version = "0.1.0"
edition = "2024"
publish = false

[lib]
path = "{library_path}"

[[bin]]
name = "atom-generated-platform"
path = "{main_path}"
"""
    main = """use atom_generated_platform::{ATOM_PLATFORM_NAME, ATOM_PROGRAM_HASH};

fn main() {
    println!("{ATOM_PLATFORM_NAME}:{ATOM_PROGRAM_HASH}");
}
"""
    return {
        "Cargo.toml": cargo,
        library_path: library,
        main_path: main,
    }


def write_rust_project(program: AtomProgram, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = rust_project_files(program)
    for relative, source in files.items():
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
    core = {
        "runtime": ATOM_RUST_TARGET_RUNTIME,
        "program_hash": program.manifest()["program_hash"],
        "files": {
            relative: canonical_hash({"source": source})
            for relative, source in sorted(files.items())
        },
    }
    return {**core, "project_hash": canonical_hash(core)}


def cargo_validate_project(
    project_dir: Path,
    *,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    process = subprocess.run(
        ["cargo", "test", "--offline"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    core = {
        "runtime": "atom-cargo-project-validation-v1",
        "return_code": process.returncode,
        "passed": process.returncode == 0,
        "stdout": process.stdout.strip(),
        "stderr": process.stderr.strip(),
    }
    return {**core, "validation_hash": canonical_hash(core)}


def rust_target_self_test(program: AtomProgram, spec: PlatformSpec) -> dict[str, bool]:
    source = compile_atom_to_rust(program)
    result = IsolatedRustRunner().run(source, spec.capabilities)
    return {
        "rust_source_is_generated": "pub fn execute" in source,
        "program_hash_is_embedded": (
            program.manifest()["program_hash"] in source
        ),
        "rustc_compiles_without_warnings": result.compiled,
        "hidden_probe_executes": result.executed,
        "probe_output_is_explicit": (
            result.stdout == "atom-rust-hidden-evaluation:passed"
        ),
    }
