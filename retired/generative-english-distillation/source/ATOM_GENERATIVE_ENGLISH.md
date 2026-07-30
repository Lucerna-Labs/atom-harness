# Atom Generative English

Detailed maintained records:

- `ATOM_GENERATIVE_ENGLISH_DEV_NOTES.md` — architecture decisions, invariants,
  defect history, verification, Kaggle lineage, and open measurements;
- `ATOM_GENERATIVE_ENGLISH_USER_NOTES.md` — plain-English behavior, results,
  limits, and current operating state.

The language target is unrestricted English generation and multi-turn
conversation. It is not a command grammar, intent classifier, or fixed set of
sentences.

The model predicts ordinary subword tokens through a sparse directed temporal
causal graph. Local and dilated causal predecessors represent what each token
can depend on. Phase locking synchronizes compatible representations,
topological persistence carries durable summaries and exact ordered landmarks
across chunks, and symbolic transition memory records causal
suffix-to-continuation edges actually observed in the current context. Phase
mixing combines features nonlinearly, controlled thermal criticality regulates
exploration, and projective measurement maps the resulting state to the next
token. The seven root primitives remain explicit trainable mechanisms.

## Language acquisition

Foundation training deterministically interleaves the `default` configuration
of full FineWeb-Edu with the complete Cosmopedia-v2 corpus. Dialogue training
uses the `all` configuration of full SmolTalk for two epochs; it does not use
the reduced `smol-smoltalk` collection intended for models below 1B parameters.
Both stages use the same exact 49,152-token mapping. The foundation and dialogue
teachers are the pinned 1.7B SmolLM2 base and instruct models. Training combines
next-token likelihood with top-k-plus-tail distillation so the new sequence
engine learns a broad language distribution without inheriting the teacher
architecture.

The default target is the 226,877,312-parameter profile with a 524,288-token
context ceiling. The first 8,192 positions retain the native phase geometry.
Beyond that boundary, phase positions extend logarithmically rather than
wrapping into periodic aliases. Each layer keeps an exact 6,144-token causal
ring and compresses older completed regions into query-recognized binary
summaries and exact position-bearing episodic landmarks. A separate bounded
symbolic cache retains observed 4-, 8-, and 16-token transition edges. Its
declared 512K ceiling is 1,572,864 transition entries, linear rather than
quadratic in context length, while the neural landmark hierarchy remains
logarithmic. Prompt prefill runs in vectorized landmark chunks rather than a
Python neural step for every prompt token.

The symbolic cache is not a second knowledge base and does not invent
continuations. It can only propose a token that followed the same causal token
pattern earlier in the active context. The proposal is accepted only when the
neural model already places it within a four-logit compatibility band. This
lets exact identifiers and ordered values survive compression without allowing
an unrelated repeated phrase to override the language model.

Foundation training targets 4.9152 billion streamed tokens over resumable
Kaggle sessions, followed by 130,490 dialogue optimizer steps at a default
2,048-token training length. That count covers both passes over all 1,043,917
training conversations, including the final non-divisible gradient
accumulation. The
40M and 82M profiles remain available for architecture comparison, but they are
not the default competence target.

Dialogue is followed by 2,600 context-conditioning optimizer steps. The
curriculum moves from 2K, 4K, and 8K spans through 16K, 32K, 64K, 128K, 264K,
and 512K. For long examples, the model scans the prefix into bounded graph
state without retaining an autograd tape for every token, then trains through
the final query-and-answer region. At 2K, the entire persistence path trains
end-to-end so the landmark gate learns what must survive before the same
mechanism is extended with bounded gradients. This directly teaches retrieval, record
selection, ordering, and state updates while keeping the gradient memory
bounded. The final five percent repeatedly includes both 264K and 512K
examples.

Dialogue training fails closed until the foundation checkpoint has consumed the
full 4.9152-billion-token target. Earlier 2,048- or 8,192-context foundation
checkpoints can seed the 524,288-context model through a strict one-way
expansion: existing weights and causal distances are copied exactly, cumulative
token and optimizer-step counters are retained, and the shape-incompatible
optimizer state is restarted. The additional reach comes from bounded recency,
multi-scale persistence, and extended phase positions; it does not add a
half-million-token dense attention matrix.

Context conditioning fails closed until the dialogue stage reaches all 130,490
optimizer steps. External evaluation and the chat runtime accept only a
context-stage checkpoint, so ordinary short-block dialogue training cannot be
mistaken for long-context training.

## Admission to conversation

`atom_english_chat.py` loads only a context-stage checkpoint. By default it
also requires an external evaluation report whose hash binds it to that exact
checkpoint and whose recomputed gate passes Wikitext-103, LAMBADA, BLiMP,
HellaSwag, all 541 IFEval prompts, and long-context probes at 32K, 64K, 128K,
264K, and 512K. The context suite covers exact retrieval, multi-record
selection, ordering, and state updates; every length must score at least 75%.
IFEval generations receive up to 2,048 tokens so long-form constraints are
actually possible. This prevents a convenient sample, a configured context
number, or an unrelated report from being presented as English competence.

The chat runtime is open-ended and stateful. A graph knowledge source can be
connected under one of three evidence policies:

- `available`: use retrieved evidence when present and report when none matched;
- `required`: abstain from unsupported factual answers;
- `off`: generate without retrieval.

Every session can write a JSON transcript and a side HTML view showing the
actual prompts, generated responses, evaluation state, and graph retrieval
surface.

## Commands

Build and test the single-source Kaggle program:

```powershell
py -3.13 scripts\build_kaggle_generative_english_bundle.py
py -3.13 -W ignore::DeprecationWarning `
  kaggle\generative-english-v1\atom_generative_english_kaggle.py `
  --mode self-test
```

Submit the private foundation-training run:

```powershell
py -3.13 -m kaggle kernels push `
  -p "C:\Projects\atom lora\kaggle\generative-english-v1"
```

After foundation, dialogue, context conditioning, and external evaluation,
start a conversation:

```powershell
py -3.13 atom_english_chat.py `
  --checkpoint <context-checkpoint-directory> `
  --evaluation <language-evaluation.json> `
  --transcript <conversation.json> `
  --side-view <conversation.html>
```

Architecture tests establish bounded state, phase reach, and equivalence among
parallel, recurrent, and chunked execution. They do not establish useful
half-million-token recall. That claim remains closed until the trained,
hash-bound dialogue checkpoint passes every long-context length and the broader
language gate, followed by an exercised conversation.
