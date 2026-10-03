# Source and environment provenance

The base repository supplies the HMA controller, blind evaluation gate, generic
MLE-bench integration, Kimi proxy and vendored `hmz` native runtime. This update
adds the fresh-run campaign, explicit goal/NTA modes, matrix, portable builds,
data preparation and analysis. No historic run artifacts are required or shipped.

Imported files and original hashes are recorded in
[`provenance.json`](../src/hma/repro/assets/provenance.json). Data catalog,
integrity, preparation worker and preparation orchestration came from
`agentkaggle/KaggleBench` at `46ebbed386c6ca4c8d580b4554ce17f8da81c918`.
Imports were changed from `flowbench` to `hma`, cluster runtime-image dependencies
were replaced with local configuration, and formatting was normalized. The
campaign interface is new; unused catalog compatibility with older Dojo tasks is
not exposed through the reproduction CLI.

| Source | Frozen revision | Use |
| --- | --- | --- |
| openai/mle-bench | `507f92e1138bb6e40dac5c6ee7a6758e6424bf97` | All 75 data preparers, checksums, leaderboards and graders |
| sjtu-sai-agents/EvoMaster | `36a52bc6c42a6b9fd710a41c52f3c3bb948b9ac9` | ML-Master 2.0 no-prior |
| InternScience/MLEvolve | `9c5c8a3b23f0361708b59a401452dddc00f97189` | MLEvolve no-prior |
| science-learner/ScienceFlow | `ad1f7ddae57873807da5a9c0070c68ba9a740515` | ScienceFlow |
| BAAI/bge-base-en-v1.5 | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | Generic MLEvolve memory encoder |

The three external `paper_runtime` adapters were copied byte-for-byte from the
local KaggleBench `experiments/harness-repro-16task-20260918/adapter/fixed` versions.
Their individual source locks verify both upstream files and adapter bytes.
Private base-image references were removed from lock metadata. Fixed adapters
restore integrity checks and use the correct ScienceFlow interpreter and resolved
candidate paths; MLEvolve's no-prior configuration keeps its search mechanisms.
The new surrounding `harness_entry.py` supplies a declared score-independent
no-final fallback and preserves pre-research infrastructure failure.

The fixed upstream tree remains read-only in `/opt/paper/upstream`; each execution
copies it into its own workspace, and the adapter applies its documented runtime
configuration/patches there. The repository does not redistribute those entire
upstream trees or pretrained weights; builds fetch the pinned revisions and
verify their allowlisted files. Consult each source's original license for use.
The adapter provenance source does not carry a separate license grant in the
imported folder; confirm the authors' redistribution terms when publishing it.
The existing repository [LICENSE](../LICENSE) is preserved rather than assigning
new third-party ownership or licensing terms.

## Hardware and protocol provenance

Section 4.1 and Appendix C.1 specify one NVIDIA A10 per task, with 30 vCPUs
backed by an Intel Xeon Platinum 8358 at 2.60 GHz and 220 GiB RAM. The reported
GPU memory is 23028 MiB; NVIDIA driver 570.148.08 reports CUDA 12.8 support.
The two alternating agents share these resources. The fresh-run defaults are
therefore one A10, 30 vCPUs and `220g` of agent memory for both native and external
16-task workflows. Appendix D.2 supplies no separate hardware profile for the
external 16-task comparison. Table 5 describes cited papers' original environments,
including MLEvolve's H200, rather than a hardware override for this campaign.

The manuscript does not specify `/dev/shm` or separate evaluator limits. This
implementation uses 64 GiB shared memory and a 2-CPU/16-GiB private evaluator,
whose overhead must fit outside the agent allocation. Its 600-second review call
limit is also an implementation choice within the paper's 900-second reserve;
the total native window remains 21,600 seconds. See the
[experiment map](experiments.md) for fresh-run repeat and terminal-window choices.

`python -m hma.repro.cli doctor` and campaign execution check selected GPUs against
`expected_gpu_model` (default `NVIDIA A10`). Explicitly changing it or setting it
to null permits a documented hardware variant. Campaign `environment.json`
records GPU UUID/name/memory/driver and host CPU model/count/RAM; these records
are part of resume identity. CPU model and driver version are not required to
equal the historical values. Low-level `hma-run` omits this hardware preflight.

The optional Swarm controller runs with host networking and the node's local
Docker socket, mounting the shared filesystem at its unchanged absolute path.
Actor/evaluator execution still uses the original local containers and isolation.
This is an added deployment implementation, not a claim about the paper's cluster
or scheduler. Every image is pinned to a registry digest. Node assignments and
source identity are frozen before deployment; GPU discovery uses a short-lived
local Docker helper. See [Swarm execution](swarm.md).

## Dependency policy

- Host/native controller: Python 3.12, `requirements-repro.lock` (runtime, tests,
  plotting and preparation tooling). Earlier `requirements-core.lock` and
  `requirements-runtime.lock` remain source-validation references, but the new
  README uses the complete reproduction lock.
- Native image: CUDA 12.6.3 runtime/Ubuntu 24.04, Node 22.23.2 (download checksum),
  Codex 0.153.0, Claude Code 2.1.259, Kimi Code 0.41.0; DeepSeek SDK/runtime
  0.1.5rc1 in the Python lock. Common ML stack is `requirements-ml.lock`.
- MLEvolve and ML-Master: the common native image plus
  `requirements-harness.lock`, derived from actual runtime imports. MLEvolve's
  full upstream environment dump mixes domain packages and conflicting framework
  requirements; installing it wholesale would replace controller dependencies.
  The smaller explicit runtime layer keeps the shared pinned training stack.
- ScienceFlow: its original `uv.lock` in its own `.venv`, including its CUDA 12.8
  PyTorch stack. It does not use the outer image's Python for the research process.
- MLEvolve: the generic BGE memory encoder is downloaded at build time to the
  exact location required by its mechanism audit. Competition-specific DINOv3
  cold-start assets are unnecessary in the no-prior arm.
- These portable environments differ from the authors' old private images.
  Docker image IDs, source fingerprints, data-manifest hashes, local resource
  settings and recorded hardware are frozen per campaign. Mutable public OS
  image tags/apt repositories
  mean builds on different dates can produce different image IDs. Preserve/export
  the built images when transferring an identical environment between machines.

To refresh dependencies intentionally, use the pinned `uv` and inspect diffs:

```bash
uv pip compile docker/requirements-ml.in --constraint requirements-runtime.lock --python-version 3.12 --output-file docker/requirements-ml.lock
uv pip compile pyproject.toml --extra runtime --extra test --extra evaluation --constraint requirements-runtime.lock --constraint docker/requirements-ml.lock --python-version 3.12 --output-file requirements-repro.lock
uv pip compile docker/requirements-harness.in --constraint requirements-repro.lock --constraint docker/requirements-ml.lock --python-version 3.12 --output-file docker/requirements-harness.lock
```

Rebuild affected images, rerun applicable tests/smokes, and start a new campaign.
Do not edit frozen runtime bytes without also documenting the protocol change and
recomputing the relevant lock; a hash mismatch must fail visibly.
