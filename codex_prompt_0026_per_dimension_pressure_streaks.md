# ENN-PyTorch Codex 작업 지침 — CPU/CUDA sustained-pressure streak 분리

## 작업 목적

현재 `ConservativeRuntimeGovernor`는 pressure가 실제로 축소하는 budget field는 CPU와 CUDA 차원별로 구분하지만, 지속성 판단에는 하나의 `consecutive_high_pressure_passes`를 사용한다.

이 구조에서는 다음처럼 서로 다른 차원의 압력이 잘못 합산될 수 있다.

```text
pass 1: CPU high
pass 2: CUDA high
```

`shrink_after_pressure_passes=2`이면 CUDA pressure가 한 번만 발생했는데도 두 번째 pass에서 device budget이 축소될 수 있다.

이번 작업의 목적은 CPU와 CUDA pressure streak를 독립적으로 추적하여 다음을 보장하는 것이다.

- CPU-only와 CUDA-only high pass는 서로의 지속 streak에 합산되지 않는다.
- 각 차원이 자체 threshold에 도달했을 때만 해당 차원에 대응하는 budget을 축소한다.
- 한 차원의 shrink가 다른 차원의 아직 완료되지 않은 streak를 제거하지 않는다.
- 기존 `consecutive_high_pressure_passes`는 호환용 aggregate로 유지한다.
- 기존 state 생성 방식, OOM 우선순위, dimension-aware field mapping, summary/history 계약과 stable namespace를 보존한다.

기대 최종 결과는 **per-dimension sustained-pressure streak tracking**이 구현되고 테스트·문서가 일치하며, PR을 자동 병합하지 않은 상태이다.

## 기준 상태

- 최신 `main`에서 새 feature branch를 만든다.
- 기준에는 PR #820 병합 결과가 포함되어 있어야 한다.
- PR #820 merge commit 참고값: `1d65f0df287f7dbb3579ebe1cffd8ccc94e71d72`
- 실제 작업 전에 반드시 다음을 확인한다.

```bash
git switch main
git pull --ff-only
git rev-parse HEAD
git status --short --branch
git switch -c codex/per-dimension-pressure-streaks
```

`main`에 직접 커밋하지 않는다.

## 제공 패치

전달된 패치:

```text
apply-patch-0026_per_dimension_pressure_streaks.diff
```

적용 전:

```bash
git apply --check apply-patch-0026_per_dimension_pressure_streaks.diff
git apply apply-patch-0026_per_dimension_pressure_streaks.diff
```

현재 main과 작은 문맥 차이로 패치가 적용되지 않으면 요청 계약을 유지하는 최소 수정으로 반영한다. 패치와 무관한 리팩터링을 하지 않는다.

적용 후 전달용 patch와 prompt 파일이 repository 안에 들어왔다면 커밋 전에 삭제한다.

## 작업 내용

### 1. State와 decision에 차원별 streak 추가

수정 대상:

```text
enn_torch_dev/runtime/governor.py
```

기존 필드 순서를 깨지 않도록 각 dataclass의 끝에 다음 필드를 append한다.

```python
consecutive_cpu_pressure_passes: int = 0
consecutive_cuda_pressure_passes: int = 0
```

적용 대상:

- `GovernorDecision`
- `RuntimeGovernorState`

두 state streak는 non-negative non-bool integer로 검증한다.

기존 필드:

```python
consecutive_high_pressure_passes: int = 0
```

는 삭제하거나 의미를 바꾸지 않는다. 새 decision/state를 생성할 때 다음 aggregate로 기록한다.

```python
max(
    consecutive_cpu_pressure_passes,
    consecutive_cuda_pressure_passes,
)
```

### 2. CPU와 CUDA streak 독립 추적

all-success이고 `recovered_oom=False`인 pass에서 `min_pressure_ratio_for_shrink`가 설정된 경우:

- CPU high 여부는 `peak_cpu_rss_ratio >= threshold`로 판단한다.
- CUDA high 여부는 다음 중 하나라도 threshold 이상인지로 판단한다.
  - `peak_cuda_allocated_ratio`
  - `peak_cuda_reserved_ratio`
  - `peak_cuda_max_allocated_ratio`
  - `peak_cuda_max_reserved_ratio`
- CPU high이면 CPU streak만 증가한다.
- CUDA high이면 CUDA streak만 증가한다.
- 한 차원이 low 또는 unknown이면 그 차원의 streak만 0으로 reset한다.
- 다른 차원의 high streak는 해당 차원의 현재 관측에 따라 독립적으로 증가하거나 유지된다.
- 두 차원이 모두 unknown인 summary와 `pressure_summary=None`은 두 streak를 모두 reset한다.

pressure가 하나라도 high인 pass는 기존처럼 success-growth를 suppress한다.

### 3. 차원별 threshold와 shrink

각 차원 streak가 `shrink_after_pressure_passes`에 도달했는지 독립적으로 판단한다.

- CPU만 도달하면 CPU-triggered shrink만 수행한다.
- CUDA만 도달하면 CUDA-triggered shrink만 수행한다.
- 둘 다 도달하면 두 차원을 함께 shrink한다.
- threshold에 도달한 차원의 streak만 0으로 reset한다.
- 아직 도달하지 않은 다른 차원의 streak는 유지한다.

budget field 선택은 PR #820 계약을 그대로 유지한다.

- CPU trigger → `max_host_bytes`
- CUDA trigger → `max_device_bytes`
- trigger된 차원 중 matching byte budget이 하나도 없을 때만 `max_items` fallback
- matching byte budget이 하나라도 있으면 `max_items`는 유지
- `pressure_shrunk_budget_fields`에는 실제 값이 바뀐 field만 기록
- minimum-bound no-op은 `budget_shrunk_by_pressure=False`, 빈 tuple

중요: budget field 선택에는 **현재 high인 모든 차원**이 아니라 **이번 pass에서 threshold에 도달한 차원만** 전달한다.

### 4. Reset 및 우선순위

다음 경우 두 dimension streak를 모두 0으로 reset한다.

- yielded OOM
- retry-recovered OOM
- empty result pass
- non-OOM fault
- all-success이지만 pressure summary가 완전히 unavailable인 경우

OOM/recovered-OOM은 기존 우선순위를 유지한다.

- 모든 configured budget field를 기존 generic shrink 경로로 한 번만 축소
- `pressure_shrunk_budget_fields == ()`
- `budget_shrunk_by_pressure is False`
- CPU/CUDA/global pressure streak 모두 0

### 5. Legacy global streak 호환

다음처럼 기존 aggregate만 설정한 state 구성을 계속 지원한다.

```python
RuntimeGovernorState(
    current_budget=...,
    consecutive_high_pressure_passes=1,
)
```

다음 조건을 모두 만족할 때만 legacy aggregate를 승계한다.

- `consecutive_high_pressure_passes > 0`
- `consecutive_cpu_pressure_passes == 0`
- `consecutive_cuda_pressure_passes == 0`
- 현재 all-success pressure observation에서 해당 차원이 high

현재 CPU만 high이면 CPU streak가 legacy 값에서 이어지고, CUDA만 high이면 CUDA streak가 이어진다. 둘 다 high이면 둘 다 legacy 값을 이어받는다.

새 observation 후에는 명시적인 CPU/CUDA streak를 기록하고 global aggregate를 다시 계산한다.

명시적인 dimension streak가 하나라도 존재하면 legacy aggregate를 추가로 합산하지 않는다.

### 6. Summary 전달

수정 대상:

```text
enn_torch_dev/runtime/summary.py
```

`RuntimePassSummary` 끝에 다음을 append한다.

```python
consecutive_cpu_pressure_passes: int = 0
consecutive_cuda_pressure_passes: int = 0
```

`summarize_runtime_pass(...)`에서 decision 값을 복사하고 formatter에 다음 scalar를 출력한다.

```text
consecutive_cpu_pressure_passes=...
consecutive_cuda_pressure_passes=...
```

기존 aggregate와 `pressure_shrunk_budget_fields` 출력은 유지한다.

### 7. History 범위

`enn_torch_dev/runtime/history.py`와 `RuntimeHistorySummary`는 변경하지 않는다.

- 기존 `pressure_shrink_passes` 집계를 그대로 유지한다.
- CPU/CUDA별 history 누적 필드를 추가하지 않는다.
- history가 보관하는 latest summary를 통해 최신 streak를 볼 수 있는 기존 구조만 유지한다.

### 8. 테스트

수정 대상:

```text
enn_torch_dev/debug/runtime/test_runtime_governor.py
enn_torch_dev/debug/runtime/test_runtime_summary.py
enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

필수 검증:

- CPU high → CUDA high 교차 pass에서 shrink 없음
- CUDA high → CPU high 교차 pass에서 shrink 없음
- CPU-only 연속 high가 CPU threshold에 도달하면 host budget 축소
- CUDA-only 연속 high가 CUDA threshold에 도달하면 device budget 축소
- CPU/CUDA가 함께 연속 high이면 두 byte budget 축소
- CPU만 threshold에 도달하면 CPU streak만 reset
- threshold 미도달 CUDA streak은 유지
- 한 차원이 low 또는 unknown이면 해당 streak만 reset
- 다른 high 차원의 streak은 계속 증가
- fully unavailable pressure는 두 streak reset
- empty/non-OOM fault도 두 streak reset
- yielded/recovered OOM은 두 streak reset 및 기존 전체-field shrink
- legacy global-only state가 현재 high 차원에 승계됨
- 명시적 dimension streak가 있으면 legacy aggregate를 중복 적용하지 않음
- aggregate는 항상 `max(cpu, cuda)`
- 신규 dataclass 필드는 기존 필드 뒤에 append
- invalid negative/bool dimension streak 거부
- summary가 CPU/CUDA streak를 복사하고 formatter에 출력
- integration session에서 pass별 CPU/CUDA/global streak 전달
- `pressure_shrunk_budget_fields`와 history count 기존 계약 유지

### 9. 문서

수정 대상:

```text
docs/dev_runtime_governor.md
docs/dev_runtime_summary.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

문서에 다음을 명시한다.

- CPU와 CUDA streak가 독립적임
- alternating dimensions가 하나의 sustained streak로 합쳐지지 않음
- low/unknown은 해당 차원만 reset
- fully unavailable/empty/fault/OOM은 둘 다 reset
- 한 차원 trigger가 다른 미완료 streak를 지우지 않음
- global streak는 max aggregate 호환 필드
- legacy global-only state 승계 조건
- per-dimension threshold/factor/learned weighting은 여전히 범위 밖

## 유지해야 하는 기존 동작

- OOM/recovered-OOM 우선순위와 전체 configured-field shrink
- growth pressure guard
- success-growth streak와 grow behavior
- dimension-aware pressure-to-budget mapping
- `max_items` fallback 계약
- minimum/maximum bounds
- actual-changed-field reporting
- summary/history retained-window 계약
- provider 및 orchestration 경계
- stable `enn_torch` namespace 비노출

## 변경 금지 범위

다음은 수정하지 않는다.

```text
enn_torch/**
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/capacity_provider.py
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/__init__.py
pyproject.toml
requirements*.txt
```

추가 금지 사항:

- 새 의존성 추가
- stable API 노출
- public export 변경
- checkpoint/파일 형식 변경
- unrelated refactor
- persistent logging/export
- 자동 capacity refresh
- free-memory admission control
- per-dimension threshold 또는 shrink factor 추가
- 자동 병합

## 테스트 방법

### Targeted

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

### Runtime 회귀

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
python -m pytest enn_torch_dev/debug/runtime -q
```

### 전체 회귀

```bash
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

### 정적 및 범위 검사

```bash
git diff --check
git status --short
git diff --name-only main...HEAD
git diff -- enn_torch
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/resources.py
git diff -- enn_torch_dev/runtime/capacity_provider.py
git diff -- enn_torch_dev/runtime/orchestration.py
git diff -- enn_torch_dev/runtime/history.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/__init__.py
git show --check --stat --oneline HEAD
```

## 예상 결과

- 교차 CPU/CUDA high pass가 threshold에 도달하지 않는다.
- 각 차원은 자체 streak로만 shrink를 trigger한다.
- trigger되지 않은 다른 차원의 incomplete streak가 유지된다.
- global compatibility streak는 새 decision/state에서 항상 CPU/CUDA streak의 max다.
- legacy global-only state가 안전하게 현재 high 차원으로 승계된다.
- OOM과 기존 field mapping/history 계약은 회귀하지 않는다.
- stable namespace와 금지 경로는 변경되지 않는다.

## Git 작업 및 PR

테스트가 통과하면:

```bash
git add \
  enn_torch_dev/runtime/governor.py \
  enn_torch_dev/runtime/summary.py \
  enn_torch_dev/debug/runtime/test_runtime_governor.py \
  enn_torch_dev/debug/runtime/test_runtime_summary.py \
  enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py \
  enn_torch_dev/debug/runtime/test_runtime_integration.py \
  docs/dev_runtime_governor.md \
  docs/dev_runtime_summary.md \
  docs/runtime_development_workflow.md \
  docs/CURRENT_STATE.md \
  docs/RUNTIME_SAFETY.md \
  docs/CHANGE_CHECKLIST.md

git commit -m "Track CPU and CUDA pressure streaks independently"
git push -u origin codex/per-dimension-pressure-streaks
```

PR 제목 권장값:

```text
Track sustained pressure independently by resource dimension
```

PR 본문에는 다음을 포함한다.

- Motivation
- Description
- CPU/CUDA independent streak contract
- legacy aggregate compatibility
- reset and OOM priority contract
- 실제 실행한 테스트 명령과 정확한 passed/skipped/warning 수
- 정적·금지 경로 검사 결과
- 실제 GitHub PR head SHA
- 다음 형식의 AI docs 목록

```text
AI docs updated:
- docs/CHANGE_CHECKLIST.md
- docs/CURRENT_STATE.md
- docs/RUNTIME_SAFETY.md
- docs/dev_runtime_governor.md
- docs/dev_runtime_summary.md
- docs/runtime_development_workflow.md
```

PR 생성 후 자동 merge하지 않는다. 검토와 명시적 병합 요청을 기다린다.

## 최종 보고

다음을 정확히 보고한다.

- 작업 branch
- 수정 파일 목록
- 구현한 streak·reset·legacy 계약
- 실행한 각 테스트 명령과 결과
- warning/skipped 수
- 실행하지 못한 테스트와 이유
- 정적/범위 검사 결과
- local commit SHA
- 실제 GitHub PR head SHA
- PR URL
- 병합하지 않았다는 사실
