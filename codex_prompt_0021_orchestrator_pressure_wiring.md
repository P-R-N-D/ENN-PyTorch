# Codex 작업 지침 — #816 Orchestrator pressure summary wiring

## 작업 목적

현재 `ConservativeRuntimeGovernor`는 opt-in pressure growth guard와
`pressure_summary` 입력을 지원한다. 그러나
`ConservativeRuntimeOrchestrator.run_pass(...)`는 아직 pass 결과와
`recovered_oom` 신호만 governor에 전달하므로, orchestrator/session 경로에서는
pressure guard에 사용할 실제 summary가 자동으로 만들어지지 않는다.

이번 작업은 호출자가 명시적으로 제공한 고정 `ResourceCapacity`를 사용해,
한 번의 finite pass에서 발생한 모든 raw runtime attempt의
`ResourceSample`을 `ResourcePressureSummary`로 축약하고 governor에 전달하는
최소 연결을 추가한다.

기대 결과는 다음과 같다.

```text
모든 raw attempt의 ResourceSample
+ 명시적으로 주입된 고정 ResourceCapacity
-> assess_resource_pressure(...)
-> governor.observe_results(..., pressure_summary=summary)
```

여기서 raw attempt에는 최종 `RuntimePassResult.results`에 남는 결과뿐 아니라,
`RuntimeRetryRunner`가 내부에서 소비한 retry 이전 OOM 결과도 포함해야 한다.

이번 작업은 pressure 기반 budget 축소나 capacity 자동 조회를 구현하지 않는다.

## Git 작업 절차

1. 최신 `main`을 기준으로 전용 branch를 만든다.

```bash
git switch main
git pull --ff-only
git switch -c codex/wire-orchestrator-pressure-summary
```

2. `main`에 직접 커밋하지 않는다.
3. 아래 외부 패치 파일을 branch에 적용한다.

```bash
git apply /mnt/data/apply-patch-0021_orchestrator_pressure_wiring.diff
```

4. 테스트와 정적 경계 검사를 실행한다.
5. 변경을 전용 branch에만 commit/push한다.
6. `main` 대상 PR을 생성한다.
7. 자동 병합하지 않는다.

중요:

- `/mnt/data/apply-patch-0021_orchestrator_pressure_wiring.diff`와
  `/mnt/data/codex_prompt_0021_orchestrator_pressure_wiring.md`는 작업 전달용
  외부 산출물이다.
- 이 두 파일을 저장소 루트나 다른 tracked 경로에 복사하거나 commit하지 않는다.
- `git status --short`에서 전달용 `.diff`, 작업 프롬프트, `__pycache__`, 임시 파일이
  나타나면 commit 전에 제거한다.
- remote 또는 push destination이 없으면 GitHub/PR 도구를 사용해 동일 branch에
  반영하되, `main`을 직접 수정하지 않는다.
- 로컬 commit SHA와 GitHub PR head SHA가 다르면 최종 보고에서 구분한다.

## 작업 내용

### 1. 수정 대상

구현과 테스트:

```text
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/debug/runtime/test_runtime_orchestration.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

AI-facing 문서:

```text
docs/dev_runtime_orchestration.md
docs/dev_runtime_governor.md
docs/dev_runtime_pressure.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

### 2. Orchestrator 생성자에 고정 capacity 추가

`ConservativeRuntimeOrchestrator.__init__(...)`의 keyword-only 인자로 다음을
추가한다.

```python
resource_capacity: ResourceCapacity | None = None
```

계약:

- 기본값 `None`은 기존 orchestration 동작을 유지한다.
- `None`이 아닌 값은 `ResourceCapacity`여야 한다.
- 잘못된 타입은 생성자에서 `TypeError`로 거부한다.
- capacity는 orchestrator 인스턴스에 고정된다.
- orchestrator가 `ResourceMonitor`를 만들거나 capacity를 자동 조회·갱신하지 않는다.
- 기존 positional 호출에는 영향이 없어야 한다.

### 3. 모든 raw attempt의 resource sample 수집

기존 `_OomTrackingRuntimeStep`은 retry layer가 보기 전의 raw `StepResult`에서
OOM 발생 여부만 기록한다. 같은 경계에서 각 raw `StepResult.resource_samples`도
pass-local 목록에 순서대로 추가한다.

```python
self.resource_samples: list[ResourceSample] = []
```

`run(...)`에서 반환값이 `StepResult`이면:

1. `result.resource_samples`를 목록에 추가한다.
2. `result.status is StepStatus.OOM_FAULT`이면 기존처럼 `saw_oom=True`를 기록한다.
3. 원래 `StepResult`를 변경 없이 반환한다.

반드시 포함해야 하는 범위:

- 최종 yielded success/fault 결과의 samples
- retry 이전에 소비된 OOM 결과의 samples
- 여러 split/retry attempt의 samples

금지 사항:

- sample을 `RuntimePassResult`의 새 필드로 그대로 보존하지 않는다.
- `StepResult`를 복사하거나 수정하지 않는다.
- retry 순서, split 결과, row identity/order를 변경하지 않는다.

### 4. Pass 종료 후 pressure summary 계산

`RuntimeRetryRunner`의 finite 결과를 기존처럼 tuple로 만든 뒤, OOM 및
recovered OOM 판정을 유지한다.

`resource_capacity`가 설정된 경우에만 다음을 수행한다.

```python
pressure_summary = assess_resource_pressure(
    tracking_step.resource_samples,
    self.resource_capacity,
)
```

설정되지 않은 경우:

```python
pressure_summary = None
```

그 후 governor를 다음처럼 호출한다.

```python
decision = self.governor.observe_results(
    results,
    recovered_oom=recovered_oom,
    pressure_summary=pressure_summary,
)
```

계약:

- capacity가 설정됐지만 sample이 없으면
  `ResourcePressureSummary()`가 생성된다.
- guard가 활성화된 all-success 경로에서 all-unknown summary는 기존 governor
  계약대로 growth를 억제한다.
- guard가 비활성화돼도 계산된 summary는 `GovernorDecision`에 기록된다.
- pressure는 budget을 직접 shrink하지 않는다.
- yielded OOM 또는 recovered OOM shrink가 pressure보다 우선한다.
- CUDA sample과 capacity device index가 다르면
  `assess_resource_pressure(...)`의 `ValueError`를 그대로 전파한다.
- pressure assessment가 실패한 경우 governor의 state/last decision을 갱신하지
  않는다.

### 5. Retention 및 실행 경계

- raw sample 목록은 `run_pass(...)` 내부 tracker가 보유하는 pass-local 데이터다.
- governor에는 scalar-only `ResourcePressureSummary`만 전달한다.
- `RuntimePassResult.results`의 기존 finite tuple 경계는 유지한다.
- session/history에 raw attempt samples나 `StepResult.store`, `loss`를 새로
  보존하지 않는다.
- Python exception을 숨기거나 fault record로 변환하지 않는다.

### 6. 테스트 보완

`test_runtime_orchestration.py`에서 최소한 다음을 검증한다.

1. capacity 미설정 시 `decision.pressure_summary is None`이고 기존 growth 동작 유지
2. 낮은 CPU pressure가 threshold 미만이면 success growth 허용
3. 높은 CPU pressure가 threshold 이상이면 growth 억제
4. capacity가 있지만 samples가 없으면 all-unknown summary 전달 및 growth 억제
5. guard가 비활성화돼도 summary가 decision에 기록되고 기존 growth 유지
6. CUDA allocated/reserved pressure 전달
7. CUDA device mismatch 예외 전파 및 governor state 미갱신
8. retry 이전 OOM attempt의 높은 sample이 최종 summary peak에 포함됨
9. recovered OOM은 pressure와 무관하게 기존 shrink 우선순위 유지
10. 잘못된 `resource_capacity` 타입 거부
11. 기존 split, result order, row identity, empty source 동작 유지

`test_runtime_integration.py`에는 session을 통한 최소 end-to-end 검증을 추가한다.

예시 흐름:

```text
pass 1: CPU ratio 0.5, threshold 0.8 -> budget growth
pass 2: CPU ratio 0.9, threshold 0.8 -> growth suppression, budget 유지
```

검증 항목:

- 각 pass decision에 summary 기록
- 첫 pass budget growth
- 두 번째 pass budget 유지 및 suppression flag
- 두 번째 pass success streak 초기화
- history/session 기존 동작 유지

### 7. 문서 수정

문서에서 다음을 명확하게 구분한다.

- pressure summary는 호출자가 주입한 고정 `ResourceCapacity`가 있을 때만
  orchestrator가 계산한다.
- governor 자체는 summary를 만들지 않는다.
- retry에 의해 최종 결과에서 사라진 raw OOM attempt의 samples도 포함한다.
- capacity는 orchestrator 인스턴스에서 고정되며 pass마다 자동 refresh하지 않는다.
- `ResourceMonitor` 자동 생성 및 capacity discovery는 아직 구현하지 않는다.
- pressure는 success-driven growth만 억제할 수 있고 직접 shrink하지 않는다.
- CUDA device mismatch는 숨기지 않는다.
- stable `enn_torch` namespace는 변경하지 않는다.

## 변경 금지 범위

다음 파일과 영역은 수정하지 않는다.

```text
enn_torch/**
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/__init__.py
pyproject.toml
requirements*.txt
lockfiles
```

구현하지 않을 것:

- pressure 기반 자동 budget shrink
- CPU/CUDA별 field-specific budget 조정
- orchestrator 내부 `ResourceMonitor` 생성
- capacity provider/callback
- pass별 capacity 자동 재조회
- dynamic/learned threshold
- persistent telemetry
- checkpoint/resume
- distributed aggregation
- stable `enn_torch` API 변경

## 테스트 방법

### 필수 targeted 테스트

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

### 회귀 테스트

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

### 정적·범위 검사

```bash
git diff --check
git diff -- enn_torch
git diff -- enn_torch_dev/runtime/governor.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/resources.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/summary.py
git diff -- enn_torch_dev/runtime/history.py
git diff -- enn_torch_dev/runtime/__init__.py
git diff -- pyproject.toml requirements.txt requirements-dev.txt
git status --short
```

`git status --short`에서 다음이 tracked/untracked 변경으로 남으면 안 된다.

```text
apply-patch-0021_orchestrator_pressure_wiring.diff
codex_prompt_0021_orchestrator_pressure_wiring.md
__pycache__/
*.pyc
임시 테스트 파일
```

## 예상 결과

- 기존 capacity 미설정 orchestrator 호출은 동작 변화 없음
- fixed capacity가 있으면 모든 raw attempt sample을 summary로 축약
- retry-consumed OOM sample peak 보존
- low pressure에서 기존 success growth 허용
- high/unknown pressure에서 opt-in growth 억제
- OOM/recovered OOM shrink 우선순위 유지
- CUDA mismatch 예외 전파 및 governor state 미갱신
- session/history/summary/identity/order 회귀 없음
- stable `enn_torch`, dependency, lockfile 변경 없음

## 최종 보고 형식

다음을 구분해 보고한다.

1. 변경 파일
2. fixed capacity 주입 계약
3. raw retry attempt sample 수집 방식
4. pressure assessment 및 governor 전달 순서
5. OOM/fault 우선순위 유지 확인
6. CUDA mismatch 처리
7. 실제 실행한 테스트 명령과 결과
8. 실행하지 못한 테스트와 이유
9. `git status --short` 결과 및 전달용 산출물 미추적 확인
10. PR URL과 실제 GitHub head SHA

```text
AI docs updated:
- docs/dev_runtime_orchestration.md
- docs/dev_runtime_governor.md
- docs/dev_runtime_pressure.md
- docs/runtime_development_workflow.md
- docs/CURRENT_STATE.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
