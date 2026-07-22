# Codex 작업 지침 — #817 Runtime summary/history pressure visibility

## 작업 목적

`ConservativeRuntimeGovernor`와 `ConservativeRuntimeOrchestrator`는 현재 다음 pressure feedback 정보를 생성하고 전달한다.

- `GovernorDecision.pressure_summary`
- `GovernorDecision.growth_suppressed_by_pressure`

그러나 `summarize_runtime_pass(...)`는 이 두 필드를 `RuntimePassSummary`로 복사하지 않으며, `RuntimePassHistory`도 pressure assessment 여부와 성장 억제 결과를 집계하지 않는다.

따라서 orchestration/session 실행 후에는 `GovernorDecision`을 직접 확인하지 않는 한 다음 정보를 알기 어렵다.

- 해당 pass에서 pressure assessment가 수행됐는지
- 알려진 최대 pressure ratio가 얼마인지
- pressure 때문에 success-driven growth가 억제됐는지
- retained history window에서 assessment·suppression이 몇 번 있었는지
- retained history window의 최대 pressure ratio가 얼마인지

이번 작업은 pressure feedback을 기존 lightweight summary/history inspection 계층에 노출한다.

기대 결과:

```text
GovernorDecision pressure feedback
  -> RuntimePassSummary
  -> RuntimeHistorySummary retained-window aggregation
  -> stable debug formatter text
```

이 작업은 governor 정책, pressure 계산, orchestration 실행, session 동작을 변경하지 않는다.

---

## Git 작업 절차

1. 최신 `main`을 기준으로 전용 branch를 만든다.

```bash
git switch main
git pull --ff-only
git switch -c codex/expose-pressure-summary-history
```

2. `main`에 직접 커밋하지 않는다.
3. 아래 외부 패치 파일을 branch에 적용한다.

```bash
git apply /mnt/data/apply-patch-0022_pressure_summary_history_visibility.diff
```

4. 테스트와 정적 범위 검사를 실행한다.
5. 변경을 전용 branch에만 commit/push한다.
6. `main` 대상 PR을 생성한다.
7. 사용자 검토 전 자동 병합하지 않는다.

중요:

- `/mnt/data/apply-patch-0022_pressure_summary_history_visibility.diff`
- `/mnt/data/codex_prompt_0022_pressure_summary_history_visibility.md`

위 두 파일은 작업 전달용 외부 산출물이다. 저장소 루트나 tracked 경로에 복사하거나 commit하지 않는다.

`git status --short`에서 전달용 `.diff`, 작업 프롬프트, `__pycache__`, `*.pyc`, 임시 테스트 파일이 나타나면 commit 전에 제거한다.

remote 또는 push destination이 없다면 GitHub/PR 도구를 사용해 같은 feature branch와 PR에 변경을 반영한다. 이 경우에도 `main`을 직접 수정하지 않는다. 로컬 commit SHA와 실제 GitHub PR head SHA가 다르면 최종 보고에서 구분한다.

---

## 작업 내용

### 1. 수정 대상

구현:

```text
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
```

테스트:

```text
enn_torch_dev/debug/runtime/test_runtime_summary.py
enn_torch_dev/debug/runtime/test_runtime_history.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

AI-facing 문서:

```text
docs/dev_runtime_summary.md
docs/dev_runtime_history.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/CHANGE_CHECKLIST.md
```

### 2. `RuntimePassSummary` pressure feedback 필드

기존 positional field 순서를 보존하기 위해 현재 필드 마지막에 다음을 추가한다.

```python
pressure_summary: ResourcePressureSummary | None = None
growth_suppressed_by_pressure: bool = False
```

계약:

- 기존 `RuntimePassSummary(...)` positional/keyword 생성은 계속 동작해야 한다.
- `summarize_runtime_pass(...)`는 `GovernorDecision`의 두 필드를 그대로 복사한다.
- `pressure_summary=None`은 assessment가 수행되지 않았음을 의미한다.
- `ResourcePressureSummary()`는 assessment는 수행됐지만 알려진 ratio가 없음을 의미한다.
- `ResourcePressureSummary`는 scalar-only frozen record이므로 summary가 raw `ResourceSample`을 보존하지 않는다.
- `StepResult`, `store`, `loss`, raw `ResourceSample` reference를 새로 보존하지 않는다.

### 3. Pass formatter pressure 출력

`format_runtime_pass_summary(...)`에 다음 안정적인 debug 필드를 추가한다.

```text
pressure_assessed=<bool>
max_pressure_ratio=<ratio 또는 unknown>
growth_suppressed_by_pressure=<bool>
```

계약:

- `pressure_summary is None`
  - `pressure_assessed=False`
  - `max_pressure_ratio=unknown`
- `pressure_summary`는 있지만 모든 ratio가 `None`
  - `pressure_assessed=True`
  - `max_pressure_ratio=unknown`
- 알려진 ratio가 있으면 `ResourcePressureSummary.max_observed_ratio` 사용
- ratio 출력은 `f"{value:.6g}"` 형식
- formatter는 machine interchange format이 아니라 기존과 같은 stable debug text이다.

### 4. `RuntimeHistorySummary` pressure aggregation

기존 positional field 순서를 보존하기 위해 현재 필드 마지막에 다음을 추가한다.

```python
pressure_assessed_passes: int = 0
pressure_growth_suppressed_passes: int = 0
peak_observed_pressure_ratio: float | None = None
```

`RuntimePassHistory.summarize()`는 현재 retained records만 대상으로 다음을 집계한다.

#### `pressure_assessed_passes`

```python
summary.pressure_summary is not None
```

인 retained pass 수.

`ResourcePressureSummary()`처럼 ratio가 전부 unknown이어도 assessment가 수행됐으므로 count에 포함한다.

#### `pressure_growth_suppressed_passes`

```python
summary.growth_suppressed_by_pressure is True
```

인 retained pass 수.

#### `peak_observed_pressure_ratio`

각 retained `RuntimePassSummary.pressure_summary.max_observed_ratio` 중 알려진 값의 최댓값.

- pressure summary가 `None`인 pass는 제외
- all-unknown pressure summary는 제외
- 알려진 ratio가 하나도 없으면 `None`
- ratio를 `1.0`으로 clamp하지 않는다
- oldest record가 trim되면 해당 record의 pressure count와 peak도 집계에서 제외돼야 한다.

### 5. History formatter pressure 출력

`format_runtime_history_summary(...)`에 다음 debug 필드를 추가한다.

```text
pressure_assessed_passes=<int>
pressure_growth_suppressed_passes=<int>
peak_observed_pressure_ratio=<ratio 또는 unknown>
latest_pressure_assessed=<bool>
latest_max_pressure_ratio=<ratio 또는 unknown>
latest_growth_suppressed_by_pressure=<bool>
```

계약:

- empty history는 count `0`, ratio `unknown`, latest bool `False`
- latest summary가 all-unknown pressure를 가진 경우:
  - `latest_pressure_assessed=True`
  - `latest_max_pressure_ratio=unknown`
- ratio 출력은 pass formatter와 동일하게 `f"{value:.6g}"`

### 6. Reference safety

Summary/history는 다음 reference를 새로 보존하면 안 된다.

```text
StepResult
StepResult.loss
StepResult.store
ResourceSample
```

허용되는 것은 scalar-only `ResourcePressureSummary`와 기존 lightweight summary 값뿐이다.

`RuntimePassHistory` 집계는 현재 `max_records` retained window에 대해서만 수행하고, 별도의 unbounded pressure history를 만들지 않는다.

### 7. 테스트

#### `test_runtime_summary.py`

최소한 다음을 검증한다.

1. 신규 pass-summary 필드가 기존 필드 뒤에 추가됐는지
2. known CPU/CUDA pressure summary 복사
3. `growth_suppressed_by_pressure` 복사
4. all-unknown summary 보존
5. unassessed `None`과 assessed-but-unknown summary 구분
6. formatter의 assessment/ratio/suppression 출력
7. summary가 raw `ResourceSample`, `StepResult`, `loss`, `store`를 보존하지 않음
8. 기존 summary 계산·오류 검증 회귀 유지

#### `test_runtime_history.py`

최소한 다음을 검증한다.

1. 신규 history-summary 필드가 기존 필드 뒤에 추가됐는지
2. empty history 기본값
3. pressure-assessed pass count
4. growth-suppressed pass count
5. 여러 pass 중 알려진 최대 ratio
6. all-unknown summary가 assessed count에는 포함되지만 peak를 오염하지 않음
7. record trim 후 제거된 old high-pressure pass가 count/peak에서 제외됨
8. formatter의 retained-window pressure 출력
9. latest pressure 상태 출력
10. history가 raw `ResourceSample`, `StepResult`, `loss`, `store`를 보존하지 않음
11. 기존 bounded retention·status/OOM/budget 집계 회귀 유지

#### `test_runtime_integration.py`

기존 pressure-aware session end-to-end 테스트를 확장해 다음을 검증한다.

```text
pass 1: pressure ratio 0.5 -> growth 허용
pass 2: pressure ratio 0.9 -> growth 억제
```

각 pass에서:

- `pass_summary.pressure_summary`
- `pass_summary.growth_suppressed_by_pressure`

를 확인한다.

각 history summary에서:

- `pressure_assessed_passes`
- `pressure_growth_suppressed_passes`
- `peak_observed_pressure_ratio`

를 확인한다.

기존 budget, session, history records 동작은 유지한다.

---

## 변경 금지 범위

다음은 수정하지 않는다.

```text
enn_torch/**
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/source_factory.py
enn_torch_dev/runtime/__init__.py
pyproject.toml
requirements*.txt
lockfiles
```

구현하지 않을 것:

```text
pressure 기반 budget shrink
governor threshold 또는 decision 규칙 변경
capacity 자동 조회·refresh
ResourceMonitor 자동 생성
새 persistence 또는 telemetry backend
JSONL/CSV/dashboard export
unbounded history
stable enn_torch API 변경
```

---

## 테스트 방법

### 필수 targeted 테스트

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
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
git diff -- enn_torch_dev/runtime/orchestration.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/source_factory.py
git diff -- enn_torch_dev/runtime/__init__.py
git diff -- pyproject.toml requirements.txt requirements-dev.txt
git status --short
```

`git status --short`에서 다음이 남으면 안 된다.

```text
apply-patch-0022_pressure_summary_history_visibility.diff
codex_prompt_0022_pressure_summary_history_visibility.md
__pycache__/
*.pyc
임시 테스트 파일
```

---

## 예상 결과

- 기존 pass/history dataclass positional field 순서 유지
- pressure assessment와 growth suppression이 pass summary에 노출
- retained history window 기준 pressure count/peak 집계
- all-unknown과 unassessed 상태 구분
- trim된 record가 pressure 집계에서 제외
- raw runtime object reference retention 없음
- governor·pressure·orchestration·session 동작 변화 없음
- stable `enn_torch`, dependency, lockfile 변경 없음

---

## PR 처리

- feature branch를 사용한다.
- `main`에 직접 커밋하지 않는다.
- 테스트가 통과한 뒤 branch에 commit/push한다.
- `main` 대상 PR을 생성하거나 기존 작업 PR을 갱신한다.
- PR 본문에 각 테스트 명령의 정확한 passed/skipped/warning 결과를 기록한다.
- 실제 GitHub PR head SHA를 확인해 보고한다.
- 사용자 검토 전에 자동 병합하지 않는다.

권장 PR 제목:

```text
Expose pressure feedback in runtime summaries
```

---

## 최종 보고 형식

다음을 구분해 보고한다.

1. 변경 파일
2. pass summary pressure 복사 계약
3. retained-window history 집계 방식
4. formatter 출력 계약
5. positional 호환 확인
6. reference retention 경계
7. 실제 실행한 테스트 명령과 정확한 결과
8. 실행하지 못한 테스트와 이유
9. 정적 범위 검사 결과
10. PR URL과 실제 GitHub head SHA

```text
AI docs updated:
- docs/dev_runtime_summary.md
- docs/dev_runtime_history.md
- docs/runtime_development_workflow.md
- docs/CURRENT_STATE.md
- docs/CHANGE_CHECKLIST.md
```
