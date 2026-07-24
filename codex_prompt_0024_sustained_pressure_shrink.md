# Codex 작업 지침 — #819 Sustained resource pressure budget shrink

## 작업 목적

현재 `ConservativeRuntimeGovernor`는 OOM 또는 retry-recovered OOM을 관찰하면 즉시 budget을 축소하고, 성공 pass의 pressure가 높거나 unknown이면 success-driven growth만 억제한다. 그러나 OOM이 발생하지 않은 상태에서 여러 pass 동안 높은 pressure가 반복되어도 budget은 유지된다.

이번 작업은 **한 번의 pressure spike가 아니라 연속된 all-success high-pressure pass**만을 보수적인 다음-pass budget shrink 신호로 사용한다.

기대 결과:

```text
OOM/recovered OOM
  > sustained-pressure shrink
  > pressure growth suppression
  > normal success growth
```

기능은 기본 비활성화여야 하며 기존 호출과 governor 동작을 보존해야 한다.

## 작업 내용

### 변경 대상

구현:

```text
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
```

테스트:

```text
enn_torch_dev/debug/runtime/test_runtime_governor.py
enn_torch_dev/debug/runtime/test_runtime_summary.py
enn_torch_dev/debug/runtime/test_runtime_history.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

문서:

```text
docs/dev_runtime_governor.md
docs/dev_runtime_summary.md
docs/dev_runtime_history.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

### 정책 필드

기존 `GovernorPolicy` 필드 뒤에 추가한다.

```python
min_pressure_ratio_for_shrink: float | None = None
shrink_after_pressure_passes: int = 2
```

계약:

- `min_pressure_ratio_for_shrink=None`이면 기존 동작을 완전히 유지한다.
- ratio는 `0 < value <= 1`이어야 한다.
- pass count는 bool이 아닌 양의 정수여야 한다.
- growth/shrink threshold가 모두 있으면 `max_pressure_ratio_for_growth <= min_pressure_ratio_for_shrink`여야 한다.

### 상태와 decision

기존 필드 뒤에 추가한다.

```python
RuntimeGovernorState.consecutive_high_pressure_passes: int = 0
GovernorDecision.consecutive_high_pressure_passes: int = 0
GovernorDecision.budget_shrunk_by_pressure: bool = False
```

`budget_shrunk_by_pressure`는 pressure shrink 정책이 실제로 `next_budget != previous_budget`을 만든 경우에만 `True`다. minimum bound 때문에 budget이 같으면 `False`다.

### decision 규칙

- OOM 또는 recovered OOM은 즉시 기존 shrink를 수행하고 high-pressure streak를 0으로 초기화한다.
- empty pass, non-OOM fault, low pressure, missing pressure, all-unknown pressure는 streak를 0으로 초기화한다.
- all-success이고 known max ratio가 shrink threshold 이상이면 streak를 1 증가시키고 growth를 억제한다.
- streak가 `shrink_after_pressure_passes`에 도달하면 기존 `_adjust_budget(..., mode="shrink")`를 재사용해 다음 budget을 축소하고 streak를 0으로 초기화한다.
- high-pressure streak 중에는 normal success growth를 허용하지 않는다.
- pressure shrink는 현재 pass 실행, retry, split을 바꾸지 않고 다음 pass budget에만 반영한다.
- OOM priority와 기존 min/max bound를 유지한다.

### summary/history

`RuntimePassSummary` 뒤에 다음을 추가하고 `GovernorDecision`에서 복사한다.

```python
consecutive_high_pressure_passes: int = 0
budget_shrunk_by_pressure: bool = False
```

pass formatter에 두 값을 출력한다.

`RuntimeHistorySummary` 뒤에 다음을 추가한다.

```python
pressure_shrink_passes: int = 0
```

현재 retained window 안에서 `budget_shrunk_by_pressure=True`인 pass만 집계하고 history formatter에 출력한다.

## 변경 금지 범위

```text
enn_torch/**
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/capacity_provider.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/__init__.py
pyproject.toml
requirements*.txt
lockfiles
```

새 의존성, stable API 변경, mid-pass budget 변경, free-memory admission control, field-specific shrink, persistence/telemetry를 추가하지 않는다.

## Git 작업 절차

1. 최신 `main`에서 feature branch를 만든다.
2. `main`에 직접 커밋하지 않는다.
3. 다음 패치를 적용한다.

```bash
git apply /mnt/data/apply-patch-0024_sustained_pressure_shrink.diff
```

4. 필요한 최소 조정만 수행한다.
5. 테스트 후 feature branch에 commit/push하고 `main` 대상 PR을 생성한다.
6. 사용자 검토 전 자동 병합하지 않는다.
7. 전달용 `.diff`와 prompt 파일을 저장소에 commit하지 않는다.

## 테스트 방법

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

필수 검증:

- 기능 미설정 시 기존 growth 동작 유지
- 첫 high-pressure success는 shrink하지 않고 streak 1
- 설정 횟수에 도달한 high-pressure success에서 items/host/device budget shrink
- low/unknown/missing/fault/empty가 streak 초기화
- OOM/recovered OOM 우선 및 streak 초기화
- high-pressure streak 중 growth 금지
- min/max bound 유지
- bound 때문에 실제 변화가 없으면 `budget_shrunk_by_pressure=False`
- provider 기반 multi-pass session에서 pressure shrink와 history count 전달
- 신규 dataclass 필드가 기존 필드 뒤에 추가됨
- stable `enn_torch`와 금지 경로에 변경 없음

정적 범위 검사:

```bash
git diff -- enn_torch
git diff -- enn_torch_dev/runtime/orchestration.py
git diff -- enn_torch_dev/runtime/capacity_provider.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/resources.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/__init__.py
git diff -- pyproject.toml requirements.txt requirements-dev.txt
git status --short
```

## 최종 보고

- 변경 파일과 정책 우선순위
- streak 증가/초기화 조건
- actual pressure shrink 판정 방식
- summary/history 전달 계약
- 실행한 각 테스트의 정확한 passed/skipped/warning 결과
- 실행하지 못한 검증과 이유
- 실제 GitHub PR head SHA
- AI docs 업데이트 목록

권장 PR 제목:

```text
Shrink budgets after sustained resource pressure
```
