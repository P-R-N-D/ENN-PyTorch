# Codex 작업 지침 — #815 Opt-in pressure guard for governor budget growth

## 작업 목적

현재 `ConservativeRuntimeGovernor`는 성공 관측 횟수만으로 설정된 budget을
성장시키며, resource pressure ratio는 정책 판단에 사용하지 않는다.

이번 작업은 `ResourcePressureSummary`를 governor가 명시적으로 전달받아,
opt-in 정책이 설정된 경우에만 성공 기반 budget 성장을 억제하는 보수적인
feedback 계약을 추가한다.

중요한 경계는 다음과 같다.

```text
높은 pressure 또는 unknown pressure -> budget 유지 + success streak 초기화
OOM 또는 recovered OOM              -> 기존처럼 budget 축소
낮은 pressure                       -> 기존 success streak/growth 허용
pressure 자체                       -> budget 자동 축소 금지
```

이번 작업에서는 orchestrator가 pressure summary를 자동 생성하거나 전달하지
않는다. 해당 연결은 별도 후속 PR로 분리한다.

## Git 작업 절차

1. 최신 `main`을 기준으로 전용 branch를 만든다.

```bash
git switch main
git pull --ff-only
git switch -c codex/add-pressure-growth-guard
```

2. `main`에 직접 커밋하지 않는다.
3. 아래 패치를 branch에 적용한다.

```bash
git apply /mnt/data/apply-patch-0020_pressure_governor_growth_guard.diff
```

4. 테스트와 정적 경계 검사를 실행한다.
5. 변경을 branch에만 commit/push한다.
6. `main` 대상 PR을 생성한다.
7. 자동 병합하지 않는다.

remote 또는 push destination이 없다면 GitHub/PR 도구를 사용해 같은 branch와
PR에 변경을 반영하되, main을 직접 수정하지 않는다. 로컬 SHA와 GitHub PR
head SHA가 다르면 최종 보고에 둘을 구분한다.

## 작업 내용

### 1. `ResourcePressureSummary.max_observed_ratio`

`enn_torch_dev/runtime/pressure.py`에 read-only property를 추가한다.

```python
@property
def max_observed_ratio(self) -> float | None:
    ...
```

계약:

- 알려진 CPU/CUDA ratio 중 최댓값 반환
- 모든 ratio가 `None`이면 `None`
- 기존 dataclass 필드와 positional 순서 변경 없음
- 새로운 ratio를 계산하거나 clamp하지 않음

### 2. `GovernorPolicy` opt-in threshold

기존 positional 순서를 보존하기 위해 마지막 필드로 추가한다.

```python
max_pressure_ratio_for_growth: float | None = None
```

검증 계약:

- `None`: 기존 governor 동작 완전 유지
- 설정값은 finite real number
- `bool` 거부
- `0 < value <= 1`
- 0, 음수, 1 초과, NaN, infinity 거부

### 3. `GovernorDecision` 관측 기록

기존 필드 뒤에 기본값이 있는 필드로 추가한다.

```python
pressure_summary: ResourcePressureSummary | None = None
growth_suppressed_by_pressure: bool = False
```

기존 `GovernorDecision(...)` 생성 호출은 수정 없이 계속 동작해야 한다.

### 4. `observe_results(...)` API

```python
def observe_results(
    self,
    results: Iterable[StepResult],
    *,
    recovered_oom: bool = False,
    pressure_summary: ResourcePressureSummary | None = None,
) -> GovernorDecision:
    ...
```

계약:

- `pressure_summary`는 `ResourcePressureSummary` 또는 `None`
- 잘못된 타입은 결과 iterable 소비 전에 `TypeError`
- 전달된 summary는 decision/state의 last decision에 기록
- `StepResult`, store, loss reference retention 계약은 유지

### 5. 결정 우선순위

기존 우선순위를 훼손하지 않는다.

1. OOM 또는 `recovered_oom=True`이면 기존 shrink 규칙 실행
2. empty stream은 기존대로 budget과 streak 유지
3. non-OOM fault는 기존대로 budget 유지 및 streak 초기화
4. all-success에서만 pressure growth guard 평가

Guard가 `None`이면:

- pressure summary 전달 여부와 관계없이 기존 success streak/growth 동작 유지
- `growth_suppressed_by_pressure=False`

Guard가 활성화돼 있으면:

- summary가 `None`이면 성장 억제
- summary의 모든 ratio가 `None`이면 성장 억제
- `max_observed_ratio >= max_pressure_ratio_for_growth`이면 성장 억제
- 성장 억제 시 budget 유지
- 성장 억제 시 `consecutive_successes=0`
- 성장 억제 시 `growth_suppressed_by_pressure=True`
- known max ratio가 threshold 미만이면 기존 streak/growth 허용

OOM, recovered OOM, non-OOM fault로 인해 성장 경로에 진입하지 않은 경우에는
`growth_suppressed_by_pressure=False`로 유지한다.

Reason 예시:

```text
success observed but resource pressure is unavailable; suppressing budget growth
resource pressure 0.9 reached growth limit 0.8; suppressing budget growth
```

### 6. 테스트

`enn_torch_dev/debug/runtime/test_runtime_pressure.py`:

- 여러 ratio 중 `max_observed_ratio` 최댓값
- 모든 ratio unknown이면 `None`

`enn_torch_dev/debug/runtime/test_runtime_governor.py`:

- 기존 `GovernorPolicy` positional field 순서 유지
- guard 기본값 `None`에서 기존 growth 완전 유지
- threshold 미만에서 growth 허용
- threshold 도달·초과에서 growth 억제
- 1.0 초과 관측 ratio도 억제에 사용
- summary 없음과 all-unknown summary에서 growth 억제
- 여러 ratio 중 최댓값 사용
- 억제 시 기존 success streak 초기화
- OOM이 pressure보다 우선해 shrink
- recovered OOM이 pressure보다 우선해 shrink
- non-OOM fault 동작 유지
- decision에 summary와 suppression flag 기록
- 기존 인자만 사용한 `GovernorDecision` 생성 및 state 복원 호환
- invalid threshold와 invalid pressure type 거부

### 7. 문서

다음을 갱신한다.

```text
docs/dev_runtime_pressure.md
docs/dev_runtime_governor.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

문서에서 반드시 구분한다.

- pressure assessment 자체는 budget을 선택하지 않음
- governor guard는 opt-in
- missing pressure는 guard 활성화 시 안전하게 growth 억제
- pressure는 budget을 직접 shrink하지 않음
- orchestrator 자동 전달은 아직 구현되지 않음

## 변경 금지 범위

다음은 수정하지 않는다.

```text
enn_torch/**
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/batching.py
pyproject.toml
requirements*.txt
lockfiles
```

구현하지 않을 것:

- pressure 기반 budget shrink
- CPU/CUDA field별 budget 조정
- ResourceMonitor를 governor에서 직접 호출
- capacity 또는 summary 자동 생성
- orchestrator pressure wiring
- dynamic/learned threshold
- persistent telemetry
- stable `enn_torch` API 변경

## 테스트 방법

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

정적 경계 검사:

```bash
git diff -- enn_torch
git diff -- enn_torch_dev/runtime/orchestration.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/resources.py
git diff -- pyproject.toml requirements.txt requirements-dev.txt
```

## 예상 결과

- guard가 비활성화된 모든 기존 governor 테스트 통과
- 신규 pressure guard 테스트 통과
- OOM/recovered OOM shrink 우선순위 유지
- summary/history/orchestration/integration 회귀 없음
- stable `enn_torch`, dependency, lockfile 변경 없음

## 최종 보고 형식

1. 변경 파일
2. threshold validation 계약
3. all-success pressure guard 결정 규칙
4. OOM/fault 우선순위 유지 확인
5. positional/state 호환 확인
6. 실행한 테스트와 결과
7. 실행하지 못한 테스트와 이유
8. PR URL과 실제 GitHub head SHA

```text
AI docs updated:
- docs/dev_runtime_pressure.md
- docs/dev_runtime_governor.md
- docs/CURRENT_STATE.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
