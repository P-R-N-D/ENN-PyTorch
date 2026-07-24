# Codex 작업 지침 — #818 Pass-scoped ResourceCapacityProvider와 capacity provenance

## 작업 목적

현재 `ConservativeRuntimeOrchestrator`는 고정된
`resource_capacity: ResourceCapacity | None`만 받을 수 있다. `RuntimeStep`은
`ResourceMonitor`를 통해 매 실행의 `ResourceSample`을 만들 수 있고,
`ResourceMonitor.capacity()`도 이미 구현돼 있지만, capacity snapshot은
orchestrator 생성 시점에 호출자가 직접 고정 주입해야 한다.

이번 작업은 pass 시작 시 capacity를 정확히 한 번 조회하는 명시적
`ResourceCapacityProvider` 경계를 추가한다. 조회된 capacity는 그 pass 전체의
pressure denominator로 고정하고, `RuntimePassResult`와 `RuntimePassSummary`에
provenance로 기록한다.

기대 흐름:

```text
run_pass(source)
  -> provider.capacity() exactly once
  -> one resolved ResourceCapacity for the whole pass
  -> raw-attempt ResourceSample collection
  -> assess_resource_pressure(...)
  -> GovernorDecision
  -> RuntimePassResult / RuntimePassSummary capacity provenance
```

이 작업은 mid-pass refresh, free-memory admission control, pressure 기반 shrink,
자동 `ResourceMonitor` 생성을 구현하지 않는다.

---

## Git 작업 절차

1. 최신 `main`을 기준으로 feature branch를 만든다.

```bash
git switch main
git pull --ff-only
git switch -c codex/pass-scoped-capacity-provider
```

2. `main`에 직접 커밋하지 않는다.
3. 외부 패치를 적용한다.

```bash
git apply /mnt/data/apply-patch-0023_resource_capacity_provider_provenance.diff
```

4. 아래 테스트와 정적 검사를 실행한다.
5. feature branch에만 commit/push한다.
6. `main` 대상 PR을 생성하거나 갱신한다.
7. 사용자 검토 전 자동 병합하지 않는다.

전달용 파일은 저장소에 복사하거나 commit하지 않는다.

```text
/mnt/data/apply-patch-0023_resource_capacity_provider_provenance.diff
/mnt/data/codex_prompt_0023_resource_capacity_provider_provenance.md
```

`git status --short`에 전달용 `.diff`, 프롬프트, `__pycache__`, `*.pyc`, 임시
테스트 파일이 남으면 commit 전에 제거한다.

---

## 작업 내용

### 1. 변경 대상

구현:

```text
enn_torch_dev/runtime/capacity_provider.py
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/__init__.py
```

테스트:

```text
enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py
enn_torch_dev/debug/runtime/test_runtime_summary.py
```

AI-facing 문서:

```text
docs/dev_runtime_capacity_provider.md
docs/dev_runtime_orchestration.md
docs/dev_runtime_summary.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

### 2. `ResourceCapacityProvider` protocol

새 active-development protocol을 추가한다.

```python
@runtime_checkable
class ResourceCapacityProvider(Protocol):
    def capacity(self) -> ResourceCapacity:
        ...
```

계약:

- `enn_torch_dev.runtime`에서 공개한다.
- stable `enn_torch`에는 노출하지 않는다.
- 기존 `ResourceMonitor`는 이미 `capacity()`를 제공하므로 별도 adapter 없이
  protocol을 만족해야 한다.
- protocol은 capacity 조회만 담당하고 sampling, budget 결정, retry를 수행하지
  않는다.

### 3. Orchestrator 생성자

keyword-only 인자를 추가한다.

```python
resource_capacity_provider: ResourceCapacityProvider | None = None
```

계약:

- `resource_capacity`와 `resource_capacity_provider`는 상호 배타적이다.
- 둘 다 `None`이면 기존 동작을 정확히 유지한다.
- fixed capacity 경로도 기존대로 유지한다.
- 잘못된 provider 타입은 생성자에서 `TypeError`로 거부한다.
- fixed와 provider를 동시에 주면 `ValueError`로 거부한다.
- positional 호출 순서를 변경하지 않는다.

### 4. Pass 시작 capacity resolution

`run_pass(...)`에서 source 타입을 검증한 뒤, source를 소비하기 전에 capacity를
해석한다.

```python
resolved_capacity = self._resolve_resource_capacity()
```

provider가 있으면:

```python
resolved_capacity = provider.capacity()
```

반환값이 `ResourceCapacity`가 아니면 `TypeError`를 발생시킨다.

필수 순서와 실패 계약:

1. source 타입 검증
2. provider를 pass당 정확히 한 번 호출
3. provider 성공 후에만 source iteration과 runtime execution 시작
4. provider 예외 또는 잘못된 반환 타입은 그대로 전파
5. provider 실패 시 source를 소비하지 않음
6. provider 실패 시 runtime step을 호출하지 않음
7. provider 실패 시 governor state/last decision을 갱신하지 않음

한 번 조회된 capacity는 해당 pass의 모든 batch, retry, split attempt에 동일하게
사용한다. retry나 subbatch마다 provider를 다시 호출하지 않는다.

### 5. Pressure assessment와 provenance

기존 raw-attempt sample 수집 여부를 다음처럼 resolved capacity에 연결한다.

```python
collect_resource_samples=resolved_capacity is not None
```

pressure assessment도 resolved capacity를 사용한다.

```python
pressure_summary = assess_resource_pressure(
    tracking_step.resource_samples,
    resolved_capacity,
)
```

`RuntimePassResult`의 기존 필드 뒤에 추가한다.

```python
resource_capacity: ResourceCapacity | None = None
```

`RuntimePassSummary`의 기존 필드 뒤에도 추가한다.

```python
resource_capacity: ResourceCapacity | None = None
```

`summarize_runtime_pass(...)`는 pass result의 capacity를 그대로 복사한다.
`format_runtime_pass_summary(...)`는 다음 debug 행을 포함한다.

```text
resource_capacity=<ResourceCapacity repr 또는 None>
```

`ResourceCapacity`는 scalar-only frozen record이므로 raw `ResourceSample`, model,
store, loss reference를 보존하지 않는다. History에는 별도 capacity aggregate를
추가하지 않는다. 각 retained `RuntimePassSummary`의 provenance만 유지한다.

### 6. 테스트

새 `test_runtime_capacity_provider.py`에서 최소한 다음을 검증한다.

1. `ResourceMonitor`와 fake provider가 protocol을 만족
2. `RuntimePassResult.resource_capacity`가 기존 필드 뒤에 추가됨
3. `RuntimePassSummary.resource_capacity`가 기존 필드 뒤에 추가됨
4. fixed capacity 기존 pressure 동작과 provenance 기록
5. provider를 pass마다 정확히 한 번 호출
6. 두 pass에서 서로 다른 capacity 반환 및 서로 다른 ratio 계산
7. retry/split 횟수와 무관하게 provider는 pass당 한 번만 호출
8. provider 예외가 source 소비 전에 전파
9. 잘못된 provider 반환 타입이 source 소비 전에 `TypeError`
10. provider 실패 시 runtime step과 governor state 미변경
11. fixed capacity와 provider 동시 설정 거부
12. 잘못된 provider 타입 거부
13. session을 통한 pass result/summary capacity provenance
14. pressure guard의 low/high ratio 동작 유지
15. dev API export 및 stable namespace 미노출

기존 `test_runtime_summary.py`의 field-order 테스트는 신규 capacity 필드가 마지막에
추가된 계약을 반영한다.

### 7. 문서

다음을 명확히 문서화한다.

- fixed capacity와 provider는 상호 배타적
- provider는 pass 시작 시 정확히 한 번 호출
- source 소비 전 호출 및 실패 전파
- 한 pass 동안 capacity 고정
- retry/split에서 재호출하지 않음
- `ResourceMonitor`가 provider로 사용 가능
- pass result/summary에 capacity provenance 기록
- raw sample reference 미보존
- mid-pass refresh와 free-memory admission control은 미구현
- stable `enn_torch` 미변경

---

## 변경 금지 범위

다음은 수정하지 않는다.

```text
enn_torch/**
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/source_factory.py
pyproject.toml
requirements*.txt
lockfiles
```

구현하지 않을 것:

```text
orchestrator 내부 ResourceMonitor 자동 생성
mid-pass capacity refresh
batch/retry별 provider 재호출
real-time free-memory admission control
pressure 기반 budget shrink
CPU/CUDA별 field-specific budget 조정
persistent telemetry 또는 export
distributed capacity aggregation
stable enn_torch API 변경
```

---

## 테스트 방법

필수 targeted 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_resource_monitor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_pressure.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_session.py -q
```

회귀 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug/data -q
python -m pytest enn_torch_dev/debug -q
```

정적·범위 검사:

```bash
git diff --check
git diff -- enn_torch
git diff -- enn_torch_dev/runtime/governor.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/resources.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/session.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/history.py
git diff -- enn_torch_dev/runtime/source_factory.py
git diff -- pyproject.toml requirements.txt requirements-dev.txt
git status --short --branch
```

테스트는 CPU-only 소형 synthetic input을 기본으로 한다. 실제 CUDA 실행은 별도 요청과
환경 확인 없이 수행하지 않는다.

---

## 예상 결과

- 기존 no-capacity 및 fixed-capacity 호출 동작 유지
- provider를 pass당 정확히 한 번 호출
- provider 실패 시 source/governor 미변경
- retry/split에서 provider 재호출 없음
- pass마다 갱신된 capacity로 pressure ratio 계산
- pass result/summary에 실제 denominator provenance 기록
- dataclass positional 호환 유지
- raw runtime reference retention 없음
- stable namespace, dependency, lockfile 변경 없음

---

## PR 처리

- feature branch를 사용한다.
- `main`에 직접 커밋하지 않는다.
- 테스트가 통과한 뒤 branch에 commit/push한다.
- `main` 대상 PR을 생성하거나 갱신한다.
- PR 본문에 각 명령의 정확한 passed/skipped/warning 수를 기록한다.
- 로컬 SHA와 실제 GitHub PR head SHA가 다르면 구분해 보고한다.
- 사용자 검토 전 자동 병합하지 않는다.

권장 PR 제목:

```text
Add pass-scoped resource capacity provider
```

---

## 최종 보고 형식

다음을 구분해 보고한다.

1. 변경 파일
2. provider protocol과 public API 경계
3. fixed/provider 상호 배타성
4. pass-start 호출 순서와 호출 횟수
5. 실패 시 source/governor 보존
6. retry/split 재호출 방지
7. capacity provenance 전달 방식
8. positional 호환과 reference-retention 경계
9. 실제 실행 테스트 명령과 정확한 결과
10. 실행하지 못한 테스트와 이유
11. `git status --short --branch`
12. PR URL과 실제 GitHub head SHA

```text
AI docs updated:
- docs/dev_runtime_capacity_provider.md
- docs/dev_runtime_orchestration.md
- docs/dev_runtime_summary.md
- docs/runtime_development_workflow.md
- docs/CURRENT_STATE.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
