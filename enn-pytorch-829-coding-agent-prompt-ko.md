# ENN-PyTorch #829: Opt-in Pre-Pass Admission Gate 구현

## 작업 전 확인

이 작업은 PR #828의 병합 결과를 전제로 한다.

기준으로 확인된 병합 커밋:

```text
2fdc0959eb5ae8e3f339f4557c510eeeb0e82e0e
Add pure pre-pass admission assessment
```

작업 시작 시 현재 `main`에 다음 API와 동작이 존재하는지 확인한다.

```text
PrePassAdmissionStatus
PrePassAdmissionPolicy
PrePassAdmissionError
PrePassAdmissionDimension
PrePassAdmissionAssessment
assess_prepass_admission
ObservedCostProfile
ConservativeRuntimeOrchestrator
RuntimeRetryRunner
RuntimePassResult
```

`main`이 위 커밋보다 진행됐더라도 #828이 포함된 최신 `main`을 기준으로 작업한다.
#828이 반영되지 않은 오래된 기준에는 이 작업을 억지로 적용하지 않는다.

별도 요청이 없는 한 다음 작업은 하지 않는다.

- 브랜치 생성 또는 변경
- 커밋
- 푸시
- PR 생성 또는 수정
- dependency 설치
- stable API 변경
- 관련 없는 리팩터링
- 기존 사용자 변경 되돌리기

제공된 패치 또는 프롬프트 파일이 작업 디렉터리에 입력 자료로 존재하더라도 최종 변경에 포함하지 않는다.

---

# 작업 목적

## 해결해야 하는 문제

현재 `assess_prepass_admission(...)`은 다음을 계산하는 순수 함수다.

```text
ResourceCapacity
+ 실행 직전 ResourceSample
+ ObservedCostProfile
+ candidate batch size
  -> ADMIT / REJECT / UNKNOWN
```

하지만 이 결과는 실제 실행 경로를 차단하지 않는다.

다음 단계에서는 `ConservativeRuntimeOrchestrator`에 명시적으로 opt-in한 경우에만, 각 runtime 실행 시도 직전에 admission을 평가하고 결과에 따라 실행하거나 구조화된 예외로 차단해야 한다.

## 작업이 필요한 이유

Admission을 `RuntimeRetryRunner` 바깥에서 한 번만 검사하면 최초 batch만 평가되고, OOM retry가 생성한 subbatch는 검사를 우회할 수 있다.

따라서 gate는 다음 호출 경계에 위치해야 한다.

```text
BudgetedBatcher
  -> RuntimeRetryRunner
       -> admission-aware RuntimeStep wrapper
            -> execution-immediate sample
            -> assess_prepass_admission(...)
            -> OOM tracking wrapper
                 -> configured RuntimeStep
```

이 구조로 최초 batch, 정적 budget split batch, OOM retry subbatch 모두 실행 직전에 평가한다.

## 기대하는 최종 결과

- gate가 비활성화되면 기존 orchestration 동작이 그대로 유지된다.
- gate가 활성화되면 각 실제 execution attempt마다 sample과 assessment가 정확히 한 번 생성된다.
- `ADMIT`은 실행한다.
- `REJECT`는 항상 실행을 차단한다.
- `UNKNOWN`은 기본적으로 차단하고, 명시적 설정에서만 허용한다.
- 차단은 `StepStatus`로 위장하지 않고 구조화된 예외로 전파한다.
- 차단된 pass는 governor 상태를 갱신하지 않는다.
- admission은 자동 split, skip, replay, rollback 또는 tuning을 하지 않는다.
- 완료된 pass는 attempt 순서의 immutable assessment를 제공한다.
- stable `enn_torch` namespace는 변경하지 않는다.

---

# 작업 내용

## 1. 신규 모듈

다음 파일을 추가한다.

```text
enn_torch_dev/runtime/admission_gate.py
```

다음 development API를 구현한다.

```text
ResourceSampleProvider
AdmissionUnknownAction
PrePassAdmissionBlocked
PrePassAdmissionGate
```

### 1.1 `ResourceSampleProvider`

`typing.Protocol` 및 `@runtime_checkable`을 사용한다.

계약:

```python
class ResourceSampleProvider(Protocol):
    def sample(self, phase: str) -> ResourceSample:
        ...
```

기존 `ResourceMonitor.sample(...)`이 구조적으로 이 protocol을 만족해야 한다.

Gate가 자동으로 `ResourceMonitor`를 만들면 안 된다. 호출자가 명시적으로 provider를 제공한다.

### 1.2 `AdmissionUnknownAction`

다음 두 값만 둔다.

```python
class AdmissionUnknownAction(Enum):
    BLOCK = "block"
    ALLOW = "allow"
```

기본 동작은 `BLOCK`이다.

처리 규칙:

```text
ADMIT                 -> 허용
UNKNOWN + BLOCK       -> 차단
UNKNOWN + ALLOW       -> 허용
REJECT                -> 항상 차단
```

`ALLOW`는 `UNKNOWN`에만 적용한다. `REJECT`를 허용하는 옵션을 추가하지 않는다.

문자열을 암묵적으로 enum으로 변환하지 않는다.

### 1.3 `PrePassAdmissionBlocked`

`RuntimeError`의 하위 클래스다.

필수 속성:

```python
assessment: PrePassAdmissionAssessment
```

생성 시 assessment 타입을 검증한다.

예외 메시지에는 최소 다음 scalar/tuple 정보를 포함할 수 있다.

```text
assessment.status
assessment.rejected_dimensions
assessment.unknown_dimensions
```

다음 객체를 예외에 저장하면 안 된다.

- `KVBatch`
- row/source/sample tensor
- baseline `ResourceSample`
- source 또는 iterator
- model
- store
- loss

예외의 사용자 정의 instance state는 assessment만 유지한다.

Admission 차단은 runtime step이 완료된 결과가 아니므로 새로운 `StepStatus`를 추가하거나 기존 fault status로 변환하지 않는다.

### 1.4 `PrePassAdmissionGate`

생성자 입력:

```python
PrePassAdmissionGate(
    capacity: ResourceCapacity,
    observed_profile: ObservedCostProfile,
    sample_provider: ResourceSampleProvider,
    *,
    policy: PrePassAdmissionPolicy | None = None,
    unknown_action: AdmissionUnknownAction = AdmissionUnknownAction.BLOCK,
)
```

입력 타입을 엄격히 검증한다.

공개 메서드:

```python
check(batch_size: int) -> PrePassAdmissionAssessment
```

정확한 순서:

1. `batch_size`가 bool이 아닌 양수 integer인지 검증한다.
2. 검증 실패 시 provider를 호출하지 않는다.
3. `sample_provider.sample("before_admission")`을 정확히 한 번 호출한다.
4. 반환값이 `ResourceSample`인지 검증한다.
5. 기존 `assess_prepass_admission(...)`을 호출한다.
6. `REJECT`이면 `PrePassAdmissionBlocked`를 raise한다.
7. `UNKNOWN`이고 unknown action이 `BLOCK`이면 raise한다.
8. 나머지는 assessment를 반환한다.

Gate는 baseline sample을 인스턴스에 보존하지 않는다.

Gate는 다음을 직접 하지 않는다.

- 모델 실행
- source 소비
- batch slicing
- retry
- governor 변경
- profile 변경
- assessment history 누적
- persistence

---

## 2. Orchestrator opt-in 연결

대상 파일:

```text
enn_torch_dev/runtime/orchestration.py
```

### 2.1 `RuntimePassResult` append-only 확장

기존 필드 순서 뒤에 다음 필드를 추가한다.

```python
admission_assessments: tuple[PrePassAdmissionAssessment, ...] = ()
```

기존 positional 필드 순서를 변경하지 않는다.

의미:

- gate 비활성: `()`
- gate 활성 및 pass 완료: 실제 execution attempt 순서의 assessments
- `UNKNOWN + ALLOW`: UNKNOWN assessment도 포함
- OOM retry에서 폐기된 원본 OOM attempt assessment도 포함

Assessment 수는 최종 `StepResult` 수보다 많을 수 있다.

예:

```text
4-item original attempt -> admitted -> OOM
2-item retry attempt     -> admitted -> success
2-item retry attempt     -> admitted -> success

admission assessments: 3
final StepResults: 2
```

### 2.2 private runtime-step wrapper

orchestration 모듈 내부에 private wrapper를 추가한다.

권장 형태:

```python
class _AdmissionRuntimeStep:
    runtime_step
    gate
    optimizer
    assessments
```

필수 동작:

```python
assessment = gate.check(batch.batch_size)
assessments.append(assessment)
return runtime_step.run(batch)
```

Gate가 차단하면 wrapped runtime step을 호출하지 않으며 assessment는 예외에 존재한다.

`optimizer`는 wrapped runtime step에서 그대로 전달한다.

```python
self.optimizer = getattr(runtime_step, "optimizer", None)
```

이 passthrough가 없으면 `RuntimeRetryRunner`가 training runtime의 OOM retry를 잘못 활성화할 수 있으므로 반드시 유지한다.

### 2.3 Wrapper 순서

다음 순서를 사용한다.

```text
RuntimeRetryRunner
  -> _AdmissionRuntimeStep
       -> _OomTrackingRuntimeStep
            -> configured runtime step
```

Admission wrapper를 retry runner 바깥에 두지 않는다.

Admission sample은 `_OomTrackingRuntimeStep`과 실제 runtime step이 호출되기 전에 생성되어야 한다.

### 2.4 Orchestrator 생성자 옵션

기존 keyword-only 필드에 다음을 추가한다.

```python
admission_profile: ObservedCostProfile | None = None
admission_sample_provider: ResourceSampleProvider | None = None
admission_policy: PrePassAdmissionPolicy | None = None
admission_unknown_action: AdmissionUnknownAction = AdmissionUnknownAction.BLOCK
```

기존 생성자 인수의 동작과 순서를 불필요하게 변경하지 않는다.

### 2.5 활성화 및 구성 검증

Gate는 `admission_profile is not None`인 경우에만 활성화한다.

Profile이 있으면 다음이 필수다.

- `admission_sample_provider`
- `resource_capacity` 또는 `resource_capacity_provider`

Profile이 없는데 다음을 지정하면 잘못된 구성으로 거부한다.

- `admission_sample_provider`
- `admission_policy`
- non-default `admission_unknown_action`

관련 option을 조용히 무시하지 않는다.

기존 규칙을 유지한다.

```text
resource_capacity와 resource_capacity_provider는 상호 배타적
```

### 2.6 Capacity/sample 호출 횟수

Capacity는 기존대로 pass 시작 시 한 번 resolve한다.

```text
resource_capacity_provider.capacity(): pass당 정확히 1회
```

Provider 실패 또는 잘못된 return type은 source 소비 전에 전파한다.

Admission sample은 각 실행 attempt 직전에 호출한다.

```text
admission_sample_provider.sample("before_admission"):
원본/정적 split/retry subbatch attempt당 정확히 1회
```

한 pass 안에서 capacity를 다시 조회하지 않는다.

### 2.7 실행 및 governor 의미

Gate가 차단하면:

- blocked candidate를 runtime step에 전달하지 않는다.
- later candidate를 소비하지 않는다.
- `RuntimePassResult`를 만들지 않는다.
- `governor.observe_results(...)`를 호출하지 않는다.
- current budget 및 governor streak/last decision을 변경하지 않는다.
- `PrePassAdmissionBlocked` 또는 기존 `PrePassAdmissionError`를 전파한다.

단, pass 안의 earlier candidate가 이미 실행됐을 수 있다.

```text
candidate 1 -> admitted and executed
candidate 2 -> blocked
```

이 경우 candidate 1의 실행 부작용을 rollback하지 않는다. Pass 전체를 transactional하다고 표현하지 않는다. Partial `RuntimePassResult`도 반환하지 않는다.

`UNKNOWN + ALLOW`로 실행한 경우에는 기존 `StepResult` 기준 governor 동작을 그대로 유지한다. Admission assessment를 governor 입력으로 사용하지 않는다.

### 2.8 기존 pressure 의미 유지

Fixed/provider-backed capacity가 존재하면 기존 pressure sample aggregation과 governor pressure feedback이 계속 동작해야 한다.

Admission sample 자체를 pressure summary에 자동 포함하지 않는다. Pressure summary는 기존 actual runtime `StepResult.resource_samples`만 사용한다.

---

## 3. Development export

대상:

```text
enn_torch_dev/runtime/__init__.py
```

다음을 import하고 `__all__`에 중복 없이 추가한다.

```text
AdmissionUnknownAction
PrePassAdmissionBlocked
PrePassAdmissionGate
ResourceSampleProvider
```

Stable `enn_torch`에는 노출하지 않는다.

---

## 4. 테스트

### 4.1 신규 테스트 파일

```text
enn_torch_dev/debug/runtime/test_prepass_admission_gate.py
```

작은 synthetic CPU 입력을 사용한다. 기본 테스트가 실제 CUDA 하드웨어를 요구하면 안 된다.

최소 다음을 검증한다.

#### Gate 단위 계약

1. structural sample provider가 protocol을 만족한다.
2. ADMIT은 sample 1회 후 assessment를 반환한다.
3. REJECT는 `PrePassAdmissionBlocked`를 raise한다.
4. UNKNOWN 기본값은 block이다.
5. UNKNOWN + explicit ALLOW는 assessment를 반환한다.
6. ALLOW가 REJECT를 허용하지 않는다.
7. invalid batch size는 sampling 전에 거부된다.
8. sample provider의 invalid return type은 assessment/runtime 전에 거부된다.
9. block exception은 assessment만 사용자 정의 state로 보존한다.
10. gate가 baseline sample 또는 batch를 보존하지 않는다.

#### Orchestration 계약

11. gate 비활성 시 기존 budget split/execution 결과와 동일하고 assessments는 빈 tuple이다.
12. BudgetedBatcher가 만든 각 candidate를 실행 직전에 평가한다.
13. 첫 candidate REJECT 시 runtime step 미호출 및 governor 불변이다.
14. UNKNOWN 기본 block 및 explicit ALLOW 동작을 검증한다.
15. OOM 원본과 모든 retry subbatch를 각각 평가한다.
16. retry가 gate를 우회하지 않는다.
17. optimizer passthrough가 training retry 제한을 유지한다.
18. capacity provider는 pass당 1회다.
19. sample provider는 execution attempt당 1회다.
20. invalid capacity provider는 source 소비 전에 실패한다.
21. CUDA provenance mismatch는 runtime/governor 전에 전파된다.
22. later candidate block 시 이후 source를 소비하지 않는다.
23. earlier candidate가 이미 실행될 수 있지만 governor는 갱신하지 않는다.
24. incomplete admission configuration을 거부한다.
25. 개발 API export와 stable namespace 비노출을 검증한다.

### 4.2 기존 통합 테스트

대상:

```text
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

기존 development API export set에 신규 API 4개를 추가한다.

기존 test 의미를 삭제하거나 약화하지 않는다.

### 4.3 기존 orchestration/retry 회귀

신규 테스트 외에도 기존 다음 동작이 그대로 통과해야 한다.

- budget split row order
- retry-recovered OOM
- optimizer가 있는 runtime의 retry 제한
- pressure aggregation
- capacity provider pass당 호출 횟수
- empty source
- invalid source/configuration
- governor budget 전이

---

## 5. 문서

### 신규 문서

```text
docs/dev_prepass_admission_gate.md
```

최소 다음을 설명한다.

- gate의 목적과 opt-in 성격
- API 4개
- ADMIT/UNKNOWN/REJECT 처리
- default UNKNOWN BLOCK
- retry runner 내부 wrapper 배치 이유
- capacity pass-scoped, sample attempt-scoped
- exception이 assessment만 보존
- StepStatus가 아님
- RuntimePassResult assessment 수와 StepResult 수 차이
- block 시 governor 미갱신
- earlier candidate rollback 없음
- 자동 split/skip/replay/persistence가 범위 밖임
- 테스트 명령

### 수정 문서

```text
docs/dev_prepass_admission.md
docs/CURRENT_STATE.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

#### `docs/dev_prepass_admission.md`

순수 assessor 자체는 계속 side-effect free임을 유지한다.
별도 gate가 assessor를 호출해 enforcement한다는 관계만 추가한다.

#### `docs/runtime_development_workflow.md`

다음을 반영한다.

- retry attempt 내부의 optional gate 위치
- opt-in 구성 예시
- REJECT/UNKNOWN 정책
- attempt별 assessment
- OOM retry subbatch도 검사
- blocked pass exception/governor/source 의미
- 자동 enforcement가 이제 존재하므로 기존 “automatic enforcement 없음” 문구 제거
- 대신 admission-driven 자동 split/skip/replay/rollback은 미지원으로 명시
- gate 테스트 명령

#### `docs/RUNTIME_SAFETY.md`

다음을 명시한다.

- gate opt-in
- capacity pass당 1회, sample attempt당 1회
- REJECT 항상 block
- UNKNOWN default block, explicit allow만 허용
- block은 StepStatus가 아님
- exception은 assessment만 보존
- optimizer passthrough
- blocked pass governor 불변
- earlier candidate rollback 없음
- 자동 split/skip/replay 없음

#### `docs/TESTING.md`

실재하는 테스트 경로만 추가한다.

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
```

표와 baseline command block 양쪽에 개별 실행 가능한 명령으로 기록한다.

#### `docs/CHANGE_CHECKLIST.md`

Gate 작업 검토 항목을 추가한다.

---

# 유지해야 하는 기존 동작

다음을 훼손하지 않는다.

- 순수 `assess_prepass_admission(...)` 계산 계약
- `REJECT > UNKNOWN > ADMIT`
- known zero와 unknown 구분
- CUDA provenance 계약
- `BudgetedBatcher` 정적 budget split
- `RuntimeRetryRunner` OOM 분할
- optimizer가 있는 runtime의 retry 제한
- `_OomTrackingRuntimeStep`의 OOM/sample 수집
- governor의 OOM/pressure/success 처리
- capacity provider pass당 1회
- pressure summary가 actual runtime samples를 사용하는 계약
- `RuntimePassResult` 기존 field positional 순서
- session/history/summary 기존 동작
- stable `enn_torch` API
- dependency manifest

---

# 변경 금지 범위

별도 문제를 발견하지 않는 한 다음 파일은 수정하지 않는다.

```text
enn_torch_dev/runtime/admission.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/step.py
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/session.py
```

다음 기능을 추가하지 않는다.

- admission REJECT 자동 split
- admission REJECT skip-and-continue
- source replay 또는 rollback
- UNKNOWN per-dimension policy
- profile 자동 calibration/update
- persistent profile 또는 admission cache
- admission-driven governor budget 변경
- summary/history admission 집계
- new `StepStatus`
- multi-GPU
- distributed coordination
- 새 dependency
- stable API 노출
- 모델 topology 변경
- 노드별 데이터 feeding 변경
- 관련 없는 리팩터링

필요한 추가 작업이 이 범위를 넘어가면 현재 패치에 임의로 포함하지 말고 별도 후속 작업으로 보고한다.

---

# 테스트 방법

## 필수 targeted 테스트

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

## Runtime 및 전체 debug 회귀

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
```

## 정적 확인

```bash
git diff --check
python -m py_compile \
  enn_torch_dev/runtime/admission_gate.py \
  enn_torch_dev/runtime/orchestration.py \
  enn_torch_dev/runtime/__init__.py \
  enn_torch_dev/debug/runtime/test_prepass_admission_gate.py
```

## 변경 범위 확인

```bash
test -z "$(git diff -- enn_torch)"
test -z "$(git diff -- pyproject.toml requirements.txt requirements-dev.txt)"
test -z "$(git diff -- \
  enn_torch_dev/runtime/admission.py \
  enn_torch_dev/runtime/batching.py \
  enn_torch_dev/runtime/retry.py \
  enn_torch_dev/runtime/step.py \
  enn_torch_dev/runtime/governor.py \
  enn_torch_dev/runtime/pressure.py \
  enn_torch_dev/runtime/resources.py \
  enn_torch_dev/runtime/summary.py \
  enn_torch_dev/runtime/history.py \
  enn_torch_dev/runtime/session.py)"
git status --short --branch
```

## CUDA 확인

먼저 CUDA 가용성을 확인한다.

```bash
python - <<'PY'
import torch
print(f"cuda_available={torch.cuda.is_available()}")
PY
```

CUDA가 없으면 실제 GPU 검증을 성공했다고 표현하지 않는다.
Synthetic `ResourceSample`, `ResourceCapacity`, `ObservedCostProfile` 기반 테스트와 실제 CUDA 실기기 검증을 명확히 구분한다.

---

# 예상 결과

다음이 모두 성립해야 한다.

- 신규 gate 테스트 통과
- 기존 pure admission 테스트 통과
- 기존 orchestration/retry/pressure/governor 테스트 통과
- gate disabled 경로 회귀 없음
- original 및 retry subbatch attempt별 검사
- blocked attempt runtime 미호출
- blocked pass governor 미갱신
- UNKNOWN default block
- UNKNOWN explicit allow
- REJECT always block
- capacity provider pass당 1회
- sample provider attempt당 1회
- stable `enn_torch` 변경 없음
- dependency 변경 없음
- 금지 파일 변경 없음
- 문서와 구현 일치

---

# 완료 보고 원칙

결과 보고에서 다음을 구분한다.

## 실제로 실행한 검증

- 실행한 명령
- 성공/실패 결과와 test count
- skipped test와 warning
- CUDA availability
- `git diff --check`
- stable package/dependency/금지 파일 변경 여부
- working tree 상태

## 실행하지 못한 검증

- 실행하지 못한 명령
- 이유
- CUDA 실기기 검증 여부
- 해당 미검증이 결과에 미치는 영향

실행하지 않은 테스트를 성공했다고 표현하지 않는다.

최종 보고에는 실제 수정한 AI 문서만 사용해 다음 형식을 정확히 하나 포함한다.

```text
AI docs updated:
- docs/dev_prepass_admission_gate.md
- docs/dev_prepass_admission.md
- docs/CURRENT_STATE.md
- docs/runtime_development_workflow.md
- docs/TESTING.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
