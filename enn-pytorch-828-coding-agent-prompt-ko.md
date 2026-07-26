# ENN-PyTorch #828 순수 Pre-Pass Admission Assessment 구현 프롬프트

## 기준 저장소와 선행 조건

- 저장소: `P-R-N-D/ENN-PyTorch`
- 기준 브랜치: `main`
- 이 프롬프트 작성 시 확인한 최신 `main`: `bc4634095b9d9a355104f5c661e4c16c171db553`
- 선행 구현:
  - PR #826: 정적 초기 `BatchBudget` 추천
  - PR #827: 관측 실행 비용 캘리브레이션
- 함께 제공된 패치:
  - `enn-pytorch-828-prepass-admission.patch`

작업을 시작하기 전에 현재 `main`과 위 기준 commit의 차이를 확인한다.

- 현재 `main`이 더 앞서 있으면 패치를 기계적으로 강제 적용하지 말고, 최신 코드 구조와 충돌을 검토해 동일 계약을 최소 범위로 반영한다.
- `ObservedCostProfile`, `ResourceCapacity`, `ResourceSample`, `BudgetedBatcher`가 없는 오래된 기준에 이 작업을 독립적으로 재구현하지 않는다.
- 기존 사용자 변경을 삭제하거나 되돌리지 않는다.
- 별도 요청 없이 브랜치 생성·전환, 커밋, push, PR 생성·수정은 수행하지 않는다.
- 패치와 프롬프트 파일이 저장소 작업 트리에 임시로 복사됐다면 최종 결과에 포함하지 않는다.

## 작업 목적

현재 개발 런타임에는 다음 두 근거가 구현돼 있다.

```text
정적 근거
ResourceCapacity + reference BatchCost + model/optimizer footprint
  -> recommend_initial_batch_budget(...)

실행 관측 근거
StepResult.resource_samples
  -> ModelCostProbe
  -> ObservedCostCalibrator
  -> ObservedCostProfile
```

그러나 `ObservedCostProfile`은 이전 실행에서 관측된 per-item 비용 envelope일 뿐, 특정 candidate batch를 현재 메모리 상태에서 실행할 수 있는지 판정하지 않는다.

이번 작업은 다음 순수 계산 경계를 추가한다.

```text
ResourceCapacity
+ 실행 직전 ResourceSample
+ ObservedCostProfile
+ candidate batch_size
  -> assess_prepass_admission(...)
  -> ADMIT / REJECT / UNKNOWN
```

최종 구현은 다음을 만족해야 한다.

1. 실행·source 소비·상태 변경 없이 하나의 candidate를 판정한다.
2. `ADMIT`, `REJECT`, `UNKNOWN`을 구분한다.
3. `REJECT`가 `UNKNOWN`보다 우선한다.
4. CPU RSS, CUDA allocated, CUDA reserved를 독립 차원으로 계산한다.
5. `None`과 관측된 0을 구분한다.
6. 현재 CUDA allocated/reserved에 관측된 실행 증분을 더한다.
7. CUDA direct delta와 peak delta 중 큰 알려진 값을 사용한다.
8. baseline의 historical `cuda_max_*`를 현재 사용량으로 더하지 않는다.
9. capacity, baseline sample, profile의 concrete CUDA provenance를 일치시킨다.
10. 판정 결과만 반환하고 실행 차단·분할·retry·governor 연결은 수행하지 않는다.
11. stable `enn_torch` API를 변경하지 않는다.

## 작업 내용

### 1. 신규 admission 모듈 추가

새 파일:

```text
enn_torch_dev/runtime/admission.py
```

다음 development API를 구현한다.

```python
PrePassAdmissionStatus
PrePassAdmissionPolicy
PrePassAdmissionError
PrePassAdmissionDimension
PrePassAdmissionAssessment
assess_prepass_admission
```

### 2. 상태 계약

```python
class PrePassAdmissionStatus(Enum):
    ADMIT = "admit"
    REJECT = "reject"
    UNKNOWN = "unknown"
```

bool 하나로 축약하지 않는다.

- `REJECT`: 현재 사용량 또는 projected 사용량이 usable capacity를 초과함
- `UNKNOWN`: 초과 증거는 없지만 capacity, baseline, profile 비용 또는 sample floor가 부족함
- `ADMIT`: 모든 applicable 차원의 필수 값이 알려져 있고 projected 사용량이 한도 이내임

전체 우선순위:

```text
REJECT > UNKNOWN > ADMIT
```

CPU가 unknown이어도 CUDA가 초과하면 전체 결과는 `REJECT`다.

### 3. Admission policy

```python
@dataclass(frozen=True, slots=True)
class PrePassAdmissionPolicy:
    host_utilization_ratio: float = 0.9
    device_utilization_ratio: float = 0.9
    host_reserve_bytes: int = 0
    device_reserve_bytes: int = 0
    min_profile_samples: int = 1
```

검증:

- bool을 숫자로 허용하지 않는다.
- utilization ratio는 finite이고 `0 < ratio <= 1`이어야 한다.
- reserve는 bool이 아닌 non-negative integer다.
- `min_profile_samples`는 bool이 아닌 positive integer다.

usable capacity 계산 순서:

```text
usable = max(0, floor(capacity * utilization_ratio) - reserve_bytes)
```

CPU capacity는 `ResourceCapacity.effective_cpu_bytes`를 사용한다. 물리 메모리와 cgroup limit가 모두 알려졌다면 작은 값을 사용해야 한다.

### 4. 입력 계약

```python
assessment = assess_prepass_admission(
    capacity,
    baseline_sample,
    observed_profile,
    batch_size=candidate_batch_size,
    policy=policy,
)
```

입력 타입:

- `ResourceCapacity`
- 실행 직전의 단일 `ResourceSample`
- `ObservedCostProfile`
- bool이 아닌 positive integer `batch_size`
- `PrePassAdmissionPolicy | None`

`BatchCost`를 입력에 추가하지 않는다.

- `BudgetedBatcher`는 기존대로 static payload/item budget을 검사한다.
- 이번 admission은 현재 RSS/CUDA 사용량과 관측 실행 증분을 검사한다.
- `batching.py`의 기존 검사를 중복 구현하지 않는다.

입력 객체를 수정하거나 결과 안에 원본 runtime 객체를 보존할 필요가 없다.

### 5. CPU 계산

```text
usable CPU bytes
= max(
    0,
    floor(capacity.effective_cpu_bytes * host_utilization_ratio)
    - host_reserve_bytes,
  )

projected CPU RSS
= baseline_sample.cpu_rss_bytes
  + observed_profile.cpu_rss.max_bytes_per_item * batch_size
```

규칙:

- capacity가 `None`이면 CPU 차원 `UNKNOWN`
- baseline RSS가 `None`이면 `UNKNOWN`
- profile CPU per-item 비용이 `None`이면 `UNKNOWN`
- profile 비용 0은 알려진 non-limiting 값
- baseline RSS 자체가 usable bytes를 넘으면 profile 비용이 unknown이어도 `REJECT`
- projected RSS가 usable bytes를 넘으면 `REJECT`

### 6. CUDA 증분 선택과 projection

CUDA allocated per-item increment:

```text
max(
    observed_profile.cuda_allocated.max_bytes_per_item,
    observed_profile.cuda_max_allocated.max_bytes_per_item,
)
```

CUDA reserved per-item increment:

```text
max(
    observed_profile.cuda_reserved.max_bytes_per_item,
    observed_profile.cuda_max_reserved.max_bytes_per_item,
)
```

`None`이 아닌 값만 대상으로 최대값을 구한다. 둘 다 `None`이면 해당 increment는 unknown이다.

Projection:

```text
projected CUDA allocated
= baseline_sample.cuda_allocated_bytes
  + allocated_increment_per_item * batch_size

projected CUDA reserved
= baseline_sample.cuda_reserved_bytes
  + reserved_increment_per_item * batch_size
```

중요:

- baseline `cuda_max_allocated_bytes`와 `cuda_max_reserved_bytes`는 현재 사용량에 더하지 않는다.
- 해당 값은 baseline sample이 CUDA-bearing인지 판단해 provenance를 요구하는 데만 사용한다.
- historical max counter를 current allocated/reserved로 오해하지 않는다.

### 7. CUDA applicability와 provenance

CUDA는 다음 중 하나라도 참이면 관련 차원으로 본다.

- baseline sample에 allocated/reserved/max allocated/max reserved 값이 하나 이상 있음
- profile에 allocated/reserved direct 또는 peak per-item envelope가 하나 이상 알려짐

CUDA가 관련되면 다음 계약을 적용한다.

1. `ResourceCapacity.cuda_total_bytes`와 `cuda_device_index`가 있어야 한다.
2. 알려진 CUDA profile metric이 있으면 `ObservedCostProfile.cuda_device_index`가 concrete index여야 한다.
3. CUDA-bearing baseline 값이 있으면 `ResourceSample.cuda_device_index`가 concrete index여야 한다.
4. capacity, profile, baseline index는 모두 같아야 한다.
5. bool, 음수, non-integer, `None` index를 concrete index로 인정하지 않는다.
6. 현재 CUDA device를 조회하거나 추정하지 않는다.
7. multi-device 값을 합치지 않는다.

누락 또는 mismatch는 `PrePassAdmissionError`로 거부하고 `dimensions`에 구조화된 정보를 제공한다.

CUDA capacity가 설정돼 있다는 이유만으로 CUDA 차원을 강제하지 않는다. baseline과 profile 모두 CUDA 근거가 없는 CPU-only candidate에서는 CUDA 차원을 non-applicable로 둔다.

### 8. Profile sample floor

```text
observed_profile.successful_samples >= policy.min_profile_samples
```

을 만족하지 않으면 profile per-item 비용을 admission 근거로 사용하지 않는다.

- 해당 increment는 `UNKNOWN`으로 취급한다.
- sample floor 부족 warning을 deterministic하게 기록한다.
- 다만 baseline current usage가 이미 usable capacity를 초과하면 `REJECT`가 우선한다.
- profile에 존재하는 잘못된 CUDA provenance를 임의로 정상화하지 않는다.

### 9. 차원 결과

각 차원은 `PrePassAdmissionDimension`으로 다음 정보를 제공한다.

```text
name
status
applicable
capacity_bytes
usable_bytes
current_bytes
incremental_bytes_per_item
projected_bytes
headroom_bytes
item_limit
item_limit_is_unbounded
reason
```

차원 순서는 항상 다음과 같다.

```text
cpu_rss
cuda_allocated
cuda_reserved
```

item limit:

- baseline이 이미 초과 상태면 0
- increment가 양수이면 `floor((usable - current) / increment)`
- increment가 알려진 0이고 current가 한도 이내면 unbounded
- 필요한 값이 unknown이면 `None`

### 10. 전체 결과

`PrePassAdmissionAssessment` 최소 필드:

```text
status
batch_size
policy
profile_successful_samples
cuda_device_index
dimensions
rejected_dimensions
unknown_dimensions
max_admissible_items
warnings
```

- `rejected_dimensions`와 `unknown_dimensions`는 고정 차원 순서를 유지한다.
- `max_admissible_items`는 applicable 차원에서 계산된 finite item limit의 최솟값이다.
- known-zero 때문에 모든 알려진 차원이 unbounded면 `None`일 수 있다.
- `unknown_dimensions`가 별도로 있으므로 unbounded와 unknown을 구분할 수 있어야 한다.
- warning 순서는 deterministic해야 한다.
- `admitted` convenience property는 status가 정확히 `ADMIT`일 때만 참이다.

### 11. Development API export

수정 파일:

```text
enn_torch_dev/runtime/__init__.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

다음을 `enn_torch_dev.runtime`에서 export한다.

```text
PrePassAdmissionStatus
PrePassAdmissionPolicy
PrePassAdmissionError
PrePassAdmissionDimension
PrePassAdmissionAssessment
assess_prepass_admission
```

요구사항:

- 기존 development export를 제거하거나 이름 변경하지 않는다.
- `__all__` 중복이 없어야 한다.
- stable `enn_torch`에는 노출하지 않는다.

### 12. 테스트

새 파일:

```text
enn_torch_dev/debug/runtime/test_prepass_admission.py
```

최소 검증 범위:

1. CPU-only `ADMIT`
2. CPU projected RSS 초과 `REJECT`
3. baseline RSS 자체 초과
4. CPU capacity unknown
5. baseline RSS unknown
6. profile CPU 비용 unknown
7. known-zero CPU 비용과 unbounded item limit
8. utilization 적용 후 reserve 차감 순서
9. physical/cgroup 중 작은 effective CPU capacity 사용
10. profile sample floor 부족
11. CUDA allocated `ADMIT` / `REJECT`
12. CUDA reserved `ADMIT` / `REJECT`
13. direct와 peak CUDA envelope 중 큰 값 선택
14. baseline `cuda_max_*`가 current usage에 더해지지 않음
15. CUDA profile provenance 누락
16. CUDA baseline provenance 누락
17. capacity/profile/sample CUDA mismatch
18. CUDA 근거가 있지만 CUDA capacity 없음
19. CPU 통과 + CUDA unknown => 전체 `UNKNOWN`
20. 한 차원 unknown + 다른 차원 reject => 전체 `REJECT`
21. 차원별 item limit와 전체 finite minimum
22. batch size bool/0/음수/non-integer 검증
23. 차원·warning 순서 결정성
24. 입력 객체 무변경
25. policy 및 public input type 검증
26. invalid negative profile metric 거부
27. 신규 API가 development namespace에만 노출

기본 테스트는 실제 CUDA 장치 없이 synthetic dataclass 입력으로 실행 가능해야 한다.

### 13. 문서 수정

대상 문서:

```text
docs/dev_prepass_admission.md
docs/dev_observed_cost_calibration.md
docs/dev_initial_batch_budget.md
docs/CURRENT_STATE.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

반드시 문서화할 내용:

- admission은 순수 판정 helper이며 실행 gate가 아님
- `ADMIT / REJECT / UNKNOWN` 계약과 우선순위
- CPU 및 CUDA projection 공식
- direct/peak CUDA envelope 선택
- baseline max counter 비사용
- unknown과 known-zero 구분
- profile sample floor
- concrete CUDA provenance agreement
- structured dimension 결과와 item limit
- orchestrator 자동 연결, fail-open/fail-closed, 자동 split/skip은 범위 밖

## 유지해야 하는 기존 동작

다음을 변경하지 않는다.

- `BatchBudget`, `BudgetedBatcher`, `BatchBudgetExceeded`
- 정적 초기 budget 추천 계약
- `ObservedCostCalibrator` 및 `ObservedCostProfile` 계약
- `ResourceCapacity`와 `ResourceSample` 필드
- `assess_resource_pressure(...)`
- retry, governor, orchestration, session, history 동작
- source iteration과 identity/order
- stable `enn_torch` API
- dependency manifest

## 변경 금지 범위

이번 작업에 포함하지 않는다.

- `batching.py` 수정
- `pressure.py` 수정
- `orchestration.py` 수정
- `retry.py` 수정
- `governor.py` 수정
- `ResourceMonitor.sample()` 자동 호출
- source에서 candidate batch 미리 소비
- `REJECT` 시 자동 split, skip 또는 retry
- `UNKNOWN` fail-open/fail-closed 정책
- runtime 실행 차단
- governor budget 자동 변경
- calibration persistence 또는 cache
- phase별 별도 admission
- multi-GPU 또는 distributed admission
- learned model, percentile, regression
- 모델 topology 또는 노드별 data feeding 변경
- 새 dependency
- stable namespace export
- 관련 없는 리팩터링

## 테스트 방법

직접 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission.py -q
python -m pytest enn_torch_dev/debug/runtime/test_observed_cost_calibration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

회귀 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

범위 확인:

```bash
test -z "$(git diff -- enn_torch)"
test -z "$(git diff -- pyproject.toml requirements.txt requirements-dev.txt)"
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/orchestration.py
git diff -- enn_torch_dev/runtime/retry.py
git diff -- enn_torch_dev/runtime/governor.py
git status --short --branch
```

위 금지 대상 runtime 파일의 diff는 비어 있어야 한다.

## 완료 보고

완료 보고에는 다음을 포함한다.

1. 수정한 파일 목록
2. `ADMIT / REJECT / UNKNOWN` 계산 계약
3. CPU와 CUDA projection 공식
4. CUDA provenance 검증 방식
5. baseline max counter를 current로 사용하지 않았다는 확인
6. 실제 실행한 테스트 명령과 결과
7. skipped test와 warning
8. CUDA 실기기 검증 여부
9. 실행하지 못한 테스트와 이유
10. stable package와 dependency manifest 변경 여부
11. 금지 대상 runtime 파일 변경 여부
12. working tree 상태

실행하지 않은 테스트를 성공했다고 표현하지 않는다.

최종 보고에는 다음 형식의 AI 문서 결과를 정확히 하나 포함한다.

```text
AI docs updated:
- docs/dev_prepass_admission.md
- docs/dev_observed_cost_calibration.md
- docs/dev_initial_batch_budget.md
- docs/CURRENT_STATE.md
- docs/runtime_development_workflow.md
- docs/TESTING.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```

실제로 수정한 문서만 기록한다.
