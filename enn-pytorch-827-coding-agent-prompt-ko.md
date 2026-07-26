# ENN-PyTorch #827 관측 실행 비용 캘리브레이션 구현 프롬프트

## 기준 저장소와 선행 조건

- 저장소: `P-R-N-D/ENN-PyTorch`
- 선행 작업: PR #826의 초기 `BatchBudget` 추천 구현
- 이 프롬프트 작성 시 확인한 선행 head: `0a09889dab4c13275faabd9e5f1bf5068f3d5bbb`
- 함께 제공된 패치: `enn-pytorch-827-observed-cost-calibration.patch`

작업을 시작하기 전에 현재 저장소 상태를 확인한다.

- PR #826이 아직 병합되지 않았다면 #826의 최신 head를 기준으로 작업한다.
- PR #826이 `main`에 병합됐다면 해당 변경이 포함된 최신 `main`을 기준으로 작업한다.
- `ModelCost`, `ModelCostProbe`, `recommend_initial_batch_budget(...)` 및 관련 개발 API가 없는 오래된 기준에는 이 작업을 독립적으로 구현하지 않는다.
- 기준 이후 변경이 있다면 패치를 기계적으로 적용하지 말고 현재 코드와 충돌 여부를 검토해 같은 계약을 최소 범위로 반영한다.

기존 사용자 변경을 삭제하거나 되돌리지 않는다. 별도 요청 없이 브랜치 생성·전환, 커밋, 푸시, PR 생성·수정은 수행하지 않는다.

## 작업 목적

현재 런타임에는 다음 경계가 구현돼 있다.

```text
StepResult.resource_samples
  -> ModelCostProbe
  -> 단일 실행의 ModelCost
```

또한 초기 `BatchBudget` 추천기는 정적 capacity, reference `BatchCost`, 모델 및 옵티마이저 footprint만 사용하므로 다음 실행 시 발생할 수 있는 activation 증가, CUDA allocator reservation, phase별 peak를 직접 반영하지 않는다.

이번 작업은 여러 완료 실행의 `ModelCost`를 다음 admission 작업에서 소비할 수 있는 보수적이고 결정적인 per-item 비용 envelope로 축약하는 개발 전용 경계를 추가한다.

```text
ModelCost observations
  -> ObservedCostCalibrator.observe(...)
  -> ObservedCostProfile
```

최종 결과는 다음을 만족해야 한다.

1. 성공한 양수 batch 관측만 수치 캘리브레이션에 사용한다.
2. byte delta를 ceiling division으로 per-item 비용으로 바꾸고 관측 최대값을 유지한다.
3. `None`, 실제 0, 음수 delta를 서로 구분한다.
4. 음수 delta를 메모리 credit으로 사용하지 않고 0으로 clamp한다.
5. 한 profile에 서로 다른 CUDA 장치의 관측을 섞지 않는다.
6. phase별 상태 수를 명시적으로 제한한다.
7. 원본 `ModelCost`, `StepResult`, `ResourceSample`, tensor, store, loss를 보존하지 않는다.
8. 모델 실행, source 소비, governor/history 변경, admission 판단을 수행하지 않는다.
9. stable `enn_torch` API를 변경하지 않는다.

## 작업 내용

### 1. `ModelCost`에 CUDA provenance 추가

대상 파일:

```text
enn_torch_dev/runtime/cost.py
enn_torch_dev/debug/runtime/test_model_cost_probe.py
```

`ModelCost`의 기존 마지막 필드 뒤에 다음 필드를 append한다.

```python
cuda_device_index: int | None = None
```

기존 positional 필드 순서는 변경하지 않는다.

`ModelCostProbe.estimate_step(...)`은 `StepResult.resource_samples`에서 CUDA 관련 값이 실제로 하나 이상 기록된 sample의 `cuda_device_index`를 확인한다.

- CUDA allocated, reserved, max allocated, max reserved 중 하나 이상이 알려진 sample만 CUDA-bearing sample로 본다.
- 모든 CUDA-bearing sample이 정확히 한 index를 가리키면 그 index를 `ModelCost.cuda_device_index`에 기록한다.
- CUDA-bearing sample이 없거나 서로 다른 index가 섞이면 `None`을 기록한다.
- 기존 endpoint field 누락 및 cross-device delta의 `None` 의미는 유지한다.
- CPU-only 실행은 `cuda_device_index=None`을 유지한다.

### 2. 관측 비용 캘리브레이션 모듈 추가

새 파일:

```text
enn_torch_dev/runtime/calibration.py
```

다음 개발 API를 구현한다.

```python
ObservedCostCalibrationPolicy
ObservedCostCalibrationError
ObservedCostMetricProfile
ObservedPhaseCostProfile
ObservedCostProfile
ObservedCostCalibrator
```

#### `ObservedCostCalibrationPolicy`

필드:

```python
min_successful_samples: int = 1
max_phase_pairs: int = 32
expected_cuda_device_index: int | None = None
```

검증:

- bool을 integer로 허용하지 않는다.
- `min_successful_samples`와 `max_phase_pairs`는 양수여야 한다.
- `expected_cuda_device_index`는 `None` 또는 bool이 아닌 non-negative integer여야 한다.

#### metric envelope

각 scalar metric은 다음 정보를 보존한다.

```python
max_bytes_per_item: int | None
known_samples: int
unknown_samples: int
zero_samples: int
negative_deltas_clamped: int
```

입력 의미:

- `None`: unknown이며 synthetic 0으로 바꾸지 않는다.
- `0`: 알려진 non-positive cost이며 실제 zero count를 증가시킨다.
- 음수: 메모리 절감 또는 credit으로 사용하지 않고 0으로 clamp하며 별도 count를 증가시킨다.
- 양수: 그대로 사용한다.

계산:

```text
normalized delta = max(delta, 0)
bytes per item = ceil(normalized delta / batch_size)
calibrated envelope = max(bytes per item over accepted observations)
```

metric별 최대값은 다음 total 필드와 phase 필드에 각각 적용한다.

- CPU RSS
- CUDA allocated
- CUDA reserved
- CUDA max allocated
- CUDA max reserved

#### 관측 수용 규칙

`ObservedCostCalibrator.observe(cost)`는 `ModelCost`만 받는다.

- `StepStatus.SUCCESS`가 아닌 관측은 수치에 사용하지 않고 status별 ignored count를 증가시킨 뒤 `False`를 반환한다.
- 성공했지만 `batch_size == 0`인 관측은 zero-batch ignored count를 증가시키고 `False`를 반환한다.
- 성공한 양수 batch 관측을 적용하면 `True`를 반환한다.
- 음수 batch size는 거부한다.
- `batch_size`와 `row_count`가 다르면 per-item 기준이 불명확하므로 거부한다.
- rejected observation은 수치 envelope나 phase accumulator에 부분 적용하지 않는다.
- 전체 observation count는 호출된 성공/ignored/rejected 관측을 구분할 수 있어야 한다.
- profile에는 `rejected_samples`를 명시적으로 제공한다.
- 각 rejected 호출은 구조화된 `ObservedCostCalibrationError.reason` 및 필요한 `dimensions`로 원인을 제공한다.

#### CUDA 규칙

- total 또는 phase의 CUDA metric 중 하나라도 알려졌다면 `ModelCost.cuda_device_index`가 반드시 있어야 한다.
- `expected_cuda_device_index`가 설정됐다면 정확히 일치해야 한다.
- 첫 accepted CUDA 관측이 profile의 CUDA index를 확정한다.
- 이후 다른 CUDA index의 관측은 거부하며 기존 profile에 부분 적용하지 않는다.
- CUDA metric이 전혀 없는 관측의 단순 device field는 profile을 임의로 bind하는 데 사용하지 않는다.
- 서로 다른 장치를 합치거나 현재 장치로 추정하지 않는다.

#### phase 규칙과 bounded state

`ResourceDelta.start_phase`와 `end_phase`는 non-empty normalized string이어야 한다.

- 하나의 `ModelCost` 안에서 같은 adjacent phase pair가 반복되면 sample count 의미가 모호하므로 거부한다.
- phase profile은 `(start_phase, end_phase)`별 metric envelope를 유지한다.
- 결과의 phase profile은 phase pair 기준으로 정렬해 결정성을 보장한다.
- 새 관측이 `max_phase_pairs`를 넘기면 수치 적용 전에 거부한다.
- fault 및 zero-batch ignored 관측은 phase bound를 소비하지 않는다.

캘리브레이터 내부에는 scalar accumulator, bounded phase-pair map, count 및 device index만 남긴다. 다음 객체를 저장하는 속성을 만들지 않는다.

- `ModelCost`
- `StepResult`
- `ResourceSample`
- tensor
- `KVStore`
- loss

#### profile 생성

`ObservedCostCalibrator.profile()`은 immutable `ObservedCostProfile` snapshot을 반환한다.

최소 포함 정보:

- 적용 policy
- total observations
- successful samples
- ignored samples
- rejected samples
- zero-batch ignored samples
- status별 ignored count의 정렬된 immutable tuple
- accepted batch size 최소/최대
- accepted CUDA device index
- total metric profiles
- 정렬된 phase metric profiles

`successful_samples < min_successful_samples`이면 profile을 만들지 않고 `ObservedCostCalibrationError`를 발생시킨다.

### 3. development API export

대상 파일:

```text
enn_torch_dev/runtime/__init__.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

다음을 `enn_torch_dev.runtime`에서 export한다.

```text
ObservedCostCalibrationPolicy
ObservedCostCalibrationError
ObservedCostMetricProfile
ObservedPhaseCostProfile
ObservedCostProfile
ObservedCostCalibrator
```

요구사항:

- `__all__` 중복이 없어야 한다.
- stable `enn_torch`에는 노출하지 않는다.
- 기존 development export를 제거하거나 이름을 변경하지 않는다.

### 4. 테스트 추가 및 보완

새 파일:

```text
enn_torch_dev/debug/runtime/test_observed_cost_calibration.py
```

수정 파일:

```text
enn_torch_dev/debug/runtime/test_model_cost_probe.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

최소 검증 사례:

1. 서로 다른 batch size의 관측을 ceiling division한 뒤 최대 per-item 값을 선택
2. unknown-only metric은 `None`
3. unknown 뒤 실제 zero가 관측되면 envelope 0과 unknown/zero count가 모두 보존
4. 음수 delta가 0으로 clamp되고 실제 zero count와 구분
5. OOM, nonfinite, data fault, runtime fault가 수치에서 제외되고 status별 count가 기록
6. zero-batch 성공이 ignored 처리
7. `min_successful_samples` 미달 시 profile 생성 실패
8. CUDA metric이 있지만 device provenance가 없을 때 실패
9. 서로 다른 CUDA index가 한 profile에 들어오면 실패하고 기존 envelope는 유지
10. expected CUDA device 불일치 실패
11. total 및 phase metric별 최대 envelope
12. phase result 정렬 결정성
13. `max_phase_pairs` 초과 시 부분 적용 없이 실패
14. 하나의 observation 내 duplicate phase pair 거부
15. 잘못된 phase name 거부
16. `batch_size != row_count` 거부
17. raw `ModelCost` 목록이나 `__dict__` 기반 observation 보존 없음
18. 같은 관측 순서에서 동일 profile 생성
19. policy의 bool, zero, negative 입력 검증
20. `ModelCostProbe`가 단일 CUDA index를 기록
21. 서로 다른 CUDA-bearing sample이 섞이면 `cuda_device_index=None`
22. CUDA 값이 없는 sample의 index는 provenance 결정에 사용하지 않음
23. 신규 API가 development namespace에만 노출

기본 테스트는 CUDA 하드웨어 없이 `ModelCost`, `ResourceDelta`, synthetic `ResourceSample`로 실행 가능해야 한다.

### 5. 문서 수정

다음 문서를 구현과 일치하도록 수정한다.

```text
docs/dev_observed_cost_calibration.md
docs/dev_cost_probe.md
docs/dev_initial_batch_budget.md
docs/CURRENT_STATE.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

문서에 반드시 명시할 내용:

- observed calibration은 현재 구현된 development helper이다.
- 성공한 양수 batch만 수치 envelope에 사용한다.
- fault와 zero-batch는 ignored count로 보존한다.
- rejected 관측은 error로 원인을 제공하며 부분 적용하지 않는다.
- unknown, 실제 zero, negative-clamped를 구분한다.
- CUDA profile은 하나의 concrete device에만 bind된다.
- phase accumulator 수는 bounded다.
- 원본 runtime 객체를 보존하지 않는다.
- profile은 future execution의 admission 증명이 아니다.
- persistence, admission, governor 자동 연결은 범위 밖이다.

`docs/dev_initial_batch_budget.md`에서는 observed calibration이 더 이상 미구현이라고 쓰지 말고, 별도 helper로 구현됐으나 pre-pass admission은 아직 범위 밖이라고 정정한다.

## 유지해야 하는 기존 동작

- `DataCostProbe`의 tensor/storage 집계
- `ModelCostProbe`의 기존 total 및 adjacent phase delta 계산
- cross-device endpoint CUDA delta의 `None` 의미
- `BatchBudget`, `BudgetedBatcher`, 초기 budget 추천 계약
- retry/OOM 동작
- governor pressure 및 history 계약
- orchestration/session/source consumption 계약
- `ModelCost`의 기존 positional 필드 순서
- stable `enn_torch` 공개 API
- dependency 및 package 설정

## 변경 금지 범위

다음은 이번 작업에 포함하지 않는다.

- 모델 forward, backward 또는 optimizer step 자동 실행
- source 샘플링 또는 소비
- pre-pass admission 허용/거부
- `ObservedCostProfile`을 `BatchBudget`으로 자동 변환
- orchestrator, session 또는 governor 자동 연결
- governor 상태 변경
- OOM retry 정책 변경
- percentile, 평균, 회귀, learned weights
- 파일, JSONL, CSV, DB 기반 calibration persistence
- checkpoint/resume
- multi-GPU profile 병합
- distributed 집계
- stable `enn_torch` export
- 새 dependency
- 관련 없는 리팩터링

## 테스트 방법

신규 기능 직접 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_observed_cost_calibration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_model_cost_probe.py -q
```

API 경계와 런타임 회귀:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

추가 정적 확인:

```bash
git diff -- enn_torch
git diff -- pyproject.toml requirements.txt requirements-dev.txt
git status --short --branch
```

필요 시 변경 Python 파일 compile 확인:

```bash
python -m py_compile \
  enn_torch_dev/runtime/calibration.py \
  enn_torch_dev/runtime/cost.py \
  enn_torch_dev/runtime/__init__.py \
  enn_torch_dev/debug/runtime/test_observed_cost_calibration.py \
  enn_torch_dev/debug/runtime/test_model_cost_probe.py \
  enn_torch_dev/debug/runtime/test_runtime_integration.py
```

테스트 결과는 실제 최신 head에서 실행한 명령과 수치를 보고한다. CUDA 실기기 검증을 실행하지 않았다면 synthetic device-provenance 검증과 CUDA 미검증 범위를 분리한다. 환경이나 의존성 문제로 테스트를 실행하지 못하면 임의 설치하지 말고 실패 원인과 미검증 범위를 기록한다.

## 완료 보고 형식

완료 보고에는 다음을 포함한다.

1. 실제 변경 파일과 핵심 계약
2. `ModelCost.cuda_device_index` 산출 규칙
3. accepted, ignored, rejected observation 의미
4. unknown, zero, negative-clamped 의미
5. bounded phase 상태 및 raw object 비보존 확인
6. 실제 실행한 테스트와 결과
7. 실행하지 못한 테스트와 CUDA 미검증 범위
8. stable package 및 dependency 변경 여부
9. 아래 형식 중 정확히 하나의 AI 문서 결과

```text
AI docs updated:
- docs/dev_observed_cost_calibration.md
- docs/dev_cost_probe.md
- docs/dev_initial_batch_budget.md
- docs/CURRENT_STATE.md
- docs/runtime_development_workflow.md
- docs/TESTING.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
