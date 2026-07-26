# ENN-PyTorch #826 초기 BatchBudget 추천기 구현 프롬프트

## 기준 저장소와 작업 원칙

- 저장소: `P-R-N-D/ENN-PyTorch`
- 기준 브랜치: `main`
- 이 프롬프트 작성 시 확인한 기준 커밋: `7f4e766852a5f10fe26dea2e496d60dc20903dde`
- 함께 제공된 패치: `enn-pytorch-826-initial-batch-budget.patch`

작업을 시작하기 전에 실제 저장소의 현재 커밋, 변경 상태, 관련 코드와 테스트를 다시 확인한다. 기준 커밋 이후 변경이 있다면 패치를 기계적으로 적용하지 말고 현재 구현과 충돌 여부를 검토해 같은 계약을 최소 범위로 반영한다.

기존 사용자의 변경을 덮어쓰거나 되돌리지 않는다. 요청과 무관한 리팩터링, 새 의존성 추가, stable `enn_torch` API 변경, 브랜치 생성·전환, 커밋, 푸시, PR 생성은 수행하지 않는다.

## 작업 목적

현재 `ConservativeRuntimeGovernor`는 실행 결과를 관찰한 뒤 다음 패스의 `BatchBudget`을 조정하지만, 첫 패스의 초기 예산은 호출자가 임의로 구성해야 한다.

다음의 정적이고 이미 알려진 정보로 첫 패스용 보수적 예산을 추천하는 개발 전용 경계를 추가한다.

```text
ResourceCapacity
+ ModelFootprint / OptimizerFootprint의 장치별 텐서 바이트
+ 기준 BatchCost
+ 명시적 utilization, reserve, item 제한
  -> recommend_initial_batch_budget(...)
  -> BatchBudgetRecommendation
```

최종 결과는 다음 조건을 만족해야 한다.

1. CPU와 지정 CUDA 장치의 용량 및 비용을 독립적으로 계산한다.
2. 추천값뿐 아니라 계산 근거를 구조화된 필드로 반환한다.
3. 모델 실행, source 소비, governor/history 상태 변경 없이 결정적으로 동작한다.
4. 알 수 없는 값(`None`)을 0으로 간주하지 않는다.
5. 계산된 item limit이 `min_items`보다 작을 때 최소값으로 강제 상향하지 않고 명시적으로 실패한다.
6. 기존 governor, retry, orchestration, session 및 stable `enn_torch` 동작을 변경하지 않는다.

## 작업 내용

### 1. 모델·옵티마이저 footprint의 장치 provenance 보존

대상 파일:

- `enn_torch_dev/runtime/footprint.py`

`ModelFootprint`와 `OptimizerFootprint`의 마지막 필드로 다음 필드를 append한다.

```python
bytes_by_device: dict[str, int] = field(default_factory=dict)
```

요구사항:

- 기존 positional 필드 순서를 변경하지 않는다.
- `from_module(...)`은 중복 storage 제거 후 parameter와 buffer 바이트를 `str(tensor.device)`별로 집계한다.
- `from_optimizer(...)`은 중복 storage 제거 후 optimizer state tensor 바이트를 장치별로 집계한다.
- `total_model_bytes == sum(bytes_by_device.values())`와 `state_bytes == sum(bytes_by_device.values())`가 probe 생성 결과에서 성립해야 한다.
- 기존 dtype 집계, alias 제거 및 빈 optimizer state 동작을 유지한다.

이 변경이 필요한 이유는 기존 footprint 객체가 총량만 보존하고 CPU/CUDA 위치를 잃기 때문이다. 추천기에서 총량을 임의로 CPU 또는 CUDA에 배정해서는 안 된다.

### 2. 순수 초기 예산 추천 모듈 추가

새 파일:

- `enn_torch_dev/runtime/budget_recommendation.py`

다음 개발 API를 구현한다.

```python
InitialBatchBudgetPolicy
BatchBudgetRecommendation
BatchBudgetRecommendationError
recommend_initial_batch_budget(...)
```

권장 함수 계약:

```python
recommend_initial_batch_budget(
    capacity: ResourceCapacity,
    batch_cost: BatchCost,
    *,
    model_footprint: ModelFootprint | None = None,
    optimizer_footprint: OptimizerFootprint | None = None,
    policy: InitialBatchBudgetPolicy | None = None,
) -> BatchBudgetRecommendation
```

#### `InitialBatchBudgetPolicy`

최소한 다음 필드를 제공한다.

- `min_items: int = 1`
- `max_items: int | None = None`
- `host_utilization_ratio: float = 0.8`
- `device_utilization_ratio: float = 0.8`
- `host_reserve_bytes: int = 0`
- `device_reserve_bytes: int = 0`
- `fallback_max_items: int | None = None`

검증 규칙:

- bool을 정수로 허용하지 않는다.
- item 수는 구성 시 양수여야 한다.
- `min_items <= max_items`를 보장한다.
- utilization ratio는 유한한 수이며 `0 < ratio <= 1`이어야 한다.
- reserve는 음수가 아니어야 한다.
- fallback은 구성할 경우 `min_items` 이상이어야 한다.

#### 장치별 고정 footprint 계산

- `cpu` footprint는 `ResourceCapacity.effective_cpu_bytes`에서 차감한다.
- `cuda` 또는 정확히 일치하는 `cuda:<cuda_device_index>` footprint만 해당 CUDA capacity에서 차감한다.
- 다른 CUDA index, MPS, XPU 등 현재 `ResourceCapacity`로 표현되지 않는 non-zero 장치는 명시적으로 실패한다.
- 총 footprint가 0보다 큰데 `bytes_by_device`가 비어 있으면 장치를 추정하지 말고 실패한다.
- footprint의 장치별 합계와 총계가 다르면 실패한다.

#### 계산식

각 차원에서 다음 순서를 유지한다.

```text
usable bytes
= floor(capacity bytes * utilization ratio)
- reserve bytes
- fixed model/optimizer bytes
```

기준 `BatchCost`를 item 비용으로 바꿀 때 보수적인 ceiling division을 사용한다.

```text
bytes per item = ceil(reference bytes / reference num_items)
items limit = floor(usable bytes / bytes per item)
```

최종 `max_items`는 알려진 host/device limit, 정책 `max_items`, 필요한 경우 fallback 중 최솟값이다.

#### `None`, 0 및 실패 의미

- `None`: 알 수 없음.
- `0`: 측정된 비용이 0이며 해당 차원은 item 수 제한 요인이 아님.
- 양수: item limit 계산 가능.
- 기준 `num_items`가 없거나 0이면 per-item 비용을 추론하지 않는다.
- 관련 차원이 불명확하면 `fallback_max_items`가 있을 때만 fallback을 사용하고 warning을 구조화해 반환한다.
- fallback도 없으면 `BatchBudgetRecommendationError`를 발생시킨다.
- 고정 footprint와 reserve가 usable capacity를 초과하면 실패한다.
- 계산된 item limit이 `min_items`보다 작으면 실패한다. `min_items`로 올려서 실행 가능한 것처럼 보이게 만들지 않는다.
- device 수요가 양수인데 CUDA capacity가 없으면 실패한다.

#### 추천 결과

`BatchBudgetRecommendation`은 최소한 다음 정보를 보존한다.

- `recommended_budget`
- `limiting_dimensions`
- 기준 `num_items`
- effective host 및 device capacity
- host/device fixed footprint
- host/device usable bytes
- host/device bytes per item
- host/device item limit
- fallback 사용 여부
- warning tuple

추천 `BatchBudget`의 host/device byte limit은 utilization, reserve, 고정 footprint를 차감한 뒤의 가변 batch headroom이다. 불명확한 차원의 byte limit을 억지로 넣어 `BudgetedBatcher`가 알 수 없는 비용 필드를 요구하게 만들지 않는다.

### 3. 개발 API export

대상 파일:

- `enn_torch_dev/runtime/__init__.py`

다음을 development namespace에서 export한다.

- `InitialBatchBudgetPolicy`
- `BatchBudgetRecommendation`
- `BatchBudgetRecommendationError`
- `recommend_initial_batch_budget`

`__all__` 중복이 없어야 한다. stable `enn_torch`에는 노출하지 않는다.

### 4. 테스트 추가 및 보완

새 파일:

- `enn_torch_dev/debug/runtime/test_budget_recommendation.py`

수정 파일:

- `enn_torch_dev/debug/runtime/test_model_footprint.py`
- `enn_torch_dev/debug/runtime/test_runtime_integration.py`

최소 검증 사례:

1. CPU-only에서 effective cgroup capacity가 사용되는지.
2. 모델 및 optimizer의 CPU 고정 footprint가 차감되는지.
3. CPU와 CUDA item limit이 독립적으로 계산되고 더 작은 차원이 선택되는지.
4. utilization과 reserve 적용 순서가 정확한지.
5. 기준 batch 총비용을 ceiling division으로 item 비용으로 바꾸는지.
6. 0 비용은 non-limiting으로 처리되는지.
7. 불명확한 차원에서 명시적 fallback이 사용되고 warning이 남는지.
8. fallback이 없을 때 실패하는지.
9. 계산 limit이 `min_items`보다 작을 때 상향 clamp하지 않는지.
10. 고정 footprint가 usable capacity를 넘을 때 실패하는지.
11. CUDA 수요가 있지만 CUDA capacity가 없을 때 실패하는지.
12. non-zero footprint에 device provenance가 없을 때 실패하는지.
13. 다른 CUDA 장치의 footprint를 거부하는지.
14. bool, 음수, 잘못된 ratio, `min_items > max_items` 검증.
15. 기존 footprint alias/dtype/empty-state 동작이 유지되면서 `bytes_by_device`가 정확한지.
16. 신규 API가 `enn_torch_dev.runtime`에는 노출되고 stable `enn_torch`에는 누출되지 않는지.

테스트는 작은 CPU synthetic 값만으로 기본 검증 가능해야 하며 CUDA 하드웨어를 요구하지 않아야 한다.

### 5. 문서 수정

다음 문서를 구현과 일치하도록 같은 변경에서 수정한다.

- `docs/dev_initial_batch_budget.md` 신규
- `docs/CURRENT_STATE.md`
- `docs/runtime_development_workflow.md`
- `docs/TESTING.md`
- `docs/RUNTIME_SAFETY.md`
- `docs/CHANGE_CHECKLIST.md`

문서에서 반드시 구분할 내용:

- 추천기는 현재 구현된 development helper이다.
- 실행 admission, 실제 SPDL pipeline, 모델 probe 실행, activation calibration은 범위 밖이다.
- 추천 결과는 첫 예산의 보수적 시작점이며 실행 가능성의 증명이 아니다.
- unobserved activation, allocator fragmentation, framework overhead, 아직 materialize되지 않은 optimizer state는 정적 추천에 포함되지 않는다.

## 유지해야 하는 기존 동작

- `BatchBudget`, `BudgetedBatcher`의 기존 검증 및 split 동작.
- governor의 OOM, recovered-OOM, 성공 증가, pressure streak 및 dimension-aware shrink 동작.
- orchestrator/session/source factory의 소비·예외·history 계약.
- `ModelFootprint`와 `OptimizerFootprint`의 기존 필드 순서와 기존 집계 의미.
- stable `enn_torch` 공개 API.
- 의존성 및 패키지 설정.

## 변경 금지 범위

다음은 이번 작업에 포함하지 않는다.

- 모델 forward 또는 `RuntimeStep` 자동 실행
- source 열람, 샘플링 또는 소비
- 실제 SPDL pipeline 구성
- pinned memory 또는 device transfer
- pass admission 또는 실행 거부 로직
- governor budget 자동 적용 또는 상태 변경
- retry/OOM 정책 변경
- observed `ModelCost` calibration
- 파일·JSONL·DB 기반 calibration 저장
- checkpoint/resume
- distributed capacity 집계
- AutoGovernor 또는 learned policy
- stable `enn_torch` export
- 새 의존성
- 관련 없는 코드 정리나 리팩터링

## 테스트 방법

먼저 신규 기능의 직접 테스트를 실행한다.

```bash
python -m pytest enn_torch_dev/debug/runtime/test_budget_recommendation.py -q
python -m pytest enn_torch_dev/debug/runtime/test_model_footprint.py -q
```

그다음 API 경계와 기존 런타임 회귀를 검증한다.

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

git status --short
```

예상 결과:

- 신규 추천 및 footprint 테스트가 통과한다.
- 기존 runtime/debug 테스트에 회귀가 없다.
- stable `enn_torch` diff가 없다.
- 의존성 파일 diff가 없다.
- 임시 파일, 캐시, 체크포인트, 산출물이 변경 목록에 포함되지 않는다.

환경 또는 의존성 문제로 전체 테스트를 실행하지 못하면 설치나 환경 변경을 임의로 수행하지 말고, 실행한 명령·실패 원인·미검증 범위를 구분해 보고한다. CUDA 테스트를 실행하지 않았다면 CPU-only 검증과 CUDA 미검증 범위를 명확히 구분한다.

## 완료 보고 형식

완료 보고에는 다음을 포함한다.

1. 실제 변경 파일과 핵심 동작.
2. 중요한 설계 판단, 특히 device provenance와 unknown/fallback 의미.
3. 실제 실행한 테스트 명령과 결과.
4. 실행하지 못한 테스트와 남은 위험.
5. stable `enn_torch`, 의존성, artifact 변경 여부.
6. 아래 형식 중 정확히 하나의 AI 문서 결과.

```text
AI docs updated:
- docs/dev_initial_batch_budget.md
- docs/CURRENT_STATE.md
- docs/runtime_development_workflow.md
- docs/TESTING.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
