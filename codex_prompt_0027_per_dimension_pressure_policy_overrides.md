# Codex 작업 프롬프트: CPU/CUDA별 sustained-pressure 정책 override

## 작업 목적

ENN-PyTorch의 `enn_torch_dev.runtime.ConservativeRuntimeGovernor`는 PR #821에서 CPU와 CUDA의 sustained-pressure streak를 독립적으로 추적하게 되었다. 그러나 현재 두 차원은 여전히 다음 공통 정책 값을 공유한다.

- `GovernorPolicy.min_pressure_ratio_for_shrink`
- `GovernorPolicy.shrink_after_pressure_passes`

CPU RSS 압력과 CUDA 메모리 압력은 서로 다른 임계값과 지속 pass 수가 필요할 수 있다. 이번 작업에서는 기존 공통 정책을 호환 fallback으로 유지하면서 CPU/CUDA별 threshold와 required-pass override를 추가한다.

최종 결과는 다음과 같아야 한다.

- override가 없으면 기존 동작과 완전히 동일하다.
- CPU와 CUDA는 각자의 effective shrink threshold와 required pass count를 사용한다.
- 한 차원의 override만으로 그 차원만 sustained-pressure shrink를 활성화할 수 있다.
- CPU와 CUDA가 서로 다른 pass에서 독립적으로 trigger될 수 있다.
- 기존 OOM, dimension-aware budget mapping, legacy streak compatibility, summary/history, stable `enn_torch` 계약은 유지된다.

## 기준 브랜치와 작업 방식

저장소:

```text
P-R-N-D/ENN-PyTorch
```

최신 `main`에는 PR #821 merge commit이 포함되어 있어야 한다.

```text
f0bac2e39e6f3d16bd68f7d6515595c18103dd48
Track sustained pressure by resource dimension
```

작업 전에 다음을 확인한다.

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
git merge-base --is-ancestor f0bac2e39e6f3d16bd68f7d6515595c18103dd48 HEAD
git status --short --branch
```

새 feature branch를 생성한다.

```bash
git checkout -b codex/per-dimension-pressure-policy-overrides
```

`main`에 직접 커밋하지 않는다. 기존 사용자 변경을 삭제하거나 되돌리지 않는다.

## 패치 적용

제공된 패치 파일:

```text
apply-patch-0027_per_dimension_pressure_policy_overrides.diff
```

적용 전 검사:

```bash
git apply --check apply-patch-0027_per_dimension_pressure_policy_overrides.diff
```

적용:

```bash
git apply apply-patch-0027_per_dimension_pressure_policy_overrides.diff
```

패치가 최신 main과 충돌하면 임의로 우회하거나 범위를 넓히지 말고, 실제 최신 코드와 아래 계약을 기준으로 최소한만 조정한다.

## 작업 내용

### 1. `GovernorPolicy` 필드 추가

수정 파일:

```text
enn_torch_dev/runtime/governor.py
```

기존 필드 뒤에 다음 필드를 append한다.

```python
min_cpu_pressure_ratio_for_shrink: float | None = None
min_cuda_pressure_ratio_for_shrink: float | None = None
cpu_shrink_after_pressure_passes: int | None = None
cuda_shrink_after_pressure_passes: int | None = None
```

기존 필드를 삭제하거나 순서를 변경하지 않는다.

```python
min_pressure_ratio_for_shrink
shrink_after_pressure_passes
```

기존 positional construction이 깨지면 안 된다.

### 2. Validation

dimension threshold는 기존 `_validate_optional_pressure_ratio(...)`와 동일한 계약을 사용한다.

- `None` 또는 finite real number
- bool 금지
- `0 < value <= 1`

dimension required-pass override는 기존 optional positive integer validation을 사용한다.

- `None` 또는 positive integer
- bool 금지
- 0과 음수 금지

effective threshold:

```python
cpu_threshold = (
    min_cpu_pressure_ratio_for_shrink
    if min_cpu_pressure_ratio_for_shrink is not None
    else min_pressure_ratio_for_shrink
)

cuda_threshold = (
    min_cuda_pressure_ratio_for_shrink
    if min_cuda_pressure_ratio_for_shrink is not None
    else min_pressure_ratio_for_shrink
)
```

effective required pass count:

```python
cpu_required = (
    cpu_shrink_after_pressure_passes
    if cpu_shrink_after_pressure_passes is not None
    else shrink_after_pressure_passes
)

cuda_required = (
    cuda_shrink_after_pressure_passes
    if cuda_shrink_after_pressure_passes is not None
    else shrink_after_pressure_passes
)
```

`max_pressure_ratio_for_growth`가 설정되어 있으면 모든 활성 effective shrink threshold에 대해 다음을 만족해야 한다.

```text
max_pressure_ratio_for_growth <= effective dimension shrink threshold
```

CPU 또는 CUDA effective threshold가 `None`이면 해당 차원은 이 validation 비교에서 제외한다.

dimension required-pass override가 설정됐지만 해당 effective threshold가 `None`인 경우에는 오류로 만들지 않는다. 해당 pass-count 값은 threshold가 활성화될 때만 사용되는 inert configuration으로 둔다.

### 3. Dimension별 판정

`ConservativeRuntimeGovernor.observe_results(...)`에서 CPU와 CUDA의 effective policy를 별도로 계산한다.

- CPU high:
  - CPU effective threshold가 설정됨
  - `peak_cpu_rss_ratio`가 알려짐
  - ratio가 CPU threshold 이상
- CUDA high:
  - CUDA effective threshold가 설정됨
  - allocated/reserved/max-allocated/max-reserved ratio 중 하나 이상이 threshold 이상

CPU threshold가 활성화되고 CUDA threshold가 비활성화된 경우, CUDA ratio가 높더라도 CUDA streak와 CUDA pressure shrink를 활성화하면 안 된다.

CUDA threshold만 활성화된 경우도 대칭적으로 처리한다.

### 4. Dimension별 trigger

CPU trigger:

```text
cpu_pressure_high
and consecutive_cpu_pressure_passes >= cpu_required
```

CUDA trigger:

```text
cuda_pressure_high
and consecutive_cuda_pressure_passes >= cuda_required
```

예:

```text
CPU threshold = 0.80
CUDA threshold = 0.95
CPU required = 2
CUDA required = 3
```

두 차원이 계속 high라면:

```text
pass 1:
CPU streak 1
CUDA streak 1
shrink 없음

pass 2:
CPU threshold 도달
max_host_bytes만 shrink
CPU streak 0
CUDA streak 2 유지

pass 3:
CUDA threshold 도달
max_device_bytes만 shrink
CUDA streak 0
CPU streak 1 유지
```

한 차원의 trigger가 다른 차원의 미완료 streak를 초기화하면 안 된다.

### 5. 기존 budget mapping 유지

trigger된 CPU:

```text
max_host_bytes
```

trigger된 CUDA:

```text
max_device_bytes
```

trigger된 차원에 matching byte budget이 하나라도 있으면 `max_items`를 추가 축소하지 않는다.

trigger된 모든 차원에 matching byte budget이 없을 때만 `max_items`를 fallback으로 사용한다.

`pressure_shrunk_budget_fields`에는 minimum bounds 적용 후 실제 값이 변경된 필드만 기록한다.

### 6. Legacy streak와 OOM 동작 유지

PR #821의 legacy aggregate 계약을 변경하지 않는다.

- dimension streak가 둘 다 0
- legacy global streak가 양수
- 현재 high 차원이 정확히 하나

인 경우에만 해당 차원에 legacy streak를 승계한다.

CPU/CUDA가 동시에 high이면 legacy aggregate를 어느 차원에도 승계하지 않는다.

yielded OOM과 retry-recovered OOM은 계속:

- 모든 configured budget field를 generic OOM path로 shrink
- CPU/CUDA pressure streak 둘 다 reset
- `pressure_shrunk_budget_fields == ()`

를 유지한다.

### 7. Decision reason

공통 threshold와 공통 required-pass 수가 하나뿐인 것처럼 표현하면 안 된다.

trigger reason에는 기존 핵심 fragment를 유지한다.

```text
triggered dimensions: ...
current triggered ratios: ...
```

추가로 trigger된 각 차원의 effective policy를 기록한다.

예:

```text
triggered policies: cpu(limit=0.8, required=2)
```

또는:

```text
triggered policies: cpu(limit=0.8, required=2), cuda(limit=0.95, required=3)
```

trigger되지 않은 차원은 triggered-policy와 triggered-ratio 부분에 포함하지 않는다.

threshold 도달 전 progress reason은 CPU/CUDA 각각의 다음 값을 구분해 표시한다.

- current streak / effective required pass count
- effective threshold
- current ratio 또는 `unknown`

global `max_observed_ratio`를 dimension별 threshold나 지속 ratio처럼 표현하지 않는다.

### 8. 테스트

수정 파일:

```text
enn_torch_dev/debug/runtime/test_runtime_governor.py
enn_torch_dev/debug/runtime/test_runtime_integration.py
```

필수 검증:

1. 신규 `GovernorPolicy` 필드는 기존 필드 뒤에 append됨
2. 기존 positional construction 유지
3. 신규 필드 기본값은 모두 `None`
4. CPU threshold override만으로 CPU shrink 활성화 가능
5. global threshold가 `None`이면 비활성 CUDA ratio가 높아도 CUDA streak 증가 없음
6. CPU 2-pass, CUDA 3-pass 정책이 각각 독립적으로 trigger됨
7. CPU trigger 후 CUDA streak 2 유지
8. 다음 pass에서 CUDA trigger 후 CPU streak 1 유지
9. CPU high / CUDA가 자체 threshold 미만이면 CPU streak만 증가
10. override가 없으면 공통 threshold/pass count로 기존과 동일하게 동작
11. 신규 ratio override의 0, 1 초과, bool 입력 거부
12. 신규 required-pass override의 0, bool 입력 거부
13. growth limit가 CPU effective threshold보다 높으면 거부
14. growth limit가 CUDA effective threshold보다 높으면 거부
15. 기존 dimension-aware mapping, max-items fallback, minimum-bound, OOM, legacy state 테스트 유지

integration test에서는 기존 provider-pressure session 경로 중 하나를 CPU-only override로 실행해 실제 orchestration/session 경로에서도 override가 적용되는지 확인한다.

### 9. 문서

수정 대상:

```text
docs/dev_runtime_governor.md
docs/runtime_development_workflow.md
docs/CURRENT_STATE.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

문서에 다음을 반영한다.

- CPU/CUDA threshold와 required-pass override
- unset override는 common policy로 fallback
- common과 dimension threshold가 모두 `None`이면 해당 차원 비활성
- growth limit는 모든 활성 effective shrink threshold 이하
- trigger reason은 effective policy와 triggered ratio를 차원별로 기록
- per-dimension shrink factor는 여전히 out of scope

## 유지해야 하는 기존 동작

- `max_pressure_ratio_for_growth`의 summary-wide growth suppression
- 공통 sustained-pressure 정책 fallback
- independent CPU/CUDA streak
- legacy global streak compatibility
- OOM/recovered-OOM priority
- dimension-aware budget field mapping
- `max_items` fallback
- minimum/maximum budget bounds
- actual changed-field reporting
- summary와 history 구조
- stable `enn_torch` namespace

## 변경 금지 범위

다음 파일과 영역은 수정하지 않는다.

```text
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/resources.py
enn_torch_dev/runtime/capacity_provider.py
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/__init__.py
enn_torch/**
pyproject.toml
requirements*.txt
lock 파일
```

다음 작업도 수행하지 않는다.

- per-dimension shrink factor 추가
- 새로운 dependency 추가
- API export 변경
- summary/history에 effective policy 필드 추가
- 관련 없는 리팩터링
- checkpoint 또는 파일 형식 변경
- 자동 merge

## 테스트 방법

targeted tests:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
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
git diff -- enn_torch_dev/runtime/governor.py
git diff -- enn_torch_dev/debug/runtime/test_runtime_governor.py
git diff -- enn_torch_dev/debug/runtime/test_runtime_integration.py
git diff -- docs/dev_runtime_governor.md
git diff -- docs/runtime_development_workflow.md
git diff -- docs/CURRENT_STATE.md
git diff -- docs/RUNTIME_SAFETY.md
git diff -- docs/CHANGE_CHECKLIST.md

git diff -- enn_torch_dev/runtime/summary.py
git diff -- enn_torch_dev/runtime/history.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch
git status --short
```

금지 경로의 `git diff` 출력은 없어야 한다.

## 전달 artifact 정리

패치와 작업 프롬프트가 저장소 root에 업로드된 경우 commit 전에 삭제한다.

```bash
rm -f apply-patch-0027_per_dimension_pressure_policy_overrides.diff
rm -f codex_prompt_0027_per_dimension_pressure_policy_overrides.md
git status --short
```

## 커밋과 PR

커밋 메시지:

```text
Add per-dimension pressure policy overrides
```

브랜치를 push한다.

```bash
git push -u origin codex/per-dimension-pressure-policy-overrides
```

PR 제목:

```text
Add per-dimension sustained-pressure policy overrides
```

PR 본문에는 다음을 포함한다.

- motivation
- 구현 계약
- common fallback과 dimension override 규칙
- validation 규칙
- 변경 파일
- 실행한 각 테스트 명령과 정확한 passed/skipped/warning 수
- 실제 GitHub head SHA
- AI docs updated 목록
- GitHub Actions가 없다면 테스트가 로컬 실행 결과임을 명시

PR을 자동 merge하지 않는다.

## 최종 보고 형식

다음 내용을 구분해서 보고한다.

### 구현

- 실제 변경 파일
- effective threshold/pass-count 규칙
- 호환성 유지 내용
- 금지 범위가 변경되지 않았는지

### 테스트

- 실행 명령
- 각 명령의 정확한 결과
- warning과 skipped 이유
- 실행하지 못한 검증이 있다면 이유

### Git

- branch
- commit SHA
- 실제 GitHub PR head SHA
- PR 번호와 URL
- merge하지 않았음을 명시

### AI 문서

```text
AI docs updated:
- docs/dev_runtime_governor.md
- docs/runtime_development_workflow.md
- docs/CURRENT_STATE.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
