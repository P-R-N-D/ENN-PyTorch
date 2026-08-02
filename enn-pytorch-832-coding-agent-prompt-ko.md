# ENN-PyTorch #832 Admission Recovery Growth Guard 적용 및 검증

## 작업 목적

PR #831이 병합된 ENN-PyTorch `main`을 기준으로, bounded admission split으로 복구된 pass가 곧바로 일반적인 clean-success 성장 근거로 누적되지 않도록 **opt-in governor growth guard**를 추가한다.

현재 admission recovery가 발생해도 최종 child `StepResult`가 모두 성공하면 governor는 해당 pass를 일반 성공으로 계산할 수 있다. 이로 인해 직전 candidate가 admission `REJECT`되어 더 작은 child로 분할됐음에도 success streak가 증가하거나 threshold에 도달해 다음 budget이 다시 성장할 수 있다.

이번 작업의 최종 결과는 다음과 같아야 한다.

- 기본 설정에서는 기존 governor 동작이 완전히 유지된다.
- 명시적으로 정책을 켠 경우에만 recovered admission `REJECT`가 있었던 성공 pass의 success streak를 0으로 재설정하고 clean-success budget growth를 억제한다.
- admission recovery limit은 구조화된 decision/summary/history provenance로 남긴다.
- yielded OOM, retry-recovered OOM, pressure shrink, non-OOM fault의 기존 우선순위와 동작은 변경하지 않는다.
- recovered limit으로 `max_items`를 직접 cap하거나 shrink하지 않는다.

기준 커밋:

```text
224bba9288569cf4b948615c650f2233de07e265
```

제공 패치:

```text
enn-pytorch-832-admission-growth-guard.patch
```

## 작업 내용

### 1. 패치 적용 전 확인

현재 브랜치가 위 기준 커밋 또는 이를 포함하는 최신 `main`에서 시작했는지 확인한다.

```bash
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor 224bba9288569cf4b948615c650f2233de07e265 HEAD
```

사용자 변경이 있으면 삭제하거나 되돌리지 않는다. 패치와 충돌하는 경우 현재 코드 구조에 맞게 같은 계약을 수동 반영하되, 무관한 리팩터링은 하지 않는다.

패치 적용:

```bash
git apply --check enn-pytorch-832-admission-growth-guard.patch
git apply enn-pytorch-832-admission-growth-guard.patch
```

패치와 이 프롬프트가 저장소 작업 트리에 복사되어 있다면 최종 PR에는 포함하지 않는다.

### 2. GovernorPolicy

대상:

```text
enn_torch_dev/runtime/governor.py
```

기존 필드 뒤에 다음 opt-in 필드를 append-only로 추가한다.

```python
suppress_growth_after_admission_recovery: bool = False
```

계약:

- 기본값 `False`는 기존 성장 동작을 그대로 유지한다.
- bool이 아닌 값은 `TypeError`로 거부한다.
- 기존 positional field 순서를 변경하지 않는다.

### 3. GovernorDecision 및 governor 입력

`GovernorDecision` 마지막에 다음 필드를 append-only로 추가한다.

```python
admission_recovery_max_items: int | None = None
growth_suppressed_by_admission_recovery: bool = False
```

`ConservativeRuntimeGovernor.observe_results(...)`에 keyword-only 입력을 추가한다.

```python
admission_recovery_max_items: int | None = None
```

입력 계약:

- `None`: recovered admission reject 없음
- bool을 제외한 양의 정수: 완료된 pass에서 관찰된 최소 recovered `max_admissible_items`
- 0, 음수, bool, float, 문자열 등은 기존 positive-int 검증 규칙에 맞게 거부

정책이 꺼져 있어도 전달받은 limit은 `GovernorDecision.admission_recovery_max_items`에 provenance로 기록한다.

### 4. Growth guard 동작

다음 조건을 모두 만족할 때만 guard를 활성화한다.

```text
관찰 결과가 하나 이상 존재
모든 최종 결과가 SUCCESS
yielded OOM 없음
retry-recovered OOM 없음
GovernorPolicy.suppress_growth_after_admission_recovery == True
admission_recovery_max_items != None
```

활성화 시:

```text
consecutive_successes = 0
growth_suppressed_by_admission_recovery = True
```

clean-success growth로 `next_budget`이 증가한 경우에는 `previous_budget`으로 되돌린다.

다음은 유지한다.

- pressure shrink로 실제 감소한 budget은 되돌리지 않는다.
- pressure growth suppression과 admission growth suppression은 동시에 `True`일 수 있다.
- pressure high/trigger streak, selected fields, applied factors, actual shrink fields를 변경하지 않는다.
- OOM 또는 retry-recovered OOM은 admission guard보다 우선한다.
- non-OOM fault와 empty result는 기존 동작을 유지하며 admission suppression flag를 세우지 않는다.

Reason에는 admission recovery에 의해 success-streak growth를 억제했다는 사실과 recovered max-items limit을 포함하되, 기존 pressure 또는 OOM reason을 잘못 덮어쓰지 않는다.

### 5. Orchestrator 연결

대상:

```text
enn_torch_dev/runtime/orchestration.py
```

완료된 `admission_step.assessments`에서 `REJECT` assessment의 valid positive reducing `max_admissible_items`를 수집하고 최솟값을 계산한다.

```text
recovered REJECT 없음 -> None
한 개 이상 -> minimum target
```

이 값을 governor에 전달한다.

```python
decision = governor.observe_results(
    results,
    recovered_oom=recovered_oom,
    pressure_summary=pressure_summary,
    admission_recovery_max_items=admission_recovery_max_items,
)
```

유지해야 할 경계:

- allowed `UNKNOWN`만 있는 pass는 recovery limit을 만들지 않는다.
- terminal block은 기존처럼 governor 호출 전에 전파된다.
- trusted preflight recovery 이외의 public exception을 recovery로 추론하지 않는다.
- admission, retry, batching 또는 runtime step 동작을 변경하지 않는다.

### 6. Summary 및 history provenance

대상:

```text
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
```

`RuntimePassSummary` 마지막에 다음을 append한다.

```python
growth_suppressed_by_admission_recovery: bool = False
governor_admission_recovery_max_items: int | None = None
```

구분:

- `minimum_recovered_admissible_items`: pass assessment에서 계산한 실행 증거
- `governor_admission_recovery_max_items`: governor가 실제 입력으로 받은 값
- `growth_suppressed_by_admission_recovery`: 정책이 실제로 success growth를 억제했는지

`RuntimeHistorySummary` 마지막에 다음을 append한다.

```python
admission_growth_suppressed_passes: int = 0
```

현재 retained window에서 suppression flag가 참인 summary만 센다. Trimming으로 제거된 summary의 기여가 남지 않아야 한다.

Formatter에 다음 provenance를 포함한다.

```text
growth_suppressed_by_admission_recovery
governor_admission_recovery_max_items
admission_growth_suppressed_passes
latest_growth_suppressed_by_admission_recovery
latest_governor_admission_recovery_max_items
```

기존 raw-assessment 비보존 계약을 유지한다.

### 7. 테스트

신규 집중 테스트:

```text
enn_torch_dev/debug/runtime/test_admission_growth_guard.py
```

최소 검증 사항:

- 정책과 decision 필드 append-only 순서
- 정책 bool 검증
- governor admission limit 입력 검증
- 기본값에서 기존 growth 유지
- opt-in recovery 시 success streak reset 및 threshold growth 억제
- 이후 clean success가 streak를 처음부터 다시 누적
- yielded OOM과 retry-recovered OOM 우선순위
- pressure suppression과 admission suppression 동시 provenance
- pressure shrink가 admission guard로 취소되지 않음
- fault와 empty path 회귀
- nested orchestrator recovery에서 최소 target 전달
- allowed `UNKNOWN`만 있는 경우 guard 미활성
- terminal child block에서 governor 미호출
- retained-window history suppression count
- stable `enn_torch` namespace 무변경

기존 field-order compatibility 테스트는 새 append-only 필드만 반영한다. 기존 테스트 의미를 약화하거나 삭제하지 않는다.

### 8. 문서

다음 문서를 구현과 일치시킨다.

```text
docs/dev_admission_governor_growth_guard.md
docs/dev_admission_observability.md
docs/dev_runtime_governor.md
docs/dev_runtime_summary.md
docs/dev_runtime_history.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
docs/CURRENT_STATE.md
```

반드시 명시할 사항:

- 기능은 opt-in이다.
- admission recovery는 OOM이나 pressure shrink가 아니다.
- 이번 범위는 clean-success growth guard뿐이다.
- recovered limit으로 `max_items`를 직접 cap하거나 새 budget 필드를 만들지 않는다.
- lower-bound 충돌, cap 해제 및 재성장 정책은 후속 범위다.

## 유지해야 하는 기존 동작

- `suppress_growth_after_admission_recovery=False`일 때 기존 governor 결과
- yielded/retry-recovered OOM shrink
- pressure growth suppression 및 sustained-pressure shrink
- CPU/CUDA 독립 pressure streak
- non-OOM fault 및 empty result 처리
- optimizer, admission split, retry, batching 및 execution 계약
- summary/history의 bounded retention과 raw-object 비보존
- stable `enn_torch` 공개 API

## 변경 금지 범위

별도 오류가 확인되지 않는 한 다음 파일은 수정하지 않는다.

```text
enn_torch_dev/runtime/admission.py
enn_torch_dev/runtime/admission_gate.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/step.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/session.py
```

다음을 추가하지 않는다.

- admission limit을 이용한 직접 `max_items` cap 또는 shrink
- `max_items=None`에서 새 item budget 생성
- governor `min_items`와 admission limit의 충돌 해결
- admission recovery streak
- skip-and-continue 또는 skipped-row record
- source replay 또는 rollback
- profile 자동 갱신·저장
- persistent telemetry
- stable API
- 새 dependency
- 관련 없는 리팩터링

## 테스트 방법

집중 테스트:

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_growth_guard.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_governor.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

전체 회귀:

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

범위 검사:

```bash
test -z "$(git diff -- enn_torch)"
test -z "$(git diff -- pyproject.toml requirements.txt requirements-dev.txt)"
test -z "$(git diff -- enn_torch_dev/runtime/admission.py)"
test -z "$(git diff -- enn_torch_dev/runtime/admission_gate.py)"
test -z "$(git diff -- enn_torch_dev/runtime/retry.py)"
test -z "$(git diff -- enn_torch_dev/runtime/batching.py)"
test -z "$(git diff -- enn_torch_dev/runtime/step.py)"
test -z "$(git diff -- enn_torch_dev/runtime/pressure.py)"
test -z "$(git diff -- enn_torch_dev/runtime/session.py)"
git status --short --branch
```

## 완료 보고 형식

다음을 구분해 보고한다.

### 실제로 실행한 검증

- 실행 명령
- 각 테스트의 pass/skip/warning 수
- 사용한 Python, PyTorch, CUDA 환경
- 실패했다면 오류와 수정 내용

### 실행하지 못한 검증

- 실행하지 못한 테스트
- 이유
- 미검증이 결과에 미치는 영향

### 범위 확인

- stable package 무변경
- dependency manifest 무변경
- 금지된 runtime 파일 무변경
- 임시 patch/prompt artifact가 최종 PR tree에 남지 않음
- working tree 상태

최종 보고에는 업데이트한 AI 문서 목록을 정확히 포함한다.
