# ENN-PyTorch #831 Admission Summary/History Observability 작업

## 기준 정보

- 저장소: `P-R-N-D/ENN-PyTorch`
- 기준 브랜치: `main`
- 기준 커밋: `79774464c6c479e82bbdbe097f755b277f743c60`
- 입력 패치: `enn-pytorch-831-admission-observability.patch`
- 이 작업은 병합된 #830 이후의 후속 작업이다.
- 이전 대화나 작업 설명을 알고 있다고 가정하지 말고 아래 요구사항만 기준으로 구현·검증한다.

## 작업 목적

완료된 `RuntimePassResult.admission_assessments`에는 admission 실행 시도별
`ADMIT`, bounded split으로 복구된 parent `REJECT`, 명시적으로 허용된
`UNKNOWN`이 순서대로 들어 있다. 현재 `RuntimePassSummary`와
`RuntimePassHistory`는 이 정보를 버리므로, 장기 보존에 안전한 scalar
provenance로 축약해 pass summary와 bounded retained-window history에서 확인할
수 있도록 한다.

최종 결과는 다음을 만족해야 한다.

1. 완료된 pass의 admission 평가 및 복구 횟수를 summary에서 확인할 수 있다.
2. bounded history가 현재 retained window 안에서 admission provenance를 집계한다.
3. summary/history는 raw assessment, exception, batch, tensor, sample 등의 객체를
   보존하지 않는다.
4. admission provenance는 governor, retry, orchestration, gate, session 실행 동작을
   변경하지 않는다.
5. terminal admission block은 여전히 `RuntimePassResult`를 만들지 않으므로
   summary/history에 추가되지 않는다.
6. stable `enn_torch` API와 dependency manifest는 변경하지 않는다.

## 작업 내용

### 1. 패치 적용과 기준 확인

먼저 현재 checkout이 기준 커밋 또는 그 후속 `main`과 호환되는지 확인한다.

```bash
git status --short --branch
git rev-parse HEAD
git apply --check enn-pytorch-831-admission-observability.patch
git apply enn-pytorch-831-admission-observability.patch
```

기준 이후의 사용자 변경이 이미 존재하면 되돌리지 않는다. 패치가 clean하게
적용되지 않으면 현재 코드 구조에 맞게 동일한 요구사항을 최소 범위로 수동
반영하되, 관련 없는 리팩터링을 하지 않는다.

입력 patch와 이 prompt 파일은 구현 참고용 산출물이다. 사용자가 별도로
요청하지 않은 이상 최종 PR 변경 목록에 포함하지 않는다.

### 2. `RuntimePassSummary` append-only admission 필드

대상:

```text
enn_torch_dev/runtime/summary.py
```

`RuntimePassSummary`의 기존 필드 뒤에 다음 기본값 필드를 append-only로
추가한다.

```python
admission_assessment_count: int = 0
admission_admit_assessment_count: int = 0
admission_recovered_reject_count: int = 0
admission_allowed_unknown_count: int = 0
admission_recovery_occurred: bool = False
minimum_recovered_admissible_items: int | None = None
```

기존 필드의 이름, 순서, 타입 또는 기본값을 변경하지 않는다.

### 3. `summarize_runtime_pass(...)` admission 축약

`RuntimePassResult.admission_assessments`를 순회해 다음 값을 계산한다.

- 전체 assessment 수
- `ADMIT` 수
- 완료된 pass에 포함된 `REJECT` 수
- 완료된 pass에 포함된 `UNKNOWN` 수
- recovered reject 존재 여부
- recovered reject들의 positive reducing `max_admissible_items` 최솟값

계약:

- 각 원소는 `PrePassAdmissionAssessment`여야 한다. 아니면 `TypeError`.
- `ADMIT`은 admit count에 포함한다.
- `UNKNOWN`은 completed pass에서 명시적으로 허용된 unknown으로 계산한다.
- `REJECT`는 terminal block이 아니라 completed recovered parent를 의미한다.
- completed-pass `REJECT`의 `batch_size`는 bool이 아닌 양의 정수여야 한다.
- `max_admissible_items`는 bool이 아닌 양의 정수여야 한다.
- `max_admissible_items < batch_size`여야 한다.
- 위 reject 계약이 깨진 수동 생성 `RuntimePassResult`는 `ValueError`로 거부한다.
- 알 수 없는 status 객체를 조용히 무시하지 않는다.

summary에는 다음 객체를 저장하지 않는다.

- `PrePassAdmissionAssessment`
- admission dimension 또는 warning tuple
- `PrePassAdmissionBlocked` 또는 private split request
- `KVBatch`, source, tensor, `ResourceSample`, store, loss

### 4. pass formatter

`format_runtime_pass_summary(...)`에 다음 라인을 추가한다.

```text
admission_assessment_count=...
admission_admit_assessment_count=...
admission_recovered_reject_count=...
admission_allowed_unknown_count=...
admission_recovery_occurred=...
minimum_recovered_admissible_items=...
```

최소 recovered limit이 없을 때는 `unknown`으로 표시한다.

기존 formatter 필드와 순서를 불필요하게 재구성하지 않는다.

### 5. `RuntimeHistorySummary` append-only admission 필드

대상:

```text
enn_torch_dev/runtime/history.py
```

기존 필드 뒤에 다음 기본값 필드를 append-only로 추가한다.

```python
admission_assessed_passes: int = 0
admission_recovery_passes: int = 0
admission_total_assessments: int = 0
admission_admit_assessments: int = 0
admission_recovered_rejects: int = 0
admission_allowed_unknowns: int = 0
minimum_recovered_admissible_items: int | None = None
```

### 6. retained-window history 집계

`RuntimePassHistory.summarize()`는 현재 `_records`에 남아 있는 summary만 사용해
다음을 계산한다.

- assessment가 하나 이상인 pass 수
- recovery가 발생한 pass 수
- total/admit/recovered-reject/allowed-unknown assessment 합계
- retained window의 minimum recovered admissible items

기존 `_trim_records()` 이후 전체 retained records를 재집계하는 방식을 유지한다.
삭제된 summary의 admission 기여가 남으면 안 된다.

한 pass에 recovered reject가 여러 개 있어도:

- recovery pass count는 1 증가
- recovered reject count는 실제 개수만큼 증가

### 7. history formatter

`format_runtime_history_summary(...)`에 다음을 추가한다.

```text
admission_assessed_passes=...
admission_recovery_passes=...
admission_total_assessments=...
admission_admit_assessments=...
admission_recovered_rejects=...
admission_allowed_unknowns=...
minimum_recovered_admissible_items=...
latest_admission_recovery_occurred=...
latest_admission_recovered_reject_count=...
```

최소 limit이 없으면 `unknown`으로 표시한다.

### 8. governor 및 실행 경계 유지

다음 파일과 동작은 변경하지 않는다.

```text
enn_torch_dev/runtime/admission.py
enn_torch_dev/runtime/admission_gate.py
enn_torch_dev/runtime/retry.py
enn_torch_dev/runtime/orchestration.py
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/session.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/step.py
```

특히 다음을 추가하지 않는다.

- recovered admission reject에 따른 budget shrink
- admission 발생 시 success growth suppression
- 다음 pass `max_items` 자동 cap
- 새로운 governor streak 또는 decision reason
- skip-and-continue 또는 skipped-row record
- source replay, rollback 또는 재처리
- raw assessment history retention
- persistent telemetry/export
- stable API
- 새 dependency

`RuntimePassResult`도 현재 `admission_assessments`로 충분하므로 변경하지 않는다.

### 9. 테스트

신규 테스트:

```text
enn_torch_dev/debug/runtime/test_admission_observability.py
```

최소 검증 항목:

1. gate 비활성 pass와 빈 pass의 admission 기본값
2. `ADMIT`, recovered `REJECT`, allowed `UNKNOWN` 개별 count
3. nested/multiple recovered reject count와 최소 limit
4. recovered OOM과 admission provenance 동시 보존
5. non-assessment 원소 `TypeError`
6. `None`, 0, 음수, bool, non-reducing reject target `ValueError`
7. pass formatter 출력
8. summary가 raw assessment 객체를 보존하지 않음
9. history pass count와 assessment count 구분
10. 한 pass의 복수 reject 집계
11. allowed unknown history 집계
12. retained-window minimum limit
13. `max_records` trim 후 폐기된 admission 기여 제거
14. history formatter와 latest recovery 필드
15. 실제 orchestrator bounded split 결과가 summary/history로 전달됨
16. `ConservativeRuntimeSession`이 summary/history provenance를 전달함
17. terminal block pass가 history에 추가되지 않음
18. 기존 pressure/OOM summary/history 필드 회귀
19. append-only dataclass field order
20. stable `enn_torch` namespace 무변경

기존 호환성 테스트의 마지막 필드 assertion을 새 append-only admission 필드를
포함하도록 갱신한다.

```text
enn_torch_dev/debug/runtime/test_runtime_summary.py
enn_torch_dev/debug/runtime/test_runtime_history.py
```

### 10. 문서

패치에 포함된 다음 AI-facing 문서를 현재 구현과 일치하도록 반영한다.

```text
docs/dev_admission_observability.md
docs/dev_runtime_summary.md
docs/dev_runtime_history.md
docs/dev_prepass_admission_gate.md
docs/dev_prepass_admission_split.md
docs/CURRENT_STATE.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

이전 문서에 남아 있는 “recovered admission은 summary/history에 전달하지 않는다”는
설명은 제거하거나 다음 계약으로 바꾼다.

- governor에는 전달하지 않음
- summary/history에는 bounded scalar observability만 전달
- raw assessment와 runtime 객체는 보존하지 않음

## 테스트 방법

### Targeted tests

```bash
python -m pytest enn_torch_dev/debug/runtime/test_admission_observability.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_summary.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_history.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
```

### Broader regression

```bash
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

### Scope checks

```bash
test -z "$(git diff -- enn_torch)"
test -z "$(git diff -- pyproject.toml requirements.txt requirements-dev.txt)"
test -z "$(git diff -- enn_torch_dev/runtime/admission.py)"
test -z "$(git diff -- enn_torch_dev/runtime/admission_gate.py)"
test -z "$(git diff -- enn_torch_dev/runtime/retry.py)"
test -z "$(git diff -- enn_torch_dev/runtime/orchestration.py)"
test -z "$(git diff -- enn_torch_dev/runtime/governor.py)"
test -z "$(git diff -- enn_torch_dev/runtime/pressure.py)"
test -z "$(git diff -- enn_torch_dev/runtime/session.py)"
test -z "$(git diff -- enn_torch_dev/runtime/batching.py)"
test -z "$(git diff -- enn_torch_dev/runtime/step.py)"
git status --short --branch
```

### CUDA 확인

```bash
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY
```

이번 변경은 scalar summary/history 로직이므로 실제 CUDA 장치 실행은 필수 구현
요건이 아니다. 다만 CUDA를 실행하지 못한 경우 해당 사실을 명확히 보고한다.

## 완료 보고 형식

다음을 구분해서 보고한다.

### 변경 내용

- 수정한 코드 파일
- 추가·수정한 테스트
- 수정한 문서
- 변경하지 않은 runtime 경계

### 실제로 실행한 검증

- 실행한 명령
- passed/failed/skipped/warning 수
- 중요한 오류 또는 경고
- CUDA availability

### 실행하지 못한 검증

- 실행하지 못한 테스트
- 이유
- 결과에 미치는 영향

### 범위 확인

- stable `enn_torch` 변경 여부
- dependency manifest 변경 여부
- 금지된 runtime 파일 변경 여부
- working tree 상태

최종 보고에는 다음 두 형식 중 정확히 하나를 포함한다.

```text
AI docs updated:
- <실제로 수정한 문서>
```

또는 문서 영향이 전혀 없을 때만:

```text
AI docs impact: none
Reason: <구체적인 이유>
```

## 금지 사항

사용자가 별도로 요청하지 않는 한 다음을 수행하지 않는다.

- 관련 없는 기능 추가 또는 리팩터링
- 기존 사용자 변경 삭제·되돌리기
- 새 dependency 추가
- stable API 변경
- checkpoint/file format 변경
- 브랜치 생성
- 커밋
- push
- PR 생성 또는 수정
- 실행하지 않은 테스트를 성공했다고 표현
