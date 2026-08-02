# ENN-PyTorch #830: Bounded Admission Reject Split Recovery

## 기준 상태

작업 저장소:

```text
P-R-N-D/ENN-PyTorch
```

이 작업은 PR #829가 병합된 `main`을 기준으로 한다.
확인 당시 기준 커밋은 다음과 같다.

```text
ab8c60c9866cd67e0420ce53095df836a5d30eda
Add opt-in pre-pass admission gate
```

실제 작업 시작 시 최신 `main`을 다시 확인한다. 최신 코드가 기준 커밋 이후 변경됐다면 무조건 덮어쓰지 말고 현재 구조에 맞게 패치를 조정한다.

입력 패치:

```text
enn-pytorch-830-bounded-admission-split.patch
```

패치와 이 프롬프트 파일은 작업 입력물이다. 최종 Git 변경 사항에 포함하지 않는다.

---

# 작업 목적

현재 opt-in `PrePassAdmissionGate`는 각 원본 candidate, 정적 budget split, OOM retry subbatch를 실행 직전에 평가하고, `REJECT` 또는 차단 대상 `UNKNOWN`이면 `PrePassAdmissionBlocked`를 전파해 pass를 중단한다.

이번 작업의 목적은 **명확한 `REJECT`에 한해서만**, 해당 assessment가 제공하는 `max_admissible_items`를 사용해 candidate를 안전하고 제한적으로 나눈 뒤 각 child를 새 baseline으로 다시 평가하는 것이다.

최종 결과는 다음을 만족해야 한다.

```text
candidate KVBatch
  -> admission REJECT
  -> assessment.max_admissible_items 검증
  -> bounded identity-preserving split
  -> 각 child를 새 ResourceSample로 다시 admission
  -> 허용된 child만 실행
```

이번 작업은 skip이나 데이터 누락 정책이 아니다. 모든 row를 유효한 child에 배치할 수 없으면 기존 block을 그대로 전파한다.

---

# 핵심 안전 계약

## 1. Opt-in 유지

`admission_split_policy=None`이면 PR #829의 기존 동작을 그대로 유지한다.

- `REJECT`: terminal block
- `UNKNOWN + BLOCK`: terminal block
- `UNKNOWN + ALLOW`: 실행
- `ADMIT`: 실행

split은 `AdmissionSplitPolicy`를 명시적으로 설정한 경우에만 활성화한다.

## 2. `REJECT`만 split

다음 조건을 모두 만족할 때만 split한다.

```text
assessment.status == PrePassAdmissionStatus.REJECT
assessment.batch_size == 현재 KVBatch.batch_size
max_admissible_items가 bool이 아닌 양의 정수
max_admissible_items < 현재 batch size
max_admissible_items >= policy.min_items
현재 admission split depth < policy.max_split_depth
필요한 child 수 <= policy.max_split_parts
모든 row를 유효한 child에 배치 가능
```

다음은 절대 split하지 않는다.

- `UNKNOWN`
- `ADMIT`
- `max_admissible_items is None`
- limit 0 또는 음수
- bool 또는 non-integer limit
- 현재 batch와 같거나 큰 limit
- assessment의 `batch_size`가 현재 candidate와 불일치
- target이 `min_items` 미만
- 필요한 child 수가 `max_split_parts` 초과
- `max_split_depth` 소진
- 모든 row를 `min_items <= child_size <= target`으로 나눌 수 없는 경우

이 경우 원래 `PrePassAdmissionBlocked`를 그대로 전파한다.

## 3. Split 크기는 assessment가 결정

다음 값을 authoritative target으로 사용한다.

```text
assessment.max_admissible_items
```

다음을 하지 않는다.

- 임의의 절반 분할
- `RetryPolicy.split_factor` 재사용
- unknown 값을 fallback target으로 변환
- target보다 큰 child 실행
- 부족한 child를 `min_items`까지 강제로 올리기

필요한 최소 part 수는 다음과 같다.

```text
part_count = ceil(batch_size / target)
```

그 part 수에 row를 가능한 균등하게 분배한다.

예:

```text
batch_size=10, target=3, min_items=2
-> 3, 3, 2, 2
```

예:

```text
batch_size=5, target=2, min_items=2
```

모든 row를 `[2, 2]` 안에 넣을 수 없으므로 terminal block이어야 한다. `2, 3`처럼 target을 넘는 child를 만들지 않는다.

## 4. Identity와 순서 보존

기존 `slice_kvbatch(...)`를 사용한다.

다음을 보존한다.

- row 순서
- `row_ids`
- `source_ids`
- `sample_ids`
- `schema_id`
- `shard_id`
- 기존 materialization 경계

전체 성공 결과의 row identity를 연결하면 원본 candidate 순서와 정확히 일치해야 한다.

## 5. Admission depth와 OOM retry depth 분리

`RuntimeRetryRunner` 내부에 두 독립 카운터를 유지한다.

```text
admission rejection -> admission_split_depth + 1
runtime OOM result   -> retry_count + 1
```

- admission split은 모델 실행 전에 발생하므로 OOM retry depth를 소비하지 않는다.
- OOM split은 admission depth를 소비하지 않는다.
- admission child가 이후 OOM retry를 사용할 수 있어야 한다.
- OOM retry subbatch가 실행 전에 admission split을 사용할 수 있어야 한다.

## 6. Optimizer 동작 유지

Admission split은 실행 전 복구이므로 optimizer가 있는 step에서도 허용한다.

반면 모델 실행 후 발생한 OOM retry는 기존 동작을 유지한다.

```text
optimizer is not None -> 기존 OOM retry 제한 유지
```

admission split을 추가하면서 optimizer 신호를 잃거나 OOM retry를 새로 허용하면 안 된다.

## 7. Assessment 기록 순서

현재 `_AdmissionRuntimeStep`은 gate가 assessment를 반환할 때만 기록한다.
이를 수정해 `PrePassAdmissionBlocked`가 발생한 경우에도 `blocked.assessment`를 먼저 기록하고 다시 raise한다.

성공적으로 복구된 pass에서는 다음 순서가 보존되어야 한다.

```text
REJECT parent
ADMIT child 1
ADMIT child 2
...
```

`RuntimePassResult.admission_assessments`는 admission 시도 순서를 기록하며, 최종 `StepResult` 수보다 많을 수 있다.

Terminal block에서는 여전히 `RuntimePassResult`를 만들지 않는다.

## 8. Traceback 보존 계약

복구 가능한 내부 `PrePassAdmissionBlocked`는 외부로 전파하지 않는다.

- assessment와 ranges를 추출한다.
- child recursion 전에 해당 내부 예외의 `__traceback__`을 `None`으로 설정한다.
- `except` 블록 밖에서 child recursion을 시작해 내부 block이 새 child 예외의 context로 연결되지 않게 한다.

Terminal block은 정상 traceback을 유지한 채 전파한다.

다음을 주장하면 안 된다.

- 모든 exception object가 assessment만 참조한다.
- terminal traceback이 raw runtime 객체를 참조하지 않는다.
- gate가 transactional rollback을 제공한다.

## 9. Governor 동작

이번 작업에서는 admission recovery를 governor feedback으로 사용하지 않는다.

성공적으로 split된 pass에서 governor는 최종 `StepResult`와 기존 OOM/pressure 신호만 관찰한다.

다음을 추가하지 않는다.

- recovered admission reject에 따른 budget shrink
- success growth suppression
- admission 기반 next-budget 추천
- summary/history admission 집계

Terminal block에서는 기존과 같이:

- `observe_results()` 호출 없음
- governor state 변경 없음
- partial `RuntimePassResult` 없음

## 10. Capacity와 sampling

- `ResourceCapacityProvider.capacity()`는 pass당 한 번만 호출한다.
- resolved capacity는 pass 내에서 고정한다.
- parent와 모든 child는 각각 fresh `sample("before_admission")`을 사용한다.
- child admission을 parent baseline으로 재사용하지 않는다.

---

# 구현 내용

## 1. `AdmissionSplitPolicy`

대상:

```text
enn_torch_dev/runtime/admission_gate.py
```

다음 frozen/slots dataclass를 추가한다.

```python
@dataclass(frozen=True, slots=True)
class AdmissionSplitPolicy:
    max_split_depth: int = 3
    min_items: int = 1
    max_split_parts: int = 16
```

검증:

- `max_split_depth`: bool 제외 integer, 0 이상
- `min_items`: bool 제외 integer, 양수
- `max_split_parts`: bool 제외 integer, 2 이상

## 2. Runtime retry runner 연결

대상:

```text
enn_torch_dev/runtime/retry.py
```

`RuntimeRetryRunner.__init__`에 append-only keyword option을 추가한다.

```python
admission_split_policy: AdmissionSplitPolicy | None = None
```

`PrePassAdmissionBlocked`를 catch해 recoverability를 계산한다.

OOM retry와 admission recovery의 recursion counter를 분리한다.

권장 private helper:

```python
_admission_split_ranges(
    batch_size,
    assessment,
    *,
    admission_split_depth,
)
```

helper는 recoverability를 충족하지 않으면 빈 tuple을 반환하고, runner는 원래 exception을 다시 raise한다.

## 3. Orchestration 연결

대상:

```text
enn_torch_dev/runtime/orchestration.py
```

`ConservativeRuntimeOrchestrator.__init__`에 추가한다.

```python
admission_split_policy: AdmissionSplitPolicy | None = None
```

검증:

- 값은 `AdmissionSplitPolicy` 또는 `None`
- profile 없이 split policy만 설정하면 구성 오류

`RuntimeRetryRunner` 생성 시 policy를 전달한다.

`_AdmissionRuntimeStep.run(...)`은 block assessment도 기록해야 한다.

## 4. Development export

대상:

```text
enn_torch_dev/runtime/__init__.py
```

다음을 export한다.

```text
AdmissionSplitPolicy
```

stable `enn_torch`에는 노출하지 않는다.

## 5. 테스트

신규 중심 테스트:

```text
enn_torch_dev/debug/runtime/test_prepass_admission_split.py
```

필요하면 기존 retry/orchestration/integration 테스트를 최소 범위에서 보완한다.

최소 검증 항목:

1. policy 미설정 시 기존 terminal block
2. assessment target 기반 split
3. 균형 분할 및 target 이하 보장
4. `UNKNOWN` 미분할
5. `None`, 0, 음수, bool, non-reducing target 미분할
6. assessment batch-size mismatch 미분할
7. `min_items`로 모든 row를 덮을 수 없는 경우 block
8. `max_split_parts` 적용
9. `max_split_depth` 적용
10. recovered internal exception traceback 정리
11. terminal child traceback 유지
12. admission split 후 OOM retry
13. OOM retry child의 admission split
14. admission/OOM depth 독립
15. optimizer가 admission split은 막지 않음
16. optimizer가 기존 OOM retry는 계속 제한
17. parent `REJECT` 후 child `ADMIT` assessment 순서
18. child별 fresh sample
19. capacity provider pass당 한 번
20. BudgetedBatcher split 후 admission split
21. row/source/sample identity 및 순서 보존
22. terminal block에서 governor 불변
23. recovered split이 governor fault feedback을 만들지 않음
24. profile 없는 split policy 구성 오류
25. invalid split policy 타입 거부
26. development-only API export
27. `UNKNOWN + ALLOW`가 split 경로를 사용하지 않음
28. gate 비활성 기존 회귀

## 6. 문서

다음을 추가한다.

```text
docs/dev_prepass_admission_split.md
```

다음을 필요한 범위에서 갱신한다.

```text
docs/dev_prepass_admission_gate.md
docs/CURRENT_STATE.md
docs/runtime_development_workflow.md
docs/TESTING.md
docs/RUNTIME_SAFETY.md
docs/CHANGE_CHECKLIST.md
```

문서에는 반드시 다음을 명시한다.

- split은 opt-in
- `REJECT`만 대상
- assessment limit이 authoritative
- `UNKNOWN` 미분할
- depth/parts/min-items bound
- OOM와 admission depth 독립
- optimizer 계약 유지
- parent/child assessment 순서
- recovered internal traceback만 정리
- governor/summary/history feedback 없음
- skip/replay/rollback 제외

---

# 유지해야 하는 기존 동작

- `assess_prepass_admission(...)` 계산식과 status precedence
- known-zero와 unknown 구분
- CUDA provenance 검증
- `AdmissionUnknownAction` 동작
- gate 미설정 및 split policy 미설정 동작
- BudgetedBatcher 정적 budget split
- OOM retry 대상 phase
- optimizer-backed OOM retry 제한
- retry-recovered OOM governor 신호
- pressure assessment 및 governor 동작
- row identity/order
- fixed/provider capacity mutual exclusion
- pass당 capacity provider 1회 호출
- stable `enn_torch` namespace

---

# 변경 금지 범위

별도 결함이 확인되지 않는 한 다음 파일은 수정하지 않는다.

```text
enn_torch_dev/runtime/admission.py
enn_torch_dev/runtime/batching.py
enn_torch_dev/runtime/step.py
enn_torch_dev/runtime/governor.py
enn_torch_dev/runtime/pressure.py
enn_torch_dev/runtime/summary.py
enn_torch_dev/runtime/history.py
enn_torch_dev/runtime/session.py
```

다음을 추가하지 않는다.

- skip-and-continue
- skipped-row `StepResult` 또는 새 `StepStatus`
- source replay
- transactional rollback
- admission-based governor feedback
- summary/history admission aggregation
- automatic profile update 또는 persistence
- batch마다 capacity refresh
- heuristic/learned split target
- multi-GPU/distributed coordination
- stable API 변경
- 새 dependency
- 관련 없는 리팩터링

브랜치 생성, commit, push, PR 생성은 사용자가 별도로 요청하지 않는 한 수행하지 않는다.

---

# 테스트 방법

실제 저장소 환경에서 다음을 실행한다.

```bash
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_split.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission_gate.py -q
python -m pytest enn_torch_dev/debug/runtime/test_prepass_admission.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_retry.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_orchestration.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_capacity_provider.py -q
python -m pytest enn_torch_dev/debug/runtime/test_runtime_integration.py -q
python -m pytest enn_torch_dev/debug/runtime -q
python -m pytest enn_torch_dev/debug -q
git diff --check
```

CUDA 환경 확인:

```bash
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_current_device={torch.cuda.current_device()}")
PY
```

CUDA가 없으면 synthetic dataclass 테스트로 검증한 범위와 실제 CUDA 미검증 범위를 분리해 보고한다.

## 범위 확인

```bash
test -z "$(git diff -- enn_torch)"
test -z "$(git diff -- pyproject.toml requirements.txt requirements-dev.txt)"
git diff -- enn_torch_dev/runtime/admission.py
git diff -- enn_torch_dev/runtime/batching.py
git diff -- enn_torch_dev/runtime/step.py
git diff -- enn_torch_dev/runtime/governor.py
git diff -- enn_torch_dev/runtime/pressure.py
git diff -- enn_torch_dev/runtime/summary.py
git diff -- enn_torch_dev/runtime/history.py
git diff -- enn_torch_dev/runtime/session.py
git status --short --branch
```

---

# 완료 보고 형식

다음을 구분해 보고한다.

## 변경 내용

- 실제 수정 파일
- admission split eligibility와 계산 방식
- OOM retry와의 상호작용
- assessment 기록 순서
- 문서 변경

## 실제로 실행한 검증

- 명령별 pass/fail
- 테스트 수
- skip 및 warning
- CUDA 가용 여부
- `git diff --check`
- stable package와 dependency manifest 무변경 확인
- 금지 파일 무변경 확인
- working tree 상태

## 실행하지 못한 검증

- 실행하지 못한 테스트
- 이유
- 실제 CUDA 미검증 여부
- 결과에 미치는 영향

최종 보고에는 정확히 하나의 AI docs block을 포함한다.

```text
AI docs updated:
- docs/dev_prepass_admission_split.md
- docs/dev_prepass_admission_gate.md
- docs/CURRENT_STATE.md
- docs/runtime_development_workflow.md
- docs/TESTING.md
- docs/RUNTIME_SAFETY.md
- docs/CHANGE_CHECKLIST.md
```
