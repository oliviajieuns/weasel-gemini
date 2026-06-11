# Gemini Export 데이터 품질 개선 — convert_gemini / prepare_scores

> **대상 브랜치/PR**: `claude/weasel-data-quality-bl94a8` → main 머지 완료 (PR #1, 커밋 `cecb921`)
> **변경 파일**: `weasel/convert_gemini.py`, `weasel/prepare_scores.py`, `scripts/convert_traindata.sh` (+202/−4)
> **검증 데이터**: `dit_task_0504_gpt54-mini_multi_harness_task_14k.jsonl` (13,738 trajectories, 2.2GB)

---

## 1. 배경: 사전 데이터 분석에서 발견된 3가지 이슈

| # | 이슈 | 판정 |
|---|---|---|
| 1 | 최종 답변 없는 trajectory가 **86.7%** | 문제 아님 — WEASEL은 **step 단위** 선택이므로 중간 tool-calling step도 유효한 학습 신호. 필터링하지 않음 |
| 2 | **task_2 빈 goal 그룹 3,456개** | 진짜 위험 — goal 추출 정규식이 Gemini 포맷과 충돌. 무관한 task들이 한 그룹으로 합쳐져 **O(n²) pairwise 연산 폭발 + 선택 왜곡** |
| 3 | 타임스탬프만 다른 중복 변형 + 루핑 trajectory (평균 tool call 20+, 성공 0) | 사전 필터링 필요 |

---

## 2. 원인 분석: "빈 goal 그룹"의 실제 메커니즘

`prepare_scores.py`의 goal 추출 정규식:

```python
GOAL_RE = re.compile(r"##\s*Goal:\s*(.*?)(?=\n##\s|\n#\s|\Z)", re.S)
```

캡처가 **goal 내부의 첫 `\n#`(마크다운 헤더)에서 멈춥니다.** Gemini user 프롬프트는
멀티라인 + 마크다운 헤더를 포함하므로, `## Goal:` 뒤에 원문 그대로 넣으면:

```
## Goal: shared template prefix
# Task: find X        ← 여기서 캡처 중단
```

→ 서로 다른 task(`find X`, `find Y`, ...)가 전부 **공유 템플릿 prefix만 goal로 잘려**
같은 키로 그룹핑됨. 이것이 task_2의 3,456-step mega-group의 정체
(합성 데이터로 재현 확인). 결과적으로:

- **O(n²) 폭발**: pairwise BERTScore가 그룹(segment) 단위라 3,456² ≈ 1,200만 쌍
- **선택 왜곡**: 무관한 task의 step끼리 importance/diversity를 경쟁

즉 "포맷이 안 맞아 데이터가 빈 것"이 아니라 **정규식 절단(truncation) 산물**.

---

## 3. 코드 변경사항

### 3.1 goal 한 줄 평탄화 — 원인 차단 (`convert_gemini.py`)

```python
def normalize_goal(text: str) -> str:
    """Flatten the goal to a single whitespace-normalized line."""
    return re.sub(r"\s+", " ", text or "").strip()

# build_steps 내부
goal = normalize_goal(first_role(messages, "user"))
```

goal에 개행이 없으므로 GOAL_RE가 항상 **전체 goal**을 캡처 → task별 그룹이 정확히 분리됨.

### 3.2 mega-group 방지 안전망 (`prepare_scores.py`)

향후 어떤 포맷 불일치가 와도 거대 그룹이 재발하지 않도록 2차 방어:

```python
goal = extract_goal(user_prompt)
if not goal or goal == "<NO_GOAL_FOUND>":
    # 파싱 불가 goal은 절대 한 그룹으로 합치지 않는다 —
    # 원본 trajectory(_traj_id) 단위로, 없으면 item 단위로 fallback
    tid = item.get("_traj_id") if isinstance(item, dict) else None
    goal = (f"<NO_GOAL_FOUND:traj#{tid}>" if tid is not None
            else f"<NO_GOAL_FOUND:item#{idx}>")
```

추가로 그룹핑 직후 **최대 segment 크기를 로그 출력**해 O(n²) 위험을 조기 감지:

```
Found N distinct goals and M trajectory segments
(largest segment: K steps; pairwise scoring is O(n^2) per segment)
```

### 3.3 사전 필터 — 중복/루핑 제거 (`convert_gemini.py`, 기본 활성화)

**(a) 타임스탬프 중복 변형 dedup**

trajectory 내용에서 타임스탬프 패턴(ISO datetime, 날짜, 시각, unix epoch
2020–2033 범위)을 `<TS>`로 마스킹한 뒤 sha1 시그니처로 비교, 첫 등장만 유지:

```python
TIMESTAMP_RES = (
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"),
    re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?\b"),
    re.compile(r"\b1[6-9]\d{8}(?:\d{3}){0,3}\b"),   # epoch s/ms/us/ns
)

def traj_signature(record) -> str:
    # (role, content, reasoning, tool_calls name+args)를 직렬화 → 타임스탬프 마스킹 → sha1
```

**(b) 루핑 trajectory 필터**

"동일한 (name, arguments) tool call 반복 + 최종 답변 없음"을 루핑으로 정의
(분석에서 본 실패 클러스터: tool 20+, 성공 0). 루프에 빠졌다가 답변에 도달한
trajectory는 보존:

```python
def is_looping(messages, repeat_threshold, max_tool_calls) -> bool:
    if final_answer_text(messages):          # 답변에 도달했으면 루핑 아님
        return False
    if max_repeated_tool_call(messages) >= repeat_threshold:   # 기본 5회
        return True
    if max_tool_calls > 0 and count_tool_calls(messages) > max_tool_calls:
        return True
    return False
```

**(c) 답변 없음 trajectory는 유지 + 카운트만**

```python
if not final_answer_text(messages):
    n_no_answer += 1   # 드랍하지 않음 — step 단위 선택이므로 유효 신호
```

### 3.4 인덱스 정합성 (정확성 핵심)

필터로 드랍된 레코드도 `_traj_id`를 소모하도록 처리 →
`select_trajectories --original-input`의 **원본 파일 인덱스 매핑이 깨지지 않음**
(convert → select → 원본 재추출 end-to-end 테스트로 검증).

### 3.5 CLI 인터페이스

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--keep-duplicates` | off (= dedup ON) | 타임스탬프 중복 dedup 비활성화 |
| `--keep-loops` | off (= 필터 ON) | 루핑 필터 비활성화 |
| `--loop-repeat-threshold` | 5 | 동일 tool call 반복 임계값 (0=비활성) |
| `--max-tool-calls` | 0 (off) | 답변 없는 trajectory의 총 tool call 상한 |

`scripts/convert_traindata.sh`는 `EXTRA_ARGS` 패스스루 지원:
`EXTRA_ARGS="--keep-duplicates --keep-loops" bash scripts/convert_traindata.sh`

---

## 4. 변환 시 레코드 분류 로직 (최종)

```
레코드 (1 trajectory)
 ├─ 첫 user 메시지 없음/빈 값        → 스킵 (skipped_empty) — goal 없음
 ├─ assistant 액션 0개 (출력 없음)    → 스킵 (skipped_empty) — 학습 타깃 없음
 ├─ 타임스탬프 제외 동일 시그니처      → 드랍 (dropped_duplicate)
 ├─ 루핑 (동일 call 5회+ & 답변 없음)  → 드랍 (dropped_loop)
 └─ 그 외                          → 변환 (steps + traj 산출)
      └─ 최종 답변 없음              → 유지하되 no_final_answer 카운트
```

---

## 5. 실데이터 검증 결과 (14k export)

### 5.1 변환 통계 (`gemini_convert_stats.json`)

| 항목 | 수치 | 비고 |
|---|---|---|
| 입력 trajectory | **13,738** | |
| `dropped_duplicate` | 3 | 타임스탬프 중복 |
| `dropped_loop` | 94 | 루핑 (분석의 실패 클러스터) |
| `skipped_empty` | 3,339 | 아래 5.2에서 분해 |
| `no_final_answer` | 12,362 (**90.0%**) | **드랍 아님**, 카운트만 — 사전 분석의 86.7%와 부합 |
| step 레코드 산출 | **53,685** | WEASEL 선택 입력 |
| trajectory 레코드 산출 | **10,302** | 학습용 native-FC |

### 5.2 skipped_empty 3,339개 분해 — 데이터 손실 아님

| 분류 | 개수 | 판정 |
|---|---|---|
| 빈 출력 (user 있음, assistant 액션 0) | 3,178 | 모델이 출력을 안 낸 빈 에피소드 — 학습 타깃이 없어 스킵이 정당 |
| user 턴 없음 (`[system, assistant, tool, ...]`) | 163 | goal 자체가 없는 변칙 포맷 |

- **검산**: 3,178 + 163 = 3,341, 이 중 2개는 중복/루핑 필터에 먼저 걸림
  → `skipped_empty` 3,339 ✓ / convertible 10,397 − 나머지 필터 95 = traj 10,302 ✓
  (13,738 전체가 숫자 단위까지 일치)
- **분포**: 빈 출력 3,178개는 특정 task 집중이 아니라 수백 개 task에 분산
  (task당 최대 10개) → 포맷 불일치가 아닌 생성 실패로 결론

### 5.3 "task_2 빈 goal 그룹 3,456개"의 최종 결론

- 데이터가 비어 있던 것이 아니라 **GOAL_RE 절단으로 무관한 task들이 공유
  prefix 아래 합쳐진 산물**
- goal 한 줄 평탄화로 원천 차단 + `_traj_id` fallback으로 재발 방지
- 최종 확인 지표: `run_select.sh`(prepare_scores) 로그의
  `largest segment` 값이 trajectory 1개 길이 수준(수십 step)이면 해소 완료

---

## 6. 원본 Gemini export 최종 검증 (dit_task_0513, 20,970 traj)

사전 분석의 수치들은 이 파일(`dit_task_0513_gemini_per_line.jsonl`) 기준이었고,
개선 후 변환에서 **전부 그대로 재현**되었다.

### 6.1 변환 통계

| 항목 | 수치 | 사전 분석과 대조 |
|---|---|---|
| 입력 trajectory | **20,970** | |
| `no_final_answer` | 18,202 (**86.8%**) | **"답변 없음 86.7%" 일치** ✓ — 유지, 카운트만 |
| `skipped_empty` | 3,466 | 아래 6.2 — **"빈 goal 그룹 3,456" 일치** ✓ |
| `dropped_duplicate` | 53 | 타임스탬프 중복 (이 export가 해당 이슈의 출처) |
| `dropped_loop` | 156 | 루핑 trajectory |
| step 레코드 산출 | **136,100** | WEASEL 선택 입력 |
| trajectory 레코드 산출 | **17,295** | 20,970 − 53 − 156 − 3,466 ✓ |

### 6.2 "빈 goal 그룹 3,456"의 정체 (확정)

원본 스캔 결과 **user 턴이 아예 없는 trajectory가 정확히 3,456개**
(`[system, assistant, tool, ...]` 구조, goal 부재). 구 파이프라인에서는 이들이
모두 빈 goal로 추출되어 한 그룹에 합쳐졌던 것. 추가로 빈 출력(assistant 액션 0)
26개가 있으며, 3,456 + 26 중 16개는 중복/루핑 필터가 선점 → `skipped_empty`
3,466 ✓ (전체 회계 일치). 둘 다 학습 타깃이 없어 제외가 정당하다.

### 6.3 멀티 rollout 데이터셋과 `--unique-goal`

이 export는 **같은 task를 평균 ~9회 rollout**한 데이터다. 기본(paper-style)
goal-텍스트 그룹핑은 rollout들을 한 segment로 병합해 largest segment가
2,629 step까지 커졌다(O(n²) 재폭발 + segment당 t0=3 예산으로 선택 왜곡).
`EXTRA_ARGS="--unique-goal"`로 변환하면 rollout 1개 = 그룹 1개가 되어
논문의 per-trajectory 예산 의미와 일치한다.

### 6.4 selection 단계 최종 확인 (RTX 4090, GPU 검증)

```
Loaded 136100 datapoints
Found 17295 distinct goals and 17295 trajectory segments
(largest segment: 50 steps; pairwise scoring is O(n^2) per segment)
```

**largest segment 50** — mega-group(3,456 / 2,629) 완전 해소. 처리 속도
~4.1 goal/s, 전체 약 70분 (roberta-large BERTScore, batch 64).

---

## 7. 재현 방법

```bash
# 변환 (필터 기본 적용; 멀티 rollout export는 --unique-goal 필수)
EXTRA_ARGS="--unique-goal" bash scripts/convert_traindata.sh
# 통계 확인
cat $WEASEL_DATA/gemini_convert_stats.json

# 선택 파이프라인 — grouping 로그에서 largest segment 확인
TRAIN_INPUT_JSON=$WEASEL_DATA/gemini_steps.jsonl bash scripts/run_select.sh --gpus 0
```
