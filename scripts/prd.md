# NSPO2 제품 요구사항 문서 (PRD)

> Null Space Preference Optimization v2 — 분산을 줄이되 ‘편향 관리’를 일급 시민으로 설계한 실사용 가능한 모듈

---

## 0. 요약 (TL;DR)

* **문제**: 정책 경사/선호최적화 학습에서 높은 분산으로 인한 느린/불안정 수렴.
* **핵심 아이디어**: 고분산 방향을 제거하는 **Null Space 투영(Projection)** 을 안정적으로 운영 환경에 붙일 수 있게 모듈화.
* **NSPO2 목표**: 분산≥25% 감소(기준: GRPO), 초기/중기 수렴 속도 ≥30% 개선, 최종 품질 저하(편향) ≤1.5%p 내로 관리.
* **주요 차별점**: (1) **Adaptive Rank** + **Trust Region NSPO** 로 편향 상한, (2) **Strategy Auto-Select**(Noise-Aware / Hybrid / Knowledge-Preserving), (3) **Bias Sentinel** 로 자동 페일오버(후반부 GRPO 재전환).

---

## 1. 배경 & 문제 정의

* 정책 경사 기반 방법은 미니배치/보상 적분/탐색 노이즈 등으로 **gradient 분산**이 높음 → 학습 불안정·느림.
* NSPO는 **공분산의 큰 고유값 방향을 제거**하여 분산을 낮추지만, **편향을 도입**할 수 있음. NSPO2는 이 트레이드오프를 **정량 관리**하고 **운영 안전장치**를 제공하는 제품화 버전.

---

## 2. 사용자 스토리

1. **RL 연구자/엔지니어**: “초기 수렴을 빨리 보고 싶다. 하이퍼 크게 만지지 않고, 분산이 낮아졌는지 바로 확인하고 싶다.”
2. **LLM 파인튜너**: “선호 데이터가 제각각이라 분산이 크다. 품질 저하 없이 안정성만 높이고 싶다.”
3. **플랫폼 오너**: “실패 모드(과도한 편향)에서 자동으로 롤백되길 원한다. 실험/대시보드도 표준화되면 좋겠다.”

---

## 3. 목표/비목표

### 3.1 성공 지표 (Acceptance Criteria)

* **분산 감소율**: GRPO 대비 **≥25%** (평균 배치 기준)
* **수렴 속도**: 동일 에폭에서 **Loss/Reward 개선 ≥30%**
* **최종 품질 저하 상한**: GRPO 대비 **절대 1.5%p 이내** (task-specific metric)
* **오버헤드**: 학습 스텝당 계산 **≤5%** 증가
* **안전장치**: Bias Sentinel이 이상 감지 시 **자동 전환**(NSPO→GRPO) 및 알림

### 3.2 비목표 (Out of Scope)

* 완전한 수렴 보장 증명 제공
* 분산 감소와 무관한 업스트림 데이터 정제/라벨링 파이프라인 구축

---

## 4. 핵심 개념 (요약)

* **Null Space Projection**: 공분산의 큰 고유값 방향(고분산)을 기저 **K**로 잡고, **P = I − K(KᵀK)⁻¹Kᵀ**로 투영하여 그 방향 성분 제거.
* **전략군**:

  * **Noise-Aware**(분산 최대 감소),
  * **Knowledge-Preserving**(중요 방향 보존, 편향 낮춤),
  * **Hybrid**(균형)
* **편향-분산 트레이드오프**: 편향 증가를 허용하되 **MSE = Bias² + Var**를 낮추는 게 목표.

---

## 5. 요구사항 (Functional)

1. **ProjectionHead v2**

   * 입력: 현재 배치 gradient/특징 표현(백엔드에 따라 vectorized).
   * 출력: 투영된 gradient.
   * 기능: (a) 공분산 추정, (b) 고유값/벡터 계산, (c) P 구성, (d) 전략별 선택적 투영.
2. **Adaptive Rank Selection**

   * 목표 바이어스 상한(target\_bias), 분산 목표치, 최근 히스토리 기반 rank 자동 탐색.
3. **Trust Region NSPO**

   * 투영 전후 거리‖Pg−g‖가 임계(trust\_radius) 초과 시 **혼합 업데이트**(αPg+(1−α)g).
4. **Bias Sentinel**

   * 편향 지표(정렬도/코사인/오류항), 최종 품질 프록시(밸리데이션 리워드, 과도한 rank 증가 신호) 모니터링.
   * 임계 초과 시: 경고 → rank 감소 → 전략 전환 → 최종적으로 **NSPO Off / GRPO Back**.
5. **VarianceTracker v2 & 대시보드**

   * 원/투영 분산, 감소율(%), 고유값 상위 r 시리즈, 수렴 곡선, 편향 추정치를 시간축으로 시각화.
6. **Auto Strategy Selector**

   * 태스크/학습 단계별 프리셋(Noise-Aware/Hybrid/Knowledge-Preserving)과 휴리스틱 스위칭.
7. **플러그형 인터페이스**

   * GRPO, PPO 기반 루프에 drop-in으로 삽입 가능(미니배치 훅, gradient hook, optimizer wrapper).

---

## 6. 비기능 요구사항 (NFR)

* **성능**: step 시간 +≤5% (d=활성 파라미터 차원, r≤32 기본).
* **안정성**: 수치 정규화(εI), 대각 안정화, 고유값 컷오프.
* **재현성**: 시드·버전·하이퍼 로그 자동 기록.
* **관측성**: 메트릭/아티팩트(그림, CSV)를 실험 폴더에 자동 저장.
* **호환성**: PyTorch 2.x, 혼합정밀/분산데이터병렬(DDP) 대응.

---

## 7. 시스템/아키텍처

```
Trainer (GRPO/PPO/...)
 └─ OptimizerWrapper
     └─ NSPO2.Engine
         ├─ CovEstimator (streaming, EWMA, shrinkage)
         ├─ EigSolver (top-k, randomized SVD)
         ├─ ProjectionHead (P build / apply)
         ├─ Strategy (NoiseAware | Hybrid | KnowledgePreserving)
         ├─ AdaptiveRank (online selection)
         ├─ TrustRegion (mixing alpha)
         ├─ BiasSentinel (guard, fallback)
         └─ VarianceTracker (metrics, plots)
```

### 7.1 데이터 플로우

1. 배치 gradient 수집 → 2) 공분산 업데이트 → 3) top‑k 고유벡터 → 4) P 구성 → 5) g↦Pg 적용 → 6) 신뢰영역 체크 → 7) Optimizer step → 8) 모니터링 기록.

---

## 8. API 설계 (초안)

```python
class NSPO2Config(TypedDict):
    dim: int                 # 특징/gradient 차원
    strategy: Literal['noise', 'hybrid', 'keep']
    projection_rank: int     # r (기본 16)
    update_freq: int         # k step마다 고유벡터 업데이트
    trust_radius: float      # ‖Pg-g‖ 상한
    target_bias: float       # 편향 추정 상한(코사인/정렬도 기준)
    max_rank: int            # r 상한 (기본 32)
    min_rank: int            # r 하한 (기본 0)
    epsilon: float           # 수치 안정화 항
    seed: int

class NSPO2:
    def __init__(self, cfg: NSPO2Config): ...
    def hook(self, grad: torch.Tensor) -> torch.Tensor: ...    # g → Pg
    def step_end(self, metrics: Dict[str, float]) -> None: ... # 히스토리 업데이트
    def state_dict(self) -> Dict: ...
    def load_state_dict(self, sd: Dict) -> None: ...
```

---

## 9. 알고리즘 설계 디테일

### 9.1 공분산 추정

* **Streaming**: 미니배치 단위로 **EMA/EWMA** 추정.
* **Shrunk** 공분산: $\Sigma_λ=(1-λ)\hat\Sigma + λ·\mathrm{diag}(\hat\Sigma)$ 로 수치 안정성.

### 9.2 고유값/벡터 계산

* 차원 큰 경우 **randomized power iteration** 로 top‑k 근사.
* r, 업데이트 주기(update\_freq)는 비용/안정성 트레이드오프.

### 9.3 투영 행렬 구성

* $P = I - K(K^T K)^{-1}K^T$, K는 정규직교화(Gram‑Schmidt) 후 역조건수 완화.

### 9.4 전략별 마스킹

* **Noise-Aware**: 상위 r 고유벡터 제거.
* **Knowledge-Preserving**: 기준 방향(직전 평균경사/성능 기여 벡터)을 **보존 제약**으로 추가.
* **Hybrid**: r을 절반씩 분할(노이즈 제거 + 보존 집합).

### 9.5 Adaptive Rank

* 최근 W 스텝의 (분산감소율, 편향추정, 검증지표)로 **목적함수** $J(r)=\widehat{Var}_{red}(r) - \gamma·\widehat{Bias}(r)$ 최대화.
* 제약: $\widehat{Bias}(r) ≤ target\_bias$, 실패 시 r 감소.

### 9.6 Trust Region 혼합 업데이트

* if ‖Pg−g‖>τ: $g' = α·Pg + (1−α)·g$, α∈(0,1) 자동 조정.

### 9.7 Bias Sentinel 규칙

* 코사인 정렬도↓, 검증 리워드 하락, 고유값 꼬리 급변 시 알람→ 단계적 완화(전략 전환→r 감소→NSPO off).

---

## 10. 실험 설계

### 10.1 벤치마크

* **RL**: CartPole, Atari 최소 2개, 로봇 연속제어 1개.
* **LLM**: 선호최적화 소규모 세트(예: 톤/사실성 혼합 데이터)로 GRPO vs NSPO2 비교.

### 10.2 메트릭

* 분산(원/투영), 감소율%, 수렴 단계 수, 최종 품질(리워드/정확도/인간평가), 오버헤드.

### 10.3 프로토콜

* 동일 시드 5회 평균, 유의성 검정.
* **학습 단계 스케줄**: 초·중기 NSPO2 on → 후기 NSPO2 off (또는 r→0).

### 10.4 어블레이션

* r ∈ {0,4,8,16,32}, update\_freq ∈ {5,10,20}, 전략군 3종, TrustRegion on/off, AdaptiveRank on/off.

### 10.5 대시보드 산출물

* eigenvalue 스펙트럼, variance\_evolution, convergence\_comparison, bias\_proxy 타임라인.

---

## 11. 운영 & 관측성

* **로그**: 각 step의 orig\_var/proj\_var/reduction, r, ‖Pg−g‖, 전략, α, sentinel flags.
* **알림**: 임계 초과 시 슬랙/메일 이벤트.
* **아티팩트**: PNG/CSV 자동 저장, 실험 폴더 규약.

---

## 12. 하이퍼파라미터 가이드 (초기값)

* projection\_rank: 16 (최대 32)
* update\_freq: 5–10
* trust\_radius: 0.1·‖g‖ 평균 기준
* target\_bias (cosine drop): 0.05
* strategy: RL=Noise-Aware, LLM=Knowledge-Preserving, 일반=Hybrid
* epsilon: 1e−6

---

## 13. 에지 케이스 & 트러블슈팅

* **메모리 부족**: dim 축소, r 축소, 주기 늘리기.
* **수치 불안정**: 공분산 대각에 εI 추가, 정규직교화 재시도.
* **수렴 실패**: lr↓, r↓, update\_freq↑, TR 혼합 강도↑.

---

## 14. 위험요인 & 완화

* **과도한 편향**: Sentinel+TR+후기 GRPO 전환.
* **탐색 저해**: 초기에는 TR 완화(α 낮춤), 주기적 NSPO off 스텝 삽입.
* **계산비용**: r 제한, randomized SVD, 업데이트 주기 조절.

---

## 15. 마일스톤

* **M1(주1)**: 프로토타입(NSPO2.Engine/Hook/Tracker) + 단위테스트
* **M2(주2)**: AdaptiveRank/TrustRegion/BiasSentinel
* **M3(주3)**: RL/LLM 벤치마크 결과 & 대시보드
* **M4(주4)**: 문서화/가이드/예제 노트북, 내부 베타

---