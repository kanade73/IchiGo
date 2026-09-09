# IchiGo make contract (docs/spec/05-validation.md §1). M0 implements the CPU targets;
# check-metal/release-check are implemented (T22+/T37); CUDA/integration targets are still
# declared and fail explicitly until their tickets land.
UV ?= uv
SWIFT ?= swift
FIXTURES := Tests/Fixtures
RELEASE_MODEL := models/p4-local-wide512-200k.ichigo

.PHONY: help build check-cpu parity-cpu check-training check-metal parity-metal check-cuda integration release-check fixtures

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

build: ## Build all Swift targets
	$(SWIFT) build

check-cpu: ## Swift Core/Features/LogicModel/Engine/GTP tests + Python pytest (no GPU)
	$(SWIFT) test --filter 'IchiGoCoreTests|IchiGoFeaturesTests|LogicModelTests|IchiGoEngineTests|IchiGoGTPTests'
	cd Training && $(UV) run pytest -q

parity-cpu: ## Regenerate Python hard fixtures and compare Swift scalar backend layer bits + heads
	cd Training && $(UV) run python -m ichigo_train make-fixtures --out ../$(FIXTURES)
	$(SWIFT) test --filter 'LogicModelTests.ParityTests|IchiGoFeaturesTests.SymmetryTests'
	$(SWIFT) build --product ichigo
	cd Training && $(UV) run python ../Scripts/check_eval_parity.py --fixture ../$(FIXTURES)/parity/tiny-9 --ichigo ../.build/debug/ichigo
	cd Training && $(UV) run python ../Scripts/check_eval_parity.py --fixture ../$(FIXTURES)/parity/tiny-19 --ichigo ../.build/debug/ichigo

check-training: ## Python loss/gradcheck/resume/freeze tests (M1: only gate/format tests exist in M0)
	cd Training && $(UV) run pytest -q

fixtures: ## Regenerate shared fixtures only
	cd Training && $(UV) run python -m ichigo_train make-fixtures --out ../$(FIXTURES)

check-metal: ## Metal kernel/loader/lifecycle tests (Mac Metal; skips cleanly with no device)
	$(SWIFT) test --filter 'LogicMetalTests'

parity-metal: ## CPU scalar vs Metal-byte vs Metal-packed parity, both board sizes (regenerates nothing)
	$(SWIFT) test --filter 'LogicMetalTests.MetalParityTests|LogicMetalTests.MetalPackedBackendTests'

check-cuda: ## CUDA forward/backward and DDP tests (T17/T26, university GPUs)
	@echo "check-cuda: not implemented until T17/T26"; exit 1

integration: ## fake teacher / GTP / fake CGOS / clock tests (T12+)
	@echo "integration: not implemented until M1/M2 tickets"; exit 1

release-check: ## CPU + Metal + cgos tests + release build/verify + 2-game hard-model smoke match (T37)
	$(MAKE) check-cpu
	$(SWIFT) build -c release --product ichigo
	BIN="$$($(SWIFT) build -c release --show-bin-path)/ichigo"; "$$BIN" doctor | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['metal']['available'] else 1)" || (echo "release-check: FAILED -- no Metal device detected by 'ichigo doctor'. docs/spec/05-validation.md §1 requires Mac Metal for release-check; refusing to silently pass." >&2; exit 1)
	$(MAKE) check-metal
	cd Training && $(UV) run pytest -q ../Tests/cgos
	DIST=$$(Scripts/release/build.sh $(RELEASE_MODEL)); \
	if [ -z "$$DIST" ]; then echo "release-check: FAILED -- Scripts/release/build.sh produced no dist directory" >&2; exit 1; fi; \
	echo "release-check: dist = $$DIST"; \
	Scripts/release/verify.sh "$$DIST" || { echo "release-check: FAILED -- Scripts/release/verify.sh" >&2; exit 1; }; \
	$(UV) run --project Training python -m ichigo_train match \
		--engine-a "$$DIST/ichigo gtp --model-9 $$DIST/models/$(notdir $(RELEASE_MODEL)) --backend auto" \
		--engine-b uniform --games 2 --size 9 --komi 7 --openings none \
		--out reports/matches/release-check-smoke --seed 20260909 --visits-a 50 \
		|| { echo "release-check: FAILED -- 2-game hard-model smoke match" >&2; exit 1; }; \
	python3 -c "import json,sys; r=json.load(open('reports/matches/release-check-smoke/report.json')); inc=sum(r['incidents'].values()); print('release-check: smoke match incidents:', r['incidents']); sys.exit(0 if inc==0 else 1)" \
		|| { echo "release-check: FAILED -- 2-game smoke match reported non-zero incidents" >&2; exit 1; }; \
	echo "release-check: PASS ($$DIST)"
