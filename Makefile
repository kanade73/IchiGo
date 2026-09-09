# IchiGo make contract (docs/spec/05-validation.md §1). M0 implements the CPU targets;
# Metal/CUDA/integration targets are declared and fail explicitly until their tickets land.
UV ?= uv
SWIFT ?= swift
FIXTURES := Tests/Fixtures

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

release-check: ## CPU + Metal + integration + hard-model smoke (T37)
	@echo "release-check: not implemented until T37"; exit 1
