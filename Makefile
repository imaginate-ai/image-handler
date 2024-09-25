.PHONY: local-setup
local-setup:
	@echo Creating virtual environment
	@poetry shell
	@$(MAKE) install

.PHONY: install
install:
	@echo Installing dependencies
	@poetry install --sync

.PHONY: lint
lint:
	@echo Linting code
	@poetry run pre-commit run -a

.PHONY: test
test:
	@echo Running tests
	@poetry run pytest -v

.PHONY: generate
generate:
	@echo Generating images
	@poetry run python image_handler/scripts/generate.py

.PHONY: health_check_fix
health_check_fix:
	@echo Checking database health
	@poetry run python image_handler/scripts/health_check.py --fix


.PHONY: health_check
health_check:
	@echo Checking database health
	@poetry run python image_handler/scripts/health_check.py