precommit: ## add pre-commit hook
	pre-commit install

install: ## Install project deps
	poetry install

format: ## Run Black formatter
	poetry run black --line-length 120 hyperec