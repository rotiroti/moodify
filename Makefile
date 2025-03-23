# Makefile for Moodify Project

.DEFAULT_GOAL := help

# Variables
PACKAGE_MANAGER := $(shell command -v uv || echo pip)
APP_ENTRY := app.py

# Phony targets
.PHONY: help clean lint format run all

##@ Moodify Commands

help: ## Show this help message
	@echo "Moodify - A Multimodal Emotion Recognition Application"
	@echo
	@echo "Usage:"
	@echo "  make <target>"
	@echo
	@echo "Available Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

clean: ## Remove Gradio and Python cache files
	@find . -type f -name "*.py[co]" -exec rm -f {} +
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@rm -rf .gradio .cache .ruff_cache .mypy_cache .pytest_cache
	@rm -rf notebooks/.ipynb_checkpoints notebooks/cache
	@rm -rf build dist *.egg-info

run: ## Run the Moodify application
	@echo "Running Moodify..."
	@$(PACKAGE_MANAGER) run python $(APP_ENTRY)

##@ Code Quality

lint: ## Check code for linting issues using ruff
	$(PACKAGE_MANAGER) run ruff check . --fix

format: ## Format code using black
	$(PACKAGE_MANAGER) run black .


##@ Combined Tasks

all: clean lint format run ## Run all tasks
