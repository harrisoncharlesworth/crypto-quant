# Agent Instructions for Crypto Quant Project

## Security Rules (CRITICAL)

- **NEVER commit secrets to git**: API keys, private keys, passwords, tokens
- Always use `.env` files for secrets and ensure they're in `.gitignore`
- Check for hardcoded secrets before any commit
- Use environment variables for all sensitive configuration

## Sub-Agent Management

- When spawning sub-agents, create `subagents.md` file to document the plan
- Include task breakdown, dependencies, and expected deliverables
- Update `subagents.md` as tasks progress

## Python Development Standards

### Code Quality
- **Formatter**: Use `black` for code formatting
- **Linter**: Use `ruff` for linting and code quality
- **Commands**:
  - Format: `black src/ scripts/ notebooks/`
  - Lint: `ruff check src/ scripts/`
  - Fix lint issues: `ruff check --fix src/ scripts/`

### Testing
- **Framework**: pytest with pytest-asyncio for async tests
- **Commands**:
  - Run tests: `pytest`
  - Run with coverage: `pytest --cov=src`
  - Run async tests: `pytest -v tests/`

### Project Structure
- Source code: `src/quantbot/`
- Scripts: `scripts/`
- Research notebooks: `notebooks/`
- Documentation: `docs/`

### Dependencies
- Use `requirements.txt` for dependency management
- Key libraries: pandas, numpy, ccxt (crypto exchange), ta (technical analysis)
- Async libraries: aiohttp, aiofiles, aiosmtplib

### Environment Setup
- Use `python-dotenv` for environment variable loading
- Use `pydantic` for configuration validation
- Create `.env` from `.env.example` template

### Crypto-Specific Guidelines
- Always use proper error handling for exchange API calls
- Implement rate limiting for API requests
- Use proper async patterns for concurrent operations
- Validate all market data before processing
- Log trading decisions and portfolio changes

### Data Handling
- Use pandas for data manipulation
- Store historical data in appropriate formats (CSV, Parquet)
- Implement proper data validation with pydantic models
- Handle timezone-aware datetime objects for market data

## Verification Commands

Before completing any task, run:
1. `black src/ scripts/` - Format code
2. `ruff check src/ scripts/` - Check for issues
3. `pytest` - Run tests
4. Check no secrets are committed: `git diff --cached`
