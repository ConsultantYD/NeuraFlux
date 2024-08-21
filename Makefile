install:
	poetry install

format:
	poetry run ruff format neuraflux
	poetry run ruff format tests

lint:
	poetry run ruff check neuraflux
	poetry run ruff check tests

security_scan:
	poetry run bandit -r neuraflux -s B403,B301,B311

test:
	poetry run pytest tests