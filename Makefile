COMPOSE_FILE=infra/docker-compose.yml

build:
	docker compose -f $(COMPOSE_FILE) build

rebuild:
	docker compose -f $(COMPOSE_FILE) build --no-cache

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

debug:
	docker compose -f $(COMPOSE_FILE) run --rm preprocessor debug $(ARGS)

input:
	docker compose -f $(COMPOSE_FILE) run --rm preprocessor input $(ARGS)

label:
	docker compose -f $(COMPOSE_FILE) run --rm preprocessor label $(ARGS)

train:
	docker compose -f $(COMPOSE_FILE) run --rm trainer train $(ARGS)

inference:
	docker compose -f $(COMPOSE_FILE) run --rm trainer inference $(ARGS)