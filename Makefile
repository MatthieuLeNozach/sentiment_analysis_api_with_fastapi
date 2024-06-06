generate-pwd-hash:
	bash run.sh generate:pwd-hash

transform-init-sql:
	bash run.sh transform:init-sql

build-api:
	bash run.sh build:api

build-inference:
	bash run.sh build:inference

build-migrate:
	bash run.sh build:migrate

build-all:
	bash run.sh build:all

push-api:
	bash run.sh push:api

push-inference:
	bash run.sh push:inference

push-migrate:
	bash run.sh push:migrate

push-all:
	bash run.sh push:all


lint:
	bash run.sh lint


purge-pycache:
	bash run.sh purge:pycache

