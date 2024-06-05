generate-pwd-hash:
	bash run.sh generate:pwd-hash

transform-init-sql:
	bash run.sh transform:init-sql

build-api:
	bash run.sh build:api

build-inference:
	bash run.sh build:inference

build-all:
	bash run.sh build:all

lint:
	bash run.sh lint


purge-pycache:
	bash run.sh purge:pycache

