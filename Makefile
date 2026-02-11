setup:
	docker-compose up --build -d
	docker-compose exec backend python scripts/init_db.py
	docker-compose exec backend python scripts/migrate_data.py
	docker-compose exec backend python scripts/seed_qdrant.py

stop:
	docker-compose down

restart:
	docker-compose restart