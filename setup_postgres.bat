@echo off
echo Stopping existing PostgreSQL container...
docker stop my-postgres 2>nul
docker rm my-postgres 2>nul

echo Creating new PostgreSQL container with admin credentials...
docker run --name adobe-postgres ^
  -e POSTGRES_DB=adobe_stock_processor ^
  -e POSTGRES_USER=admin ^
  -e POSTGRES_PASSWORD=admin123 ^
  -p 5432:5432 ^
  -d postgres:15

echo Waiting for PostgreSQL to start...
timeout /t 10

echo Checking container status...
docker ps

echo PostgreSQL is ready!
echo Connection string: postgresql+asyncpg://admin:admin123@localhost:5432/adobe_stock_processor