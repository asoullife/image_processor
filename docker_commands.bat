@echo off
echo Current Docker containers:
docker ps -a

echo.
echo Stopping and removing existing PostgreSQL container...
docker stop my-postgres 2>nul
docker rm my-postgres 2>nul
docker stop adobe-postgres 2>nul  
docker rm adobe-postgres 2>nul

echo.
echo Creating new PostgreSQL container...
docker run --name adobe-postgres -e POSTGRES_DB=adobe_stock_processor -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin123 -p 5432:5432 -d postgres:15

echo.
echo Waiting for PostgreSQL to start...
timeout /t 5 /nobreak >nul

echo.
echo Checking new container:
docker ps

echo.
echo PostgreSQL is ready!
echo Connection: postgresql+asyncpg://admin:admin123@localhost:5432/adobe_stock_processor