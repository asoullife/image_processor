# การติดตั้งและตั้งค่า PostgreSQL

## วิธีที่ 1: ติดตั้ง PostgreSQL บน Windows

### 1. ดาวน์โหลดและติดตั้ง
```bash
# ดาวน์โหลดจาก https://www.postgresql.org/download/windows/
# หรือใช้ Chocolatey
choco install postgresql

# หรือใช้ winget
winget install PostgreSQL.PostgreSQL
```

### 2. สร้าง Database
```sql
-- เชื่อมต่อ PostgreSQL ด้วย psql
psql -U postgres

-- สร้าง database
CREATE DATABASE adobe_stock_processor;

-- สร้าง user (ถ้าต้องการ)
CREATE USER adobe_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE adobe_stock_processor TO adobe_user;
```

### 3. ตั้งค่า Environment Variable
```bash
# Windows
set DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/adobe_stock_processor

# หรือสร้างไฟล์ .env
echo DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/adobe_stock_processor > .env
```

## วิธีที่ 2: ใช้ Docker

### 1. ติดตั้ง Docker Desktop
ดาวน์โหลดจาก https://www.docker.com/products/docker-desktop

### 2. รัน PostgreSQL Container
```bash
docker run --name adobe-postgres \
  -e POSTGRES_DB=adobe_stock_processor \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  -d postgres:15
```

### 3. ตรวจสอบ Container
```bash
docker ps
docker logs adobe-postgres
```

## วิธีที่ 3: ใช้ SQLite สำหรับ Development

### 1. แก้ไข Database URL
```python
# ใน backend/database/connection.py
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite+aiosqlite:///./adobe_stock_processor.db"
)
```

### 2. ติดตั้ง aiosqlite
```bash
pip install --user aiosqlite
```

## การทดสอบการเชื่อมต่อ

```bash
# ทดสอบ health check
python scripts/startup.py health

# รัน migration
python scripts/run_migrations.py migrate

# เริ่ม API server
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## การตั้งค่าเพิ่มเติม

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@host:port/database

# API Settings
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=true

# Security
SECRET_KEY=your-secret-key-here
```

### Production Settings
```bash
# Production database
DATABASE_URL=postgresql+asyncpg://user:password@prod-host:5432/adobe_stock_processor

# Security
SECRET_KEY=strong-random-secret-key
DEBUG=false
```