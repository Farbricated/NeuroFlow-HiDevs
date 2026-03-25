# NeuroFlow — Deployment Guide

## Target: Railway (Recommended)

### Step 1: Provision Databases
In Railway dashboard:
- Add PostgreSQL (pgvector) → note DATABASE_URL
- Enable pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
- Add Redis → note REDIS_URL

### Step 2: Set Environment Variables
In Railway → Variables, add all from `.env.example`.

### Step 3: Deploy API
```bash
railway up --service api
# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2
```

### Step 4: Deploy Worker
Add second Railway service, same repo, start command: `python -m worker`

### Step 5: Run Migrations
```bash
railway run python -c "
import asyncpg, asyncio, os
async def m():
    c = await asyncpg.connect(os.environ['POSTGRES_URL'])
    with open('infra/init/001_schema.sql') as f: await c.execute(f.read())
asyncio.run(m())
"
```

## Production Verification

```bash
BASE=https://your-app.railway.app
curl $BASE/health
curl -X POST $BASE/api/v1/ingest -H "Authorization: Bearer $TOKEN" -F "file=@tests/fixtures/test_doc.pdf"
curl $BASE/api/v1/evaluations -H "Authorization: Bearer $TOKEN"
curl $BASE/metrics
```

## Rollback Procedure
1. Find previous image SHA in GitHub Packages
2. `railway up --service api --image ghcr.io/<repo>/api:<prev-sha>`
3. Verify: `curl $BASE/health`
