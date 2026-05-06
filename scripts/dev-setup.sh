#!/usr/bin/env bash
set -e

echo "==> E-Commerce Agent Monorepo Setup"

# Navigate to backend directory
cd apps/backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "--> Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "--> Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "--> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Start PostgreSQL + pgvector via Docker
echo "--> Starting PostgreSQL with pgvector..."
docker run -d --name pgvector-dev -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16 2>/dev/null || echo "Container may already exist"

echo "--> Waiting a few seconds for Postgres to start..."
sleep 3

# Run DB setup
echo "--> Running database setup..."
python -m backend.db.setup
python -m backend.db.vector_setup

cd ../..

echo ""
echo "==> Setup complete!"
echo "To run the agent:"
echo "  cd apps/backend"
echo "  source venv/bin/activate"
echo "  python main.py"
