"""Citadel — Root-level app re-export for Docker/uvicorn."""

from server.app import app, main

if __name__ == "__main__":
    main()
