import uvicorn

if __name__ == "__main__":
    print("Starting SONARQUESTV2 Server...")
    # Run the FastAPI app with Uvicorn programmatically
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, log_level="info")
