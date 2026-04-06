import uvicorn
import logging

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting TanitAI OpenAI Compatible API Server...")
    
    # Run the FastAPI app via Uvicorn
    uvicorn.run("src.view.api:app", host="0.0.0.0", port=10000, reload=True)
