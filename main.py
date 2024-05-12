import uvicorn
from decouple import config

if __name__ == "__main__":
    if config("MODE") == "prod":
        uvicorn.run(
            "app.app:app",
            host="0.0.0.0",
            port=80,
            reload=False,
            workers=4,
            timeout_keep_alive=1000,
        )
    else:
        uvicorn.run(
            "app.app:app",
            host=config("HOST"),
            port=int(config("PORT")),
            reload=True,
            timeout_keep_alive=10000,
        )
