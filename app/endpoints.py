from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()


@app.get("/time", response_model=datetime)
async def get_current_time():
    current_time = datetime.now()
    return current_time
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime

app = FastAPI()


@app.get("/time", response_model=datetime)
async def get_current_time():
    current_time = datetime.now()
    return current_time
