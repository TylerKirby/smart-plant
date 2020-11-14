import numpy as np
import uvicorn
from fastapi import FastAPI


app = FastAPI()


@app.get('/health')
def health():
    return 200

@app.get('/temp/')
def get_temp(start: int = 0, end: int = 0):
    return {
        'temps': [np.sin(i)*3+50 for i in np.arange(0, 24, 1)]
    }

@app.get('/soil/')
def get_temp(start: int = 0, end: int = 0):
    return {
        'temps': [np.sin(i)*3+50 for i in np.arange(0, 24, 1)]
    }

@app.get('/light/')
def get_temp(start: int = 0, end: int = 0):
    return {
        'temps': [np.sin(i)*3+50 for i in np.arange(0, 24, 1)]
    }


if __name__ == '__main__':
    uvicorn.run(app, port=5000)
