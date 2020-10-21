import numpy as np
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
