Analytics Server
=
General server for retrieving data and analytics on plant from AWS.
## Set up
1. Create a virtual environment: `python3 -m venv venv`
2. Source the virtual env: `source venv/bin/activate`
3. Install the dependencies: `pip install -r requirements.txt`
4. Start the server: `uvicorn main:app --reload`

The server will be hosted at `http://localhost:8000` when running locally. See `http://localhost:8000/docs`
for available endpoints.