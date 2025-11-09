# To run api, follow these steps:

1. Active anaconda environment
 ```bash
    conda actiavte <env_name>
 ```

2. Run api in background
```bash
   nohup flask --app main run --host=0.0.0.0 --port=5000
```

# To call api, follow these steps:

1. Login to app to get access token:

  - POST `http://171.246.173.54:443/login`
  - Body: 
   ```json
   {
      "username": "<username>",
      "password": "<password>"
   }
   ```

2. Call to api to answer the question using access token in step 1
   - POST `http://171.246.173.54:443/query`
   - Using Bearer with access token to authenticate: `Brearer <access_token>`
   - Body:
   ```json
   {
      "premises": [<list of premises in array of string>],
      "questions": ["<question>"]
   }
   ```
   Note: Require only one question, if there are multiple provided questions, an error are returned.

# To convert premises and Yes/No/Uncertain question to FOL representaion:
  1. Login to app to get access token.
  2. Api to convert yes/no/uncertain question.
   - POST `http://171.246.173.54:443/convert-fol-yn`
   - Using Bearer with access token to authenticate: `Brearer <access_token>`
   - Body:
   ```json
   {
      "premises": [<list of premises in array of string>],
      "questions": ["<yes/no question>"]
   }
   ```
   Note: Require only one question, if there are multiple provided questions, an error are returned.