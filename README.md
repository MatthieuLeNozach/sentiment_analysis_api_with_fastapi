
# **Sentiment and emotion analysis API with FastAPI**
### ***sentiment_analysis_api_with_fastapi (v0.1)***

Project started from [this template](https://github.com/MatthieuLeNozach/api_basemodel_for_machine_learning_with_fastapi/tree/main)


1. [**Installation**](#installation)
2. [**Getting started**](#getting-started)





## **Installation** <a name="installation"></a>
```bash
git clone https://github.com/MatthieuLeNozach/api_basemodel_for_machine_learning_with_fastapi
```

**[CAUTION]**: This project uses PyTorch based models, which rely on GPU computing with CUDA toolkit and Nvidia drivers. As some package installations are CUDA-specific, incompatible hardware and drivers may raise issues at install or import.  
You may have to build your own python / torch environment.

Cuda drivers are also mentioned in the Docker image, you may want to remove this line from the Dockerfile:
```Dockerfile
RUN conda install -c conda-forge cudatoolkit=11.7
```

### **Set up a Python virtual environment**
**Python version: 3.10.14**


```bash
cd /path/to/repository/root
```

#### OptionA: Using `venv`:


```bash
python3 -m venv venv

# Virtual environment activation:
source venv/bin/activate

# Installation of required packages from the requirements file
pip install -r requirements.txt
```
#### Option B: Using `miniconda`:

```bash
conda create --name fastapi_nlp python=3.10.14

# Virtual environment activation
conda activate fastapi_nlp

# Installation of required packages from the requirements file
conda install --file requirements.txt
```

### **Alternative: Mount project as a DevContainer in VSCode**

#### Create image from the `Dockerfile`:
```bash
docker build -t nlp_api_image:latest
```

#### Use image as a DevContainer: #TODO- instructions
  #TODO

### **Grant permissions to `run.sh`**

```bash

chmod +x run.sh

```


## **Getting started** <a name="getting-started"></a>

### **A. Securize the project**
- **Add `.environment` folder to the `.gitignore` file** to stop exposing sensitive information to git commits
- **Access the `.environment` folder** from repository root, the folder may be hidden.
```bash
nano .environment/env.dev

nano .environment/env.test
```

- **Replace the dummy secret keys with your own keys**. You can generate base64 secrets with this command:
```bash
openssl rand -base64 32
```

- **Change the `superuser` password**. Modify the file `app/devtools.py` with the credentials of your choice:
```py
# from app/devtools.py
def create_superuser(db: Session):
    superuser = User(
        username='superuser@example.com',
        first_name='Super',
        last_name='User',
        hashed_password=pwd_context.hash('8888'),
...
```
Only an admin can register another admin. A permanent admin can be created once logged-in as the `superuser`, by sending a `post` request to this address:
`http://localhost:8080/admin/create`

### **B. Run app in development mode**

`superuser` argument initializes an admin at startup and deletes it at shutdown.

```bash
./run.sh dev superuser
```
superuser credentials (from app/dev_tools.py):
```py
def create_superuser(db: Session):
    superuser = User(
        username='superuser@example.com',
        first_name='Super',
        last_name='User',
        hashed_password=pwd_context.hash('8888'),
        is_active=True,
        role='admin',
        has_access_sentiment=True,
        has_access_emotion=True
    )
    db.add(superuser)
    db.commit()
```


### **The authentication flow**


#### **Authentication as Superuser**

```bash
curl -X 'POST' \
  'http://localhost:8080/auth/token' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'grant_type=&username=superuser%40example.com&password=8888&scope=&client_id=&client_secret='
  ```


![alt text](<readme/2.png>)




#### **Access Machine Learning services**

From `http://localhost:8080/docs`:
1. **Authenticate:**
    - Click on the lock
    - Fill username (ex: `superuser@example.com`)
    - Fill password (ex `8888`)
2. **Send input text**
    - Sentiment route: [localhost:8080/mlservice/sentiment/predict/interpreted](http://localhost:8080/mlservice/sentiment/predict/interpreted):
![alt text](<readme/5.png>)
    - Emotion route [localhost:8080/mlservice/emotion/predict](http://localhost:8080/mlservice/emotion/predict)

3. **Get output**
    - Sentiment:
![alt text](<readme/6.png>)
    - Emotion:
![alt text](<readme/9.png>)





## **What's ready to use?** <a name="whats-ready-to-use"></a>

### **3.1 Endpoints and routers**

#### **A. Router structure**

To achieve separation of concerns and improved code organization, endpoints and helper functions are grouped into logical modules with their own namespaces (ex `/admin/...`):

- **auth** router file for registration / security related helpers and endpoint functions (see below)
- **admin** router file for admin specific actions, like grant/revoke access rights, delete user
- **user** router file for user specific actions (change password, #TODO get service call history)
- **bert_sentiment**
- **roberta_emotion**

#### **B. Endpoints**
API endpoints documentation (see below), and HTTP request templates are available at this address:
`http://localhost:8080/docs`

![alt text](<readme/1.png>)


### **3.2 Natural Language Processing classifiers**

#### **A. Sentiment: BERT model**
This multilanguage pretrained model returns sentiment probabilities for 5 target labels (0 = very negative to 4 = very positive).  

More information on the model [here](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)


#### **B. Emotion: Roberta base model**
This pretrained model returns emotion probabilities for a total of 28 different emotions, the service selects the 5 most significant labels, returns their name and probability  

More information on the model [here](https://huggingface.co/SamLowe/roberta-base-go_emotions)
