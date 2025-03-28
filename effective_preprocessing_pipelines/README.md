# Reproducible experiments for generating pre-processing pipelines for AutoETL

This is the repository for the companion reproducible paper of [1].
To implement the optimization process, we departed from the code provided in [2] (GitHub repository: [https://github.com/aquemy/DPSO_experiments](https://github.com/aquemy/DPSO_experiments)).

[1] J. Giovanelli, B. Bilalli, A. Abelló, "Data pre-processing pipeline generation for AutoETL", Inf. Syst. (2021) 101957. http://dx.doi.org/10.1016/j.is.2021.101957

[2] A. Quemy, "Data Pipeline Selection and Optimization." DOLAP. 2019. http://ceur-ws.org/Vol-2324/Paper19-AQuemy.pdf
# Requirements

In order to reproduce the experiments in any operating systems, Docker is required: [https://www.docker.com/](https://www.docker.com/).
Install it, and be sure that it is running when trying to reproduce the experiments.

To test if Docker is installed correctly:

- open the terminal;
- run ```docker run hello-world```.

***Expected output:***

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
2db29710123e: Pull complete 
Digest: sha256:7d246653d0511db2a6b2e0436cfd0e52ac8c066000264b3ce63331ac66dca625
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

# Reproducing the experiments

1. Get the repository

	You can either clone this repository from GitHub (which contains only the code and the datasets to start the experiments, without the intermediary results) or download the full material (code, datasets, and intermediary results) from the Harvard DataVerse.
	
	To clone the repository, open the terminal and type:
	```
	git clone https://github.com/josephgiovanelli/effective_preprocessing_pipelines.git
	```
	This will download the repository under the folder effective_preprocessing_pipelines.
	
	
	To download the full material from the Harvard DataVerse, you can access the link [https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O2XQ1P](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O2XQ1P) and press the "Access Dataset" button. This will download an archive (i.e., effective_preprocessing_pipelines.zip); unzip it and go to the next step.

2. Run the experiments
	- Go to the created folder.
		```
		cd effective_preprocessing_pipelines
		```
	- Run the start script.
	
		 If you have downloaded the full material from the Harvard Dataverse, you have the intermediary results for reproducing all the paper artifacts without a time-consuming computation; otherwise we suggest you to add the ```-toy``` parameter to the running script, which allows to test the reproducibility of the experiments -- but with a less time-consuming computation.
	  - on Unix-like systems run the script ***start.sh*** in the root folder;
		  ```
		  ./start.sh
		  ``` 
		  or 
		  ```
		  ./start.sh -toy
		  ```
	  - on Windows systems run the script ***start.bat*** in the root folder.

		- If using the Windows PowerShell:
			```
			./start.bat
			```  
			or 
			```
			./start.bat -toy
			``` 
		- If using the Command Prompt:
			```
			start.bat
			``` 
			or 
			```
			start.bat -toy
			```
  
	The script will search for an image of the needed [pre-built container](https://github.com/josephgiovanelli/effective_preprocessing_pipelines/releases/tag/0.1.0); if it is not found, it is downloaded. 

***Expected output:***

```
Error response from daemon: No such container: effective_preprocessing_pipelines
Error: No such container: effective_preprocessing_pipelines
Unable to find image 'ghcr.io/josephgiovanelli/effective-preprocessing-pipelines:0.1.0' locally
0.1.0: Pulling from josephgiovanelli/effective-preprocessing-pipelines
1671565cc8df: Pull complete 
3e94d13e55e7: Pull complete 
fa9c7528c685: Pull complete 
53ad072f9cd1: Pull complete 
d6b983117533: Pull complete 
d8092d56ded5: Pull complete 
43a9f01be008: Pull complete 
4c44fb36b5a8: Pull complete 
8b9de5c6c1b4: Pull complete 
e473a4b91685: Pull complete 
ebed27060700: Pull complete 
36ebe683df3b: Pull complete 
01d78e0efc1d: Pull complete 
4d3cc9b83039: Pull complete 
077b35bf5e42: Pull complete 
Digest: sha256:4e45e01f960f6a9df1fac28e2ce1c7a10aa7b7480d88a9d233d63965ede0a5ac
Status: Downloaded newer image for ghcr.io/josephgiovanelli/effective-preprocessing-pipelines:0.1.0
```

Afterwards, the workflows illustrated in the paper are run. 

***Expected output:***

```
### EXPLORATORY ANALYSIS (EA) ###

Creating scenarios...
        Done.

EA01. SMBO on fixed pre-processing prototypes and ML algorithms

                Checking scenarios with results...

                num invalid scenarios: 0
                num scenarios with results: 0
                num scenarios to run: 3
                        estimated time: 0:01:00 (60s)
:   0%|          | 0/3 [00:00<?, ?it/s]
```

# Possible issues

## Docker permissions

The code is executed in a Docker container, which communicates with the file system through a [volume](https://docs.docker.com/storage/volumes/). It might be possible that Docker needs the permissions (usually on Windows).

Luckily, Docker asks for it when attaching the volume.

<img width="356" alt="permission" src="https://user-images.githubusercontent.com/41596745/188275585-5ad52e6f-05be-44d1-bad7-10f15eeb4af1.png">

In that case, give the permissions by clicking on "Share it".
  
In the worst case, you have to give permissions manually:
- open Docker Desktop;
- click on the engine icon in the top right corner;
- click on "Resources" in the menu on the left;
- click on "FILE SHARING" section;
- click on the "+" button to add the repository folder;
- and finally click on "Apply & Restart".

<img width="1433" alt="Schermata 2022-09-03 alle 16 56 57" src="https://user-images.githubusercontent.com/41596745/188276254-a0ca6872-07fe-4964-9f82-9fca9c45b5d3.png">

## Running permissions

It might be possible that your operating system needs the authorization to run the scripts (usually on Unix-like systems).

***Possible output***:
```
./wrapper_experiments.sh: ./scripts/prototype_construction.sh: /bin/bash: bad interpreter: Permission denied
./wrapper_experiments.sh: ./scripts/experimental_evaluation.sh: /bin/bash: bad interpreter: Permission denied
./wrapper_experiments.sh: ./scripts/exploratory_analysis.sh: /bin/bash: bad interpreter: Permission denied
```

To give the necessary permissions to the files at hand, you can type:
```
chmod 777 ./scripts/*
```

If the error is given also for the scripts ***start.sh*** and/or ***wrapper_experiments.sh***, you can type:
```
chmod 777 start.sh
chmod 777 wrapper_experiments.sh
```
