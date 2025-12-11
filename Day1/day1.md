#Day 001: Environment Setup

## 1.Github operation procedures
Some basic operations in github website and local github:  
1. **create a repository in github.com**  
2. **download github Desktop software**  
3. **clone**: clone the remote repository to local github Desktop.  
4. **write code in vscode**: find the path of the repository and edit code in vscode.  
5. **commit and push** :click commit and push in github desktop.

## 2.Markdown grammar and Shortcut
1. grammar  
\# ：headers  
**** : Bold text  
\$$ : equations. Math in line  
\```bash XXXXX ``` : Code Block

2. Shortcut
Ctrl+Shift+V : Preview the md file

3. commands
   ls: list all files in the current directory
   cd: enter the file folder


## 3. python virtual environment 
1. developers environment  
   **OS**: Windows 11  
   **VSCODE**: 1.106.3  
   **python**: 3.12.9  
2. command 
```bash
python -m venv venvscheduling

``` 

3. execute this in Powershell(click "+" in terminal)
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

```bash
.\venvscheduling\Scripts\activate
```

4. Use Ctrl+Shift+P,then search python:Select Interpreter, select the virtual environment.

5. Use pip to install the needed packages.   
```bash
pip install pipreqs
```

Export the installed packages:
```bash
pipreqs . --encoding=utf-8 --force
```
 Install all packages in requirement.txt
 ```bash
 pip install -r requirement.txt
 ```