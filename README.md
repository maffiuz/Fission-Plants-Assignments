# Nuclear Fission Plant Assignment 1.2

This repository collects all assignments of the Nuclear Fission Plants course. Each assignment is contained in its own dedicated branch. Here, the solution of assignment 1.2 is contained

## Setup
1. Make sure you have `pipenv` installed. Otherwise you can install it using the regular `pip` command:
    ```bash
   pip install pipenv
   ```

2. Create a virtual environment using `pipenv`:
   ```bash
   pipenv shell
   ```

3. Update the `Pipfile.lock` (not mandatory, do only if the next step is not working properly):
   ```bash
   pipenv lock
   ```

4. Install the required dependencies from `Pipfile.lock`:
   ```bash
   pipenv sync
   ```

## Usage
Run the script `main.py` to execute the script. For example:
```bash
python main.py
```