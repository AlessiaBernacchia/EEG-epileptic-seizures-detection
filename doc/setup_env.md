## Setup Environment and Packages
To set up the environment and prepare the data for analysis, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AlessiaBernacchia/EEG-epileptic-seizures-detection.git
   ```
2. **Navigate to the project directory**:
   ```bash
    cd EEG-epileptic-seizures-detection
    ```
3. **Install environment with required dependencies**:
   If nvidia GPU is available:
   ```bash
    conda env create --file environment-gpu.yml
   ```
   else:
   ```bash
    conda env create --file environment-cpu.yml
   ```

4. **Activate the environment**:
   ```bash
     conda activate eeg-seizure-detection
   ```
> To update the environment with new dependencies, simply add them to the `environment.yml` file and run `conda env update -n eeg-seizure-detection --file ./environment-cpu.yml --prune` or `conda env update -n eeg-seizure-detection --file ./environment-cpu.yml --prune` to apply the changes.

5. **Make kernel available in Jupyter**:
    ```bash
    python -m ipykernel install --user --name eeg-seizure-detection --display-name "Python (eeg-seizure-detection)"
    ```

6. **Install the project as a package**:
if you want to use the project as a package and import its modules, you can install it in editable mode:
   ```bash
    python -m pip install -e .
   ```
   If you want to install the package without editable mode, simply run:
      ```bash
      python -m pip install .
      ```

   > Important: the package name `eeg_seizure_detection` is the install target, not the Python import root. Your actual import root is `utils` because that is the package folder found by `find_packages()`.