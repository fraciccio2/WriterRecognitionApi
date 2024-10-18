# Writer Recognition Api
Writer Recognition Api (written in Python) for identifying the writer of a handwritten paragraph image.

## How To Use
1. Install Python 3 interpreter
2. Clone the repository
   ```Console
   git clone https://github.com/fraciccio2/WriterRecognitionApi.git
   ```
3. Install project dependencies
   ```Console
   pip install -r requirements.txt
   ```
4. Run the project
   ```Console
   python ./src/main.py
   ```
5. Use the path
   ```Console
   http://127.0.0.1/files
   ```
   and send a FormData with an array of string with all writers name, a test file with the name test0 and all the files of the various writers with the name of the script and the file number. Example pippo0.

**Note:** the system is tested on Windows using `Python 3.8.6` but should work on other platforms as well.

## Starting Library
   ```Console
   https://github.com/OmarBazaraa/WriterIdentifier
   ```
