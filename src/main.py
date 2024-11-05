from flask import Flask, jsonify, request
import os
import numpy as np
import time
from sklearn.svm import SVC
from collections import Counter

from data.test_generator import TestGenerator
from features.feature_extractor import FeatureExtractor
from pre_processing.pre_processor import PreProcessor
from segmentation.line_segmentor import LineSegmentor
from utils.constants import *
from utils.utils import *

# Create Flask's app
app = Flask(__name__)

#
# Variables
#
result_file = None
total_time = 0.0
testcase_time = []
results = []
accuracies = []

UPLOAD_FOLDER = './data/testcases/001/'


# =====================================================================
#
# Pipeline functions
#
# =====================================================================


def run_pipeline():
    # Set global variables
    global result_file, total_time, testcase_time, results, accuracies
    results = []
    accuracies = []

    # Start timer
    start = time.time()

    # Open files
    result_file = open(PREDICTED_RESULTS_PATH, 'w')

    # Iterate on every testcase
    for root, dirs, files in os.walk(TESTCASES_PATH):
        for d in dirs:
            print('Running test iteration \'%s\'...' % d)

            # Start timer of test iteration
            t = time.time()

            # Solve test iteration
            process_testcase(TESTCASES_PATH + d + '/')

            # Finish timer of test iteration
            t = (time.time() - t)
            testcase_time.append(t)

            print('Finish test iteration \'%s\' in %.02f seconds\n' % (d, t))
        break

    # Close files
    result_file.close()

    # End timer
    total_time = (time.time() - start)
    return results, accuracies


@app.route('/files', methods=['POST'])
def run():
    if os.path.exists(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    writers_string = request.form.get('writers')
    writers = [writer.strip() for writer in writers_string.split(',')] if writers_string else []

    for writer in writers:
        writer_dir = os.path.join(UPLOAD_FOLDER, writer)
        os.makedirs(writer_dir, exist_ok=True)

        training_files = [f for f in request.files if f.startswith(writer)]
        for idx, training_file in enumerate(training_files):
            file = request.files[training_file]
            if file:
                original_extension = os.path.splitext(file.filename)[1]
                file.save(os.path.join(writer_dir, f'training_image_{idx + 1}{original_extension}'))

    test_files = [f for f in request.files if f.startswith('test')]
    for idx, test_file in enumerate(test_files):
        file = request.files[test_file]
        if file:
            original_extension = os.path.splitext(file.filename)[1]
            file.save(os.path.join(UPLOAD_FOLDER, f'test_image_{idx + 1}{original_extension}'))
    results, accuracies = run_pipeline()

    return jsonify({"message": "Processed files successfully", "results": results, "accuracies": accuracies}), 200


def process_testcase(path):
    global results, accuracies
    features, labels = [], []

    # Loop over every writer in the current test iteration
    for root, dirs, files in os.walk(path):
        for d in dirs:
            print('    Processing writer \'%s\'...' % d)
            x, y = get_writer_features(path + d + '/', d)
            features.extend(x)
            labels.extend(y)

    # Train the SVM model
    classifier = SVC(C=5.0, gamma='auto', probability=True)
    classifier.fit(features, labels)

    # Loop over test images in the current test iteration
    for root, dirs, files in os.walk(path):
        for filename in files:
            # Extract the features of the test image
            x = get_features(path + filename)

            # Get the most likely writer
            p = classifier.predict_proba(x)
            p = np.sum(p, axis=0)
            predict = classifier.predict(x)
            f = Counter(predict).most_common(1)[0][1]
            r = classifier.classes_[np.argmax(p)]

            # Write result
            result_file.write(str(r) + '\n')

            results.append(str(r))
            accuracies.append(f/len(predict))

            print('    Classifying test image \'%s\' as writer \'%s\'' % (path + filename, r))
        break


def get_writer_features(path, writer_id):
    # All lines of the writer
    total_gray_lines, total_bin_lines = [], []

    # Read and append all lines of the writer
    for root, dirs, files in os.walk(path):
        for filename in files:
            gray_img = cv.imread(path + filename, cv.IMREAD_GRAYSCALE)
            gray_img, bin_img = PreProcessor.process(gray_img)
            gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()
            total_gray_lines.extend(gray_lines)
            total_bin_lines.extend(bin_lines)
        break

    # Extract features of every line separately
    x, y = [], []
    for g, b in zip(total_gray_lines, total_bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)
        y.append(writer_id)

    return x, y


def get_features(path):
    # Read and pre-process the image
    gray_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    gray_img, bin_img = PreProcessor.process(gray_img)
    gray_lines, bin_lines = LineSegmentor(gray_img, bin_img).segment()

    # Extract features of every line separately
    x = []
    for g, b in zip(gray_lines, bin_lines):
        f = FeatureExtractor([g], [b]).extract()
        x.append(f)

    # Return list of features for every line in the image
    return x


# =====================================================================
#
# Generate test iterations from IAM data set
#
# =====================================================================

if GENERATE_TEST_ITERATIONS:
    gen = TestGenerator()
    gen.generate(TESTCASES_PATH, 10, 3, 2)

# =====================================================================
#
# Main
#
# =====================================================================

# run_pipeline()
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

if DEBUG_SAVE_WRONG_TESTCASES:
    print_wrong_testcases()
    save_wrong_testcases()
