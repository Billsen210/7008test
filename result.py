rf_file_path = '/opt/data/rf_result.txt'
svm_file_path = '/opt/data/svm_result.txt'
output_file_path = '/opt/data/merged_result.txt'
with open(rf_file_path, 'r') as rf_file, open(svm_file_path, 'r') as svm_file, open(output_file_path, 'w') as output_file:

    output_file.write(rf_file.read())

    output_file.write('\n\n')

    output_file.write(svm_file.read())
