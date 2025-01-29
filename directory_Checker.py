import os

def get_directory_report(directory):
    report = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            report.append((file_path, file_size))
    return report

def print_report(report):
    total_size = 0
    for file_path, file_size in report:
        print(f"File: {file_path}, Size: {file_size} bytes")
        total_size += file_size
    print(f"Total size: {total_size} bytes")

def print_summary(report):
    total_files = len(report)
    total_size = sum(file_size for _, file_size in report)
    print("\nSummary:")
    print(f"Total number of files: {total_files}")
    print(f"Total size of all files: {total_size} bytes")

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    report = get_directory_report(directory)
    print_report(report)
    print_summary(report)