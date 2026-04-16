import os

file_names = ['seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7', 'seg8', 'seg9',
             'seg10', 'seg11', 'seg12', 'seg13', 'seg14', 'seg15', 'seg16', 'seg17', 
             'seg18', 'test1', 'test2', 'test3', 'test4']

def rename_files(root_directory):
    # Walk through all directories and files in root_directory
    for dirpath, dirnames, filenames in os.walk(root_directory):
        print('dirpath is {}'.format(dirpath))
        for filename in filenames:
            # Check if the filename matches the pattern you want to change
            if filename.startswith('E') and '_nt' in filename:
                old_file_path = os.path.join(dirpath, filename)
                # Define new filename by removing the part after '_nt' up to '.npy'
                new_filename = filename.split('_nt')[0] + '.npy'
                new_file_path = os.path.join(dirpath, new_filename)
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print('Renamed {} to {}'.format(old_file_path, new_file_path))
            elif filename.startswith('Ahat') and '_nt' in filename:
                old_file_path = os.path.join(dirpath, filename)
                # Define new filename by removing the part after '_nt' up to '.npy'
                new_filename = filename.split('_nt')[0] + '.npy'
                new_file_path = os.path.join(dirpath, new_filename)
                # Rename the file
                os.rename(old_file_path, new_file_path)
                print('Renamed {} to {}'.format(old_file_path, new_file_path))

# Example usage
root_directory = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/04_AlexNet/result/'  # Specify your root directory here
rename_files(root_directory)
