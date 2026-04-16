# -*- coding: utf-8 -*-
import os
import glob
import sys

def reindex_and_create_video():
    """
    1. Renames all images in the folder to be sequential (image_000001.png, ...).
       This fixes gaps in numbering (e.g., missing 170238).
    2. Runs FFmpeg to create the video.
    """

    # --- Configuration ---
    base_dir = '/local_raid1/03_user/jungmin/02_data/02_Movie/NNDB/06_concat_500SUM_CIT/val/frames'
    output_video_name = 'check_concat_video_fixed.mp4'
    output_path = os.path.join('/local_raid1/03_user/jungmin/02_data/02_Movie/NNDB/06_concat_500SUM_CIT/val/', output_video_name)

    print "------------------------------------------------"
    print "Step 1: Re-indexing files in %s" % base_dir
    print "------------------------------------------------"

    # 1. Get all image files and sort them
    # Sorting is crucial to maintain the correct frame order
    pattern = os.path.join(base_dir, 'image_*.png')
    files = sorted(glob.glob(pattern))

    if not files:
        print "Error: No images found."
        return

    total_files = len(files)
    print "Found %d files. Starting rename process..."

    # 2. Rename loop
    # This closes any gaps (e.g., if 17038 is missing, 17039 becomes 17038)
    count = 0
    for i, old_path in enumerate(files):
        # New index starts from 1
        new_index = i + 1
        
        # Define the new filename (e.g., image_000001.png)
        new_filename = "image_%06d.png" % new_index
        new_path = os.path.join(base_dir, new_filename)

        # Only rename if the name is actually different
        if old_path != new_path:
            os.rename(old_path, new_path)
            count += 1

        # Print progress every 5000 files
        if new_index % 5000 == 0:
            sys.stdout.write("\rProcessing: %d / %d" % (new_index, total_files))
            sys.stdout.flush()
    
    print "\n\nRe-indexing complete. %d files renamed." % count
    print "File sequence is now continuous from 000001 to %06d." % total_files

    print "------------------------------------------------"
    print "Step 2: Running FFmpeg"
    print "------------------------------------------------"

    # 3. Run FFmpeg
    input_pattern = os.path.join(base_dir, 'image_%06d.png')
    
    # Command: 60fps, H.264 codec
    cmd = 'ffmpeg -framerate 60 -i "{0}" -c:v libx264 -pix_fmt yuv420p -crf 23 "{1}"'.format(input_pattern, output_path)
    
    print "Executing: %s" % cmd
    ret = os.system(cmd)

    if ret == 0:
        print "\n[Success] Video saved to: %s" % output_path
    else:
        print "\n[Error] FFmpeg failed."

if __name__ == '__main__':
    reindex_and_create_video()