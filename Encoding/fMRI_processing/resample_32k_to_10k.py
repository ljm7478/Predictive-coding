import os

subjects = ['01', '02', '03', '04', '05', '06', '09',
            '10', '14', '15', '16', '17', '18', '19', '20']
segments = [1, 2, 3, 4, 5, 6, 7, 8]

std_mesh_atlas_path = '/local_raid1/01_software/HCPpipelines/global/templates/standard_mesh_atlases/'
Gordon_path = '/local_raid1/02_data/99_parcellation/Goldon2016/10k_cifti_separate_Gordon333'

Gordon_10k = f'{Gordon_path}/Gordon333_FreesurferSubcortical.10k_fs_LR.dlabel.nii'
L_sphere_32k = f'{std_mesh_atlas_path}/L.sphere.32k_fs_LR.surf.gii'
L_sphere_10k = f'{std_mesh_atlas_path}/L.sphere.10k_fs_LR.surf.gii'
R_sphere_32k = f'{std_mesh_atlas_path}/R.sphere.32k_fs_LR.surf.gii'
R_sphere_10k = f'{std_mesh_atlas_path}/R.sphere.10k_fs_LR.surf.gii'
L_surf_32k = f'{std_mesh_atlas_path}/cortical_surface/S900.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
L_surf_10k = f'{std_mesh_atlas_path}/cortical_surface/S900.L.midthickness_MSMAll.10k_fs_LR.surf.gii'
R_surf_32k = f'{std_mesh_atlas_path}/cortical_surface/S900.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
R_surf_10k = f'{std_mesh_atlas_path}/cortical_surface/S900.R.midthickness_MSMAll.10k_fs_LR.surf.gii'


for sub in subjects:
    for seg in segments:

        data_path = f'/ares-share/local_raid3/user/sunghyoung/01_project/04_encoding_model/02_data/Results_MNI152NLin6Asym/sub-{sub}'
        dtseries_32k = f'{data_path}/ses-movie_task-movie_run-{seg}_Atlas_s3.dtseries.nii'
        dtseries_10k = f'{data_path}/ses-movie_task-movie_run-{seg}_Atlas_s3.10k.dtseries.nii'

        os.system(f'wb_command -cifti-resample {dtseries_32k} COLUMN {Gordon_10k} COLUMN ADAP_BARY_AREA TRILINEAR {dtseries_10k} \
            -left-spheres {L_sphere_32k} {L_sphere_10k} -left-area-surfs {L_surf_32k} {L_surf_10k} \
            -right-spheres {R_sphere_32k} {R_sphere_10k} -right-area-surfs {R_surf_32k} {R_surf_10k}')
