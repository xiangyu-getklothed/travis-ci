PYOPENGL_PLATFORM=osmesa python3 main.py \
						--config ../cfg_files/fit_smplx.yaml \
						--data_folder /Monolith/People/XiangyuWang/examples/webchayan/ \
						--output_folder /Monolith/People/XiangyuWang/examples/webchayan/smplify/ \
						--model_folder /Monolith/People/IlyaKavalerov/models/third_party/smplify_x/models/smplx/SMPLX_NEUTRAL.npz \
						--vposer_ckpt /Monolith/People/ChayanPatodi/PeopleWatcher/DockerSetup/vposer_v1_0 \
						--part_segm_fn /Monolith/People/ChayanPatodi/PeopleWatcher/DockerSetup/smplx_parts_segm.pkl \
						--visualize False --use_cuda True --interpenetration False \
						--use_densepose True
