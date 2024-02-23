import os

from ..... import MODEL_FILES_PATH
from .body_models import SMPLXLayer

CACHE_MODELS = None


def load_models():
    global CACHE_MODELS

    if CACHE_MODELS is None:
        smplx_layer_neutral = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_NEUTRAL.npz"),
            num_betas=10,
            use_pca=False,
            age="adult",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )
        # smplx_faces = torch.from_numpy(smplx_layer.faces.astype(np.int32))

        smplx_layer_male = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_MALE.npz"),
            num_betas=10,
            use_pca=False,
            age="adult",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )

        smplx_layer_female = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_FEMALE.npz"),
            num_betas=10,
            use_pca=False,
            age="adult",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )

        smplx_layer_neutral_kid = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_NEUTRAL.npz"),
            num_betas=10,
            use_pca=False,
            age="kid",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )
        # smplx_faces = torch.from_numpy(smplx_layer.faces.astype(np.int32))

        smplx_layer_male_kid = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_MALE.npz"),
            num_betas=10,
            use_pca=False,
            age="kid",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )

        smplx_layer_female_kid = SMPLXLayer(
            model_path=os.path.join(MODEL_FILES_PATH, "smplx/SMPLX_FEMALE.npz"),
            num_betas=10,
            use_pca=False,
            age="kid",
            kid_template_path=os.path.join(MODEL_FILES_PATH, "smplx_kid_template.npy"),
        )

        CACHE_MODELS = (
            smplx_layer_neutral,
            smplx_layer_male,
            smplx_layer_female,
            smplx_layer_neutral_kid,
            smplx_layer_male_kid,
            smplx_layer_female_kid,
        )
    return CACHE_MODELS
