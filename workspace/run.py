from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils import *

load_dotenv()
Migrator().run()

query = "volume1"
delete = Model.find(Model.filename % query).all()
for r in tqdm(delete, desc="Deleting existing keys"):
    try:
        Model.delete(r.pk)
    except Exception as e:
        print("There was an exception with {}\n{}".format(r, e))


filenames = index_directory(
    CONTAINER_IMAGE_FOLDER, formats=(
        ".jpeg", ".jpg", ".png"))
for filename in tqdm(filenames, desc="Writing filenames to redis db."):
    Model(filename=filename).save()
models = Model.find(Model.filename % query).all()
triton_client = grpcclient.InferenceServerClient(
    url=TRITON_SERVER_URL, verbose=False)
model_metadata = triton_client.get_model_metadata(
    model_name=FACE_DETECT_MODEL_NAME, model_version=MODEL_VERSION
)
model_config = triton_client.get_model_config(
    model_name=FACE_DETECT_MODEL_NAME, model_version=MODEL_VERSION
).config
input_names = [i.name for i in model_config.input]
output_names = [i.name for i in model_config.output]

pbar = tqdm(
    total=len(models),
    desc="Submitting photos to {} at {}".format(
        FACE_DETECT_MODEL_NAME, TRITON_SERVER_URL
    ),
    unit_scale=True, unit_divisor=1024, miniters=1, smoothing=1
)

results = []

with ThreadPoolExecutor() as executor:
    for chunk in chunked(models, THREAD_CHUNKS):
        futures = []
        for model in chunk:
            futures.append(
                executor.submit(
                    submit_to_facedetect,
                    model.filename,
                    input_names,
                    output_names,
                    model.pk,
                )
            )

        for future in as_completed(futures):
            pbar.update()
            try:
                infer_result = future.result()
                model = Model.get(infer_result.get_response().id)
                image_wise_bboxes = infer_result.as_numpy(
                    output_names[0]).reshape(-1, 4)
                image_probas = infer_result.as_numpy(
                    output_names[1]).reshape(-1, 1)
                for bbox, proba in zip(image_wise_bboxes, image_probas):
                    model.faces = [
                        {
                            "bbox": {
                                "x1": int(bbox[0]),
                                "y1": int(bbox[1]),
                                "x2": int(bbox[2]),
                                "y2": int(bbox[3]),
                            },
                            "probability": int(proba[0]),
                        }
                        for bbox, proba in zip(image_wise_bboxes, image_probas)
                    ]
                model.save()

            except Exception as e:
                print("There was an exception: {}.".format(e))


triton_client = grpcclient.InferenceServerClient(
    url=TRITON_SERVER_URL, verbose=False)

model_metadata = triton_client.get_model_metadata(
    model_name=FPENET_MODEL_NAME, model_version=MODEL_VERSION
)

model_config = triton_client.get_model_config(
    model_name=FPENET_MODEL_NAME, model_version=MODEL_VERSION
).config

input_names = [i.name for i in model_config.input]
output_names = [i.name for i in model_config.output]


models = Model.find(Model.filename % query).all()
models = [model for model in models if model.faces]

pbar = tqdm(
    total=len(models),
    desc="Submitting photos to {} at {}".format(
        FPENET_MODEL_NAME,
        TRITON_SERVER_URL),
)

results = []

with ThreadPoolExecutor() as executor:
    for chunk in chunked(models, THREAD_CHUNKS):
        futures = []
        for model in chunk:
            if model.faces:
                futures.append(
                    executor.submit(
                        submit_to_fpenet,
                        model,
                        input_names[0],
                        output_names,
                        request_id=model.pk,
                    )
                )

        for future in as_completed(futures):
            pbar.update()
            try:
                infer_results = future.result()
                for i, infer_result in enumerate(infer_results):
                    model = Model.get(infer_result.get_response().id)
                    image_points = infer_result.as_numpy(
                        output_names[1]).squeeze()
                    model.faces[i].rotation = get_fpenet_rotation(image_points)
                    model.faces[i].descriptors = parse_descriptors(
                        image_points)
                    model.save()

            except Exception as e:
                print(
                    "There was an exception: {}\n{}.".format(
                        e, infer_result.get_response().id
                    )
                )
