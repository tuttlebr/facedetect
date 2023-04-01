from datetime import datetime
from typing import List, Optional

from redis_om import Field, JsonModel, Migrator, get_redis_connection


class ImagesModel(JsonModel):
    """
    The images referenced by a COCO dataset are listed in the images array.
    Each image object contains information about the image such as the image
    file name.

    id: (Required) A unique identifier for the image. The id field maps to
        the id field in the annotations array (where bounding box information
        is stored).
    license: (Not Required) Maps to the license array.
    coco_url: (Optional) The location of the image.
    flickr_url: (Not required) The location of the image on Flickr.
    width: (Required) The width of the image.
    height: (Required) The height of the image.
    file_name: (Required) The image file name. In this example, file_name and
        id match, but this is not a requirement for COCO datasets.
    date_captured: (Required) the date and time the image was captured.
    """

    id: int = None
    license: Optional[int]
    coco_url: Optional[str]
    flickr_url: Optional[str]
    height: int
    width: int
    channels: int
    file_name: str = Field(index=True)
    date_captured: Optional[str] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class CategoriesModel(JsonModel):
    """
    Label information is stored the categories array.

    supercategory: (Not required) The parent category for a label.
    id: (Required) The label identifier. The id field maps to the category_id f
    ield in an annotation object.
    name: (Required) the label name.
    """

    supercategory: Optional[str] = Field(index=True)
    id: Optional[int] = None
    name: Optional[str] = Field(index=True)


class AnnotationsModel(JsonModel):
    """
    Bounding box information for all objects on all images is stored the
    annotations list. A single annotation object contains bounding box
    information for a single object and the object's label on an image. There is
    an annotation object for each instance of an object on an image.

    id: (Not required) The identifier for the annotation.
    image_id: (Required) Corresponds to the image id in the images array.
    category_id: (Required) The identifier for the label that identifies the
        object within a bounding box. It maps to the id field of the categories array.
    iscrowd: (Not required) Specifies if the image contains a crowd of objects.
    segmentation: (Not required) Segmentation information for objects on an image.
    area: (Not required) The area of the annotation.
    bbox: (Required) Contains the coordinates, in pixels, of a bounding box
        around an object on the image.
    """

    segmentation: Optional[List[List[float]]] = None
    iscrowd: Optional[bool] = None
    area: Optional[float] = None
    image_id: Optional[int] = None
    bbox: Optional[List[float]] = None
    rotation: Optional[float] = None
    center: Optional[List[float]] = None
    category_id: Optional[int] = None
    id: Optional[int] = None


class COCOModel(JsonModel):
    images: Optional[List[ImagesModel]] = None
    categories: Optional[List[CategoriesModel]] = None
    annotations: Optional[List[AnnotationsModel]] = None

    class Meta:
        database = get_redis_connection()
