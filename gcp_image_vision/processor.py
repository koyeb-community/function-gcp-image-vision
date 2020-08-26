import json
import logging
import google.oauth2.service_account

from google.cloud import vision
from google.protobuf.json_format import MessageToJson


VISION_OPERATIONS = {
    "face-detection": vision.enums.Feature.Type.FACE_DETECTION,
    "image-properties": vision.enums.Feature.Type.IMAGE_PROPERTIES,
    "label-detection": vision.enums.Feature.Type.LABEL_DETECTION,
    "landmark-detection": vision.enums.Feature.Type.LANDMARK_DETECTION,
    "logo-detection": vision.enums.Feature.Type.LOGO_DETECTION,
    "object-localization": vision.enums.Feature.Type.OBJECT_LOCALIZATION,
    "safe-search-detection": vision.enums.Feature.Type.SAFE_SEARCH_DETECTION,
    "web-detection": vision.enums.Feature.Type.WEB_DETECTION,
    "text-detection": vision.enums.Feature.Type.TEXT_DETECTION,
    "document-text-detection": vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION,
}


class GCPVisionProcessor(object):
    def __init__(self, gcp_key, features, store):
        self.logger = logging.getLogger()
        self.client = self._get_gcp_authenticated_client(gcp_key)
        self.features = features
        self.request_features = []
        self.store = store
        for feat in features:
            assert feat in VISION_OPERATIONS
            self.request_features.append({"type": VISION_OPERATIONS[feat]})

        self._file_ext = "{}{}".format("gcp-vision", "-".join(features))

    def _append_file_ext(self, name):
        return "{}.{}.json".format(name, self._file_ext)

    def _get_gcp_authenticated_client(self, gcp_key):
        service_account_info = json.loads(gcp_key)
        credentials = (
            google.oauth2.service_account.Credentials.from_service_account_info(
                service_account_info
            )
        )
        # FIXME: Add support of regions when GCP do support them
        return vision.ImageAnnotatorClient(credentials=credentials)

    def process(self, obj_key):
        self.logger.debug("Reading image")
        obj = self.store.Object(obj_key).get()
        content = obj["Body"].read()
        self.logger.debug("Preparing image")
        image = vision.types.Image(content=content)
        self.logger.info("Contacting Google")
        response = self.client.annotate_image(
            {"image": image, "features": self.request_features}
        )

        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(
                    response.error.message
                )
            )

        # Save JSON response
        new_json_file = self.store.put_object(
            Body=MessageToJson(response),
            Key=self._append_file_ext(obj_key),
            ContentType="application/json",
        )
        self.logger.info("File saved {}".format(new_json_file))

        return [new_json_file]
