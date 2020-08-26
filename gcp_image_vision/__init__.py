import logging
import os

import boto3
from .processor import GCPVisionProcessor


def store_client(store: str) -> boto3:
    boto_session = boto3.Session(region_name=os.environ[f"KOYEB_STORE_{store}_REGION"])
    client = boto_session.resource(
        "s3",
        aws_access_key_id=os.environ[f"KOYEB_STORE_{store}_ACCESS_KEY"],
        aws_secret_access_key=os.environ[f"KOYEB_STORE_{store}_SECRET_KEY"],
        endpoint_url=os.environ[f"KOYEB_STORE_{store}_ENDPOINT"],
    )
    return client.Bucket(store)


def handler(event, context):
    logging.getLogger().setLevel(logging.DEBUG)
    print("Got data", event)
    print("Got context", context)
    #    print("Got context event", context.event)
    operation = os.environ["GCP_VISION_OPERATION"]
    logging.info("New operation {} starting".format(operation))

    obj_name = event["object"]["key"]
    gcp_key = os.environ["GCP_KEY"]
    store_name = event["bucket"]["name"]

    processor = GCPVisionProcessor(gcp_key, [operation], store_client(store_name))
    processor.process(obj_name)
    return {"response": "ok"}
