import os
import sys
import uuid
import base64

import requests

# NVAI endpoint for the NV-DINOv2 NIM
nvai_url="https://ai.api.nvidia.com/v1/cv/nvidia/nv-dinov2"

header_auth = f"Bearer {os.getenv('NVIDIA_API_KEY')}"


def _upload_asset(input, description):
    """
    Uploads an asset to the NVCF API.
    :param input: The binary asset to upload
    :param description: A description of the asset

    """
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"

    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }

    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)

    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=input,
        headers=s3_headers,
        timeout=300,
    )

    response.raise_for_status()
    return uuid.UUID(asset_id)


if __name__ == "__main__":
    """Uploads an image of your choosing to the NVCF API and sends a
    request to the NV-DINOv2 classification model.
    The response is written to stdout.

    Note: You must set up an environment variable, NVIDIA_API_KEY.
    """

    if len(sys.argv) != 2:
        print("Usage: python test.py <image>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if len(image_b64) < 200_000 :
        # For images of size less than 200 KB send as base64 string
        payload = {
          "messages": [
            {
              "content": {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{image_b64}"
                  }
              }
            }
          ],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": header_auth,
            "Accept": "application/json"
        }
    else:
        # For images of size more than 200 KB use the NVCF assets API
        asset_id = _upload_asset(open(sys.argv[1], "rb"), "Input Image")

        payload = {"messages": []}

        asset_list = f"{asset_id}"

        headers = {
            "Content-Type": "application/json",
            "NVCF-INPUT-ASSET-REFERENCES": asset_list,
            "NVCF-FUNCTION-ASSET-IDS": asset_list,
            "Authorization": header_auth,
        }

    response = requests.post(nvai_url, headers=headers, json=payload)

    if response.status_code == 200:
        print(response.json())
    else:
        print('Inference failed.')