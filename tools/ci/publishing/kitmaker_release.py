# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "requests",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Create and monitor releases on the Kitmaker portal.

All configuration (portal URL, project ID, etc.) is read from environment
variables, with optional CLI argument overrides.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class KitmakerClient:
    """Client for the Kitmaker release API with retry and monitoring support."""

    def __init__(self, base_url: str, api_token: str, verify_ssl: bool = True):
        self.base_url = base_url
        self.verify_ssl = verify_ssl
        if not verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)

    def create_release(self, project_id, project_name, pic_email, wheels, upload=True):
        """Create a release and return the release UUID."""
        url = f"{self.base_url}/projects/{project_id}/releases"
        payload = {
            "project_name": project_name,
            "payload": [
                {
                    "pic": pic_email,
                    "job_type": "wheel-release-job",
                    "publish_to": "both_devzone_pypi",
                    "url": wheel_url,
                    "size": size,
                    "upload": upload,
                }
                for wheel_url, size in wheels
            ],
        }

        print("Creating release...")
        response = self.session.post(url, json=payload, verify=self.verify_ssl, timeout=60)

        print(f"\n--- API RESPONSE ---")
        print(f"HTTP Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
        print("-------------------\n")

        if response.status_code >= 400:
            print("--- REQUEST DETAILS (for debugging) ---")
            print(f"URL: {url}")
            print(f"Payload sent:")
            print(json.dumps(payload, indent=2))
            print("---------------------------------------\n")

        response.raise_for_status()

        data = response.json()
        release_uuid = data.get("release_uuid")
        if not release_uuid:
            raise ValueError(f"No release_uuid in response: {data}")

        print(f"Created release: {release_uuid}")
        return release_uuid

    def monitor_release(self, release_uuid, poll_interval=30, timeout=None):
        """Poll release status until completed, failed, or timeout."""
        print(f"\nMonitoring release progress (polling every {poll_interval}s)...")
        start_time = time.time()

        while True:
            if timeout is not None and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Release monitoring exceeded {timeout}s timeout")

            try:
                url = f"{self.base_url}/status/{release_uuid}"
                response = self.session.get(url, verify=self.verify_ssl, timeout=30)
                response.raise_for_status()
                status_data = response.json()
                status = status_data.get("status", "unknown")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Release status: {status}")

                if status == "completed":
                    print("Release completed successfully!")
                    return True
                elif status == "failed":
                    print(f"Release failed! Response: {status_data}")
                    return False

            except (requests.RequestException, ValueError) as e:
                print(f"Warning: Status check failed: {e}")
                print(f"Retrying in {poll_interval} seconds...")

            time.sleep(poll_interval)


def infer_wheel_size(wheel_url):
    """Return ``"small"`` for macOS wheels, ``"medium"`` otherwise."""
    url_lower = wheel_url.lower()
    if "macosx" in url_lower:
        return "small"
    return "medium"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create and monitor releases on the Kitmaker portal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python kitmaker_release.py \\
    https://<host>/<path>/warp_lang-1.0.0-py3-none-manylinux_2_28_x86_64.whl \\
    https://<host>/<path>/warp_lang-1.0.0-py3-none-macosx_11_0_arm64.whl \\
    --dry-run

Environment Variables:
  KITMAKER_API_TOKEN      Required. API token for authentication.
  KITMAKER_BASE_URL       Required. Base URL for the Kitmaker API.
  KITMAKER_PROJECT_ID     Required. Kitmaker project ID.
  KITMAKER_PROJECT_NAME   Required. Kitmaker project name.
  KITMAKER_PIC_EMAIL      Required. Person-in-charge email address.
        """,
    )

    parser.add_argument("wheel_urls", nargs="+", metavar="WHEEL_URL",
                        help="HTTPS URL(s) to .whl file(s) (1-4 URLs)")
    parser.add_argument("--size", choices=["small", "medium", "large"],
                        help="Job size override for all wheels (default: auto-detect)")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status checks (default: 30)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max seconds to wait for completion (default: 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate without uploading")
    parser.add_argument("--base-url", default=None,
                        help="Kitmaker API base URL (default: KITMAKER_BASE_URL env var)")
    parser.add_argument("--project-id", default=None,
                        help="Kitmaker project ID (default: KITMAKER_PROJECT_ID env var)")
    parser.add_argument("--project-name", default=None,
                        help="Kitmaker project name (default: KITMAKER_PROJECT_NAME env var)")
    parser.add_argument("--pic-email", default=None,
                        help="Person-in-charge email (default: KITMAKER_PIC_EMAIL env var)")
    parser.add_argument("--no-verify-ssl", action="store_true",
                        help="Disable SSL certificate verification")

    args = parser.parse_args(argv)

    # Resolve config: CLI args take priority over env vars
    config = {
        "base_url": args.base_url or os.environ.get("KITMAKER_BASE_URL"),
        "project_id": args.project_id or os.environ.get("KITMAKER_PROJECT_ID"),
        "project_name": args.project_name or os.environ.get("KITMAKER_PROJECT_NAME"),
        "pic_email": args.pic_email or os.environ.get("KITMAKER_PIC_EMAIL"),
    }
    missing = [k for k, v in config.items() if not v]
    if missing:
        print(f"Error: Missing required configuration: {', '.join(missing)}")
        return 1

    api_token = os.environ.get("KITMAKER_API_TOKEN")
    if not api_token:
        print("Error: KITMAKER_API_TOKEN environment variable is not set")
        return 1

    client = KitmakerClient(config["base_url"], api_token, verify_ssl=not args.no_verify_ssl)

    try:
        if len(args.wheel_urls) > 4:
            print(f"Error: Too many wheel URLs ({len(args.wheel_urls)}). Maximum is 4.")
            return 1

        wheels = []
        for wheel_url in args.wheel_urls:
            if not wheel_url.startswith("https://") or not wheel_url.lower().endswith(".whl"):
                print(f"Error: Invalid wheel URL (must be https:// and end with .whl): {wheel_url}")
                return 1
            wheels.append((wheel_url, args.size or infer_wheel_size(wheel_url)))

        # Configuration summary
        print(f"\nProject: {config['project_name']} (ID: {config['project_id']})")
        print(f"PIC: {config['pic_email']}")
        print(f"Upload: {'No (dry-run)' if args.dry_run else 'Yes'}")
        print(f"Timeout: {args.timeout}s | Poll interval: {args.poll_interval}s")
        for url, size in wheels:
            print(f"  {url} ({size})")

        release_uuid = client.create_release(
            project_id=config["project_id"],
            project_name=config["project_name"],
            pic_email=config["pic_email"],
            wheels=wheels,
            upload=not args.dry_run,
        )

        success = client.monitor_release(
            release_uuid,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
        return 0 if success else 1

    except requests.HTTPError as e:
        print(f"\nAPI Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
        return 1
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
