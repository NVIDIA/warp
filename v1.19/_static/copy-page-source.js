// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

(function () {
    "use strict";

    async function fetchPageSource(sourceUrl, fetchSource) {
        const response = await fetchSource(sourceUrl);
        if (!response.ok) {
            throw new Error(`Unable to fetch page source (${response.status})`);
        }
        return response.text();
    }

    function copyPageSource(sourceText, clipboard) {
        if (typeof sourceText !== "string") {
            return Promise.reject(new Error("Page source is not ready"));
        }

        try {
            return Promise.resolve(clipboard.writeText(sourceText));
        } catch (error) {
            return Promise.reject(error);
        }
    }

    function bindCopyPageSourceControl(control, { fetchSource, clipboard, scheduleReset }) {
        const button = control.querySelector("[data-copy-page-source]");
        const label = control.querySelector("[data-copy-page-source-label]");
        const status = control.querySelector("[data-copy-page-source-status]");
        const defaultLabel = label.textContent;
        let sourceText;
        const resetStatus = () => {
            label.textContent = defaultLabel;
            status.textContent = "";
        };

        button.disabled = true;
        // Preload source so the click handler can call the Clipboard API before
        // Safari's transient user-activation window expires.
        const sourceReady = fetchPageSource(button.dataset.pageSourceUrl, fetchSource)
            .then((text) => {
                sourceText = text;
                button.disabled = false;
            })
            .catch(() => {
                label.textContent = "Copy unavailable";
                button.title = "Unable to load page source. Use View page source instead.";
            });

        button.addEventListener("click", () => {
            button.disabled = true;
            return copyPageSource(sourceText, clipboard)
                .then(() => {
                    label.textContent = "Copied";
                    status.textContent = "Page source copied to the clipboard.";
                    scheduleReset(resetStatus, 2000);
                })
                .catch(() => {
                    label.textContent = "Copy failed";
                    status.textContent = "Unable to copy page source. Use View page source instead.";
                    scheduleReset(resetStatus, 2000);
                })
                .finally(() => {
                    button.disabled = false;
                });
        });

        return sourceReady;
    }

    function initializeCopyPageSource(root, options) {
        return Promise.all(
            Array.from(root.querySelectorAll("[data-page-source-control]"), (control) =>
                bindCopyPageSourceControl(control, options),
            ),
        );
    }

    if (typeof window === "object" && typeof document === "object") {
        const initialize = () => {
            initializeCopyPageSource(document, {
                clipboard: window.navigator.clipboard,
                fetchSource: window.fetch.bind(window),
                scheduleReset: window.setTimeout.bind(window),
            });
        };

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", initialize);
        } else {
            initialize();
        }
    }
})();
