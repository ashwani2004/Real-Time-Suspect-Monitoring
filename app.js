const videoFeed = document.getElementById("videoFeed");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");

const suspectCount = document.getElementById("suspectCount");
const logCount = document.getElementById("logCount");
const systemStatus = document.getElementById("systemStatus");
const latestDetection = document.getElementById("latestDetection");
const logsList = document.getElementById("logsList");
const suspectsGrid = document.getElementById("suspectsGrid");

let cameraActive = false;

function startCamera() {
    if (cameraActive) {
        return;
    }
    videoFeed.src = "/video_feed";
    videoFeed.classList.add("active");
    videoPlaceholder.style.display = "none";
    cameraActive = true;
}

function stopCamera() {
    videoFeed.removeAttribute("src");
    videoFeed.classList.remove("active");
    videoPlaceholder.style.display = "grid";
    cameraActive = false;
}

function renderLatestDetection(item) {
    if (!item) {
        latestDetection.textContent = "No detections yet.";
        return;
    }

    latestDetection.innerHTML = `
        <strong>${item.name}</strong>
        <div class="log-meta">
            ${item.timestamp}<br>
            Confidence: ${item.confidence}<br>
            Status: ${item.status}
        </div>
    `;
}

function renderLogs(items) {
    if (!items.length) {
        logsList.innerHTML = '<div class="empty-state">No alert logs yet.</div>';
        return;
    }

    logsList.innerHTML = items.map((item) => `
        <article class="log-card">
            ${item.image_url ? `<img class="log-thumb" src="${item.image_url}" alt="${item.name} alert">` : '<div class="log-thumb"></div>'}
            <div class="log-body">
                <strong>${item.name}</strong>
                <div class="log-meta">
                    ${item.timestamp}<br>
                    Confidence: ${item.confidence}<br>
                    Status: ${item.status}
                </div>
            </div>
        </article>
    `).join("");
}

function renderSuspects(items) {
    if (!items.length) {
        suspectsGrid.innerHTML = '<div class="empty-state">No suspects found in MongoDB.</div>';
        return;
    }

    suspectsGrid.innerHTML = items.map((item) => `
        <article class="suspect-card">
            ${item.image_url ? `<img src="${item.image_url}" alt="${item.name}">` : '<div class="empty-state">No image</div>'}
            <div class="suspect-content">
                <strong>${item.name}</strong>
                <div class="suspect-meta">
                    Samples: ${item.sample_count}<br>
                    Processed: ${item.processed_images}<br>
                    Skipped: ${item.skipped_images}<br>
                    Model: ${item.model_name}
                </div>
            </div>
        </article>
    `).join("");
}

async function loadStats() {
    const response = await fetch("/api/stats");
    const data = await response.json();
    suspectCount.textContent = data.suspect_count;
    logCount.textContent = data.log_count;
    systemStatus.textContent = data.status;
    renderLatestDetection(data.latest_detection);
}

async function loadLogs() {
    const response = await fetch("/api/logs");
    renderLogs(await response.json());
}

async function loadSuspects() {
    const response = await fetch("/api/suspects");
    renderSuspects(await response.json());
}

async function refreshDashboard() {
    try {
        await Promise.all([loadStats(), loadLogs(), loadSuspects()]);
    } catch (error) {
        console.error("Dashboard refresh failed", error);
    }
}

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);

refreshDashboard();
setInterval(refreshDashboard, 5000);
