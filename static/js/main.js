var socketConnections = {};
var activeTabName = '';
var cameraOn;

function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tabcontent");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tablinks");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";

    activeTabName = tabName; // Update the active tab name
    manageWebsockets(tabName);
}
let flash_status = "On"; // Global variable to keep track of flash state
let holdTimeout;
let disableTimeout;
let flashDiv; // Global variable to keep track of the flash div

function toggleFlash() {
    const flashIcon = document.getElementById('flashIcon');
    const flashButton = document.getElementById('flashButton');
    
    if (flash_status === "On") {
        flashIcon.src = "/static/flash_off.png";
        flash_status = "Off";
    } else if (flash_status === "Off") {
        flashIcon.src = "/static/flash_on.png";
        flash_status = "On";
    }
    console.log('toggling flash: ', flash_status);

    disableFlashButton(flashButton);
}

function startHold() {
    const flashIcon = document.getElementById('flashIcon');
    const flashButton = document.getElementById('flashButton');
    const createDatasetElement = document.getElementById('CreateDataset');

    if (flash_status === "Off") {
        holdTimeout = setTimeout(() => {
            flashIcon.src = "/static/flash_hold.png";
            flash_status = "hold";

            flashDiv = document.createElement('div');
            flashDiv.className = 'flash';
            createDatasetElement.appendChild(flashDiv);

            disableFlashButton(flashButton);
        }, 3000); // Change to hold icon after 3 seconds of holding
    } else if (flash_status === "hold") {
        holdTimeout = setTimeout(() => {
            flashIcon.src = "/static/flash_off.png";
            flash_status = "Off";
            if (flashDiv && flashDiv.parentNode === createDatasetElement) {
                createDatasetElement.removeChild(flashDiv);
            }
            disableFlashButton(flashButton);
        }, 3000); // Change to off icon after 3 seconds of holding
    }
}

function stopHold() {
    clearTimeout(holdTimeout);
}

function disableFlashButton(button) {
    button.disabled = true;
    disableTimeout = setTimeout(() => {
        button.disabled = false;
    }, 1000); // Disable the button for 1 second
}






function manageWebsockets(activeTab) {
    const createDatasetStreams = ['live-video-left-and-right-eye', 'live-video-1'];
    const testStreams = ['live-video-left', 'live-video-right'];
    console.log("Managing sockets, camera state: ", cameraOn);
    if (!cameraOn) {
        const streams = ['left', 'right', 'left-eye', 'right-eye', 'view1'];
        const videoElements = {
            'left': document.getElementById('live-video-left'),
            'right': document.getElementById('live-video-right'),
            'left-eye'  : document.getElementById('live-video-left-eye'),
            'right-eye' : document.getElementById('live-video-right-eye'),
            'view1'     : document.getElementById('live-video-1')
        };
        streams.forEach((stream) => {
            // Set placeholder image when camera is off
            videoElements[stream].src = '/static/no-video.png';
        });
        closeWebsockets(createDatasetStreams);
        closeWebsockets(testStreams);
    } else {
        console.log("my activeTab: ", activeTab);
        if (activeTab === 'CreateDataset') {
            openWebsockets(createDatasetStreams);
            closeWebsockets(testStreams);
        } else if (activeTab === 'Test') {
            openWebsockets(testStreams);
            closeWebsockets(createDatasetStreams);
        } else {
            closeWebsockets(createDatasetStreams);
            closeWebsockets(testStreams);
        }
    }
}

function openWebsockets(streams) {
    streams.forEach((stream) => {
        if (!socketConnections[stream]) {
            const socket = io('http://127.0.0.1:8000');
            socketConnections[stream] = socket;

            socket.on('connect', () => {
                console.log(`WebSocket connection established for ${stream}`);
                socket.emit('start_video', { stream: stream });
            });

            socket.on(stream, (data) => {
                var arrayBuffer = '';
                var left_eye_arrayBuffer = '';
                var right_eye_arrayBuffer = '';
                var blob = '';
                var left_eye_blob = '';
                var right_eye_blob = '';
                var url = '';
                var left_eye_url = '';
                var right_eye_url = '';
                var videoElement = '';
                var left_eye_videoElement = '';
                var right_eye_videoElement = '';

                if (stream === "live-video-left-and-right-eye"){
                    if (data.type === "left_and_right_eye"){
                        left_eye_arrayBuffer = data.left_eye;
                        right_eye_arrayBuffer = data.right_eye;

                        left_eye_blob = new Blob([left_eye_arrayBuffer], { type: 'image/jpeg' });
                        left_eye_url = URL.createObjectURL(left_eye_blob);
                        right_eye_blob = new Blob([right_eye_arrayBuffer], { type: 'image/jpeg' });
                        right_eye_url = URL.createObjectURL(right_eye_blob);


                        left_eye_videoElement = document.getElementById("live-video-left-eye");
                        right_eye_videoElement = document.getElementById("live-video-right-eye");

                        left_eye_videoElement.src = left_eye_url;
                        left_eye_videoElement.classList.add('visible');
                        right_eye_videoElement.src = right_eye_url;
                        right_eye_videoElement.classList.add('visible');
                    }else{
                        console.log("ERROR openWebsockets: Stream Type live-video-left-and-right-eye should have type left_eye or right_eye")
                    }
                }else{
                    arrayBuffer = data.frame;
                    blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                    url = URL.createObjectURL(blob);
                
                    videoElement = document.getElementById(stream);
                    videoElement.src = url;
                    videoElement.classList.add('visible');
                }
            });

            socket.on('error', (error) => {
                console.error(`WebSocket error for ${stream}:`, error);
            });

            socket.on('disconnect', () => {
                console.log(`WebSocket connection closed for ${stream}`);
                const leftVideoElement          = document.getElementById('live-video-left-eye');
                const rightVideoElement         = document.getElementById('live-video-right-eye');
                const leftEyeVideoElement       = document.getElementById('live-video-left-eye');
                const rightEyeVideoElement      = document.getElementById('live-video-right-eye');
                const live1VideoElement         = document.getElementById('live-video-1');

                leftVideoElement.classList.remove('visible');
                rightVideoElement.classList.remove('visible');
                leftEyeVideoElement.classList.remove('visible');
                rightEyeVideoElement.classList.remove('visible');
                live1VideoElement.classList.remove('visible');
                
            });
        }
    });
}




function closeWebsockets(streams) {
    streams.forEach((stream) => {
        if (socketConnections[stream]) {
            var videoElement = '';
            socketConnections[stream].disconnect();
            delete socketConnections[stream];
            if (stream === "live-video-left-and-right-eye"){
                videoElement = document.getElementById('live-video-left-eye');
                videoElement.src = '/static/no-video.png';
                videoElement.classList.remove('visible');

                videoElement = document.getElementById('live-video-right-eye');
                videoElement.src = '/static/no-video.png';
                videoElement.classList.remove('visible');
            }else{
                videoElement = document.getElementById(stream);
                videoElement.src = '/static/no-video.png';
                videoElement.classList.remove('visible');
            }
            
            
        }
    });
}

function toggleDatasetMode() {
    const newDatasetOptions = document.getElementById('newDatasetOptions');
    const existingDatasetOptions = document.getElementById('existingDatasetOptions');

    const datasetModeElement = document.querySelector('input[name="datasetMode"]:checked');
    if (!datasetModeElement) {
        alert('Please select a dataset mode.');
        return;
    }
    
    const datasetMode = datasetModeElement.value;

    if (datasetMode === 'new') {
        newDatasetOptions.classList.remove('hidden');
        existingDatasetOptions.classList.add('hidden');
    } else {
        newDatasetOptions.classList.add('hidden');
        existingDatasetOptions.classList.remove('hidden');
        
    }
}

document.addEventListener('DOMContentLoaded', (event) => {
    document.getElementById('correction-select-model').addEventListener('change', function () {
        const selectedModel = this.value;
        const checkpointDropdown = document.getElementById('correction-select-model-checkpoint');

        if (selectedModel === 'disabled')
        {
            checkpointDropdown.style.display = "none";
        }
        else{
            checkpointDropdown.style.display = "inline-block";
        }

        sendSelectedModel(selectedModel);
    });

    document.getElementById('correction-select-model-checkpoint').addEventListener('change', function () {
        const selectedCheckpoint = this.value;

        sendSelectedCheckpoint(selectedCheckpoint);
    });
});

function sendSelectedCheckpoint(selectedCheckpoint) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/set_checkpoint", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                let message = response.message;
                console.log('Set checkpoint result:', message);
            } else {
                console.error('Error:', xhr.responseText);
            }
        }
    };

    xhr.send(JSON.stringify({ checkpoint: selectedCheckpoint }));
}

function sendSelectedModel(model) {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/set_correction_model", true);
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
    
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                let checkpoint_list = response.checkpoint_list;
                console.log('Received Checkpoints:', checkpoint_list);

                const checkpointDropdown = document.getElementById('correction-select-model-checkpoint');
                
                // Clear any existing options
                checkpointDropdown.innerHTML = '';
                
                // Create a default option
                let defaultOption = document.createElement('option');
                defaultOption.text = "Latest";
                defaultOption.value = "latest";
                checkpointDropdown.add(defaultOption);

                // Populate dropdown with new checkpoints
                checkpoint_list.forEach(function(checkpoint) {
                    let option = document.createElement('option');
                    option.text = checkpoint;
                    option.value = checkpoint.split('-')[1];
                    checkpointDropdown.add(option);
                });

            } else {
                console.error('Error:', xhr.responseText);
            }
        }
    };

    xhr.send(JSON.stringify({ model: model }));
}



// Function to update the select options
function updateDatasetOptions() {
    // Create a new XMLHttpRequest object
    const xhr = new XMLHttpRequest();

    // Configure it: GET-request for the URL /backend/get_datasets
    xhr.open('GET', '/backend/get_models_and_datasets', true);

    // Set up the callback function for when the request completes
    xhr.onload = function() {
        if (xhr.status === 200) {
            // Parse the JSON response
            const newOptions = JSON.parse(xhr.responseText);

            // Get the select element by its id
            const selectElement = document.getElementById('existingDatasets');
            const trainingSelectElement = document.getElementById('training-tab-select-dataset');
            const correctionModelSelectElement = document.getElementById('correction-select-model');
            

            // Clear the existing options
            selectElement.innerHTML = '';
            trainingSelectElement.innerHTML = '';
            correctionModelSelectElement.innerHTML = '';

            // Add new options to the select element
            newOptions.datasets.forEach(function(option) {
                // Create a new option element
                const newOption = document.createElement('option');
                // Set the value and text content
                newOption.value = option.value;
                newOption.textContent = option.text;
                // Append the new option to the select element
                selectElement.appendChild(newOption);

                // Create a new option element
                const newOption2 = document.createElement('option');
                // Set the value and text content
                newOption2.value = option.value;
                newOption2.textContent = option.text;
                // Append the new option to the select element
                trainingSelectElement.appendChild(newOption2);
            });

            // Create a DISABLED option element
            const newOption3 = document.createElement('option');
            // Set the value and text content
            newOption3.value = 'disabled';
            newOption3.textContent = 'disabled';
            newOption3.selected = true;
            // Append the new option to the select element
            correctionModelSelectElement.appendChild(newOption3);

            // Add new options to the select element models
            newOptions.models.forEach(function(option) {
                // Create a new option element
                const newOption4 = document.createElement('option');
                // Set the value and text content
                newOption4.value = option.value;
                newOption4.textContent = option.text;
                // Append the new option to the select element
                correctionModelSelectElement.appendChild(newOption4);
            });
        } else {
            console.error('Failed to fetch dataset options:', xhr.statusText);
        }
    };

    // Set up the callback function for when the request fails
    xhr.onerror = function() {
        console.error('Request error');
    };

    // Send the request
    xhr.send();
}

// Call the function to update the select options when the page loads
document.addEventListener('DOMContentLoaded', updateDatasetOptions);

function downloadDataset() {
    const selectElement = document.getElementById('existingDatasets');
    const selectedDataset = selectElement.value;

    if (selectedDataset) {
        const downloadUrl = `/backend/download_dataset/${selectedDataset}`;
        window.location.href = downloadUrl;
    } else {
        alert('Please select a dataset to download.');
    }
}

function startNewDataset() {
    const datasetName = document.getElementById('datasetName').value;
    const selectElement = document.getElementById('existingDatasets');

    if (!datasetName) {
        alert('Please enter a dataset name.');
        return;
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/backend/create_dataset', true);
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');

    xhr.onload = function() {
        if (xhr.status === 201) {
            alert('Dataset created successfully.');
            document.querySelector('input[name="datasetMode"][value="existing"]').checked = true;
            updateDatasetOptions();
            // Wait for the update to complete, then set the value
            setTimeout(() => {
                selectElement.value = datasetName;
                console.log("selectElement.value: ", selectElement.value);
                toggleDatasetMode();
            }, 100); 
        } else {
            const response = JSON.parse(xhr.responseText);
            alert('Error creating dataset: ' + response.error);
        }
    };

    xhr.onerror = function() {
        alert('Request error');
    };

    xhr.send(JSON.stringify({ datasetName: datasetName }));
}

let eventSource;
let datasetLoadingSource;

function createPopup(id, message) {
    const existingPopupBackground = document.getElementById(`${id}-background`);
    if (existingPopupBackground) {
        document.body.removeChild(existingPopupBackground);
    }

    const popupBackground = document.createElement('div');
    popupBackground.className = 'training_popup-background';
    popupBackground.id = `${id}-background`;

    const popup = document.createElement('div');
    popup.className = 'training_popup';
    popup.id = id;

    const content = document.createElement('div');
    content.className = 'training_popup-content';

    const text = document.createElement('p');
    text.textContent = message;
    content.appendChild(text);

    if (id != 'training-started-popup') {
        const closeButton = document.createElement('button');
        closeButton.className = 'training_popup-close-button';
        closeButton.textContent = 'Continue';
        closeButton.onclick = () => {
            const popupBackground = document.getElementById(`${id}-background`);
            if (popupBackground) {
                document.body.removeChild(popupBackground);
            }
        };
        content.appendChild(closeButton);
    }

    popup.appendChild(content);
    popupBackground.appendChild(popup);
    document.body.appendChild(popupBackground);
}

function startTraining() {
    const modelName = document.getElementById('training-tab-model-name').value;
    const dataset = document.getElementById('training-tab-select-dataset').value;
    const epochs = document.getElementById('training-tab-epochs').value;
    const learningRate = document.getElementById('training-tab-learning-rate').value;

    if (!modelName || !dataset || !epochs || !learningRate) {
        console.log('Please fill in all fields.');
        alert('Please fill in all fields.');
        return;
    }

    const data = {
        model_name: modelName,
        dataset: dataset,
        epochs: parseInt(epochs),
        learning_rate: parseFloat(learningRate)
    };

    console.log('Sending data to backend:', data);
    updateProgressBar(1);
    
    if (eventSource) {
        eventSource.close();
    }
    if (datasetLoadingSource) {
        datasetLoadingSource.close();
    }

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/load_dataset', true);
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                console.log('Dataset loading started successfully.');
                createPopup('loading-dataset-popup', 'Loading Dataset...');

                // Handle dataset loading progress
                datasetLoadingSource = new EventSource('/dataset_loading_progress_stream');
                datasetLoadingSource.onmessage = function (event) {
                    const data = JSON.parse(event.data);
                    const progress = data.progress;
                    const file_count = data.file_count;
                    const total_files = data.total_files;

                    console.log('Dataset Loading Progress:', progress);
                    updateProgressBar(progress);
                    updateDatasetFileProgress(file_count, total_files, progress);

                    if (progress >= 100) {
                        datasetLoadingSource.close();
                        const loadingDatasetPopup = document.getElementById('loading-dataset-popup-background');
                        if (loadingDatasetPopup) {
                            document.body.removeChild(loadingDatasetPopup);
                        }
                        
                        createPopup('training-started-popup', 'Starting training, please wait...');
                        updateProgressBar(1);

                        updateEpochProgress("0", "0", "0")
                        // Start training after dataset loading is complete
                        console.log('Starting actual training.');
                        startActualTraining(`datasets/${dataset}`, epochs, learningRate);
                    }
                };

                resetGraphs();
            } else {
                console.log('Error loading dataset:', xhr.responseText);
            }
        }
    };

    xhr.send(JSON.stringify(data));
}

function startActualTraining(datasetPath, epochs, learningRate) {
    const data = {
        dataset_path: datasetPath,
        epochs: epochs,
        learning_rate: learningRate
    };

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/start_training', true);
    xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                eventSource = new EventSource('/training_progress');
                eventSource.onmessage = function (event) {
                    const data = JSON.parse(event.data);
                    const progress = data.progress;
                    const epoch_count = data.epoch_count; 
                    const total_epochs = data.total_epochs;
                    const inputImage = data.input_image;
                    const targetImage = data.target_image;
                    const predictedImage = data.predicted_image;
                    const generatorLoss = data.generator_loss;
                    const discriminatorLoss = data.discriminator_loss;

                    console.log('Training started successfully.');
                    const trainingStartedPopup = document.getElementById('training-started-popup-background');
                    if (trainingStartedPopup) {
                        document.body.removeChild(trainingStartedPopup);
                    }

                    console.log('Training Progress:', progress);
                    updateProgressBar(progress);
                    updateImages(inputImage, targetImage, predictedImage);
                    updateLossGraphs(generatorLoss, discriminatorLoss);
                    updateEpochProgress(epoch_count, total_epochs, progress);

                    if (progress >= 100) {
                        eventSource.close();
                        createPopup('training-completed-popup', 'Training is complete.');
                        updateDatasetOptions();
                    }
                };
            } else {
                console.log('Error starting training:', xhr.responseText);
            }
        }
    };

    xhr.send(JSON.stringify(data));
}

function updateEpochProgress(currentEpoch, totalEpochs, progress) {
    document.getElementById('epoch-progress').innerText = `| Progress ${progress}% | Epoch ${currentEpoch}/${totalEpochs}`;
}

function updateDatasetFileProgress(currentFile, totalFiles, fileProgresss) {
    document.getElementById('epoch-progress').innerText = `| Progress ${fileProgresss}% | Files ${currentFile}/${totalFiles} |`;
}

function updateProgressBar(progress) {
    const progressBar = document.getElementById('training-tab-loading-bar');
    if (progress === 0){progress = 1}
    progressBar.style.width = progress + '%';
    console.log('Updated progress bar to', progress, '%');
}


// Function to update the before and after images
function updateImages(inputImage, targetImage, predictedImage) {
    const beforeImageElement = document.getElementById('training-tab-before-image');
    const targetImageElement = document.getElementById('training-tab-target-image');
    const afterImageElement = document.getElementById('training-tab-after-image');
    console.log("Updating Images...")
    beforeImageElement.src = `${inputImage}`;
    targetImageElement.src = `${targetImage}`;
    afterImageElement.src = `${predictedImage}`;
    console.log('Updated images');
}

// // Call this function when the page loads
// document.addEventListener('DOMContentLoaded', (event) => {
//     updateImages(inputImage="models/Test1_Model/image_checkpoints/image_input_at_epoch_0_step_489.png",targetImage="models/Test1_Model/image_checkpoints/image_target_at_epoch_0_step_489.png", predictedImage="models/Test1_Model/image_checkpoints/image_predicted_at_epoch_0_step_489.png");
// });

function fetchImageList(epoch) {
    return fetch(`/image_checkpoints/${epoch}/`)
        .then(response => response.json());
}

function handleChartClick(event, chart) {
    const points = chart.getElementsAtEventForMode(event, 'nearest', { intersect: true }, false);

    if (points.length) {
        const firstPoint = points[0];
        const epoch = chart.data.labels[firstPoint.index]-1;

        fetchImageList(epoch).then(files => {
            const inputImage = files.input_image_path;
            const targetImage = files.target_image_path;
            const predictedImage = files.predicted_image_path;

            updateImages(inputImage, targetImage, predictedImage);
        });
    }
}



// Function to initialize the loss graphs
let genLossChart, discLossChart;

function initializeLossGraphs() {
    const genLossCtx = document.getElementById('training-tab-generator-loss-graph').getContext('2d');
    const discLossCtx = document.getElementById('training-tab-discriminator-loss-graph').getContext('2d');

    const commonOptions = {
        scales: {
            x: {
                title: {
                    display: true,
                    text: 'Epoch',
                    color: '#ffffff'
                },
                ticks: {
                    color: '#ffffff'
                },
                grid: {
                    color: '#ffffff'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Loss',
                    color: '#ffffff'
                },
                ticks: {
                    color: '#ffffff'
                },
                grid: {
                    color: '#ffffff'
                },
                beginAtZero: true
            }
        },
        plugins: {
            legend: {
                labels: {
                    color: '#ffffff'
                }
            }
        }
    };

    genLossChart = new Chart(genLossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Generator Loss',
                data: [],
                borderColor: 'rgb(99, 255, 104)',
                borderWidth: 1
            }]
        },
        options: {
            ...commonOptions,
            onClick: (event) => handleChartClick(event, genLossChart) // Add click event listener for genLossChart
        }
    });

    discLossChart = new Chart(discLossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Discriminator Loss',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            ...commonOptions,
            onClick: (event) => handleChartClick(event, discLossChart) // Add click event listener for discLossChart
        }
    });
}


// Function to update the loss graphs
function updateLossGraphs(generatorLoss, discriminatorLoss) {
    const currentEpoch = genLossChart.data.labels.length + 1;
    
    genLossChart.data.labels.push(currentEpoch);
    genLossChart.data.datasets[0].data.push(generatorLoss);
    genLossChart.update();

    discLossChart.data.labels.push(currentEpoch);
    discLossChart.data.datasets[0].data.push(discriminatorLoss);
    discLossChart.update();
}

// Function to reset the graphs
function resetGraphs() {
    genLossChart.data.labels = [];
    genLossChart.data.datasets[0].data = [];
    genLossChart.update();

    discLossChart.data.labels = [];
    discLossChart.data.datasets[0].data = [];
    discLossChart.update();
}

// Call this function when the page loads
document.addEventListener('DOMContentLoaded', (event) => {
    initializeLossGraphs();
});



document.addEventListener('DOMContentLoaded', (event) => {
    const tabButtons = document.querySelectorAll(".tab button");
    const activeIndicator = document.createElement("div");
    activeIndicator.classList.add("active-indicator");
    document.querySelector(".tab").appendChild(activeIndicator);
    
    function moveActiveIndicator(button) {
        const buttonRect = button.getBoundingClientRect();
        const tabRect = document.querySelector(".tab").getBoundingClientRect();
        activeIndicator.style.width = `${buttonRect.width}px`;
        activeIndicator.style.left = `${buttonRect.left - tabRect.left}px`;
    }
    
    tabButtons.forEach(button => {
        button.addEventListener("click", function() {
            tabButtons.forEach(btn => btn.classList.remove("active"));
            button.classList.add("active");
            moveActiveIndicator(button);
        });
    });
    
    // Initialize the indicator position based on the active tab
    const initialActiveButton = document.querySelector(".tab button.active") || tabButtons[0];
    moveActiveIndicator(initialActiveButton);

    const defaultTab = document.querySelector('.tablinks.active');
    if (defaultTab) {
        defaultTab.click();
    } else {
        openTab({ currentTarget: document.querySelector('.tablinks') }, 'Test');
    }

    document.getElementById('left-panel').classList.remove('hidden');
    document.getElementById('right-panel').classList.remove('hidden');
    panelsVisible = true;

    const bigRedButton = document.getElementById('bigRedButton');
    if (bigRedButton) {
        bigRedButton.addEventListener('click', () => captureImages(1)); // Default to 1 image
    } else {
        console.log('bigRedButton not found');
    }

    
    const numImages = document.getElementById('numImages');
    let nr_images;
    if (numImages) {
        nr_images = parseInt(numImages.value, 10);
        console.log("nr_images: ", nr_images);
    } else {
        console.log('numImages input not found');
    }

    const startExistingDatasetButton = document.getElementById('startExistingDatasetButton');
    if (startExistingDatasetButton) {
        startExistingDatasetButton.addEventListener('click', () => captureImages(nr_images)); // Pass the validated number of images
    } else {
        console.log('startExistingDatasetButton not found');
    }


    toggleDatasetMode();

    const statusMessage = document.getElementById('statusMessage');
    if (statusMessage) {
        statusMessage.addEventListener('click', toggleCamera); // Ensure this is present
    } else {
        console.log('statusMessage not found');
    }

    // Initialize WebSocket connections
    refreshLiveViews();

    async function captureImages() {
        const datasetModeElement = document.querySelector('input[name="datasetMode"]:checked');
        if (!datasetModeElement) {
            alert('Please select a dataset mode.');
            return;
        }
    
        const datasetMode = datasetModeElement.value;
        let datasetName;
    
        if (datasetMode === 'new') {
            datasetName = document.getElementById('datasetName').value;
            if (!datasetName) {
                alert('Please enter a dataset name.');
                return;
            }
        } else {
            datasetName = document.getElementById('existingDatasets').value;
            if (!datasetName) {
                alert('Please select an existing dataset.');
                return;
            }
        }
    
        const showPopup = (message, buttonText, inputField = false) => {
            return new Promise((resolve) => {
                const popup = document.createElement('div');
                popup.className = 'capture_image_popup';
    
                const background = document.createElement('div');
                background.className = 'capture_image_popup-background';
    
                const content = document.createElement('div');
                content.className = 'capture_image_popup-content';
    
                const infoText = document.createElement('p');
                infoText.innerHTML = message;
                content.appendChild(infoText);
    
                let inputElement;
                if (inputField) {
                    inputElement = document.createElement('input');
                    inputElement.type = 'number';
                    inputElement.min = '1';
                    inputElement.style.width = '100%';
                    inputElement.style.padding = '12px';
                    inputElement.style.marginBottom = '20px';
                    inputElement.style.border = '2px solid #3d5a80';
                    inputElement.style.borderRadius = '8px';
                    inputElement.style.boxSizing = 'border-box';
                    inputElement.style.fontSize = '14px';
                    inputElement.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                    inputElement.style.color = '#fff';
                    inputElement.style.transition = 'border-color 0.3s ease, box-shadow 0.3s ease';
                    
                    inputElement.onmouseover = () => {
                        inputElement.style.borderColor = '#2a3e5c';
                    };
    
                    inputElement.onfocus = () => {
                        inputElement.style.borderColor = '#1e2a38';
                        inputElement.style.boxShadow = '0 0 8px rgba(30, 42, 56, 0.7)';
                        inputElement.style.outline = 'none';
                    };
    
                    inputElement.onblur = () => {
                        inputElement.style.borderColor = '#3d5a80';
                        inputElement.style.boxShadow = 'none';
                    };
    
                    content.appendChild(inputElement);
                }
    
                const confirmButton = document.createElement('button');
                confirmButton.textContent = buttonText;
                confirmButton.className = 'capture_image_popup-confirm-button';
                confirmButton.onclick = () => {
                    const result = inputField ? inputElement.value : null;
                    document.body.removeChild(popup);
                    document.body.removeChild(background);
                    resolve(result);
                };
                content.appendChild(confirmButton);
    
                popup.appendChild(content);
                document.body.appendChild(background);
                document.body.appendChild(popup);
            });
        };
    
        const nr_images = await showPopup(`
            You will take 2 sets of pictures.<br><br>
            <strong>Set 1:</strong><br>
            Direct your head in a direction and keep your eyes focused on the camera.<br><br>
            <strong>Set 2:</strong><br>
            Keep your head still and point your eyes away from the camera.<br><br>
            Please enter the number of images per set:
        `, 'Continue to Set 1', true);
    
        if (!nr_images) {
            alert('Please enter a valid number of images.');
            return;
        }
    
        const captureSet = async (cameraDirection, setDescription) => {
            await showPopup(setDescription, 'Start');
    
            const payload = {
                datasetMode: datasetMode,
                datasetName: datasetName,
                cameraDirection: cameraDirection
            };
    
            const createDatasetElement = document.getElementById('CreateDataset');
    
            const progressDiv = document.createElement('div');
            progressDiv.className = 'progress-display';
            progressDiv.style.position = 'fixed';
            progressDiv.style.top = '10%';
            progressDiv.style.left = '50%';
            progressDiv.style.transform = 'translateX(-50%)';
            progressDiv.style.backgroundColor = 'black';
            progressDiv.style.color = 'white';
            progressDiv.style.padding = '10px 20px';
            progressDiv.style.fontSize = '24px';
            progressDiv.style.zIndex = '1000';
            document.body.appendChild(progressDiv);
    
            let flashDiv = '';
            if (flash_status === 'on') {
                flashDiv = document.createElement('div');
                flashDiv.className = 'flash';
                createDatasetElement.appendChild(flashDiv);
            }
    
            const minimumDuration = (duration) => {
                return new Promise(resolve => setTimeout(resolve, duration));
            };
    
            for (let i = 0; i < nr_images; i++) {
                progressDiv.textContent = `Image ${i + 1}/${nr_images}`;
                try {
                    const response = await fetch('/backend/capture-images', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(payload)
                    });
    
                    if (response.ok) {
                        const result = await response.json();
                        console.log(result.message);
                    } else {
                        const errorData = await response.json();
                        console.log(`Failed to capture images: ${errorData.error}`);
                    }
                } catch (error) {
                    console.error('Error capturing images:', error);
                    alert('Error capturing images.');
                } finally {
                    // if (flash_status === 'on') {
                    //     setTimeout(() => {
                    //         flashDiv.classList.remove('flash');
                    //         setTimeout(() => {
                    //             flashDiv.classList.add('flash');
                    //             setTimeout(() => {
                    //                 flashDiv.classList.remove('flash');
                    //                 if (i === nr_images - 1) {
                    //                     createDatasetElement.removeChild(flashDiv);
                    //                 }
                    //             }, 100);
                    //         }, 100);
                    //     }, 100);
                    // }
                }
    
                // await minimumDuration(500);  // Ensure at least 500ms between images
            }
    
            document.body.removeChild(progressDiv);
        };
    
        await captureSet('lookingAtCamera', 'Set 1: Direct your head in any direction and keep your eyes pointed at the camera.');
        await captureSet('awayFromCamera', 'Set 2: Keep your head in the same direction as Set 1 and point your eyes in the same direction as your head.');
        alert("Dataset Captured Successfully")
    }
    
     
    

    async function toggleCamera() {
        console.log('toggleCamera function called');
        const statusMessage = document.getElementById('statusMessage');

        // Clear existing status classes
        statusMessage.className = '';

        statusMessage.classList.add('disabled');  // Add disabled class to indicate processing
        statusMessage.classList.add('info'); // Add info class

        // Start the animation
        startDotDotDotAnimation(statusMessage, 'Waiting for camera');

        try {
            console.log('Sending POST request to /backend/toggle-camera');
            const response = await fetch('/backend/toggle-camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            console.log('Response received');
            if (response.ok) {
                const result = await response.json();
                console.log('Response OK:', result);
                statusMessage.textContent = `${result.status}`;  // Update status message
                statusMessage.classList.remove('info');  // Remove info class
                statusMessage.classList.add('success');  // Add success class

                // Refresh live view parts
                console.log("my toggle camera before: ", cameraOn)
                if (result.status === 'On'){
                    cameraOn =  true;
                }else{
                    cameraOn =  false;
                }
                
                console.log("my toggle camera after: ", cameraOn)
                refreshLiveViews();

            } else {
                const errorData = await response.json();
                console.log('Response error:', errorData);
                statusMessage.textContent = `Failed to toggle camera: ${errorData.error}`;  // Update status message
                statusMessage.classList.remove('info');  // Remove info class
                statusMessage.classList.add('error');  // Add error class
            }
        } catch (error) {
            console.error('Error toggling camera:', error);
            statusMessage.textContent = 'Error toggling camera.';  // Update status message
            statusMessage.classList.remove('info');  // Remove info class
            statusMessage.classList.add('error');  // Add error class
        } finally {
            clearInterval(animationInterval);  // Stop the animation
            setTimeout(() => {
                statusMessage.classList.remove('disabled');  // Remove disabled class after processing
            }, 2000);  // Delay to give user feedback
        }
    }

    function refreshLiveViews() {
        manageWebsockets(activeTabName);
    }

    function startDotDotDotAnimation(element, baseText) {
        let dotCount = 0;
        animationInterval = setInterval(() => {
            let dots = '.'.repeat(dotCount % 4);
            element.textContent = `${baseText}${dots}`;
            dotCount++;
        }, 500);  // Adjust the interval as needed
    }

    function toggleDatasetMode() {
        const newDatasetOptions = document.getElementById('newDatasetOptions');
        const existingDatasetOptions = document.getElementById('existingDatasetOptions');

        const datasetModeElement = document.querySelector('input[name="datasetMode"]:checked');
        if (!datasetModeElement) {
            alert('Please select a dataset mode.');
            return;
        }

        const datasetMode = datasetModeElement.value;

        if (datasetMode === 'new') {
            newDatasetOptions.classList.remove('hidden');
            existingDatasetOptions.classList.add('hidden');
        } else {
            newDatasetOptions.classList.add('hidden');
            existingDatasetOptions.classList.remove('hidden');
        }
    }

    (async () => {
        console.log('Checking initial camera state');
        const statusMessage = document.getElementById('statusMessage');

        // Clear existing status classes
        statusMessage.className = '';

        statusMessage.classList.add('disabled');  // Add disabled class to indicate processing
        statusMessage.classList.add('info'); // Add info class

        try {
            const response = await fetch('/backend/camera-status', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Initial camera state retrieved successfully:', result);
                statusMessage.textContent = `${result.camera_on ? 'On' : 'Off'}`;
                statusMessage.classList.remove('info');  // Remove info class
                statusMessage.classList.add('success');  // Add success class

                // Initialize WebSocket connections
                
                cameraOn = result.camera_on;
                manageWebsockets(activeTabName)

            } else {
                const errorData = await response.json();
                console.log('Error checking initial camera state:', errorData);
                statusMessage.textContent = `Failed to check camera state: ${errorData.error}`;
                statusMessage.classList.remove('info');  // Remove info class
                statusMessage.classList.add('error');  // Add error class
            }
        } catch (error) {
            console.error('Error checking initial camera state:', error);
            statusMessage.textContent = 'Error checking camera state.';
            statusMessage.classList.remove('info');  // Remove info class
            statusMessage.classList.add('error');  // Add error class
        } finally {
            statusMessage.classList.remove('disabled');  // Remove disabled class after processing
        }
    })();    
});
