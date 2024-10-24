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


// Initial state for each button
// Assign variables to the window object
window.face_mesh = false;
window.mouth_grid = false;
window.eye_grids = false;
window.grid_labels = false;
window.classify_gaze = false;
window.head_outline = false;
window.eye_bounding_boxes = false;
window.eye_extraction = false;

let currentStream = null; // Store the current stream to use in the emit

// Function to toggle the state and apply CSS
function toggleFeature(button, button_description, feature, socket, stream) {
    // Log the feature to ensure it's being passed correctly
    console.log(`Toggling feature: ${feature}`);

    // Check if the feature exists on window
    if (typeof window[feature] !== 'undefined') {
        // Toggle the feature variable
        window[feature] = !window[feature];
        console.log(`Feature ${feature} is now ${window[feature]} for stream ${stream}`);
    } else {
        console.error(`Feature ${feature} does not exist on window`);
        return;  // Exit if feature doesn't exist
    }

    // Toggle the button's visual state
    if (window[feature]) {
        button.style.transform = 'scale(1.15) translateY(-15px)';
        button_description.style.textDecoration = 'underline';  // Apply underline
        button_description.style.textDecorationColor = 'white'; // Set underline color to white
        button_description.style.transition = 'text-decoration-color 0.3s ease'; // Smooth transition
    } else {
        button.style.transform = 'none'; // Reset the transformation
        button_description.style.textDecoration = 'none'; // Remove underline
    }

    // Only emit updated settings if the stream is "live-video-left"
    if (stream === 'live-video-left') {
        // Log current state before emitting
        console.log(`Emitting updated settings to server for ${stream}:`);
        socket.emit('update_settings', {
            stream: currentStream,
            show_face_mesh: window.face_mesh,
            classify_gaze: window.classify_gaze,
            draw_rectangles: window.eye_bounding_boxes,
            show_eyes: window.eye_grids,
            show_mouth: window.mouth_grid,
            show_face_outline: window.head_outline,
            show_text: window.grid_labels,
            extract_eyes: window.eye_extraction
        });
        console.log(`Settings updated for ${stream}`);
    }
}


function openWebsockets(streams) {
    streams.forEach((stream) => {
        if (!socketConnections[stream]) {
            console.log(`Opening WebSocket connection for stream: ${stream}`);
            const socket = io('http://127.0.0.1:8000');
            socketConnections[stream] = socket;
            currentStream = stream; // Set the current stream
            console.log(`Current stream set to ${currentStream}`);

            socket.on('connect', () => {
                console.log(`WebSocket connection established for ${stream}`);

                // Initial emit of the settings when the connection is established
                console.log("Sending initial settings to server:");
                socket.emit('start_video', {
                    stream: stream,
                    show_face_mesh: window.face_mesh,
                    classify_gaze: window.classify_gaze,
                    draw_rectangles: window.eye_bounding_boxes,
                    show_eyes: window.eye_grids,
                    show_mouth: window.mouth_grid,
                    show_face_outline: window.head_outline,
                    show_text: window.grid_labels,
                    extract_eyes: window.eye_extraction
                });
                console.log("Initial settings sent");

                // If the current stream is 'live-video-left', bind the button events
                if (stream === 'live-video-left') {
                    console.log('Assigning button click listeners for live-video-left stream.');

                    const buttons = document.querySelectorAll('.performance-button');
                    const buttonDescriptions = document.querySelectorAll('.button-description');

                    buttons[0].addEventListener('click', () => {
                        console.log('Face mesh button clicked');
                        toggleFeature(buttons[0], buttonDescriptions[0], 'face_mesh', socket, stream);
                    });
                    buttons[1].addEventListener('click', () => {
                        console.log('Mouth grid button clicked');
                        toggleFeature(buttons[1], buttonDescriptions[1], 'mouth_grid', socket, stream);
                    });
                    buttons[2].addEventListener('click', () => {
                        console.log('Eye grids button clicked');
                        toggleFeature(buttons[2], buttonDescriptions[2], 'eye_grids', socket, stream);
                    });
                    buttons[3].addEventListener('click', () => {
                        console.log('Grid labels button clicked');
                        toggleFeature(buttons[3], buttonDescriptions[3], 'grid_labels', socket, stream);
                    });
                    buttons[4].addEventListener('click', () => {
                        console.log('Classify gaze button clicked');
                        toggleFeature(buttons[4], buttonDescriptions[4], 'classify_gaze', socket, stream);
                    });
                    buttons[5].addEventListener('click', () => {
                        console.log('Head outline button clicked');
                        toggleFeature(buttons[5], buttonDescriptions[5], 'head_outline', socket, stream);
                    });
                    buttons[6].addEventListener('click', () => {
                        console.log('Eye bounding boxes button clicked');
                        toggleFeature(buttons[6], buttonDescriptions[6], 'eye_bounding_boxes', socket, stream);
                    });
                    buttons[7].addEventListener('click', () => {
                        console.log('Eye extraction button clicked');
                        toggleFeature(buttons[7], buttonDescriptions[7], 'eye_extraction', socket, stream);
                    });
                }
            });

            socket.on(stream, (data) => {
                // Handle incoming frames for left and right eyes
                if (stream === "live-video-left-and-right-eye") {
                    if (data.type === "left_and_right_eye") {
                        console.log("Received left and right eye frames");
            
                        // Process left eye frame
                        let leftArrayBuffer = data.left_frame;
                        let leftBlob = new Blob([leftArrayBuffer], { type: 'image/jpeg' });
                        let leftUrl = URL.createObjectURL(leftBlob);
                        let leftEyeElement = document.getElementById("live-video-left");
                        leftEyeElement.src = leftUrl;
                        leftEyeElement.classList.add('visible');
                        console.log("Left eye frame received and displayed");
            
                        // Process right eye frame
                        let rightArrayBuffer = data.right_frame;
                        let rightBlob = new Blob([rightArrayBuffer], { type: 'image/jpeg' });
                        let rightUrl = URL.createObjectURL(rightBlob);
                        let rightEyeElement = document.getElementById("live-video-right");
                        rightEyeElement.src = rightUrl;
                        rightEyeElement.classList.add('visible');
                        console.log("Right eye frame received and displayed");
            
                    } else {
                        console.log("ERROR: Stream Type live-video-left-and-right-eye should have type left_and_right_eye");
                    }
                } else {
                    // Handle other streams (non-eye streams)
                    let arrayBuffer = data.frame;
                    let blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                    let url = URL.createObjectURL(blob);
                    let videoElement = document.getElementById(stream);
                    videoElement.src = url;
                    videoElement.classList.add('visible');
                    console.log(`Frame received and displayed for ${stream}`);
                }
            });

            socket.on('error', (error) => {
                console.error(`WebSocket error for ${stream}:`, error);
            });

            socket.on('disconnect', () => {
                console.log(`WebSocket connection closed for ${stream}`);
                // Handle cleanup after disconnect
                document.getElementById('live-video-left-eye').classList.remove('visible');
                document.getElementById('live-video-right-eye').classList.remove('visible');
                document.getElementById('live-video-1').classList.remove('visible');
            });
        }
    });
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

const correctionSelectModel = document.getElementById('correction-select-model');
const checkpointDropdown = document.getElementById('correction-select-model-checkpoint');
let selectedModel = '';
// Function to toggle the correction select dropdown based on cameraOn value
function toggleCorrectionSelectModel() {
    correctionSelectModel.disabled = !cameraOn;
}

document.addEventListener('DOMContentLoaded', (event) => {
    // sendSelectedModel(selectedModel);
    // document.getElementById('correction-select-model').addEventListener('click', function () {
    //     if (!cameraOn) {
    //         alert('Please turn on the camera to activate the gaze correction.');
    //         return; // Prevent further execution if the camera is off
    //     }
    // });
    
    document.getElementById('correction-select-model').addEventListener('change', function () {
        selectedModel = this.value;
        console.log('selectedModel: ', selectedModel);
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
    const calibrationBtn = document.getElementById('calibration-btn');
    // Show calibration button only if the selected value is 'auto'
    if (model === 'Auto') {
        console.log('calibration button enabled');
        calibrationBtn.style.display = 'inline-block';
    } else {
        console.log('calibration button disabled');
        calibrationBtn.style.display = 'none';
    }

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

// Function to create the popup with Yes/No buttons
function createPopup(id, message, yesCallback, noCallback) {
    const existingPopupBackground = document.getElementById(`${id}-background`);
    if (existingPopupBackground) {
        document.body.removeChild(existingPopupBackground);
    }

    // Create the darkened background
    const popupBackground = document.createElement('div');
    popupBackground.className = 'training_popup-background';
    popupBackground.id = `${id}-background`;

    // Create the popup container
    const popup = document.createElement('div');
    popup.className = 'training_popup';
    popup.id = id;

    // Create the popup content container
    const content = document.createElement('div');
    content.className = 'training_popup-content';

    // Add the message text
    const text = document.createElement('p');
    text.textContent = message;
    content.appendChild(text);

    if ((message==='Loading Dataset...') || (message==='Starting training, this can take up to 5 minutes please wait...')){
        // Add the Yes button
        const continueButton = document.createElement('button');
        continueButton.className = 'training_popup-yes-button';
        continueButton.textContent = 'Continue';
        continueButton.onclick = () => {
            document.body.removeChild(popupBackground);
            if (yesCallback) yesCallback(); // Trigger the Yes callback
        };
        content.appendChild(continueButton);
    }else if(message ==='Training is complete.'){
        // Add the Yes button
        const doneButton = document.createElement('button');
        doneButton.className = 'training_popup-yes-button';
        doneButton.textContent = 'Done';
        doneButton.onclick = () => {
            document.body.removeChild(popupBackground);
            if (yesCallback) yesCallback(); // Trigger the Yes callback
        };
        content.appendChild(doneButton);
    }
    else{
        // Add the Yes button
        const yesButton = document.createElement('button');
        yesButton.className = 'training_popup-yes-button';
        yesButton.textContent = 'Yes';
        yesButton.onclick = () => {
            document.body.removeChild(popupBackground);
            if (yesCallback) yesCallback(); // Trigger the Yes callback
        };
        content.appendChild(yesButton);

        // Add the No button
        const noButton = document.createElement('button');
        noButton.className = 'training_popup-no-button';
        noButton.textContent = 'No';
        noButton.onclick = () => {
            document.body.removeChild(popupBackground);
            if (noCallback) noCallback(); // Trigger the No callback
        };
        content.appendChild(noButton);
    }
        

    // Append content to the popup and popup to the background
    popup.appendChild(content);
    popupBackground.appendChild(popup);
    document.body.appendChild(popupBackground);
}

function deleteAutoDataset() {
    fetch('/delete_auto_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ dataset_name: 'Auto' })  // Add body content if required
    })
    .then(response => {
        if (response.ok) {
            console.log('[INFO] Dataset deletion request successful.');
            return response.json();
        } else {
            throw new Error('Dataset deletion request failed.');
        }
    })
    .then(data => {
        console.log(data.message);
    })
    .catch(error => {
        console.error('[ERROR] ' + error.message);
    });
}



function start_calibration_procedure() {
    sendSelectedModel('disabled');

    const pingSound = new Audio('static/done-sound.mp3');  // Replace with your audio file path

    // Function to play the sound
    function playPing() {
        pingSound.play();
    }

    // Remove any existing calibration popup if present
    const existingCalibrationBackground = document.getElementById('calibration-procedure-background');
    if (existingCalibrationBackground) {
        document.body.removeChild(existingCalibrationBackground);
    }

    // Create a darkened background for the calibration procedure
    const calibrationBackground = document.createElement('div');
    calibrationBackground.className = 'calibration-background';
    calibrationBackground.id = 'calibration-procedure-background';

    // Create a popup container
    const calibrationPopup = document.createElement('div');
    calibrationPopup.className = 'calibration-popup';
    calibrationPopup.id = 'calibration-procedure-popup';

    // Create a container for the content (message and buttons)
    const contentContainer = document.createElement('div');
    contentContainer.className = 'calibration-content-container';

    // Add the big white text message
    const message = document.createElement('h1');
    message.className = 'calibration-message';
    message.textContent = 'Click start to begin calibration procedure';
    contentContainer.appendChild(message);

    // Create a container for the buttons
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'calibration-button-container';

    // Create the Start button
    const startButton = document.createElement('button');
    startButton.className = 'calibration-start-button';
    startButton.textContent = 'Start';
    buttonContainer.appendChild(startButton);

    // Create the Cancel button
    const cancelButton = document.createElement('button');
    cancelButton.className = 'calibration-cancel-button';
    cancelButton.textContent = 'Cancel';
    buttonContainer.appendChild(cancelButton);

    // Append the button container to the content container
    contentContainer.appendChild(buttonContainer);

    // Append the content container to the calibration popup
    calibrationPopup.appendChild(contentContainer);

    // Append the popup to the darkened background
    calibrationBackground.appendChild(calibrationPopup);

    // Add the darkened background and popup to the body
    document.body.appendChild(calibrationBackground);

    // Event handler for Cancel button
    cancelButton.onclick = () => {
        console.log('Calibration canceled');
        document.body.removeChild(calibrationBackground);
    };

    // Start the calibration capture when the user clicks the Start button
    startButton.onclick = () => {
        // Hide the Start and Cancel buttons
        startButton.style.display = 'none';
        cancelButton.style.display = 'none';

        startCalibrationCapture();
    };

    // Shared variables for progress display
    let stepProgressDiv;
    let progressDiv;

    async function startCalibrationCapture() {
        deleteAutoDataset();

        // Create a div to show step progress
        stepProgressDiv = document.createElement('div');
        stepProgressDiv.className = 'calibration-step-progress';
        contentContainer.appendChild(stepProgressDiv);

        // Create a div to show image capture progress
        progressDiv = document.createElement('div');
        progressDiv.className = 'calibration-progress';
        contentContainer.appendChild(progressDiv);

        let image_count = 200;

        const steps = [
            {
                totalImages: image_count,
                message: 'Look at the camera and DO NOT BLINK...',
                payload: {
                    datasetMode: 'new',
                    datasetName: 'Auto',
                    cameraDirection: 'lookingAtCamera'
                }
            },
            {
                totalImages: image_count / 2,
                message: 'Turn your eyes to the left of your screen and DO NOT BLINK.',
                payload: {
                    datasetMode: 'existing',
                    datasetName: 'Auto',
                    cameraDirection: 'awayFromCamera'
                }
            },
            {
                totalImages: image_count / 2,
                message: 'Turn your eyes to the right of your screen and DO NOT BLINK.',
                payload: {
                    datasetMode: 'existing',
                    datasetName: 'Auto',
                    cameraDirection: 'awayFromCamera'
                }
            }
        ];

        let currentStepIndex = 0;

        async function proceedToNextStep() {
            playPing();
            const popupMessage = document.querySelector('.calibration-message');
            if (currentStepIndex < steps.length) {
                const step = steps[currentStepIndex];

                // Update the popup message
                popupMessage.textContent = step.message;

                // Update step progress
                stepProgressDiv.textContent = `Step ${currentStepIndex + 1} of ${steps.length}`;

                // Show the Start and Cancel buttons
                startButton.style.display = 'inline-block';
                cancelButton.style.display = 'inline-block';

                // Update Start button onclick to start capturing images for this step
                startButton.onclick = async () => {
                    // Hide the Start and Cancel buttons
                    startButton.style.display = 'none';
                    cancelButton.style.display = 'none';

                    await captureImages(step.totalImages, step.payload);

                    currentStepIndex++;
                    await proceedToNextStep(); // Proceed to next step
                };

                // Update Cancel button onclick
                cancelButton.onclick = () => {
                    console.log('Calibration canceled');
                    document.body.removeChild(calibrationBackground);
                };
            } else {
                // All steps completed, start retraining model step
                await startRetrainingModel();
            }
        }

        await proceedToNextStep(); // Start the first step
    }

    async function captureImages(totalImages, payload) {
        const progressDiv = document.querySelector('.calibration-progress');

        for (let i = 0; i < totalImages; i++) {
            progressDiv.textContent = `Capturing Image ${i + 1}/${totalImages}`;
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
                break; // Exit the loop if there's an error
            }
        }

        // Clear the progress text after capturing images
        progressDiv.textContent = '';
    }
    
    // Function to handle retraining model step
    async function startRetrainingModel() {
        // Update the popup message
        const popupMessage = document.querySelector('.calibration-message');
        popupMessage.textContent = 'Applying Calibration...';
    
        // Clear previous step progress
        stepProgressDiv.textContent = '';
        // Show progressDiv
        progressDiv.innerHTML = 'Progress: 0% <div class="loading-spinner"></div>'; // Add the spinner
    
        // Start retraining by making a POST request to the backend
        try {
            const startRetrainingResponse = await fetch('/start_retraining', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
    
            if (!startRetrainingResponse.ok) {
                const errorData = await startRetrainingResponse.json();
                console.error('Failed to start retraining:', errorData.message);
                return;
            }
    
            console.log('Retraining started successfully!');
    
            // Now start checking /get_calibration_progress for retraining progress
            let progress = 0;
            while (progress < 100) {
                const progressResponse = await fetch('/get_calibration_progress');
                if (progressResponse.ok) {
                    const data = await progressResponse.json();
                    progress = data.progress; // Assuming the backend returns {progress: <number>}
                    const calibrationMessage = data.calibration_message; // Assuming backend sends this
                    progressDiv.textContent = `${calibrationMessage}: ${progress}%`;
                    
                    // Append the spinner again
                    progressDiv.innerHTML = `
                                            <div style="display: flex; flex-direction: column; align-items: center;">
                                                <div class="loading-spinner"></div>
                                                <div>${calibrationMessage}: ${progress}%</div>
                                            </div>
                                        `;

                } else {
                    console.error('Failed to get calibration progress');
                }
                await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds before next check
            }
    
            // Retraining complete
            progressDiv.textContent = 'Retraining complete!';
            console.log('Retraining process finished!');
    
            // Play the ping sound to indicate completion
    
            // Optionally, remove the calibration popup after completion
            setTimeout(() => {
                document.body.removeChild(calibrationBackground);
            }, 2000); // Remove after 2 seconds
    
        } catch (error) {
            console.error('Error during retraining process:', error);
        }
    }
    

}

// Event listener for the calibration button
document.getElementById('calibration-btn').addEventListener('click', () => {
    if (!cameraOn) {
        alert('Please turn on the camera to start the calibration procedure.');
        return; // Prevent further execution if the camera is off
    }
    
    // Proceed with showing the popup if the camera is on
    createPopup(
        'calibration-popup',
        'Do you want to start the calibration procedure?',
        start_calibration_procedure, // Call the calibration procedure if Yes is clicked
        () => {
            console.log('Calibration canceled');
        }
    );
});







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
                        
                        createPopup('training-started-popup', 'Starting training, this can take up to 5 minutes please wait...');
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
                    
                    if (data.error) {
                        console.log('Error:', data.error);
                        // You can optionally show an error message or handle this case differently
                        return;
                    }
                
                    const progress = data.progress;
                    const epoch_count = data.epoch_count; 
                    const total_epochs = data.total_epochs;
                    const inputImage = data.input_image;
                    const targetImage = data.target_image;
                    const predictedImage = data.predicted_image;
                    const generatorLoss = data.generator_loss;
                    const discriminatorLoss = data.discriminator_loss;
                
                    // Check if all images are provided, do nothing if they are missing
                    if (inputImage && targetImage && predictedImage) {
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
                    } else {
                        console.log('Checkpoint images are missing, skipping update.');
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

        if (!cameraOn) {
            alert("Please turn on camera first.");
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
    
        let nr_images = await showPopup(`
            You will take 3 sets of pictures.<br><br>
            <strong>Notes:</strong><br>
            - There will be a pause between each set.<br>
            - <strong>Do Not Blink</strong> or move during capture!<br><br>
            Please enter the number of images per set:
        `, 'Continue to Set 1', true);
    
        if (!nr_images) {
            alert('Please enter a valid number of images.');
            return;
        }
    
        const captureSet = async (cameraDirection, setDescription) => {
            // Create the red circle if it doesn't exist yet
            let bigRedCircle = document.getElementById('bigRedCircle');
            if (!bigRedCircle) {
                bigRedCircle = document.createElement('div');
                bigRedCircle.id = 'bigRedCircle';
                bigRedCircle.style.position = 'fixed';
                bigRedCircle.style.width = '100px';
                bigRedCircle.style.height = '100px';
                bigRedCircle.style.borderRadius = '50%';
                bigRedCircle.style.backgroundColor = 'red';
                bigRedCircle.style.zIndex = '999';
                document.body.appendChild(bigRedCircle);
            }

            // Show and position the red circle based on the camera direction
            if (cameraDirection === 'lookingAtCamera') {
                bigRedCircle.style.display = 'none';  // Hide the red circle when looking at the camera
            } else if (cameraDirection === 'awayFromCamera') {
                bigRedCircle.style.display = 'block';
                bigRedCircle.style.top = '50%';  // Position towards the right for 'awayFromCamera'
                bigRedCircle.style.left = '97%';
                bigRedCircle.style.transform = 'translate(-50%, -50%)';
            } else if (cameraDirection === 'awayFromCameraLeft') {
                bigRedCircle.style.display = 'block';
                bigRedCircle.style.top = '50%';  // Position towards the left for 'awayFromCameraLeft'
                bigRedCircle.style.left = '3%';
                bigRedCircle.style.transform = 'translate(-50%, -50%)';
            }


            await showPopup(setDescription, 'Start');
        
            let payload;
            if (cameraDirection === 'awayFromCameraLeft') {
                payload = {
                    datasetMode: datasetMode,
                    datasetName: datasetName,
                    cameraDirection: 'awayFromCamera'
                };
            } else {
                payload = {
                    datasetMode: datasetMode,
                    datasetName: datasetName,
                    cameraDirection: cameraDirection
                };
            }
        
            
        
            const progressDiv = document.createElement('div');
            progressDiv.className = 'progress-display';
            progressDiv.style.position = 'fixed';
            progressDiv.style.backgroundColor = 'black';
            progressDiv.style.color = 'white';
            progressDiv.style.padding = '10px 20px';
            progressDiv.style.fontSize = '24px';
            progressDiv.style.zIndex = '1000';
        
            // Adjust position of the progressDiv based on camera direction
            if (cameraDirection === 'lookingAtCamera') {
                progressDiv.style.top = '1vh';
                progressDiv.style.left = '50vw';
                progressDiv.style.transform = 'translateX(-50%)';
            } else if (cameraDirection === 'awayFromCamera') {
                progressDiv.style.top = '55vh';
                progressDiv.style.left = '100vw';
                progressDiv.style.transform = 'translateX(-80%)';
            } else {
                progressDiv.style.top = '55vh';
                progressDiv.style.left = '0vw';
                progressDiv.style.transform = 'translateX(0)';
            }
        
            document.body.appendChild(progressDiv);
        
            if (cameraDirection !== 'lookingAtCamera'){
                total_images = nr_images/2;
            }else{
                total_images = nr_images;
            }

            for (let i = 0; i < total_images; i++) {
                progressDiv.textContent = `Image ${i + 1}/${total_images}`;
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
                }
            }

            bigRedCircle.style.display = 'none';  // Hide the red circle when looking at the camera
        
            document.body.removeChild(progressDiv);
            
            if (cameraDirection === 'awayFromCamera') {
                console.log("Running third set");
                await captureSet('awayFromCameraLeft', '<strong>Set 3:</strong><br> Keep your head still and point your eyes at the RED DOT on the LEFT.');
            }
        };
    
        await captureSet('lookingAtCamera', '<strong>Set 1:</strong><br> Direct your at the screen and keep your eyes pointed at the camera. Do Not Blink!');
        await captureSet('awayFromCamera', '<strong>Set 2:</strong><br> Keep your head still and point your eyes in at the red dot on the RIGHT.');
        alert("Dataset Captured Successfully")
    }
    
    // const statusMessage = document.getElementById('statusMessage');
    const contextMenu = document.getElementById('contextMenu');
    
    // Add event listener to show context menu on right-click
    statusMessage.addEventListener('contextmenu', (event) => {
      event.preventDefault(); // Prevent default context menu
      showContextMenu(event);
    });
    
    // Add event listener to handle context menu clicks
    contextMenu.addEventListener('click', (event) => {
      if (event.target.classList.contains('context-menu-item')) {
        const cameraIndex = event.target.getAttribute('data-index');
        setCamera(cameraIndex);
        hideContextMenu();
      }
    });
    
    // Hide the context menu when clicking outside
    window.addEventListener('click', (event) => {
      if (event.target !== contextMenu && !contextMenu.contains(event.target)) {
        hideContextMenu();
      }
    });
    
    function showContextMenu(event) {
        event.preventDefault();
      
        // Get the height of the context menu
        const menuHeight = contextMenu.offsetHeight;
        const menuWidth = contextMenu.offsetWidth;
      
        // Calculate available space below the cursor
        const availableSpaceBelow = window.innerHeight - event.pageY;
        const availableSpaceRight = window.innerWidth - event.pageX;
      
        // Default to showing the menu below the click
        let top = event.pageY;
        let left = event.pageX;
      
        // If there's not enough space below, show it above the cursor
        if (availableSpaceBelow < menuHeight) {
          top = event.pageY - menuHeight;
        }
      
        // If there's not enough space to the right, show it to the left
        if (availableSpaceRight < menuWidth) {
          left = event.pageX - menuWidth;
        }
      
        // Apply the calculated position
        contextMenu.style.display = 'block';
        contextMenu.style.left = `${left}px`;
        contextMenu.style.top = `${top}px`;
      }
    
    // Function to hide the context menu
    function hideContextMenu() {
      contextMenu.style.display = 'none';
    }
    
    // Function to send the selected camera index to the backend
    function setCamera(index) {
        console.log("Setting camera: ", index)
        if (cameraOn){
            toggleCamera();
        }
    
        fetch('/set_camera', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ cameraIndex: index }),
        })
        .then(response => response.json())
        .then(data => {
          toggleCamera();
          console.log('Camera set successfully:', data);
        })
        .catch(error => {
          console.error('Error setting camera:', error);
          alert("Error: Camera not found, try a different camera.")
        });
    }
    

    async function toggleCamera() {
        console.log('toggleCamera function called');
        const statusMessage = document.getElementById('statusMessage');

        // Clear existing status classes
        statusMessage.className = '';

        statusMessage.classList.add('disabled');  // Add disabled class to indicate processing
        statusMessage.classList.add('info'); // Add info class

        // Start the animation
        startLoadingAnimation(statusMessage, '');

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
                
                
                statusMessage.classList.remove('info');  // Remove info class
                statusMessage.classList.add('success');  // Add success class

                // Refresh live view parts
                console.log("my toggle camera before: ", cameraOn)
                if (result.status === 'On'){
                    cameraOn =  true;
                    sendSelectedModel(selectedModel);
                    statusMessage.textContent = '🛑';
                    toggleCorrectionSelectModel();
                }else{
                    cameraOn =  false;
                    statusMessage.textContent = '📷';
                    toggleCorrectionSelectModel();
                }
                                



                console.log("my toggle camera after: ", cameraOn)
                refreshLiveViews();

            } else {
                const errorData = await response.json();
                console.log('Response error:', errorData);
                // statusMessage.textContent = `Failed to toggle camera: ${errorData.error}`;  // Update status message
                statusMessage.textContent = `ERROR`;
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

    function startLoadingAnimation(element, baseText) {
        // Create a container to hold the baseText and loading icon
        const loadingContainer = document.createElement('span');
        loadingContainer.textContent = baseText;
    
        // Create the loading icon element
        const loadingIcon = document.createElement('div');
        loadingIcon.classList.add('loading-icon');
    
        // Append the loading icon to the container
        loadingContainer.appendChild(loadingIcon);
    
        // Clear any existing content in the element and append the new container
        element.innerHTML = '';
        element.appendChild(loadingContainer);
    
        // Store the interval ID globally for later use if needed
        animationInterval = setInterval(() => {
            // Animation logic handled by CSS, so this is a placeholder to keep consistency
        }, 1000);  // Adjust interval if needed, but the CSS handles the animation
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
                statusMessage.textContent = `${result.camera_on ? '🛑' : '📷'}`; // Camera emoji for "On" and crossed circle for "Off"
                statusMessage.style.fontSize = '48px';
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
