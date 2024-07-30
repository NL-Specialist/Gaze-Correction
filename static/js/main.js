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
                        left_eye_arrayBuffer = data.frame['left_eye'];
                        right_eye_arrayBuffer = data.frame['right_eye'];

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

// Function to update the select options
function updateDatasetOptions() {
    // Create a new XMLHttpRequest object
    const xhr = new XMLHttpRequest();

    // Configure it: GET-request for the URL /backend/get_datasets
    xhr.open('GET', '/backend/get_datasets', true);

    // Set up the callback function for when the request completes
    xhr.onload = function() {
        if (xhr.status === 200) {
            // Parse the JSON response
            const newOptions = JSON.parse(xhr.responseText);

            // Get the select element by its id
            const selectElement = document.getElementById('existingDatasets');

            // Clear the existing options
            selectElement.innerHTML = '';

            // Add new options to the select element
            newOptions.forEach(function(option) {
                // Create a new option element
                const newOption = document.createElement('option');
                // Set the value and text content
                newOption.value = option.value;
                newOption.textContent = option.text;
                // Append the new option to the select element
                selectElement.appendChild(newOption);
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



document.addEventListener('DOMContentLoaded', (event) => {
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

    async function captureImages(nr_images = 1) {
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
    
        // Check if the popup already exists
        if (document.querySelector('.capture_image_popup') || document.querySelector('.capture_image_popup-background')) {
            return;
        }
    
        // Create a popup for camera direction
        const cameraDirection = await new Promise((resolve) => {
            const popup = document.createElement('div');
            popup.className = 'capture_image_popup';
    
            const background = document.createElement('div');
            background.className = 'capture_image_popup-background';
    
            const content = document.createElement('div');
            content.className = 'capture_image_popup-content';
    
            const question = document.createElement('p');
            question.textContent = 'Are you looking at the camera or away from the camera?';
            content.appendChild(question);
    
            const lookingAtCameraOption = document.createElement('input');
            lookingAtCameraOption.type = 'radio';
            lookingAtCameraOption.name = 'cameraDirection';
            lookingAtCameraOption.value = 'lookingAtCamera';
            lookingAtCameraOption.id = 'lookingAtCamera';
            const lookingAtCameraLabel = document.createElement('label');
            lookingAtCameraLabel.htmlFor = 'lookingAtCamera';
            lookingAtCameraLabel.textContent = 'Looking at the camera';
            
            const lookingAtCameraDiv = document.createElement('div');
            lookingAtCameraDiv.className = 'radio-group';
            lookingAtCameraDiv.appendChild(lookingAtCameraOption);
            lookingAtCameraDiv.appendChild(lookingAtCameraLabel);
            content.appendChild(lookingAtCameraDiv);
    
            const awayFromCameraOption = document.createElement('input');
            awayFromCameraOption.type = 'radio';
            awayFromCameraOption.name = 'cameraDirection';
            awayFromCameraOption.value = 'awayFromCamera';
            awayFromCameraOption.id = 'awayFromCamera';
            const awayFromCameraLabel = document.createElement('label');
            awayFromCameraLabel.htmlFor = 'awayFromCamera';
            awayFromCameraLabel.textContent = 'Away from the camera';
            
            const awayFromCameraDiv = document.createElement('div');
            awayFromCameraDiv.className = 'radio-group';
            awayFromCameraDiv.appendChild(awayFromCameraOption);
            awayFromCameraDiv.appendChild(awayFromCameraLabel);
            content.appendChild(awayFromCameraDiv);
    
            const confirmButton = document.createElement('button');
            confirmButton.textContent = 'Take Picture';
            confirmButton.className = 'capture_image_popup-confirm-button';
            confirmButton.onclick = () => {
                const selectedOption = document.querySelector('input[name="cameraDirection"]:checked');
                if (selectedOption) {
                    resolve(selectedOption.value);
                    document.body.removeChild(popup);
                    document.body.removeChild(background);
                } else {
                    alert('Please select an option.');
                }
            };
            content.appendChild(confirmButton);
    
            popup.appendChild(content);
            document.body.appendChild(background);
            document.body.appendChild(popup);
        });
    
        const payload = {
            datasetMode: datasetMode,
            datasetName: datasetName,
            cameraDirection: cameraDirection // Add this line to include the new data
        };
    
        console.log('Sending payload:', payload);
    
        const createDatasetElement = document.getElementById('CreateDataset');
    
        // Create a flash div and append it to the createDatasetElement
        var flashDiv = '';
        if (flash_status === 'on')
        {
            flashDiv = document.createElement('div');
            flashDiv.className = 'flash';
            createDatasetElement.appendChild(flashDiv);
        }
    
        // Helper function to ensure minimum duration
        const minimumDuration = (duration) => {
            return new Promise(resolve => setTimeout(resolve, duration));
        };
    
        for (let i = 0; i < nr_images; i++) {
            try {
                // Wait for at least 500ms before making the API call
                // await minimumDuration(500);
    
                // Make the API call
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
                // Mimic camera flashes
                if (flash_status === 'on')
                {
                    setTimeout(() => {
                        flashDiv.classList.remove('flash');
                        setTimeout(() => {
                            flashDiv.classList.add('flash');
                            setTimeout(() => {
                                flashDiv.classList.remove('flash');
                                if (i === nr_images - 1) { // Remove flashDiv only after the last image is captured
                                    createDatasetElement.removeChild(flashDiv);
                                }
                            }, 100); // Duration of the second flash
                        }, 100); // Interval between the two flashes
                    }, 100); // Duration of the first flash
                }      
            }
        }
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
